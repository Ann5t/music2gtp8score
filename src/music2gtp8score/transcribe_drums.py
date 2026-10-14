"""Drum transcription from separated stems.

Input is the MSST 53-stem output: one wav per instrument group (kick, snare,
hh, toms, ...). With per-piece stems the transcription problem collapses to
onset detection + velocity estimation on nearly-solo signals; no ML needed
for the baseline. Toms are split into pitch classes by spectral peak.

Backend interface: ``transcribe_stems(stems_dir, rules) -> list[DrumEvent]``.
Heavy deps (librosa/numpy) are imported lazily so the core package stays light.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .layout import LayoutRules

# spectral-peak band (Hz) used to tell toms apart on the shared "toms" stem
_TOM_BANDS = [  # (low_hz, high_hz) -> drum key; first match wins
    ((200, 400), "tom_very_high"),
    ((150, 200), "tom_high"),
    ((110, 150), "tom_mid"),
    ((80, 110), "tom_low"),
    ((40, 80), "tom_very_low"),
]


@dataclass
class DrumEvent:
    time: float       # seconds
    drum: str         # key into LayoutRules.drum_map
    velocity: int     # 1..127


def _detect_onsets(y, sr, hop: int = 512):
    import librosa
    import numpy as np

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop, onset_envelope=onset_env,
        backtrack=True, delta=0.06, wait=3, pre_max=3, post_max=3, pre_avg=3, post_avg=5,
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    strengths = onset_env[np.clip(frames, 0, len(onset_env) - 1)]
    return times, strengths, frames


def _velocities(strengths, target: int) -> list[int]:
    import numpy as np

    if len(strengths) == 0:
        return []
    lo, hi = np.percentile(strengths, [10, 95])
    span = max(hi - lo, 1e-6)
    return [int(np.clip(target * (0.55 + 0.75 * (s - lo) / span), 1, 127)) for s in strengths]


def _tom_key(y, sr, onset_time: float, window: float = 0.09) -> str:
    """Classify one tom hit by the peak frequency of its first ~90 ms."""
    import numpy as np

    i0 = int(onset_time * sr)
    seg = y[i0:i0 + int(window * sr)]
    if len(seg) < 256:
        return "tom_mid"
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    peak = float(freqs[int(np.argmax(spec))]) if spec.size else 0.0
    for (lo, hi), key in _TOM_BANDS:
        if lo <= peak < hi:
            return key
    return "tom_mid" if 80 <= peak < 200 else ("tom_very_low" if peak < 80 else "tom_very_high")


def transcribe_stems(stems_dir: str | Path, rules: LayoutRules,
                     hh_key: str = "hihat_closed") -> list[DrumEvent]:
    """Onset-detect every mapped stem; merge into one sorted event list."""
    import librosa

    stems_dir = Path(stems_dir)
    events: list[DrumEvent] = []
    for category, stem_names in rules.stem_map.items():
        for stem in stem_names:
            wav = stems_dir / f"{stem}.wav"
            if not wav.exists():
                print(f"[transcribe] stem missing, skipped: {wav.name}")
                continue
            y, sr = librosa.load(wav, sr=22050, mono=True)
            times, strengths, _ = _detect_onsets(y, sr)
            for t, v in zip(times, _velocities(strengths, rules.velocity)):
                if category == "toms":
                    key = _tom_key(y, sr, float(t))
                elif category == "hihat":
                    key = hh_key
                else:
                    key = category
                events.append(DrumEvent(time=float(t), drum=key, velocity=v))
            print(f"[transcribe] {wav.name}: {len(times)} onsets")
    events.sort(key=lambda e: e.time)
    return _merge_close(events, min_gap=0.03)


def _merge_close(events: list[DrumEvent], min_gap: float) -> list[DrumEvent]:
    """Drop same-drum hits closer than min_gap (onset-detector double fires)."""
    out: list[DrumEvent] = []
    last: dict[str, float] = {}
    for e in events:
        if e.time - last.get(e.drum, -1e9) >= min_gap:
            out.append(e)
            last[e.drum] = e.time
    return out
