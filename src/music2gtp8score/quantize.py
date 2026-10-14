"""Tempo estimation and grid quantization."""
from __future__ import annotations

from dataclasses import dataclass

from .transcribe_drums import DrumEvent


@dataclass
class TempoMap:
    tempo: float
    beats_per_bar: int = 4

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.tempo

    @property
    def seconds_per_bar(self) -> float:
        return self.seconds_per_beat * self.beats_per_bar


@dataclass
class QuantizedNote:
    bar: int
    slot: int      # 0..resolution-1 within the bar
    drum: str
    velocity: int


def normalize_tempo(tempo: float, lo: float = 70.0, hi: float = 150.0) -> float:
    """Fold octave errors of tempo trackers into a musical range."""
    while tempo < lo:
        tempo *= 2
    while tempo > hi:
        tempo /= 2
    return tempo


def _flat_prior_tempo(audio_path, hop: int = 512) -> float:
    """librosa dynamic tempo with a nearly-flat prior (the default prior is
    glued to ~120 BPM and drags slow songs up)."""
    import librosa  # lazy heavy dep
    import numpy as np
    from librosa.feature.rhythm import tempo as tempo_fn

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    values = tempo_fn(onset_envelope=onset_env, sr=sr, hop_length=hop,
                      start_bpm=100.0, std_bpm=4.0, aggregate=None)
    if values is None or len(values) == 0:
        raise RuntimeError(f"tempo estimation failed on {audio_path}")
    return normalize_tempo(float(np.median(values)))


def estimate_tempo(audio_paths, event_times: list[float] | None = None,
                   resolution: int = 16) -> TempoMap:
    """Coarse tempo = median of flat-prior estimates over the given sources
    (kick + hh stems are the most reliable); then optionally refine by
    fitting the onset times to the grid."""
    import numpy as np

    if isinstance(audio_paths, (str, bytes)) or not isinstance(audio_paths, (list, tuple)):
        audio_paths = [audio_paths]
    estimates = [_flat_prior_tempo(p) for p in audio_paths]
    coarse = float(np.median(estimates))
    if not event_times:
        return TempoMap(tempo=coarse)
    refined, phase = refine_tempo(np.asarray(event_times), coarse, resolution)
    tm = TempoMap(tempo=refined)
    tm.grid_origin = phase  # type: ignore[attr-defined]
    return tm


def refine_tempo(times, coarse: float, resolution: int = 16,
                 search: float = 0.06, steps: int = 61) -> tuple[float, float]:
    """Fit tempo+phase to the onsets themselves: for each candidate tempo in
    ±search around coarse, pick the grid phase minimizing the median distance
    of onsets to the nearest grid point. Returns (tempo, grid_origin)."""
    import numpy as np

    best = (float("inf"), coarse, 0.0)
    for factor in np.linspace(1 - search, 1 + search, steps):
        tempo = coarse * float(factor)
        sec_per_slot = 60.0 / tempo / (resolution / 4)
        phase = float(np.median(times % sec_per_slot))
        dist = np.abs((times - phase + sec_per_slot / 2) % sec_per_slot - sec_per_slot / 2)
        err = float(np.median(dist))
        if err < best[0]:
            best = (err, tempo, phase)
    return best[1], best[2]


def quantize(events: list[DrumEvent], tempo_map: TempoMap,
             resolution: int = 16, origin: float | None = None) -> list[QuantizedNote]:
    """Snap events to the 16th-note grid. ``origin`` = time of bar 0 beat 1
    (defaults to the tempo-fit grid origin, else the first event)."""
    if not events:
        return []
    if origin is None:
        origin = getattr(tempo_map, "grid_origin", None)
    t0 = events[0].time if origin is None else origin
    slots_per_beat = resolution // tempo_map.beats_per_bar
    sec_per_slot = tempo_map.seconds_per_beat / slots_per_beat
    notes = []
    for e in events:
        pos = max(0, round((e.time - t0) / sec_per_slot))
        bar, slot = divmod(pos, resolution)
        notes.append(QuantizedNote(bar=bar, slot=slot, drum=e.drum, velocity=e.velocity))
    return notes
