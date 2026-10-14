"""Grid-search onset-detection parameters per stem against the reference scores.

Methodology notes:
- quantize at the GROUND-TRUTH tempo (from the reference score) so tempo
  estimation errors don't confound detection quality;
- onset envelopes are computed once per (song, stem), then delta/wait/strength
  filtering are searched cheaply on top;
- score = F1 at (bar, 16th-slot, class) with +/-1 slot tolerance, best over
  bar shifts in [-2, 2], averaged across the 4 reference songs.

Usage:
    python evals/tune_onsets.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GT_BASE = Path(r"C:/Users/dongy/Documents/Kimi Project/Score/extracted")

SONGS = {
    "郭源潮": (REPO / "out/smoke/stems/guoyuanchao", GT_BASE / "郭源潮/score.json"),
    "大石碎胸口": (REPO / "out/大石碎胸口/stems/backing", GT_BASE / "大石碎胸口 造访版/score.json"),
    "杀死那个石家庄人": (REPO / "out/杀死那个石家庄人/stems/backing", GT_BASE / "杀死那个石家庄人 造访版/score.json"),
    "崇明海边": (REPO / "out/我等你们在崇明海边/stems/backing", GT_BASE / "我等你们在崇明海边/score.json"),
}

STEMS = ["kick", "snare", "hh", "toms"]
GRID = {
    "delta": [0.03, 0.05, 0.07, 0.1, 0.15, 0.25],
    "wait": [3, 5, 8],
    "strength_pct": [0, 30, 50, 65],
}
COARSE = {36: "kick", 35: "kick", 38: "snare", 37: "snare", 40: "snare",
          42: "hh", 44: "hh", 46: "hh",
          43: "toms", 45: "toms", 47: "toms", 48: "toms", 50: "toms"}


def gt_notes(gt_path: Path, cls: str) -> tuple[set, float]:
    j = json.loads(gt_path.read_text(encoding="utf-8"))
    tempo = j["masterbars"][0]["tempoautomations"][0]["value"]
    arts = j["tracks"][0]["percussionarticulations"]
    out = set()
    for bi, bar in enumerate(j["tracks"][0]["staves"][0]["bars"]):
        for v in bar["voices"]:
            for beat in v["beats"]:
                slot = round(beat["displaystart"] / 240)
                for n in beat["notes"] or []:
                    if COARSE.get(arts[n["percussionarticulation"]]["outputmidinumber"]) == cls:
                        out.add((bi, slot))
    return out, float(tempo)


def f1(gt: set, ours: set, shift: int, tol: int = 1) -> float:
    ours_by_bar: dict[int, list[int]] = {}
    for b, s in ours:
        ours_by_bar.setdefault(b, []).append(s)
    matched = sum(1 for gb, gs in gt
                  if any(abs(s - gs) <= tol for s in ours_by_bar.get(gb + shift, [])))
    p = matched / max(len(ours), 1)
    r = matched / max(len(gt), 1)
    return 2 * p * r / max(p + r, 1e-9)


def best_f1(gt: set, ours: set) -> float:
    return max(f1(gt, ours, s) for s in range(-2, 3))


def main() -> None:
    import librosa
    import numpy as np

    from music2gtp8score.quantize import refine_tempo

    print("loading songs + computing onset envelopes ...")
    data: dict[str, dict] = {}
    for song, (stems, gt_path) in SONGS.items():
        envs = {}
        for stem in STEMS:
            wav = stems / f"{stem}.wav"
            y, sr = librosa.load(wav, sr=22050, mono=True)
            envs[stem] = (librosa.onset.onset_strength(y=y, sr=sr, hop_length=512), sr)
        gt = {stem: gt_notes(gt_path, stem) for stem in STEMS}
        data[song] = {"envs": envs, "gt": gt}

    def sec_per_slot(tempo: float) -> float:
        return 60.0 / tempo / 4

    for stem in STEMS:
        print(f"\n=== {stem} (grid: {len(GRID['delta']) * len(GRID['wait']) * len(GRID['strength_pct'])} combos)")
        results = []
        for delta in GRID["delta"]:
            for wait in GRID["wait"]:
                for pct in GRID["strength_pct"]:
                    scores = []
                    for song, d in data.items():
                        oe, sr = d["envs"][stem]
                        gt_set, tempo = d["gt"][stem]
                        frames = librosa.onset.onset_detect(
                            onset_envelope=oe, sr=sr, hop_length=512, backtrack=True,
                            delta=delta, wait=wait, pre_max=3, post_max=3, pre_avg=3, post_avg=5)
                        strengths = oe[np.clip(frames, 0, len(oe) - 1)]
                        if pct and len(strengths):
                            thr = np.percentile(strengths, pct)
                            frames = frames[strengths >= thr]
                        times = librosa.frames_to_time(frames, sr=sr, hop_length=512)
                        if len(times) == 0:
                            scores.append(0.0)
                            continue
                        _, phase = refine_tempo(np.asarray(times), tempo)
                        grid = sec_per_slot(tempo)
                        ours = {(int((t - phase) // (grid * 16)), round(((t - phase) / grid) % 16))
                                for t in times if t >= phase}
                        scores.append(best_f1(gt_set, ours))
                    results.append((sum(scores) / len(scores), delta, wait, pct))
        results.sort(reverse=True)
        for mean_f1, delta, wait, pct in results[:5]:
            print(f"  F1={mean_f1:.3f}  delta={delta:<5} wait={wait} strength>={pct}%")


if __name__ == "__main__":
    main()
