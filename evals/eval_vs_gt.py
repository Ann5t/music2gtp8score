"""Evaluate transcription quality against a ground-truth score.

Usage:
    python evals/eval_vs_gt.py --stems out/smoke/stems/guoyuanchao \
        --gt "C:/Users/dongy/Documents/Kimi Project/Score/extracted/郭源潮/score.json"

Compares pipeline output to a reference score at (bar, 16th-slot, coarse drum
class) level with +/-1 slot tolerance and a small bar-shift search (our bar 0
may be offset from the reference). The ground truth is an alphaTab score.json
produced from a real .gp file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from music2gtp8score.layout import LayoutRules
from music2gtp8score.quantize import estimate_tempo, quantize
from music2gtp8score.transcribe_drums import transcribe_stems

COARSE = {36: "kick", 35: "kick", 38: "snare", 37: "snare", 40: "snare",
          42: "hh", 44: "hh", 46: "hh",
          43: "toms", 45: "toms", 47: "toms", 48: "toms", 50: "toms",
          49: "cym", 57: "cym", 52: "cym", 55: "cym",
          51: "ride", 53: "ride", 59: "ride"}


def coarse(key: str) -> str:
    if key.startswith("tom"):
        return "toms"
    if key in ("hihat_closed", "hihat_open", "hihat_pedal"):
        return "hh"
    if key == "crash":
        return "cym"
    if key == "ride":
        return "ride"
    return key


def gt_notes(gt_path: Path) -> set[tuple[int, int, str]]:
    j = json.loads(gt_path.read_text(encoding="utf-8"))
    arts = j["tracks"][0]["percussionarticulations"]
    out = set()
    for bi, bar in enumerate(j["tracks"][0]["staves"][0]["bars"]):
        for v in bar["voices"]:
            for beat in v["beats"]:
                slot = round(beat["displaystart"] / 240)
                for n in beat["notes"] or []:
                    cls = COARSE.get(arts[n["percussionarticulation"]]["outputmidinumber"])
                    if cls:
                        out.add((bi, slot, cls))
    return out


def our_notes(stems: Path) -> set[tuple[int, int, str]]:
    rules = LayoutRules.from_yaml()
    events = transcribe_stems(stems, rules)
    sources = [stems / n for n in ("kick.wav", "hh.wav") if (stems / n).exists()]
    tempo = estimate_tempo(sources, [e.time for e in events], rules.grid_resolution)
    print(f"tempo: {tempo.tempo:.2f}")
    return {(n.bar, n.slot, coarse(n.drum)) for n in quantize(events, tempo, rules.grid_resolution)}


def f1(gt: set, ours: set, shift: int, tol: int = 1) -> tuple[float, float, float]:
    matched = 0
    ours_by_key: dict[tuple[int, str], list[int]] = {}
    for b, s, c in ours:
        ours_by_key.setdefault((b, c), []).append(s)
    for gb, gs, c in gt:
        for s in ours_by_key.get((gb + shift, c), []):
            if abs(s - gs) <= tol:
                matched += 1
                break
    p = matched / max(len(ours), 1)
    r = matched / max(len(gt), 1)
    return (2 * p * r / max(p + r, 1e-9), p, r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path, help="ground-truth score.json (alphaTab)")
    args = ap.parse_args()

    gt, ours = gt_notes(args.gt), our_notes(args.stems)
    print(f"GT notes: {len(gt)} | ours: {len(ours)}")
    for shift in range(-2, 3):
        f, p, r = f1(gt, ours, shift)
        print(f"bar shift {shift:+d}: F1={f:.3f} precision={p:.3f} recall={r:.3f}")
    for c in ("kick", "snare", "hh", "toms"):
        g = {x for x in gt if x[2] == c}
        o = {x for x in ours if x[2] == c}
        best = max((f1(g, o, s) for s in range(-2, 3)), key=lambda t: t[0])
        print(f"  {c:5s} best: F1={best[0]:.3f} P={best[1]:.3f} R={best[2]:.3f} "
              f"(GT {len(g)}, ours {len(o)})")


if __name__ == "__main__":
    main()
