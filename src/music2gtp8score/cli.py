"""music2gtp8score command line: audio -> .gp score.

    music2gtp8score song.mp3 -o out/song --title 歌名
    music2gtp8score --stems-dir out/song/stems/song ...   # reuse existing MSST stems
    music2gtp8score --check
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

from .layout import LayoutRules
from .separate import MsstConfig, separate

REPO_ROOT = Path(__file__).resolve().parents[2]


def _check() -> int:
    ok = True

    def line(good: bool, msg: str) -> None:
        nonlocal ok
        ok = ok and good
        print(("  OK  " if good else " FAIL ") + msg)

    cfg = MsstConfig.from_env()
    problems = cfg.check()
    line(not problems, "MSST paths" + ("" if not problems else " -> " + "; ".join(problems)))
    if cfg.python.exists():
        import subprocess
        proc = subprocess.run([str(cfg.python), "-c",
                               "import torch;print(torch.cuda.is_available())"],
                              check=False, capture_output=True, text=True)
        line(proc.stdout.strip().endswith("True"), "MSST runtime torch+CUDA")
    for mod in ("librosa", "soundfile", "yaml"):
        line(importlib.util.find_spec(mod) is not None, f"python module: {mod}")
    line(shutil.which("node") is not None, "node on PATH")
    line((REPO_ROOT / "tools" / "node_modules" / "@coderline" / "alphatab").exists(),
         "tools/: @coderline/alphatab (npm install --prefix tools)")
    print("\nready" if ok else "\nmissing pieces — see FAIL lines above")
    return 0 if ok else 1


def run(args: argparse.Namespace) -> int:
    from .notate_gp import build_model, write_gp
    from .quantize import estimate_tempo, quantize
    from .transcribe_drums import transcribe_stems

    audio = Path(args.audio) if args.audio else None
    title = args.title or (audio.stem if audio else Path(args.stems_dir).name)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rules = LayoutRules.from_yaml(args.config)

    if args.stems_dir:
        stems = Path(args.stems_dir)
    else:
        print("[1/4] MSST 53-stem separation ...")
        stems = separate(audio, out_dir / "stems")
    print(f"      stems: {stems}")

    print("[2/4] onset detection per drum stem ...")
    events = transcribe_stems(stems, rules)
    print(f"      {len(events)} drum events")

    tempo_sources = [stems / name for name in ("kick.wav", "hh.wav", "snare.wav")
                     if (stems / name).exists()]
    if not tempo_sources:
        tempo_sources = [next(stems.glob("*.wav"))]
    tempo = estimate_tempo(tempo_sources, [e.time for e in events], rules.grid_resolution)
    print(f"[3/4] tempo ~ {tempo.tempo:.2f} BPM (grid-refined)")

    notes = quantize(events, tempo, rules.grid_resolution)
    model = build_model(notes, tempo, rules, title=title, artist=args.artist)
    gp_path = out_dir / f"{title}.gp"
    summary = write_gp(model, gp_path)
    print(f"[4/4] wrote {gp_path}")
    print(f"      verify: {summary['masterBars']} bars, {summary['notes']} notes, "
          f"{summary['bytes'] / 1024:.0f} KB, title={summary['title']!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="music2gtp8score",
                                description="audio -> Guitar Pro 7/8 (.gp) drum score")
    p.add_argument("audio", nargs="?", help="input audio (mp3/wav/flac ...)")
    p.add_argument("-o", "--out", default="out", help="output directory (default: out/)")
    p.add_argument("--title", default="", help="score title (default: audio file name)")
    p.add_argument("--artist", default="", help="score artist")
    p.add_argument("--config", default=None, help="layout rules yaml (default: config/layout.default.yaml)")
    p.add_argument("--stems-dir", default=None,
                   help="reuse an existing MSST stems directory (skip separation)")
    p.add_argument("--check", action="store_true", help="check all pipeline dependencies and exit")
    args = p.parse_args(argv)

    if args.check:
        return _check()
    if not args.audio and not args.stems_dir:
        p.error("give an audio file or --stems-dir")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
