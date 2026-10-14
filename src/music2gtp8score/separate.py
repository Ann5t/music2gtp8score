"""Stem separation via MSST (ZFTurbo Music-Source-Separation-Training, MIT).

We do not re-implement inference: we drive the upstream ``inference.py`` with
a Python interpreter that already has torch+CUDA (e.g. the MSST-GUI runtime),
and the user's local checkpoints. Paths are configurable via environment
variables or CLI flags; defaults point at the desktop MSST-GUI install.

Output layout (filename_template ``{file_name}/{instr}``)::

    <out_dir>/<song>/<instr>.wav      # kick.wav snare.wav hh.wav toms.wav ...
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MSST_GUI = Path(os.environ.get(
    "M2G_MSST_GUI", r"C:\Users\dongy\Desktop\MSST-GUI-乔大峰版"))


@dataclass
class MsstConfig:
    python: Path          # interpreter with torch+CUDA (MSST-GUI runtime)
    repo: Path            # cloned ZFTurbo Music-Source-Separation-Training
    model_yaml: Path      # e.g. mvsep_mega_model_bs_roformer_53_stems-fast.yaml
    checkpoint: Path      # e.g. mvsep_mega_model_bs_roformer_53_stems_v1.ckpt
    ffmpeg: Path | None   # for non-wav input conversion
    device_ids: int = 0

    @classmethod
    def from_env(cls) -> MsstConfig:
        ckpts = _DEFAULT_MSST_GUI / "ckpts"
        ffmpeg = _DEFAULT_MSST_GUI / "runtime" / "ffmpeg.exe"
        return cls(
            python=Path(os.environ.get("M2G_MSST_PYTHON", _DEFAULT_MSST_GUI / "runtime" / "python.exe")),
            repo=Path(os.environ.get("M2G_MSST_REPO", _REPO_ROOT / "vendor" / "Music-Source-Separation-Training")),
            model_yaml=Path(os.environ.get("M2G_MSST_YAML", ckpts / "mvsep_mega_model_bs_roformer_53_stems-fast.yaml")),
            checkpoint=Path(os.environ.get("M2G_MSST_CKPT", ckpts / "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt")),
            ffmpeg=ffmpeg if ffmpeg.exists() else shutil.which("ffmpeg"),
        )

    def check(self) -> list[str]:
        problems = []
        for label, p in [("MSST python", self.python), ("MSST repo", self.repo),
                         ("model yaml", self.model_yaml), ("checkpoint", self.checkpoint)]:
            if not p or not Path(p).exists():
                problems.append(f"{label} not found: {p}")
        if not self.ffmpeg:
            problems.append("ffmpeg not found (needed for non-wav input)")
        return problems


def _to_wav(audio_path: Path, work_dir: Path, ffmpeg: Path | None) -> Path:
    """MSST reads wav/flac via soundfile; mp3 etc. go through ffmpeg first."""
    if audio_path.suffix.lower() in {".wav", ".flac"}:
        return audio_path
    if not ffmpeg:
        raise RuntimeError(f"cannot convert {audio_path.suffix} to wav: ffmpeg not found")
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / (audio_path.stem + ".wav")
    subprocess.run([str(ffmpeg), "-y", "-i", str(audio_path), "-ar", "44100", "-ac", "2", str(wav)],
                   check=True, capture_output=True)
    return wav


def separate(audio_path: str | Path, out_dir: str | Path, cfg: MsstConfig | None = None) -> Path:
    """Run MSST 53-stem separation; return the directory containing <instr>.wav files."""
    cfg = cfg or MsstConfig.from_env()
    problems = cfg.check()
    if problems:
        raise RuntimeError("MSST is not usable:\n  " + "\n  ".join(problems))
    audio_path, out_dir = Path(audio_path).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="m2g_in_") as tmp:
        wav = _to_wav(audio_path, Path(tmp), cfg.ffmpeg)
        cmd = [str(cfg.python), "inference.py",
               "--model_type", "bs_roformer",
               "--config_path", str(cfg.model_yaml),
               "--start_check_point", str(cfg.checkpoint),
               "--input_folder", str(wav.parent),
               "--store_dir", str(out_dir),
               "--device_ids", str(cfg.device_ids)]
        print("[separate]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=cfg.repo, check=True)

    stems = out_dir / wav.stem
    if not stems.is_dir():
        raise RuntimeError(f"MSST finished but {stems} is missing")
    return stems
