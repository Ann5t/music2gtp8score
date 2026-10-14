"""Layout rule engine: loads the engraving standard from a YAML config.

Every notation decision the pipeline makes (staff style, voice assignment,
grid, dynamics, barlines, sections, page layout) comes from here, so the
"house style" lives in one explicit, diff-able file. Defaults below mirror
config/layout.default.yaml; keep the two in sync.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "layout.default.yaml"

DEFAULT_DRUM_MAP = {
    "kick": 36, "snare": 38,
    "hihat_closed": 42, "hihat_open": 46, "hihat_pedal": 44,
    "crash": 49, "ride": 51,
    "tom_very_high": 50, "tom_high": 48, "tom_mid": 47,
    "tom_low": 45, "tom_very_low": 43,
}

CYMBAL_KEYS = {"hihat_closed", "hihat_open", "crash", "ride"}
DRUM_KEYS = {"kick", "snare", "hihat_pedal",
             "tom_very_high", "tom_high", "tom_mid", "tom_low", "tom_very_low"}
TOM_KEYS = {"tom_very_high", "tom_high", "tom_mid", "tom_low", "tom_very_low"}


@dataclass
class LayoutRules:
    tablature: bool = False
    line_count: int = 5
    cymbal_voice: int = 0
    drum_voice: int = 1
    grid_resolution: int = 16
    min_gap_slots: int = 1
    show_dynamics: bool = False
    velocity: int = 96
    plain_notes_only: bool = True
    plain_barlines_only: bool = True
    sections_enabled: bool = True
    sections_auto_detect: bool = False
    sections_manual: list[dict] = field(default_factory=list)
    bars_per_system: int = 4
    drum_map: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_DRUM_MAP))
    stem_map: dict[str, list[str]] = field(default_factory=lambda: {
        "kick": ["kick"], "snare": ["snare"], "hihat": ["hh"],
    })
    onset: dict[str, dict] = field(default_factory=lambda: {
        "kick": {"delta": 0.03, "wait": 8, "strength_pct": 0},
        "snare": {"delta": 0.03, "wait": 8, "strength_pct": 0},
        "hihat": {"delta": 0.07, "wait": 8, "strength_pct": 0},
        "toms": {"delta": 0.07, "wait": 8, "strength_pct": 50},
    })

    def voice_of(self, drum_key: str) -> int:
        return self.cymbal_voice if drum_key in CYMBAL_KEYS else self.drum_voice

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> LayoutRules:
        cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8")) or {}
        r = cls()
        staff = cfg.get("staff", {})
        r.tablature = bool(staff.get("tablature", r.tablature))
        r.line_count = int(staff.get("line_count", r.line_count))
        voices = cfg.get("voices", {})
        r.cymbal_voice = int(voices.get("cymbal_voice", r.cymbal_voice))
        r.drum_voice = int(voices.get("drum_voice", r.drum_voice))
        grid = cfg.get("grid", {})
        r.grid_resolution = int(grid.get("resolution", r.grid_resolution))
        r.min_gap_slots = int(grid.get("min_gap_slots", r.min_gap_slots))
        dynamics = cfg.get("dynamics", {})
        r.show_dynamics = bool(dynamics.get("show", r.show_dynamics))
        r.velocity = int(dynamics.get("velocity", r.velocity))
        techniques = cfg.get("techniques", {})
        r.plain_notes_only = bool(techniques.get("plain_notes_only", r.plain_notes_only))
        barlines = cfg.get("barlines", {})
        r.plain_barlines_only = bool(barlines.get("plain_only", r.plain_barlines_only))
        sections = cfg.get("sections", {})
        r.sections_enabled = bool(sections.get("enabled", r.sections_enabled))
        r.sections_auto_detect = bool(sections.get("auto_detect", r.sections_auto_detect))
        r.sections_manual = list(sections.get("manual", r.sections_manual))
        page = cfg.get("page", {})
        r.bars_per_system = int(page.get("bars_per_system", r.bars_per_system))
        if cfg.get("drums"):
            r.drum_map.update({k: int(v) for k, v in cfg["drums"].items()})
        if cfg.get("stems"):
            r.stem_map.update({k: list(v) for k, v in cfg["stems"].items()})
        if cfg.get("onset"):
            for k, v in cfg["onset"].items():
                r.onset.setdefault(k, {}).update({kk: float(vv) if kk == "delta" else vv
                                                  for kk, vv in v.items()})
        return r
