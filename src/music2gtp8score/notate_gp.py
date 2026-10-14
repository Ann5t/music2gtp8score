"""Notation layer: build an alphaTab score model (JSON) from quantized drum
events, then export a Guitar Pro 7/8 (.gp) file through alphaTab's
Gp7Exporter (tools/build_gp.mjs, Node.js).

The JSON schema mirrors what alphaTab's JsonConverter produces/consumes
(lowercase keys), learned from real GP7 files written by Guitar Pro.
Voices follow the house style: voice 0 = cymbals, voice 1 = drums,
voices 2/3 empty — exactly like the reference scores.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .layout import LayoutRules
from .quantize import QuantizedNote, TempoMap

TICKS_PER_QUARTER = 960
_TICKS = {1: 4 * TICKS_PER_QUARTER, 2: 2 * TICKS_PER_QUARTER, 4: TICKS_PER_QUARTER,
          8: TICKS_PER_QUARTER // 2, 16: TICKS_PER_QUARTER // 4}

# slot-run length -> (duration enum, dots); a 16-slot bar run is a whole note
_RUNS = {16: (1, 0), 12: (2, 1), 8: (2, 0), 6: (4, 1), 4: (4, 0), 3: (8, 1), 2: (8, 0), 1: (16, 0)}


def _art(a_id: int, element: str, staffline: int, head: int, midi: int,
         head_half: int | None = None, head_whole: int | None = None) -> dict:
    return {"id": a_id, "elementtype": element, "staffline": staffline,
            "noteheaddefault": head, "noteheadhalf": head_half if head_half is not None else head,
            "noteheadwhole": head_whole if head_whole is not None else head,
            "techniquesymbol": -1, "techniquesymbolplacement": 3, "outputmidinumber": midi}


# palette order defines the percussionarticulation index used on notes
PALETTE: dict[str, dict] = {
    "kick":          _art(36, "Kick Drum", 7, 57508, 36, 57507, 57506),
    "snare":         _art(38, "Snare", 3, 57508, 38, 57507, 57506),
    "hihat_closed":  _art(42, "Charley", -1, 57513, 42),
    "hihat_open":    _art(46, "Charley", -1, 57523, 46),
    "hihat_pedal":   _art(44, "Charley", 9, 57513, 44),
    "crash":         _art(49, "Crash High", -2, 57592, 49),
    "ride":          _art(51, "Ride", 0, 57513, 51),
    "tom_very_high": _art(50, "Tom Very High", 1, 57508, 50, 57507, 57506),
    "tom_high":      _art(48, "Tom High", 2, 57508, 48, 57507, 57506),
    "tom_mid":       _art(47, "Tom Medium", 4, 57508, 47, 57507, 57506),
    "tom_low":       _art(45, "Tom Low", 5, 57508, 45, 57507, 57506),
    "tom_very_low":  _art(43, "Tom Very Low", 6, 57508, 43, 57507, 57506),
}
_PALETTE_ORDER = list(PALETTE)

_STYLESHEET = {
    "hidedynamics": True, "bracketextendmode": 0, "usesystemsignseparator": False,
    "globaldisplaytuning": True, "globaldisplaychorddiagramsontop": True,
    "globaldisplaychorddiagramsinscore": False, "singletracktracknamepolicy": 1,
    "multitracktracknamepolicy": 1, "firstsystemtracknamemode": 1, "othersystemstracknamemode": 1,
    "firstsystemtracknameorientation": 1, "othersystemtracknameorientation": 1,
    "multitrackmultibarrest": False, "extendbarlines": False, "hideemptystaves": False,
    "hideemptystavesinfirstsystem": False, "showsinglestaffbrackets": False, "barnumberdisplay": 0,
}

_HEADER_FOOTER = {
    "0": {"template": "%TITLE%", "isvisible": True, "textalign": 1},
    "1": {"template": "%SUBTITLE%", "isvisible": True, "textalign": 1},
    "2": {"template": "%ARTIST%", "isvisible": True, "textalign": 1},
    "3": {"template": "%ALBUM%", "isvisible": True, "textalign": 1},
    "4": {"template": "Words by %WORDS%", "isvisible": True, "textalign": 0},
    "5": {"template": "Music by %MUSIC%", "isvisible": True, "textalign": 2},
    "6": {"template": "Words & Music by %MUSIC%", "isvisible": True, "textalign": 2},
    "7": {"template": "Tabbed by %TABBER%", "isvisible": False, "textalign": 2},
    "8": {"template": "%COPYRIGHT%", "isvisible": True, "textalign": 1},
    "9": {"template": "All Rights Reserved - International Copyright Secured", "isvisible": True, "textalign": 1},
}


@dataclass
class _Ids:
    next: int = 0

    def __call__(self) -> int:
        self.next += 1
        return self.next


def _note(ids: _Ids, drum_key: str, velocity: int) -> dict:
    return {
        "id": ids(), "accentuated": 0, "bendtype": 0, "bendstyle": 0, "iscontinuedbend": False,
        "fret": -1, "string": -1, "showstringnumber": False, "octave": -1, "tone": -1,
        "percussionarticulation": _PALETTE_ORDER.index(drum_key), "isvisible": True,
        "islefthandtapped": False, "ishammerpullorigin": False, "isslurdestination": False,
        "harmonictype": 0, "harmonicvalue": 0, "isghost": False, "isletring": False,
        "ispalmmute": False, "isdead": False, "isstaccato": False, "slideintype": 0,
        "slideouttype": 0, "vibrato": 0, "istiedestination": False, "lefthandfinger": -2,
        "righthandfinger": -2, "trillvalue": -1, "trillspeed": 32, "durationpercent": 1,
        "accidentalmode": 2, "dynamics": 4, "ornament": 0,
    }


def _beat(ids: _Ids, start_ticks: int, duration: int, dots: int, notes: list[dict]) -> dict:
    ticks = _TICKS[duration] * (3 if dots else 2) // 2
    return {
        "id": ids(), "notes": notes, "isempty": not notes, "whammystyle": 0, "ottava": 2,
        "islegatoorigin": False, "duration": duration, "automations": [], "dots": dots,
        "fade": 0, "lyrics": None, "pop": False, "slap": False, "tap": False, "text": None,
        "slashed": False, "deadslapped": False, "brushtype": 0, "brushduration": 0,
        "tupletdenominator": -1, "tupletnumerator": -1, "iscontinuedwhammy": False,
        "whammybartype": 0, "vibrato": 0, "chordid": None, "gracetype": 0, "pickstroke": 0,
        "crescendo": 0, "displaystart": start_ticks, "playbackstart": start_ticks,
        "displayduration": ticks, "playbackduration": ticks, "golpe": 0, "dynamics": 4,
        "invertbeamdirection": False, "preferredbeamdirection": None, "beamingmode": 0,
        "wahpedal": 0, "barrefret": -1, "barreshape": 0, "rasgueado": 0,
        "showtimer": False, "timer": None,
    }


def _runs(length: int) -> list[int]:
    """Split a slot run into notatable pieces (16 12 8 6 4 3 2 1), greedy."""
    out = []
    for d in (16, 12, 8, 6, 4, 3, 2, 1):
        while length >= d:
            out.append(d)
            length -= d
    return out


def _voice_beats(ids: _Ids, grid: dict[int, list[QuantizedNote]], resolution: int) -> list[dict]:
    """Turn {slot: [notes]} into beats: hits capped at a quarter, rests in gaps."""
    beats, s = [], 0
    slots = sorted(grid)
    while s < resolution:
        nxt = min((k for k in slots if k > s), default=resolution)
        if s in grid:
            run = min(nxt - s, 4) if nxt == resolution else nxt - s
        else:
            run = nxt - s
        for d in _runs(run):
            duration, dots = _RUNS[d]
            start = sum(_TICKS[b["duration"]] * (3 if b["dots"] else 2) // 2 for b in beats)
            if s in grid:
                notes = [_note(ids, n.drum, n.velocity) for n in grid[s]]
            else:
                notes = []
            beats.append(_beat(ids, start, duration, dots, notes))
            s += d
    return beats


def build_model(notes: list[QuantizedNote], tempo_map: TempoMap, rules: LayoutRules,
                title: str = "", artist: str = "") -> dict:
    """Build the alphaTab score-model JSON (schema compatible with jsonToScore)."""
    ids = _Ids()
    resolution = rules.grid_resolution

    grids: dict[int, dict[int, dict[int, list[QuantizedNote]]]] = {}
    for n in notes:
        if n.drum not in PALETTE:
            continue
        voice = rules.voice_of(n.drum)
        grids.setdefault(n.bar, {}).setdefault(voice, {}).setdefault(n.slot, []).append(n)
    n_bars = max(grids, default=-1) + 1

    sections = {int(s["bar"]): str(s["text"]) for s in rules.sections_manual} if rules.sections_enabled else {}

    masterbars = []
    for i in range(n_bars):
        mb = {"alternateendings": 0, "isdoublebar": False, "isrepeatstart": False, "repeatcount": 0,
              "timesignaturenumerator": tempo_map.beats_per_bar, "timesignaturedenominator": 4,
              "timesignaturecommon": False, "isfreetime": False, "tripletfeel": 0,
              "tempoautomations": [], "start": i * resolution * TICKS_PER_QUARTER // 4,
              "isanacrusis": False, "displayscale": 1, "displaywidth": -1}
        if i == 0:
            mb["tempoautomations"] = [{"islinear": False, "type": 0, "value": round(tempo_map.tempo, 2),
                                       "ratioposition": 0, "text": "", "isvisible": True}]
        if i in sections:
            mb["section"] = {"marker": chr(ord("A") + len([k for k in sections if k < i])), "text": sections[i]}
        masterbars.append(mb)

    bars = []
    for i in range(n_bars):
        voice_grids = grids.get(i, {})
        voices = []
        for vi in range(4):  # house style: 0 cymbals, 1 drums, 2/3 always empty
            beats = _voice_beats(ids, voice_grids.get(vi, {}), resolution) if vi in voice_grids else []
            if vi in (rules.cymbal_voice, rules.drum_voice) and vi not in voice_grids:
                beats = _voice_beats(ids, {}, resolution)  # whole-bar rest keeps the voice alive
            voices.append({"id": ids(), "beats": beats})
        bars.append({"id": i, "clef": 0, "clefottava": 2, "voices": voices, "similemark": 0,
                     "displayscale": 1, "displaywidth": -1, "sustainpedals": [], "barlineleft": 0,
                     "barlineright": 0, "keysignature": 0, "keysignaturetype": 0})

    return {
        "title": title, "artist": artist, "album": "", "words": "", "music": "",
        "copyright": "", "tab": "", "instructions": "", "notices": "", "subtitle": "",
        "masterbars": masterbars,
        "tracks": [{
            "staves": [{
                "bars": bars, "capo": 0, "transpositionpitch": 0, "displaytranspositionpitch": 0,
                "stringtuning": "", "showslash": False, "shownumbered": False,
                "showtablature": rules.tablature, "showstandardnotation": True,
                "ispercussion": True, "standardnotationlinecount": rules.line_count,
            }],
            "playbackinfo": {"volume": 12, "balance": 8, "port": 0, "program": 0, "bank": 0,
                             "primarychannel": 9, "secondarychannel": 9, "ismute": False, "issolo": False},
            "name": "Drumkit", "isvisibleonmultitrack": True, "shortname": "drm.",
            "defaultsystemslayout": rules.bars_per_system, "systemslayout": [],
            "percussionarticulations": [PALETTE[k] for k in _PALETTE_ORDER],
        }],
        "defaultsystemslayout": rules.bars_per_system, "systemslayout": [rules.bars_per_system],
        "stylesheet": _STYLESHEET, "backingtrack": {}, "style": {"headerandfooter": _HEADER_FOOTER},
    }


def write_gp(model: dict, out_path: str | Path, tools_dir: str | Path | None = None) -> dict:
    """Export the model to .gp via tools/build_gp.mjs; returns the verification summary."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node not found on PATH (needed for the GP7 exporter)")
    tools = Path(tools_dir) if tools_dir else Path(__file__).resolve().parents[2] / "tools"
    script = tools / "build_gp.mjs"
    if not (tools / "node_modules" / "@coderline" / "alphatab").exists():
        raise RuntimeError(f"alphaTab not installed — run: npm install --prefix {tools}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(model, f)
        model_path = f.name
    try:
        proc = subprocess.run([node, str(script), model_path, str(out_path)],
                              check=True, capture_output=True, text=True, encoding="utf-8")
        return json.loads(proc.stdout.strip().splitlines()[-1])
    finally:
        Path(model_path).unlink(missing_ok=True)
