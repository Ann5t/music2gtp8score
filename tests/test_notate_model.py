from music2gtp8score.layout import LayoutRules
from music2gtp8score.notate_gp import _PALETTE_ORDER, PALETTE, build_model
from music2gtp8score.quantize import QuantizedNote, TempoMap


def _rules():
    return LayoutRules.from_yaml()


def test_model_shape_and_counts():
    notes = [
        QuantizedNote(bar=0, slot=0, drum="kick", velocity=100),
        QuantizedNote(bar=0, slot=0, drum="hihat_closed", velocity=90),
        QuantizedNote(bar=0, slot=4, drum="snare", velocity=100),
        QuantizedNote(bar=1, slot=8, drum="crash", velocity=110),
        QuantizedNote(bar=1, slot=8, drum="kick", velocity=100),
    ]
    model = build_model(notes, TempoMap(tempo=82), _rules(), title="t")
    assert model["title"] == "t"
    assert len(model["masterbars"]) == 2
    mb0 = model["masterbars"][0]
    assert mb0["tempoautomations"][0]["value"] == 82
    assert model["masterbars"][1]["tempoautomations"] == []

    track = model["tracks"][0]
    assert track["percussionarticulations"] == [PALETTE[k] for k in _PALETTE_ORDER]
    bars = track["staves"][0]["bars"]
    assert len(bars) == 2
    assert all(len(b["voices"]) == 4 for b in bars)

    def voice_notes(bar, voice):
        return [n for v_idx, v in enumerate(bars[bar]["voices"]) if v_idx == voice
                for b in v["beats"] for n in b["notes"]]

    # house style: cymbals -> voice 0, drums -> voice 1
    assert [n["percussionarticulation"] for n in voice_notes(0, 0)] == [
        _PALETTE_ORDER.index("hihat_closed")]
    assert [n["percussionarticulation"] for n in voice_notes(0, 1)] == [
        _PALETTE_ORDER.index("kick"), _PALETTE_ORDER.index("snare")]
    assert [n["percussionarticulation"] for n in voice_notes(1, 0)] == [
        _PALETTE_ORDER.index("crash")]
    assert [n["percussionarticulation"] for n in voice_notes(1, 1)] == [
        _PALETTE_ORDER.index("kick")]


def test_every_voice_fills_the_bar():
    notes = [QuantizedNote(bar=0, slot=0, drum="kick", velocity=100)]
    model = build_model(notes, TempoMap(tempo=100), _rules())
    bar = model["tracks"][0]["staves"][0]["bars"][0]
    for v in bar["voices"][:2]:
        total = sum(b["displayduration"] for b in v["beats"])
        assert total == 4 * 960, f"voice must fill a whole bar, got {total}"
