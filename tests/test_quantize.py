from music2gtp8score.quantize import TempoMap, normalize_tempo, quantize
from music2gtp8score.transcribe_drums import DrumEvent


def test_normalize_tempo_folds_octaves():
    assert normalize_tempo(82) == 82
    assert normalize_tempo(164) == 82
    assert normalize_tempo(41) == 82
    assert normalize_tempo(200) == 100


def test_quantize_snaps_to_grid():
    tm = TempoMap(tempo=120)  # 0.5s per beat, 0.125s per 16th slot
    events = [
        DrumEvent(time=0.00, drum="kick", velocity=100),    # bar 0 slot 0
        DrumEvent(time=0.49, drum="snare", velocity=100),   # ~beat 1 -> slot 4
        DrumEvent(time=1.01, drum="kick", velocity=100),    # ~beat 2 -> slot 8
        DrumEvent(time=2.06, drum="hihat_closed", velocity=100),  # bar 1 slot ~0
    ]
    notes = quantize(events, tm, resolution=16)
    got = [(n.bar, n.slot, n.drum) for n in notes]
    assert got == [(0, 0, "kick"), (0, 4, "snare"), (0, 8, "kick"), (1, 0, "hihat_closed")]


def test_quantize_empty():
    assert quantize([], TempoMap(tempo=100)) == []
