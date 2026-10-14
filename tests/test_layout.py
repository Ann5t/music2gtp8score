from music2gtp8score.layout import DEFAULT_CONFIG, LayoutRules


def test_defaults_match_yaml():
    r = LayoutRules.from_yaml(DEFAULT_CONFIG)
    assert r.tablature is False
    assert r.grid_resolution == 16
    assert r.bars_per_system == 4
    assert r.plain_notes_only is True
    assert r.plain_barlines_only is True
    assert r.show_dynamics is False
    assert r.drum_map["kick"] == 36
    assert r.stem_map["hihat"] == ["hh"]


def test_voice_assignment():
    r = LayoutRules.from_yaml(DEFAULT_CONFIG)
    for key in ("hihat_closed", "hihat_open", "crash", "ride"):
        assert r.voice_of(key) == r.cymbal_voice, key
    for key in ("kick", "snare", "hihat_pedal", "tom_high", "tom_low"):
        assert r.voice_of(key) == r.drum_voice, key


def test_override(tmp_path):
    cfg = tmp_path / "my.yaml"
    cfg.write_text("page:\n  bars_per_system: 6\ndrums:\n  snare: 40\n", encoding="utf-8")
    r = LayoutRules.from_yaml(cfg)
    assert r.bars_per_system == 6
    assert r.drum_map["snare"] == 40
    assert r.drum_map["kick"] == 36  # untouched defaults survive
