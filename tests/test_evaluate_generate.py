from jev_like.evaluate_generate import generated_index


def test_generated_index_accepts_valid_marker_prefixes():
    assert generated_index("A", 3) == 0
    assert generated_index("  B. explanation", 3) == 1
    assert generated_index("C: selected", 3) == 2


def test_generated_index_rejects_invalid_or_out_of_range_markers():
    assert generated_index("weather", 3) is None
    assert generated_index("D", 3) is None
    assert generated_index("AB", 3) is None
