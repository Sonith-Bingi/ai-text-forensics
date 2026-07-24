from forensics.adversarial.reward import _length_penalty, _lexical_overlap


def test_lexical_overlap_identical_text_is_one():
    assert _lexical_overlap("the quick fox", "the quick fox") == 1.0


def test_lexical_overlap_disjoint_text_is_zero():
    assert _lexical_overlap("apples oranges", "bicycles trains") == 0.0


def test_lexical_overlap_empty_string_is_zero():
    assert _lexical_overlap("", "some words here") == 0.0
    assert _lexical_overlap("some words here", "") == 0.0


def test_length_penalty_within_acceptable_range_is_one():
    original = "one two three four five six seven eight"
    paraphrase = "one two three four five six"  # ratio 6/8 = 0.75, within [0.4, 2.5]
    assert _length_penalty(original, paraphrase) == 1.0


def test_length_penalty_degenerate_short_output_is_penalized():
    original = "one two three four five six seven eight nine ten"
    paraphrase = "one"  # ratio 1/10 = 0.1, well below min_length_ratio
    assert _length_penalty(original, paraphrase) < 1.0


def test_length_penalty_runaway_long_output_is_penalized():
    original = "one two three"
    paraphrase = " ".join(["word"] * 50)  # ratio way above max_length_ratio
    assert _length_penalty(original, paraphrase) < 1.0
