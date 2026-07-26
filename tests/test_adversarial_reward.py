from forensics.adversarial.reward import _length_penalty, _wellformedness_gate, semantic_similarity


def test_semantic_similarity_real_paraphrase_scores_high():
    original = "The mayor was asked to explain the difference between deficit and debt in a short segment."
    paraphrase = "London's mayor was asked to briefly explain the difference between deficit and debt."
    assert semantic_similarity(original, paraphrase) > 0.6


def test_semantic_similarity_incoherent_text_scores_low():
    # The actual reward-hacked output from the first (v1 reward) training run --
    # this is the concrete case the fidelity rewrite exists to catch.
    original = (
        "'Get your fingers out of that nook, dick.' I knew it was a bad idea the moment "
        "I heard it, but damned if I let that put a boogie in my game."
    )
    garbled = (
        '"Give me your k*** a snuff bobby cut!" \'Kind you stand up shay bob" '
        '"Shark! Hurst!" cheers Under a Spectacle T-Shirt: Bobby Bobby\'s happy camp'
    )
    assert semantic_similarity(original, garbled) < 0.5


def test_semantic_similarity_unrelated_text_scores_very_low():
    original = "The clinical pharmacist develops a detailed drug therapy plan for patient-specific problems."
    unrelated = "I like cats and dogs very much, they are great pets for everyone."
    assert semantic_similarity(original, unrelated) < 0.3


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


def test_wellformedness_gate_catches_repeated_character_spam():
    # The actual v3 (threshold_gate reward) failure case: a coherent prefix
    # with a repeated-character/ALL-CAPS garbage tail bolted on, which barely
    # moves whole-text semantic similarity but is exactly what stylometric
    # features (upper_ratio, rep3_rate) flag hard in the real detector.
    text = (
        '"But i kinda hit you," she whispered, regretting, "OMG NO NO '
        'MEEEEEEEEEEEEEEEEAKE US OUR BONUS STATION THAT SHOCK'
    )
    assert _wellformedness_gate(text) == 0.0


def test_wellformedness_gate_allows_normal_text():
    assert _wellformedness_gate("The mayor explained the difference between deficit and debt.") == 1.0


def test_wellformedness_gate_allows_scattered_acronyms():
    # Must not false-positive on ordinary text just because it has a few
    # capitalized acronyms -- only a *consecutive run* of all-caps words
    # (the actual degenerate spam pattern) should trigger the gate.
    text = "NASA and the USA collaborated with the UN on this project, according to the report."
    assert _wellformedness_gate(text) == 1.0


def test_wellformedness_gate_allows_two_consecutive_acronyms():
    text = "The US UK alliance has been strong since world war two, according to historians."
    assert _wellformedness_gate(text) == 1.0


def test_wellformedness_gate_empty_string_is_penalized():
    assert _wellformedness_gate("") == 0.0
