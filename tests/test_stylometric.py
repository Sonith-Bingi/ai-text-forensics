from forensics.features.stylometric import (
    burstiness,
    char_entropy,
    digit_ratio,
    punct_ratio,
    stylometric_feature_names,
    stylometric_features,
    type_token_ratio,
    upper_ratio,
)


def test_punct_ratio_all_punct():
    assert punct_ratio("!!!") == 1.0


def test_punct_ratio_empty():
    assert punct_ratio("") == 0.0


def test_digit_ratio():
    assert digit_ratio("abc123") == 0.5


def test_upper_ratio_no_letters():
    assert upper_ratio("123 !!!") == 0.0


def test_upper_ratio_all_caps():
    assert upper_ratio("ABC") == 1.0


def test_char_entropy_uniform_higher_than_repetitive():
    assert char_entropy("abcdefgh") > char_entropy("aaaaaaaa")


def test_type_token_ratio_no_repeats_is_one():
    assert type_token_ratio("the quick brown fox") == 1.0


def test_type_token_ratio_all_repeats_is_low():
    assert type_token_ratio("the the the the") < 0.5


def test_burstiness_empty_is_zero():
    assert burstiness("") == 0.0


def test_stylometric_features_returns_all_expected_keys():
    feats = stylometric_features("A short test sentence. With two sentences!")
    assert set(feats.keys()) == set(stylometric_feature_names())
    assert all(isinstance(v, (int, float)) for v in feats.values())


def test_stylometric_features_handles_empty_string():
    feats = stylometric_features("")
    assert feats["len_chars"] == 0
    assert all(v == 0 for k, v in feats.items() if k not in ("flesch_reading_ease", "gunning_fog", "smog_index"))
