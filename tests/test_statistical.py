import math

from forensics.features.statistical import statistical_feature_names, statistical_features


def test_statistical_features_returns_finite_values():
    feats = statistical_features("The quick brown fox jumps over the lazy dog near the river bank.")
    assert set(feats.keys()) == set(statistical_feature_names())
    assert all(math.isfinite(v) for v in feats.values())


def test_statistical_features_empty_string_is_safe_default():
    feats = statistical_features("")
    assert feats == {"stat_loglik": 0.0, "stat_logrank": 0.0, "stat_lrr": 0.0, "stat_curvature": 0.0}


def test_statistical_features_are_cached_by_text():
    a = statistical_features("Repeatable input text for cache verification.")
    b = statistical_features("Repeatable input text for cache verification.")
    assert a == b
