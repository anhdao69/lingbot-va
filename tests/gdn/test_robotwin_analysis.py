"""Catch a misleading speedup denominator or accidental promotion at <1.10x."""

from benchmarks.robotwin_cache.analyze import decision, speedup_over_dense_fast


def test_speedup_uses_dense_fast_not_original_dense():
    summaries = {
        "dense": {"policy_ms": 10},
        "dense_fast": {"policy_ms": 8},
        "gdn50": {"policy_ms": 4},
    }
    assert speedup_over_dense_fast(summaries, "gdn50") == 2
    assert speedup_over_dense_fast(summaries, "dense_fast") == 1


def test_go_thresholds_do_not_promote_small_speedups():
    assert decision([1.3, 1.26]) == "GO"
    assert decision([1.3, 1.15]) == "CONDITIONAL"
    assert decision([1.09, 1.09]) == "STOP speed direction"


def test_comparison_rejects_original_dense_in_hybrid():
    import pytest

    from benchmarks.robotwin_cache.analyze import validate_configuration

    with pytest.raises(ValueError, match="dense_fast"):
        validate_configuration(
            {"variant": "gdn50", "dense_backend": "full", "config": "robotwin"}, "gdn50"
        )
    validate_configuration(
        {"variant": "gdn50", "dense_backend": "dense_fast", "config": "robotwin"},
        "gdn50",
    )
