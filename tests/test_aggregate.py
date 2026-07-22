"""Offline tests for plantforge.aggregate's pure helpers -- no checkpoints,
no GPU, no network.

    python -m plantforge.tests.test_aggregate     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import math

from plantforge.aggregate import (
    transfer_cells, aggregate_matrices, mean_std_str, matrix,
)
from plantforge.evaluate import FAMILIES, HOLD_FAMILY, TRAIN_RATES


def test_transfer_cells_structure():
    for mode in ("wh_only", "corpus"):
        cells = transfer_cells(mode)
        labels = [c[0] for c in cells]
        assert labels[0] == "reference (train-like)"
        ref_family = cells[0][1]
        assert ref_family == ("wh" if mode == "wh_only" else "stribeck")
        # 1 ref + 3 held-out-family rates + 2 held-out excitations
        # + (len(FAMILIES)-1) held-out-rate families
        assert len(cells) == 1 + 3 + 2 + (len(FAMILIES) - 1)
        assert all(c[1] != HOLD_FAMILY or "family" in c[0] for c in cells)
    print("  PASS  test_transfer_cells_structure")


def test_aggregate_matrices_mean_std_n():
    mats = [{"a": 1.0, "b": 10.0}, {"a": 3.0, "b": 10.0}]
    agg = aggregate_matrices(mats)
    m, s, n = agg["a"]
    assert abs(m - 2.0) < 1e-12 and n == 2
    # sample std (ddof=1) of [1, 3]: variance = ((1-2)**2+(3-2)**2)/(2-1) = 2,
    # so std = sqrt(2) -- NOT 1.0 (that's the *population*, ddof=0, std).
    assert abs(s - 2 ** 0.5) < 1e-12
    m, s, n = agg["b"]
    assert abs(m - 10.0) < 1e-12 and abs(s - 0.0) < 1e-12
    print("  PASS  test_aggregate_matrices_mean_std_n")


def test_aggregate_matrices_single_seed_no_nan_std():
    agg = aggregate_matrices([{"a": 2.0}])
    m, s, n = agg["a"]
    assert m == 2.0 and n == 1
    assert not math.isnan(s) and s == 0.0   # n=1 reports std 0, not NaN
    print("  PASS  test_aggregate_matrices_single_seed_no_nan_std")


def test_mean_std_str_format():
    s = mean_std_str([0.1, 0.2, 0.3])
    assert "0.2000" in s and "±" in s and "n=3" in s
    print("  PASS  test_mean_std_str_format")


def test_matrix_averages_reference_over_train_rates_for_corpus_mode():
    """The bug this task fixes: corpus mode's 'reference (train-like)' cell
    used to be nmse(model, 'stribeck', 'multisine', 0.05) -- dt=0.05 is a
    held-out rate, not a trained one. matrix() must now average over
    TRAIN_RATES (0.10, 0.02) instead, mirroring the evaluate.report() fix."""
    import plantforge.aggregate as agg
    calls = []

    def fake_nmse(model, fam, exc, dt):
        calls.append((fam, exc, dt))
        return dt   # deterministic: returned value IS dt, so the mean is checkable

    original = agg.nmse
    agg.nmse = fake_nmse
    try:
        result = agg.matrix(None, "corpus")
    finally:
        agg.nmse = original

    assert abs(result["reference (train-like)"] - 0.06) < 1e-9, \
        f"expected mean of TRAIN_RATES (0.06), got {result['reference (train-like)']}"
    ref_calls = [c for c in calls if c[0] == "stribeck" and c[1] == "multisine"
                 and c[2] in TRAIN_RATES]
    assert len(ref_calls) == 2, \
        f"expected exactly 2 calls (one per TRAIN_RATES entry) for the reference cell, got {ref_calls}"
    assert all(c[2] != 0.05 for c in ref_calls), "reference calls must not use the held-out rate 0.05"
    print("  PASS  test_matrix_averages_reference_over_train_rates_for_corpus_mode")


def test_matrix_wh_only_reference_unchanged_at_dt05():
    """wh_only mode's reference (wh/multisine/dt=0.05) IS wh_only's only
    trained combination -- it was already correct and must stay a single
    dt=0.05 call, not an average."""
    import plantforge.aggregate as agg
    calls = []

    def fake_nmse(model, fam, exc, dt):
        calls.append((fam, exc, dt))
        return 1.0

    original = agg.nmse
    agg.nmse = fake_nmse
    try:
        result = agg.matrix(None, "wh_only")
    finally:
        agg.nmse = original

    assert result["reference (train-like)"] == 1.0
    assert calls[0] == ("wh", "multisine", 0.05)
    print("  PASS  test_matrix_wh_only_reference_unchanged_at_dt05")


def _run_all():
    test_transfer_cells_structure()
    test_aggregate_matrices_mean_std_n()
    test_aggregate_matrices_single_seed_no_nan_std()
    test_mean_std_str_format()
    test_matrix_averages_reference_over_train_rates_for_corpus_mode()
    test_matrix_wh_only_reference_unchanged_at_dt05()


if __name__ == "__main__":
    print("PLANTFORGE aggregate -- offline helper tests:")
    _run_all()
