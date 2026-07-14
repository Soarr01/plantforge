"""Offline tests for plantforge.ident_exp -- untrained model, synthetic data.

    python -m plantforge.tests.test_ident_exp     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from plantforge.ident_exp import qry_stats, nmse_per_instance, quartile_table
from plantforge.evaluate import InContextSysID, D, DEV
from plantforge.realbench import nmse_on_windows


def test_qry_stats_reconstruct_batch_nmse():
    torch.manual_seed(0)
    model = InContextSysID().to(DEV)
    model.eval()
    u = torch.randn(8, D, device=DEV)
    y = torch.randn(8, D, device=DEV)
    num, den = qry_stats(model, u, y)
    assert num.shape == (8,) and den.shape == (8,)
    batch = nmse_on_windows(model, u, y)
    assert abs((num.mean() / den.mean()).item() - batch) < 1e-5
    per_inst = nmse_per_instance(model, u, y)
    assert per_inst.shape == (8,) and torch.isfinite(per_inst).all()
    print("  PASS  test_qry_stats_reconstruct_batch_nmse")


def test_quartile_table_monotone_construction():
    metric = np.arange(1.0, 9.0)          # 1..8
    nmse = metric * 2.0                   # perfectly increasing with metric
    rows = quartile_table(metric, nmse)
    assert len(rows) == 4
    # rows: (bin mean metric, bin mean nmse, bin median nmse, bin count)
    assert all(rows[i][1] < rows[i + 1][1] for i in range(3))   # mean col
    assert all(rows[i][2] < rows[i + 1][2] for i in range(3))   # median col
    assert sum(r[3] for r in rows) == 8
    print("  PASS  test_quartile_table_monotone_construction")


def test_quartile_table_empty_bin_guard():
    # Heavy ties: quartile cut points can coincide, leaving a bin empty.
    # Must not raise and must not report NaN for populated bins.
    metric = np.array([1.0] * 6 + [2.0] * 2)
    nmse = np.array([10.0] * 6 + [20.0] * 2)
    rows = quartile_table(metric, nmse)
    assert len(rows) == 4
    for mm, mn, md, n in rows:
        if n == 0:
            continue
        assert not np.isnan(mm) and not np.isnan(mn) and not np.isnan(md)
    assert sum(r[3] for r in rows) == 8
    print("  PASS  test_quartile_table_empty_bin_guard")


def _run_all():
    test_qry_stats_reconstruct_batch_nmse()
    test_quartile_table_monotone_construction()
    test_quartile_table_empty_bin_guard()


if __name__ == "__main__":
    print("PLANTFORGE ident_exp -- offline tests:")
    _run_all()
