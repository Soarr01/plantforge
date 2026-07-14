"""Offline tests for plantforge.baselines -- no GPU training, no network.

    python -m plantforge.tests.test_baselines     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from plantforge.baselines import (
    predict_batch, baseline_nmse_batch, select_order, LAGS,
)
from plantforge.evaluate import T_CTX, D


def _arx_windows(B=4, seed=0):
    """Noise-free windows from a known ARX process (within model class for
    k>=2): y_t = 0.6 y_{t-1} - 0.2 y_{t-2} + 0.5 u_t + 0.1 u_{t-1}."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((B, D))
    y = np.zeros((B, D))
    for t in range(2, D):
        y[:, t] = (0.6 * y[:, t - 1] - 0.2 * y[:, t - 2]
                   + 0.5 * u[:, t] + 0.1 * u[:, t - 1])
    return (torch.tensor(u, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


def test_arx_recovers_arx_process():
    u, y = _arx_windows()
    v = baseline_nmse_batch(u, y, poly=False)
    assert v < 1e-3, v
    print("  PASS  test_arx_recovers_arx_process")


def test_narx_handles_linear_process():
    u, y = _arx_windows()
    v = baseline_nmse_batch(u, y, poly=True)
    assert v < 1e-2, v
    print("  PASS  test_narx_handles_linear_process")


def test_select_order_returns_candidate():
    u, y = _arx_windows(B=1)
    k = select_order(u[0].numpy().astype(np.float64),
                     y[0].numpy().astype(np.float64), poly=False)
    assert k in LAGS, k
    print("  PASS  test_select_order_returns_candidate")


def test_freerun_never_peeks_at_query_y():
    u, y = _arx_windows()
    y2 = y.clone()
    y2[:, T_CTX:] = torch.randn_like(y2[:, T_CTX:])
    for poly in (False, True):
        p1 = predict_batch(u, y, poly)
        p2 = predict_batch(u, y2, poly)
        assert np.array_equal(p1, p2), "query-horizon y leaked into prediction"
    print("  PASS  test_freerun_never_peeks_at_query_y")


def _run_all():
    test_arx_recovers_arx_process()
    test_narx_handles_linear_process()
    test_select_order_returns_candidate()
    test_freerun_never_peeks_at_query_y()


if __name__ == "__main__":
    print("PLANTFORGE baselines -- offline tests:")
    _run_all()
