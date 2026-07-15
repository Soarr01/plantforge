"""Offline-safe tests for plantforge.realbench's decimation/windowing
helpers -- no network access, no trained checkpoints required.

    python -m plantforge.tests.test_realbench     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from plantforge.realbench import (
    decimate_to_factor, best_decimation_factor, make_windows,
    pooled_windows, nmse_on_windows, load_model, D, DEV,
)
from plantforge.corpus import RATES


def test_make_windows_shapes_and_cap():
    n = D * 5 + 30          # 5 full windows + a remainder
    u = np.arange(n, dtype=np.float64)
    y = np.arange(n, dtype=np.float64) * 2
    windows = make_windows(u, y, cap=3)
    assert windows is not None
    u_win, y_win = windows
    assert u_win.shape == (3, D)
    assert y_win.shape == (3, D)
    assert torch.equal(u_win[0], torch.tensor(u[:D], dtype=torch.float32, device=DEV))
    print("  PASS  test_make_windows_shapes_and_cap")


def test_make_windows_too_short_returns_none():
    u = np.zeros(D - 1, dtype=np.float64)
    y = np.zeros(D - 1, dtype=np.float64)
    assert make_windows(u, y) is None
    print("  PASS  test_make_windows_too_short_returns_none")


def test_decimate_to_factor_single_call():
    t = np.linspace(0, 10, 20000)
    x = np.sin(2 * np.pi * 0.5 * t)
    out = decimate_to_factor(x, 12)
    assert np.isfinite(out).all()
    assert abs(len(out) - len(x) // 12) <= 2
    assert out.std() > 0.1          # signal survives decimation, not degenerate
    print("  PASS  test_decimate_to_factor_single_call")


def test_decimate_to_factor_chained():
    t = np.linspace(0, 10, 200000)
    x = np.sin(2 * np.pi * 0.1 * t)
    out = decimate_to_factor(x, 100)   # forces chaining (10 * 10)
    assert np.isfinite(out).all()
    assert abs(len(out) - len(x) // 100) <= 5
    print("  PASS  test_decimate_to_factor_chained")


def test_best_decimation_factor_matches_silverbox():
    native_dt = 0.0016384041943147375     # measured Silverbox sampling_time
    result = best_decimation_factor(native_dt, RATES)
    assert result is not None
    q, achieved_dt = result
    assert q == 12
    assert abs(achieved_dt - 0.02) < 0.001
    print("  PASS  test_best_decimation_factor_matches_silverbox")


def test_best_decimation_factor_none_when_native_already_coarser():
    result = best_decimation_factor(4.0, RATES)   # Cascaded_Tanks case
    assert result is None
    print("  PASS  test_best_decimation_factor_none_when_native_already_coarser")


def test_decimate_to_factor_prime_terminates():
    t = np.linspace(0, 10, 20000)
    x = np.sin(2 * np.pi * 0.5 * t)
    out = decimate_to_factor(x, 13)      # prime > 10, no clean small-factor split
    assert np.isfinite(out).all()
    assert abs(len(out) - len(x) // 13) <= 2
    print("  PASS  test_decimate_to_factor_prime_terminates")


def test_best_decimation_factor_no_crash_when_native_much_finer():
    result = best_decimation_factor(0.001, RATES)   # 1kHz, finer than every target
    assert result is not None
    q, achieved_dt = result
    assert q >= 1
    assert np.isfinite(achieved_dt)
    print("  PASS  test_best_decimation_factor_no_crash_when_native_much_finer")


def test_best_decimation_factor_matches_boucwen():
    native_dt = 1.0 / 750.0
    result = best_decimation_factor(native_dt, RATES)
    assert result is not None
    q, achieved_dt = result
    assert q == 15, q
    assert abs(achieved_dt - 0.02) < 1e-9, achieved_dt   # exact match, no rounding error
    print("  PASS  test_best_decimation_factor_matches_boucwen")


def test_pooled_windows_caps_across_records():
    rec1 = (np.zeros(D * 2, dtype=np.float64), np.zeros(D * 2, dtype=np.float64))
    rec2 = (np.ones(D * 5, dtype=np.float64), np.ones(D * 5, dtype=np.float64))
    windows = pooled_windows([rec1, rec2], cap=5)
    assert windows is not None
    u_win, y_win = windows
    assert u_win.shape == (5, D)
    assert (u_win[:2] == 0).all() and (u_win[2:] == 1).all()
    print("  PASS  test_pooled_windows_caps_across_records")


def test_pooled_windows_none_when_all_records_too_short():
    rec1 = (np.zeros(D - 1, dtype=np.float64), np.zeros(D - 1, dtype=np.float64))
    rec2 = (np.ones(D - 1, dtype=np.float64), np.ones(D - 1, dtype=np.float64))
    assert pooled_windows([rec1, rec2]) is None
    print("  PASS  test_pooled_windows_none_when_all_records_too_short")


def test_nmse_on_windows_finite_untrained_model():
    from plantforge.evaluate import InContextSysID
    model = InContextSysID().to(DEV)
    model.eval()
    u = torch.randn(4, D, device=DEV)
    y = torch.randn(4, D, device=DEV)
    v = nmse_on_windows(model, u, y)
    assert np.isfinite(v)
    print("  PASS  test_nmse_on_windows_finite_untrained_model")


def test_load_model_missing_checkpoint_returns_none():
    assert load_model("eval_this_checkpoint_does_not_exist.pt") is None
    print("  PASS  test_load_model_missing_checkpoint_returns_none")


def _run_all():
    test_make_windows_shapes_and_cap()
    test_make_windows_too_short_returns_none()
    test_decimate_to_factor_single_call()
    test_decimate_to_factor_chained()
    test_best_decimation_factor_matches_silverbox()
    test_best_decimation_factor_none_when_native_already_coarser()
    test_decimate_to_factor_prime_terminates()
    test_best_decimation_factor_no_crash_when_native_much_finer()
    test_best_decimation_factor_matches_boucwen()
    test_pooled_windows_caps_across_records()
    test_pooled_windows_none_when_all_records_too_short()
    test_nmse_on_windows_finite_untrained_model()
    test_load_model_missing_checkpoint_returns_none()


if __name__ == "__main__":
    print("PLANTFORGE realbench -- offline decimation/windowing tests:")
    _run_all()
