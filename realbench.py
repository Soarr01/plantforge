"""Zero-shot evaluation of trained in-context SysID models on real measured
plants from nonlinearbenchmark.org, via the `nonlinear_benchmarks` package.
`nonlinear_benchmarks` is imported lazily (inside functions) so this module
only requires it when actually run, not merely imported.

    python -m plantforge.realbench
"""
from __future__ import annotations

import numpy as np
import scipy.signal
import torch

from .corpus import RATES
from .evaluate import InContextSysID, _norm, T_CTX, D, DEV, CKPT_DIR

WINDOW_CAP = 8


def decimate_to_factor(x: np.ndarray, q: int) -> np.ndarray:
    """Anti-aliased decimation by integer factor q. Chains calls in steps of
    at most 10 when q factors evenly into such steps (scipy.signal.decimate
    recommends q<=13 per call); when q (or what remains of it) has no
    divisor <=10 other than 1 -- e.g. a prime like 13, 17, 23 -- decimates
    by the remaining factor in a single call instead of looping forever
    looking for a clean split that doesn't exist."""
    factors = []
    remaining = q
    while remaining > 1:
        step = None
        for f in range(min(remaining, 10), 1, -1):
            if remaining % f == 0:
                step = f
                break
        if step is None:
            step = remaining
        factors.append(step)
        remaining //= step
    out = x
    for f in factors:
        out = scipy.signal.decimate(out, f, ftype="iir", zero_phase=True)
    return out


def best_decimation_factor(native_dt: float, target_dts=RATES):
    """Integer decimation factor q>=1 that brings native_dt down to the
    FINEST (smallest dt) trained rate reachable by decimating. Finer
    retains more of the real signal's bandwidth than coarser, so among
    reachable targets the finest is preferred outright -- picking by raw
    abs(dt) error instead favors coarse targets by round-off coincidence
    for some native rates (e.g. native_dt=0.0016384s lands closer in raw
    error to the 0.10s target than to the 0.02s target, despite 0.02s
    being reachable and far more informative). Returns None if native_dt
    is already coarser than every target (nothing to decimate to -- the
    Cascaded_Tanks case)."""
    if native_dt >= max(target_dts):
        return None
    finest = min(target_dts)
    q = max(1, round(finest / native_dt))
    achieved = native_dt * q
    return q, achieved


def make_windows(u: np.ndarray, y: np.ndarray, cap: int = WINDOW_CAP):
    """Slice 1-D (u, y) arrays into non-overlapping length-D windows, stacked
    as (B, D) float32 tensors on DEV. Returns None if fewer than 1 full
    window is available."""
    n_windows = min(len(u), len(y)) // D
    if n_windows < 1:
        return None
    n_windows = min(n_windows, cap)
    u_win = np.stack([u[i * D:(i + 1) * D] for i in range(n_windows)])
    y_win = np.stack([y[i * D:(i + 1) * D] for i in range(n_windows)])
    return (torch.tensor(u_win, dtype=torch.float32, device=DEV),
            torch.tensor(y_win, dtype=torch.float32, device=DEV))


def pooled_windows(records, cap: int = WINDOW_CAP):
    """Slice windows from each (u, y) record and concatenate across records,
    stopping once `cap` total windows are collected."""
    u_all, y_all = [], []
    remaining = cap
    for u, y in records:
        if remaining <= 0:
            break
        w = make_windows(u, y, cap=remaining)
        if w is None:
            continue
        u_all.append(w[0]); y_all.append(w[1])
        remaining -= w[0].shape[0]
    if not u_all:
        return None
    return torch.cat(u_all, dim=0), torch.cat(y_all, dim=0)
