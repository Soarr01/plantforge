"""Classical SysID baselines (ARX, degree-2 polynomial NARX) under the SAME
in-context protocol as the transformer: per window, fit on the 192-sample
context only, then FREE-RUN the 32-sample query horizon given only u (the
transformer's yprev input is zeroed there too -- neither method sees
ground-truth y in the query horizon). Zero new dependencies.

    python -m plantforge.baselines          # synthetic transfer cells
    python -m plantforge.baselines real     # also real-plant windows (network)
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from .evaluate import make_batch, _norm, T_CTX, D
from .aggregate import transfer_cells

LAGS = (2, 4, 8)      # order candidates (na = nb = k)
VAL = 32              # last VAL context samples -> order selection
RIDGE = 1e-6          # Tikhonov regularization for conditioning
CLIP = 1e6            # free-run state bound: a diverged NARX rollout reports a
                      # huge-but-finite nMSE instead of overflowing to nan; NaN
                      # predictions are zeroed before clamping to ensure finiteness
EVAL_SEEDS = range(900, 906)   # same eval seeds as evaluate.nmse


def _lag_vector(u, y, k, t):
    """Regressor for y[t]: k lags of y (t-1..t-k), k values of u (t..t-k+1)."""
    return np.concatenate([y[t - k:t][::-1], u[t - k + 1:t + 1][::-1]])


def _phi(z, poly):
    """[1, z] for ARX; [1, z, upper-triangle degree-2 products] for NARX."""
    if not poly:
        return np.concatenate([[1.0], z])
    quad = np.outer(z, z)[np.triu_indices(len(z))]
    return np.concatenate([[1.0], z, quad])


def _fit(u, y, k, poly, t_lo, t_hi):
    X = np.stack([_phi(_lag_vector(u, y, k, t), poly)
                  for t in range(max(k, t_lo), t_hi)])
    target = y[max(k, t_lo):t_hi]
    A = X.T @ X + RIDGE * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ target)


def _one_step_mse(u, y, w, k, poly, t_lo, t_hi):
    err = 0.0
    for t in range(t_lo, t_hi):
        pred = _phi(_lag_vector(u, y, k, t), poly) @ w
        err += (pred - y[t]) ** 2
    return err / (t_hi - t_lo)


def select_order(u, y, poly):
    """Pick k from LAGS by one-step-ahead error on the last VAL context
    samples; fit uses only the context before them. Never touches t>=T_CTX."""
    best_k, best_err = LAGS[0], np.inf
    for k in LAGS:
        w = _fit(u, y, k, poly, k, T_CTX - VAL)
        err = _one_step_mse(u, y, w, k, poly, T_CTX - VAL, T_CTX)
        if err < best_err:
            best_k, best_err = k, err
    return best_k


def _freerun(u, y, w, k, poly):
    """Simulate t = T_CTX..D-1 with predicted y fed back as lags; true y is
    used only for t < T_CTX. The fed-back state is clipped to +-CLIP so an
    unstable fit (common for poly-NARX in free-run) yields a huge-but-finite
    error instead of overflowing float64 to nan. NaN predictions are zeroed
    before clamping to ensure the fed-back state is always finite."""
    yhat = y.copy()
    for t in range(T_CTX, D):
        pred = _phi(_lag_vector(u, yhat, k, t), poly) @ w
        if not np.isfinite(pred):
            pred = 0.0
        yhat[t] = min(max(pred, -CLIP), CLIP)
    return yhat[T_CTX:]


def predict_batch(u_b: torch.Tensor, y_b: torch.Tensor, poly: bool) -> np.ndarray:
    """Free-run query predictions for a batch of NORMALIZED (B, D) windows."""
    u_np = u_b.detach().cpu().numpy().astype(np.float64)
    y_np = y_b.detach().cpu().numpy().astype(np.float64)
    preds = []
    for i in range(u_np.shape[0]):
        u, y = u_np[i], y_np[i]
        k = select_order(u, y, poly)
        w = _fit(u, y, k, poly, k, T_CTX)
        preds.append(_freerun(u, y, w, k, poly))
    return np.stack(preds)


def baseline_nmse_batch(u_b, y_b, poly) -> float:
    """Ratio of batch-mean query MSE to batch-mean query power -- the same
    aggregation as evaluate.nmse / realbench.nmse_on_windows."""
    pred = predict_batch(u_b, y_b, poly)
    y_q = y_b.detach().cpu().numpy().astype(np.float64)[:, T_CTX:]
    return float(((pred - y_q) ** 2).mean() / (y_q ** 2).mean())


def synthetic_report():
    print("=== classical baselines on the corpus transfer cells "
          "(context-fit + free-run, nMSE) ===")
    for label, fam, exc, dt in transfer_cells("corpus"):
        vals = {"ARX": [], "NARX2": []}
        for sd in EVAL_SEEDS:
            u, y = make_batch(fam, exc, dt, 96, sd)   # already _norm-alized
            if not (torch.isfinite(u).all() and torch.isfinite(y).all()):
                continue                              # skip diverged draws
            vals["ARX"].append(baseline_nmse_batch(u, y, poly=False))
            vals["NARX2"].append(baseline_nmse_batch(u, y, poly=True))
        arx = sum(vals["ARX"]) / max(len(vals["ARX"]), 1)
        narx = sum(vals["NARX2"]) / max(len(vals["NARX2"]), 1)
        print(f"  {label}: ARX {arx:.4f} | NARX2 {narx:.4f}")


def real_report():
    from .realbench import silverbox_windows, cascaded_tanks_windows, boucwen_windows
    print("=== classical baselines on real-plant windows (nMSE) ===")
    sb, sb_dt, sb_q = silverbox_windows()
    ct, ct_dt = cascaded_tanks_windows()
    bw, bw_dt, bw_q = boucwen_windows()
    for name, windows in (("Silverbox", sb), ("Cascaded_Tanks", ct), ("Bouc-Wen", bw)):
        if windows is None:
            print(f"  {name}: SKIPPED (no windows)")
            continue
        u_n, y_n = _norm(*windows)                    # raw -> same norm as model eval
        arx = baseline_nmse_batch(u_n, y_n, poly=False)
        narx = baseline_nmse_batch(u_n, y_n, poly=True)
        print(f"  {name}: ARX {arx:.4f} | NARX2 {narx:.4f}")


if __name__ == "__main__":
    synthetic_report()
    if len(sys.argv) > 1 and sys.argv[1] == "real":
        real_report()
