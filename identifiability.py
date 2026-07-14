"""Per-instance practical-identifiability annotation via the Fisher information matrix.

For each instance and excitation record: sensitivities s_i = dy/dtheta_i by central
finite differences (relative step), FIM = S^T S / sigma^2 with a reference noise floor,
and the annotation = per-parameter relative CRLB sqrt(diag(FIM^-1)) / |theta| plus the
FIM log-condition number. High rel-CRLB -> that parameter is practically unidentifiable
FROM THIS excitation — which is exactly what the excitation-design axis is for.
"""
from __future__ import annotations

import torch

from .families import simulate, param_vector

SIGMA_REF = 0.05                    # reference measurement-noise std for the FIM scale
REL_STEP = 0.02


@torch.no_grad()
def identifiability(family: str, p: dict, u: torch.Tensor, dt: float):
    """Returns dict: rel_crlb (B,K), log10_cond (B,), keys (list of K names)."""
    theta, keys = param_vector(family, p)              # (B,K)
    B, K = theta.shape
    T = u.shape[0]
    sens = torch.zeros(B, K, T)
    for i, key in enumerate(keys):
        h = (p[key].abs() * REL_STEP).clamp(min=1e-3)
        pp = {k: v.clone() for k, v in p.items()}
        pp[key] = p[key] + h
        y_hi = simulate(family, pp, u, dt)
        pp[key] = p[key] - h
        y_lo = simulate(family, pp, u, dt)
        sens[:, i] = ((y_hi - y_lo) / (2 * h).unsqueeze(0)).t()
    F = torch.einsum("bkt,blt->bkl", sens, sens) / SIGMA_REF ** 2
    F = F + 1e-9 * torch.eye(K)                        # numerical floor
    eigs = torch.linalg.eigvalsh(F)
    log_cond = torch.log10(eigs[:, -1].clamp(min=1e-30) / eigs[:, 0].clamp(min=1e-30))
    Finv = torch.linalg.pinv(F)
    rel_crlb = torch.sqrt(torch.diagonal(Finv, dim1=1, dim2=2).clamp(min=0)) \
        / theta.abs().clamp(min=1e-3)
    return {"rel_crlb": rel_crlb, "log10_cond": log_cond, "keys": keys}
