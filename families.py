"""The plant families. Each instance is a continuous-time system discretized EXACTLY by
ZOH at any requested rate (the corpus's multi-rate axis), with named physical parameters
(the ground-truth axis). All batched, pure torch, CPU-friendly.

Design invariants (they ARE the corpus's claims — do not regress):
- (u, y) pairs are PHYSICALLY CONSISTENT with the stated parameters: no post-hoc output
  normalization inside the plant (normalization is a method choice, done in eval
  preprocessing, never in ground truth).
- State-nonlinear families substep at a fine internal dt (<=2 ms) with the input
  ZOH-held, so every sampling rate observes the SAME continuous-time plant.
- Every family exposes a STEPPER (init/step) so closed-loop excitation is a true
  sequential loop, and simulate() is built on the same code path.

Families:
  wh        — LTI → static tanh → LTI (the incumbent generators' only family; kept for
              headline experiments, excluded from the corpus's 5)
  stribeck  — Stribeck+Coulomb velocity friction in the loop (state NL)
  backlash  — input deadzone before the LTI core (input NL)
  saturate  — input saturation before the LTI core (input NL)
  boucwen   — Bouc-Wen hysteresis on the output (output NL with memory)
  drivetrain— two-inertia motor/gear/compliant load, Coulomb friction on the load
              (mechatronics staple: resonant 4th-order core + state NL)
"""
from __future__ import annotations

import torch

FAMILIES = ["stribeck", "backlash", "saturate", "boucwen", "drivetrain"]
ALL = FAMILIES + ["wh"]
DT_INT = 0.002                      # internal substep target for state-NL families


def _core2(p1, p2, c0, c1):
    B = p1.shape[0]
    A = torch.zeros(B, 2, 2)
    A[:, 0, 1] = 1.0
    A[:, 1, 0] = -(p1 * p2)
    A[:, 1, 1] = p1 + p2
    b = torch.zeros(B, 2); b[:, 1] = 1.0
    c = torch.stack([c0, c1], dim=1)
    return A, b, c


def _zoh(A, b, dt):
    B, n = A.shape[0], A.shape[1]
    M = torch.zeros(B, n + 1, n + 1)
    M[:, :n, :n] = A * dt
    M[:, :n, n] = b * dt
    Md = torch.matrix_exp(M)
    return Md[:, :n, :n], Md[:, :n, n]


def sample(family: str, B: int, gen: torch.Generator) -> dict:
    def U(lo, hi):
        return lo + (hi - lo) * torch.rand((B,), generator=gen)

    if family == "drivetrain":
        return {"Jm": U(0.2, 0.8), "Jl": U(0.5, 2.0), "k": U(20.0, 80.0),
                "bm": U(0.1, 0.5), "bl": U(0.2, 0.8), "N": U(1.0, 3.0), "Fc": U(0.1, 0.5)}
    p = {"p1": -U(1.0, 12.0), "p2": -U(1.0, 12.0),
         "c0": torch.randn((B,), generator=gen).sign() * U(2.0, 6.0), "c1": U(-2.0, 2.0)}
    if family == "wh":
        p.update({"g": U(0.6, 1.6), "p3": -U(1.0, 12.0), "p4": -U(1.0, 12.0),
                  "d0": torch.randn((B,), generator=gen).sign() * U(2.0, 6.0),
                  "d1": U(-2.0, 2.0)})
    elif family == "stribeck":
        p.update({"Fc": U(0.3, 0.8), "Fs_extra": U(0.2, 0.6), "vs": U(0.1, 0.3)})
    elif family == "backlash":
        p.update({"db": U(0.3, 0.8)})
    elif family == "saturate":
        p.update({"sat": U(0.8, 2.0)})
    elif family == "boucwen":
        p.update({"alpha": U(0.5, 0.8), "beta": U(0.1, 0.5), "gam": U(0.1, 0.5)})
    else:
        raise ValueError(family)
    return p


class Stepper:
    """Sequential interface: y_k = step(u_k). One ZOH hold interval per call."""

    def __init__(self, family: str, p: dict, dt: float):
        self.family, self.p, self.dt = family, p, dt
        B = next(iter(p.values())).shape[0]
        self.B = B
        if family == "drivetrain":
            Jm, Jl, k_, bm, bl, N = p["Jm"], p["Jl"], p["k"], p["bm"], p["bl"], p["N"]
            A = torch.zeros(B, 4, 4)
            A[:, 0, 1] = 1.0
            A[:, 1, 0] = -k_ / (Jm * N * N); A[:, 1, 1] = -bm / Jm; A[:, 1, 2] = k_ / (Jm * N)
            A[:, 2, 3] = 1.0
            A[:, 3, 0] = k_ / (Jl * N); A[:, 3, 2] = -k_ / Jl; A[:, 3, 3] = -bl / Jl
            b = torch.zeros(B, 4); b[:, 1] = 1.0 / Jm
            self.n_sub = max(1, round(dt / DT_INT))
            self.Ad, self.bd = _zoh(A, b, dt / self.n_sub)
            self.x = torch.zeros(B, 4)
            return
        A, b, self.c = _core2(p["p1"], p["p2"], p["c0"], p["c1"])
        if family in ("stribeck", "wh"):
            # wh's intermediate tanh(z) varies WITHIN a hold interval (z is a state),
            # so the second block sees a non-held input — substep for rate-consistency
            self.n_sub = max(1, round(dt / DT_INT))
            self.Ad, self.bd = _zoh(A, b, dt / self.n_sub)
        else:
            self.Ad, self.bd = _zoh(A, b, dt)
        self.x = torch.zeros(B, 2)
        if family == "wh":
            A2, b2, self.d = _core2(p["p3"], p["p4"], p["d0"], p["d1"])
            self.Ad2, self.bd2 = _zoh(A2, b2, dt / self.n_sub)
            self.x2 = torch.zeros(B, 2)
        if family == "boucwen":
            self.h = torch.zeros(B)
            self.uprev = torch.zeros(B)

    def step(self, u: torch.Tensor) -> torch.Tensor:
        f, p = self.family, self.p
        if f == "wh":
            for _ in range(self.n_sub):
                self.x = torch.einsum("bij,bj->bi", self.Ad, self.x) + self.bd * u.unsqueeze(-1)
                z = (self.x * self.c).sum(1)
                w = torch.tanh(p["g"] * z)
                self.x2 = torch.einsum("bij,bj->bi", self.Ad2, self.x2) + self.bd2 * w.unsqueeze(-1)
            return (self.x2 * self.d).sum(1)
        if f == "backlash":
            uu = torch.sign(u) * (u.abs() - p["db"]).clamp(min=0.0)
            self.x = torch.einsum("bij,bj->bi", self.Ad, self.x) + self.bd * uu.unsqueeze(-1)
            return (self.x * self.c).sum(1)
        if f == "saturate":
            uu = torch.minimum(torch.maximum(u, -p["sat"]), p["sat"])
            self.x = torch.einsum("bij,bj->bi", self.Ad, self.x) + self.bd * uu.unsqueeze(-1)
            return (self.x * self.c).sum(1)
        if f == "stribeck":
            Fc, Fs, vs = p["Fc"], p["Fc"] + p["Fs_extra"], p["vs"]
            for _ in range(self.n_sub):
                vel = self.x[:, 1]
                fric = (Fc + (Fs - Fc) * torch.exp(-(vel / vs) ** 2)) * torch.tanh(20.0 * vel)
                self.x = torch.einsum("bij,bj->bi", self.Ad, self.x) \
                    + self.bd * (u - fric).unsqueeze(-1)
            return (self.x * self.c).sum(1)
        if f == "boucwen":
            self.x = torch.einsum("bij,bj->bi", self.Ad, self.x) + self.bd * u.unsqueeze(-1)
            z = (self.x * self.c).sum(1)
            du = (u - self.uprev).clamp(-1.0, 1.0)
            self.uprev = u.clone()
            self.h = (self.h + du - p["beta"] * du.abs() * self.h
                      - p["gam"] * du * self.h.abs()).clamp(-3, 3)
            return p["alpha"] * z + (1 - p["alpha"]) * self.h
        if f == "drivetrain":
            dts = self.dt / self.n_sub
            for _ in range(self.n_sub):
                self.x = torch.einsum("bij,bj->bi", self.Ad, self.x) + self.bd * u.unsqueeze(-1)
                self.x[:, 3] = self.x[:, 3] - (p["Fc"] / p["Jl"]) \
                    * torch.tanh(20.0 * self.x[:, 3]) * dts
            return self.x[:, 3]
        raise ValueError(f)


def simulate(family: str, p: dict, u: torch.Tensor, dt: float) -> torch.Tensor:
    """u (T,B) -> y (T,B). Physically consistent (no normalization)."""
    st = Stepper(family, p, dt)
    return torch.stack([st.step(u[k]) for k in range(u.shape[0])])


def param_vector(family: str, p: dict):
    keys = sorted(p.keys())
    return torch.stack([p[k] for k in keys], dim=1), keys
