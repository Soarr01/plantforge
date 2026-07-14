"""The excitation-signal axis — 4 classes, all defined in PHYSICAL time so the same
signal composes with the multi-rate axis.

  prbs      — pseudo-random binary sequence, fixed physical hold time (0.25 s)
  multisine — sum of 8 sines, random band-limited frequencies/phases
  chirp     — linear frequency sweep 0.05..3 Hz
  closedloop— TRUE sequential loop: a PI controller tracks a random reference against
              the actual plant stepper (u correlated with y — the hard ID case)
"""
from __future__ import annotations

import torch

from .families import Stepper

EXCITATIONS = ["prbs", "multisine", "chirp", "closedloop"]


def open_loop_input(kind: str, T: int, B: int, dt: float, gen: torch.Generator):
    t = torch.arange(T).float().unsqueeze(1) * dt                    # (T,1)
    if kind == "prbs":
        hold = 0.25
        n_holds = int(T * dt / hold) + 2
        bits = (torch.rand((n_holds, B), generator=gen) < 0.5).float() * 2 - 1
        idx = (t.squeeze(1) / hold).long().clamp(max=n_holds - 1)
        amp = 1.0 + 0.8 * torch.rand((1, B), generator=gen)
        return bits[idx] * amp
    if kind == "multisine":
        K = 8
        f = 0.05 + 1.8 * torch.rand((K, B), generator=gen)
        ph = 2 * torch.pi * torch.rand((K, B), generator=gen)
        a = (0.4 + 0.8 * torch.rand((K, B), generator=gen)) / K ** 0.5
        return (a * torch.sin(2 * torch.pi * f * t.unsqueeze(1) + ph)).sum(1)
    if kind == "chirp":
        f0 = 0.05 + 0.1 * torch.rand((1, B), generator=gen)
        f1 = 1.5 + 1.5 * torch.rand((1, B), generator=gen)
        Tend = T * dt
        amp = 1.0 + 0.6 * torch.rand((1, B), generator=gen)
        return amp * torch.sin(2 * torch.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * Tend)))
    raise ValueError(kind)


def generate(family: str, p: dict, kind: str, T: int, B: int, dt: float,
             gen: torch.Generator):
    """Returns (u, y), both (T,B), physically consistent with p at rate 1/dt."""
    if kind != "closedloop":
        u = open_loop_input(kind, T, B, dt, gen)
        st = Stepper(family, p, dt)
        y = torch.stack([st.step(u[k]) for k in range(T)])
        return u, y
    # true sequential closed loop: PI on the actual plant stepper, one-step output lag
    ref = open_loop_input("multisine", T, B, dt, gen) * 0.7
    kp = 0.8 + 0.8 * torch.rand((B,), generator=gen)
    ki = 0.3 + 0.5 * torch.rand((B,), generator=gen)
    st = Stepper(family, p, dt)
    u = torch.zeros(T, B)
    y = torch.zeros(T, B)
    e_int = torch.zeros(B)
    y_prev = torch.zeros(B)
    for k in range(T):
        e = ref[k] - y_prev
        e_int = (e_int + e * dt).clamp(-10, 10)
        u[k] = (kp * e + ki * e_int).clamp(-6, 6)
        y[k] = st.step(u[k])
        y_prev = y[k]
    return u, y
