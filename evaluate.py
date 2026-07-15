"""Track-2 evaluation: in-context SysID transfer across the corpus axes.

Two shipped experiments (GPU recommended, checkpoint-resumable):
  headline  — model trained on WH-only white-noise (the incumbent recipe, from the
              gate) evaluated across corpus families/rates: reproduces the 32x/10x
              degradation that motivates the corpus.
  corpus    — model trained on 4-of-5 corpus families (multisine+prbs, 2 rates)
              evaluated on the HELD-OUT family + held-out rate + held-out excitation:
              the corpus's value claim (broad training transfers where narrow fails).

    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python -m plantforge.evaluate corpus
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import torch
import torch.nn as nn

from .families import FAMILIES, sample
from .excitation import generate

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T_CTX, T_QRY = 192, 32
D = T_CTX + T_QRY
CKPT_DIR = pathlib.Path(os.environ.get("PLANTFORGE_DATA", "/home/coder/plantforge_data"))
SEED = int(os.environ.get("PF_SEED", "0"))   # training seed: init + data pool draws
WIDTH = int(os.environ.get("PF_WIDTH", "160"))    # InContextSysID d
LAYERS = int(os.environ.get("PF_LAYERS", "5"))    # InContextSysID layers
HOLD_FAMILY = os.environ.get("PF_HOLD_FAMILY", "backlash")   # held-out family
TRAIN_RATES = [0.10, 0.02]          # held-out rate: 0.05
TRAIN_EXC = ["multisine", "prbs"]   # held-out excitation: chirp, closedloop


class InContextSysID(nn.Module):
    def __init__(self, d=160, layers=5, heads=8):
        super().__init__()
        self.inp = nn.Linear(2, d)
        self.pos = nn.Parameter(0.02 * torch.randn(D, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(d, 1)
        self.register_buffer("mask", torch.triu(torch.ones(D, D) * float("-inf"), 1))

    def forward(self, u, y):
        yprev = torch.zeros_like(y)
        yprev[:, 1:] = y[:, :-1]
        yprev[:, T_CTX:] = 0.0
        x = self.inp(torch.stack([u, yprev], dim=-1)) + self.pos[None]
        return self.out(self.tr(x, mask=self.mask)).squeeze(-1)


def _norm(u, y):
    """Per-series normalization — a METHOD choice, applied identically everywhere."""
    return u / (u.std(1, keepdim=True) + 1e-6), y / (y.std(1, keepdim=True) + 1e-6)


def make_batch(family, exc, dt, B, seed):
    gen = torch.Generator().manual_seed(seed)
    p = sample(family, B, gen)
    u, y = generate(family, p, exc, D, B, dt, gen)
    u, y = u.t().contiguous(), y.t().contiguous()
    u, y = _norm(u, y)
    return u.to(DEV), y.to(DEV)


POOL_N = 240                        # fresh batches per run; iterated (data gen dominates
                                    # wall-clock otherwise — reuse is fine for in-context)


def _ckpt_name(mode: str) -> str:
    """eval_{mode}_s{SEED}.pt for the default architecture (160/5) and
    default held-out family (backlash) -- unchanged from before this plan,
    so the already-trained default checkpoints stay loadable with zero
    retraining. Non-default width/layers and/or non-default held family
    each add an explicit suffix component so they never collide with the
    default name or each other."""
    suffix = ""
    if (WIDTH, LAYERS) != (160, 5):
        suffix += f"_d{WIDTH}L{LAYERS}"
    if HOLD_FAMILY != "backlash":
        suffix += f"_ho{HOLD_FAMILY.upper()}"
    return f"eval_{mode}_s{SEED}{suffix}.pt"


def build_pool(mode, run_salt):
    pool = []
    i, tries = 0, 0
    while len(pool) < POOL_N and tries < 3 * POOL_N:
        tries += 1
        if mode == "wh_only":
            u, y = make_batch("wh", "multisine", 0.05, 96, 50_000 + run_salt + tries)
        else:
            fams = [f for f in FAMILIES if f != HOLD_FAMILY]
            fam = fams[i % len(fams)]
            exc = TRAIN_EXC[(i // len(fams)) % len(TRAIN_EXC)]
            dt = TRAIN_RATES[(i // (len(fams) * len(TRAIN_EXC))) % len(TRAIN_RATES)]
            u, y = make_batch(fam, exc, dt, 96, 60_000 + run_salt + tries)
        if torch.isfinite(u).all() and torch.isfinite(y).all() and y.abs().max() < 1e4:
            pool.append((u, y))
            i += 1
        # else: silently drop the diverged batch (e.g. closed-loop PI unstable on a
        # resonant drivetrain instance) — logged rarity, not a crash
    return pool


def nmse(model, family, exc, dt, seeds=range(900, 906), B=96):
    tot, n = 0.0, 0
    for sd in seeds:
        u, y = make_batch(family, exc, dt, B, sd)
        if not (torch.isfinite(u).all() and torch.isfinite(y).all()):
            continue                                   # skip diverged closed-loop draws
        with torch.no_grad():
            pred = model(u, y)
        tot += (((pred[:, T_CTX:] - y[:, T_CTX:]) ** 2).mean()
                / (y[:, T_CTX:] ** 2).mean()).item()
        n += 1
    return tot / max(n, 1)


def run(mode: str, total_steps=10000, budget_s=500.0):
    torch.manual_seed(SEED)
    ck_path = CKPT_DIR / _ckpt_name(mode)
    model = InContextSysID(d=WIDTH, layers=LAYERS).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    done = 0
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=DEV)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); done = ck["step"]
        print(f"[{mode}] resumed at step {done}")
    t0 = time.time()
    print(f"[{mode}] building fresh data pool ({POOL_N} batches) ...")
    pool = build_pool(mode, run_salt=done * 13 + SEED * 1_000_000)
    print(f"[{mode}] pool ready ({time.time()-t0:.0f}s); training ...")
    train_t0 = time.time()
    i = done
    while i < total_steps and time.time() - train_t0 < budget_s:
        lr = 5e-4 * min(1.0, (i + 1) / 300) * (0.02 + 0.98 * 0.5 *
             (1 + torch.cos(torch.tensor(3.14159 * min(i / total_steps, 1.0))).item()))
        for g in opt.param_groups:
            g["lr"] = lr
        u, y = pool[i % POOL_N]
        loss = ((model(u, y)[:, T_CTX:] - y[:, T_CTX:]) ** 2).mean()
        if not torch.isfinite(loss):
            print(f"  [{mode}] step {i}: non-finite loss, skipping batch {i % POOL_N}")
            pool[i % POOL_N] = pool[(i + 1) % POOL_N]   # drop the poisoned batch
            i += 1
            continue
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if i % 1000 == 0:
            print(f"  [{mode}] step {i:5d} mse {loss.item():.4f}")
        i += 1
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": i}, ck_path)
    print(f"[{mode}] checkpoint at step {i}")
    if i < total_steps:
        print(f"[{mode}] NOT DONE — rerun to continue ({i}/{total_steps})")
        return model, False
    return model, True


def report(model, mode):
    print(f"\n=== {mode}: transfer matrix (in-context nMSE) ===")
    ref = nmse(model, "wh" if mode == "wh_only" else "stribeck", "multisine", 0.05)
    print(f"  reference (train-like): {ref:.4f}")
    print(f"  held-out family {HOLD_FAMILY}:")
    for dt in (0.10, 0.05, 0.02):
        v = nmse(model, HOLD_FAMILY, "multisine", dt)
        print(f"    dt={dt:.2f}: {v:.4f}  ({v/ref:.1f}x)")
    print(f"  held-out excitation (chirp/closedloop, stribeck, dt=0.05):")
    for exc in ("chirp", "closedloop"):
        v = nmse(model, "stribeck", exc, 0.05)
        print(f"    {exc}: {v:.4f}  ({v/ref:.1f}x)")
    print(f"  held-out rate dt=0.05 across trained families:")
    for fam in [f for f in FAMILIES if f != HOLD_FAMILY]:
        v = nmse(model, fam, "multisine", 0.05)
        print(f"    {fam}: {v:.4f}  ({v/ref:.1f}x)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    budget = float(os.environ.get("PF_BUDGET", "500"))
    model, finished = run("wh_only" if mode == "headline" else "corpus", budget_s=budget)
    if finished:
        report(model, "wh_only" if mode == "headline" else "corpus")
