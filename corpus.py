"""Corpus generation — shards of (u, y, params, identifiability) across the three axes:
family x excitation x rate. Resumable (skips existing shards). CPU-dominant.

    python -m plantforge.corpus --instances 200        # v0 demo corpus (~10 min CPU)
    python -m plantforge.corpus --instances 4000       # full-scale (leave running)

Shard = one (family, excitation, rate) cell holding all instances:
  u (T,B), y (T,B), theta (B,K), keys, rel_crlb (B,K), log10_cond (B)
Same instances (same seed) appear across all excitations and rates of a family — that
alignment is what makes cross-excitation / cross-rate comparisons well-posed.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import torch

from .families import FAMILIES, sample
from .excitation import EXCITATIONS, generate
from .identifiability import identifiability

OUT = pathlib.Path(os.environ.get("PLANTFORGE_DATA", "/home/coder/plantforge_data")) / "corpus"
RATES = [0.10, 0.05, 0.02]          # dt values: 10 / 20 / 50 Hz
T_PHYS = 12.8                        # physical seconds per trajectory


_FAMILY_IDX = {f: i for i, f in enumerate(FAMILIES)}
_EXC_IDX = {e: i for i, e in enumerate(EXCITATIONS)}


def gen_cell(family, exc, dt, n_inst, seed=0, ident=True, chunk=64):
    """One shard. Instances are drawn with a per-family seed so the SAME instances
    appear across all (exc, rate) cells of that family.

    Uses a fixed FAMILIES/EXCITATIONS index lookup for seeding (not
    hash(family)/hash(exc), which are randomized per Python process unless
    PYTHONHASHSEED is pinned -- the prior version was not actually
    deterministic across separate `python -m plantforge.corpus` runs).

    Non-finite (diverged) draws are filtered out and the chunk is topped up
    by re-drawing with a fresh seed, so gen_cell always returns exactly
    n_inst instances with finite u/y -- matching the datasheet's documented
    behavior (previously the datasheet's claim did not match this
    function's actual, filter-less behavior)."""
    T = round(T_PHYS / dt)
    us, ys, thetas, crlbs, conds = [], [], [], [], []
    keys = None
    collected = 0
    lo = 0
    while collected < n_inst:
        B = min(chunk, n_inst - collected)
        attempt = 0
        got = 0
        while got < B:
            attempt += 1
            gen_p = torch.Generator().manual_seed(
                seed * 7919 + _FAMILY_IDX[family] * 10007 + lo * 1000 + attempt)
            p = sample(family, B - got, gen_p)
            gen_u = torch.Generator().manual_seed(
                seed * 104729 + _EXC_IDX[exc] * 3571 + lo * 1000 + attempt)
            u, y = generate(family, p, exc, T, B - got, dt, gen_u)
            finite = torch.isfinite(u).all(dim=0) & torch.isfinite(y).all(dim=0)
            if not finite.all():
                u, y = u[:, finite], y[:, finite]
                p = {k: v[finite] for k, v in p.items()}
            if u.shape[1] == 0:
                continue   # whole sub-chunk diverged, retry with a fresh seed
            us.append(u); ys.append(y)
            from .families import param_vector
            th, keys = param_vector(family, p)
            thetas.append(th)
            if ident:
                idn = identifiability(family, p, u, dt)
                crlbs.append(idn["rel_crlb"]); conds.append(idn["log10_cond"])
            got += u.shape[1]
        collected += B
        lo += chunk
    shard = {"u": torch.cat(us, dim=1), "y": torch.cat(ys, dim=1),
             "theta": torch.cat(thetas), "keys": keys, "dt": dt,
             "family": family, "excitation": exc}
    if ident:
        shard["rel_crlb"] = torch.cat(crlbs)
        shard["log10_cond"] = torch.cat(conds)
    return shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=200)
    ap.add_argument("--no-ident", action="store_true")
    ap.add_argument("--families", default=",".join(FAMILIES))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fams = args.families.split(",")
    reg = {"families": fams, "excitations": EXCITATIONS, "rates_dt": RATES,
           "t_phys": T_PHYS, "instances_per_cell": args.instances,
           "identifiability": not args.no_ident}
    with open(OUT / "registry.json", "w") as f:
        json.dump(reg, f, indent=1)

    n_cells = len(fams) * len(EXCITATIONS) * len(RATES)
    done = 0
    for fam in fams:
        for exc in EXCITATIONS:
            for dt in RATES:
                done += 1
                path = OUT / f"{fam}_{exc}_dt{int(1/dt)}hz.pt"
                if path.exists():
                    print(f"[{done:2d}/{n_cells}] skip (exists) {path.name}")
                    continue
                shard = gen_cell(fam, exc, dt, args.instances,
                                 ident=not args.no_ident)
                torch.save(shard, path)
                mb = path.stat().st_size / 1e6
                print(f"[{done:2d}/{n_cells}] {path.name}  ({mb:.1f} MB)")
    print(f"\ncorpus -> {OUT}")


if __name__ == "__main__":
    main()
