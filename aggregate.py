"""Multi-seed aggregation: recompute the transfer matrix for every finished
seed checkpoint and print each cell as mean +/- std, plus the real-plant
zero-shot numbers aggregated the same way.

    PLANTFORGE_DATA=... python -m plantforge.aggregate
Env: PF_SEEDS (default "0,1,2,3,4") -- seeds to look for.
"""
from __future__ import annotations

import os

import torch

from .evaluate import (FAMILIES, HOLD_FAMILY, TRAIN_RATES, CKPT_DIR, nmse)
from .realbench import load_model, nmse_on_windows

TOTAL_STEPS = 10000


def transfer_cells(mode: str):
    """(label, family, excitation, dt) rows, mirroring evaluate.report()."""
    ref_family = "wh" if mode == "wh_only" else "stribeck"
    cells = [("reference (train-like)", ref_family, "multisine", 0.05)]
    for dt in (0.10, 0.05, 0.02):
        cells.append((f"held-out family {HOLD_FAMILY} dt={dt:.2f}",
                      HOLD_FAMILY, "multisine", dt))
    for exc in ("chirp", "closedloop"):
        cells.append((f"held-out excitation {exc} (stribeck, dt=0.05)",
                      "stribeck", exc, 0.05))
    for fam in [f for f in FAMILIES if f != HOLD_FAMILY]:
        cells.append((f"held-out rate dt=0.05 {fam}", fam, "multisine", 0.05))
    return cells


def matrix(model, mode: str) -> dict:
    out = {}
    for label, fam, exc, dt in transfer_cells(mode):
        if label == "reference (train-like)" and mode != "wh_only":
            # dt=0.05 is a held-out rate (TRAIN_RATES = [0.10, 0.02]) --
            # average over the actually-trained rates instead, mirroring the
            # fix in evaluate.report() (2026-07-20 adversarial-review-fixes,
            # Finding 1). wh_only mode is unaffected: dt=0.05 IS wh_only's
            # only trained rate, so it was already a valid reference.
            out[label] = sum(nmse(model, fam, exc, r) for r in TRAIN_RATES) / len(TRAIN_RATES)
        else:
            out[label] = nmse(model, fam, exc, dt)
    return out


def aggregate_matrices(mats: list) -> dict:
    """label -> (mean, sample std [0 when n=1], n) across per-seed matrices."""
    out = {}
    for label in mats[0]:
        vals = [m[label] for m in mats]
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
        out[label] = (mean, std, n)
    return out


def mean_std_str(values: list) -> str:
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f} (n={n})"


def _finished_models(mode: str, seeds):
    models = []
    for s in seeds:
        name = f"eval_{mode}_s{s}.pt"
        ck_path = CKPT_DIR / name
        if not ck_path.exists():
            print(f"  (seed {s}: {name} missing -- skipped)")
            continue
        step = torch.load(ck_path, map_location="cpu")["step"]
        if step < TOTAL_STEPS:
            print(f"  (seed {s}: {name} unfinished at step {step} -- skipped)")
            continue
        models.append(load_model(name))
    return models


def main():
    seeds = [int(s) for s in os.environ.get("PF_SEEDS", "0,1,2,3,4").split(",")]
    realplant = {}
    try:
        from .realbench import silverbox_windows, cascaded_tanks_windows, boucwen_windows
        sb, sb_dt, sb_q = silverbox_windows()
        ct, ct_dt = cascaded_tanks_windows()
        bw, bw_dt, bw_q = boucwen_windows()
        realplant = {f"Silverbox (decimated {sb_q}x -> dt={sb_dt:.4f}s)": sb,
                     f"Cascaded_Tanks (native dt={ct_dt:.2f}s, extrapolation)": ct,
                     f"Bouc-Wen (decimated {bw_q}x -> dt={bw_dt:.4f}s)": bw}
    except Exception as e:
        print(f"(real-plant section skipped: {type(e).__name__}: {e})")

    for mode in ("wh_only", "corpus"):
        print(f"\n=== {mode}: transfer matrix, mean ± std over seeds ===")
        models = _finished_models(mode, seeds)
        if not models:
            print("  no finished checkpoints -- nothing to aggregate")
            continue
        mats = [matrix(m, mode) for m in models]
        agg = aggregate_matrices(mats)
        ref_mean = agg["reference (train-like)"][0]
        for label, (mean, std, n) in agg.items():
            print(f"  {label}: {mean:.4f} ± {std:.4f} (n={n})"
                  f"  [{mean / ref_mean:.1f}x ref]")
        for ds_label, windows in realplant.items():
            if windows is None:
                continue
            u, y = windows
            vals = [nmse_on_windows(m, u, y) for m in models]
            print(f"  real-plant {ds_label}: {mean_std_str(vals)}")


if __name__ == "__main__":
    main()
