"""Do the corpus's identifiability annotations predict in-context difficulty?
Scores corpus-model checkpoints per-instance on corpus shards (which carry
per-instance rel-CRLB and FIM condition annotations) and reports Spearman
correlations plus a quartile table.

Only dt in {0.05, 0.02} cells are usable: shards hold T_PHYS=12.8 s, so
10 Hz cells have 128 < D=224 samples.

    PLANTFORGE_DATA=... python -m plantforge.ident_exp
Env: PF_SEEDS (default "0,1,2,3,4"), PF_IDENT_N (instances/cell, default 1000).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from scipy.stats import spearmanr

from .corpus import OUT
from .evaluate import _norm, T_CTX, D, DEV, CKPT_DIR
from .families import FAMILIES
from .excitation import EXCITATIONS
from .realbench import load_model

USABLE_DTS = (0.05, 0.02)
CHUNK = 256
TOTAL_STEPS = 10000


def qry_stats(model, u, y):
    """Per-instance query-horizon (mse, power): num/den are (B,) tensors and
    num.mean()/den.mean() reproduces nmse_on_windows' batch ratio exactly."""
    u_n, y_n = _norm(u, y)
    with torch.no_grad():
        pred = model(u_n, y_n)
    num = ((pred[:, T_CTX:] - y_n[:, T_CTX:]) ** 2).mean(dim=1)
    den = (y_n[:, T_CTX:] ** 2).mean(dim=1)
    return num, den


def nmse_per_instance(model, u, y):
    num, den = qry_stats(model, u, y)
    return num / (den + 1e-12)


def quartile_table(metric: np.ndarray, nmse: np.ndarray):
    """Bin instances into metric quartiles; return
    [(bin mean metric, bin mean nMSE, bin count)] x 4."""
    qs = np.quantile(metric, [0.25, 0.5, 0.75])
    bins = np.digitize(metric, qs)
    rows = []
    for b in range(4):
        mask = bins == b
        rows.append((float(metric[mask].mean()), float(nmse[mask].mean()),
                     int(mask.sum())))
    return rows


def _corpus_models(seeds):
    models = []
    for s in seeds:
        name = f"eval_corpus_s{s}.pt"
        ck_path = CKPT_DIR / name
        if not ck_path.exists():
            continue
        if torch.load(ck_path, map_location="cpu")["step"] < TOTAL_STEPS:
            continue
        models.append(load_model(name))
    return models


def _score_cell(models, u, y):
    """Mean-over-seeds per-instance nMSE, chunked to bound GPU memory."""
    per_model = []
    for m in models:
        chunks = [nmse_per_instance(m, u[i:i + CHUNK], y[i:i + CHUNK])
                  for i in range(0, u.shape[0], CHUNK)]
        per_model.append(torch.cat(chunks))
    return torch.stack(per_model).mean(dim=0).cpu().numpy()


def main():
    seeds = [int(s) for s in os.environ.get("PF_SEEDS", "0,1,2,3,4").split(",")]
    n_per_cell = int(os.environ.get("PF_IDENT_N", "1000"))
    models = _corpus_models(seeds)
    if not models:
        print("no finished corpus checkpoints found -- train first")
        return
    print(f"=== identifiability vs in-context difficulty "
          f"(corpus model, {len(models)} seed(s), {n_per_cell}/cell) ===")

    fams, crlbs, conds, nmses = [], [], [], []
    for fam in FAMILIES:
        for exc in EXCITATIONS:
            for dt in USABLE_DTS:
                path = OUT / f"{fam}_{exc}_dt{int(1 / dt)}hz.pt"
                if not path.exists():
                    print(f"  (shard missing, skipped: {path.name})")
                    continue
                shard = torch.load(path, map_location="cpu")
                u = shard["u"].t()[:n_per_cell, :D].to(DEV)   # (B, D)
                y = shard["y"].t()[:n_per_cell, :D].to(DEV)
                v = _score_cell(models, u, y)
                crlb = shard["rel_crlb"][:n_per_cell].max(dim=1).values.numpy()
                cond = shard["log10_cond"][:n_per_cell].numpy()
                fams += [fam] * len(v)
                crlbs.append(crlb); conds.append(cond); nmses.append(v)

    crlb = np.concatenate(crlbs); cond = np.concatenate(conds)
    nmse = np.concatenate(nmses); fams = np.array(fams)
    ok = np.isfinite(crlb) & np.isfinite(cond) & np.isfinite(nmse)
    crlb, cond, nmse, fams = crlb[ok], cond[ok], nmse[ok], fams[ok]
    print(f"  instances scored: {len(nmse)} (dropped {int((~ok).sum())} non-finite)")

    r, p = spearmanr(crlb, nmse)
    print(f"  Spearman(max rel-CRLB, nMSE): r={r:.3f} (p={p:.1e})")
    r2, p2 = spearmanr(cond, nmse)
    print(f"  Spearman(log10 FIM cond, nMSE): r={r2:.3f} (p={p2:.1e})")
    for fam in FAMILIES:
        m = fams == fam
        if m.sum() > 10:
            rf, pf = spearmanr(crlb[m], nmse[m])
            print(f"    {fam}: r={rf:.3f} (p={pf:.1e}, n={int(m.sum())})")

    print("  quartiles of max rel-CRLB -> mean nMSE:")
    for i, (mm, mn, n) in enumerate(quartile_table(crlb, nmse), 1):
        print(f"    Q{i}: metric~{mm:.3g}  nMSE={mn:.4f}  (n={n})")


if __name__ == "__main__":
    main()
