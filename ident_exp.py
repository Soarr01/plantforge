"""Do the corpus's identifiability annotations predict in-context difficulty?
Scores corpus-model checkpoints per-instance on corpus shards (which carry
per-instance rel-CRLB and FIM condition annotations) and reports Spearman
correlations plus a quartile table.

Analysis structure (PRIMARY vs SECONDARY), per statistical review:
Pooling instances across all 40 cells (family x excitation x rate) before
correlating confounds the hypothesis: cell structure changes both
identifiability and model difficulty simultaneously, which can manufacture
a pooled correlation of either sign (Simpson's paradox) independent of the
within-cell relationship. The hypothesis under test is intrinsically
within-cell -- holding (family, excitation, rate) fixed, do harder-to-identify
parameter draws yield higher model difficulty? So the PRIMARY statistic is
each cell's own Spearman(max rel-CRLB, nMSE), aggregated (median/mean) across
cells. The pooled-across-cells Spearman (and its per-family breakdown) is kept
as a SECONDARY, clearly-labeled diagnostic, since it mixes within-cell signal
with between-cell confounds.

Two instance-level confounds can distort the within-cell r and are controlled
for, per reviewer request:
1. Query-horizon power (den) confound: per-instance nMSE = num/den, and a
   near-zero den (query horizon nearly flat after normalization) mechanically
   inflates nMSE regardless of identifiability. `filter_low_power` drops each
   cell's bottom decile of den and the correlation is recomputed
   ("power-controlled").
2. Annotation dynamic range: cells where max rel-CRLB is nearly constant
   contribute pure rank noise to r. Per-cell rel-CRLB IQR is reported, and a
   "range-filtered" aggregate excludes near-zero-IQR cells.

The quartile table reports both bin-mean and bin-median nMSE: per-instance
nMSE is heavy-tailed (can reach ~1e9 for near-flat query-horizon instances,
e.g. deadzone outputs, that can dominate a bin's mean), so the median is the
defensible summary; the mean is kept alongside for reference.

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
from .identifiability import identifiability
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
    [(bin mean metric, bin mean nMSE, bin median nMSE, bin count)] x 4.
    Bin-mean nMSE is heavy-tail-dominated (a single ~1e9 instance can swamp
    a 2000-instance bin); bin-median nMSE is the defensible summary and is
    reported alongside it. Empty bins (possible with heavy quantile ties)
    are reported with n=0 and nan placeholders rather than raising a numpy
    empty-slice warning."""
    qs = np.quantile(metric, [0.25, 0.5, 0.75])
    bins = np.digitize(metric, qs)
    rows = []
    for b in range(4):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            rows.append((float("nan"), float("nan"), float("nan"), 0))
            continue
        rows.append((float(metric[mask].mean()), float(nmse[mask].mean()),
                     float(np.median(nmse[mask])), n))
    return rows


def filter_low_power(crlb, v, den, q=0.1):
    """Drop instances in the bottom q-quantile of query-horizon power (den):
    near-flat query horizons mechanically explode nMSE = num/den and are a
    live confound for the identifiability correlation."""
    thresh = np.quantile(den, q)
    keep = den > thresh
    return crlb[keep], v[keep], den[keep]


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
    cell_rows = []  # (fam, exc, dt, r_cell, p_cell, n, r_pow, p_pow, n_pow, iqr)
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
                # Recompute identifiability on the SAME 224-sample window the
                # prediction task uses, rather than reusing the shard's
                # full-trajectory annotation (128-640 samples depending on
                # rate) -- the two must cover the same record length for the
                # comparison to be meaningful. theta/keys in the shard fully
                # capture the sampled p dict (param_vector's forward
                # direction), so this reconstruction is exact and lossless.
                theta_window = shard["theta"][:n_per_cell]           # (n_per_cell, K)
                keys = shard["keys"]
                p_window = {k: theta_window[:, i] for i, k in enumerate(keys)}
                # identifiability() expects u as (T, B); shard["u"] is
                # already stored (T_full, B_total) uncut, so slice directly
                # without transposing -- this is a DIFFERENT slice of the
                # same underlying tensor than the (B, D)-shaped `u` above,
                # which is used for the model's forward pass.
                u_for_ident = shard["u"][:D, :n_per_cell]            # (D, n_per_cell) = (T, B)
                idn = identifiability(fam, p_window, u_for_ident, dt)
                crlb = idn["rel_crlb"].max(dim=1).values.numpy()
                cond = idn["log10_cond"].numpy()
                # Query-horizon power, model-independent (no forward pass needed):
                # same normalization qry_stats uses for `den`, computed once per cell.
                with torch.no_grad():
                    _, y_n = _norm(u, y)
                den = (y_n[:, T_CTX:] ** 2).mean(dim=1).cpu().numpy()
                fams += [fam] * len(v)
                crlbs.append(crlb); conds.append(cond); nmses.append(v)

                # Deliberately omits `cond`: cond isn't used in the within-cell stat.
                ok_cell = np.isfinite(crlb) & np.isfinite(v) & np.isfinite(den)
                crlb_f, v_f, den_f = crlb[ok_cell], v[ok_cell], den[ok_cell]
                n_cell = len(v_f)
                if n_cell > 10:
                    r_cell, p_cell = spearmanr(crlb_f, v_f)
                    iqr = float(np.subtract(*np.percentile(crlb_f, [75, 25])))
                    crlb_p, v_p, _ = filter_low_power(crlb_f, v_f, den_f)
                    n_pow = len(v_p)
                    if n_pow > 10:
                        r_pow, p_pow = spearmanr(crlb_p, v_p)
                    else:
                        r_pow, p_pow = float("nan"), float("nan")
                    cell_rows.append((fam, exc, dt, r_cell, p_cell, n_cell,
                                       r_pow, p_pow, n_pow, iqr))

    crlb = np.concatenate(crlbs); cond = np.concatenate(conds)
    nmse = np.concatenate(nmses); fams = np.array(fams)
    ok = np.isfinite(crlb) & np.isfinite(cond) & np.isfinite(nmse)
    crlb, cond, nmse, fams = crlb[ok], cond[ok], nmse[ok], fams[ok]
    print(f"  instances scored: {len(nmse)} (dropped {int((~ok).sum())} non-finite)")

    cell_r = np.array([c[3] for c in cell_rows])
    n_cells = len(cell_rows)
    n_pos = int((cell_r > 0).sum())
    n_sig_pos = sum(1 for c in cell_rows if c[3] > 0 and c[4] < 0.05)
    print("  PRIMARY -- within-cell Spearman(max rel-CRLB, nMSE), "
          "aggregated over cells:")
    if n_cells:
        print(f"    median r = {np.median(cell_r):.3f}  "
              f"mean r = {cell_r.mean():.3f}  "
              f"(cells: {n_cells}, positive r: {n_pos}/{n_cells}, "
              f"p<0.05 & r>0: {n_sig_pos}/{n_cells} "
              f"(descriptive; no multiplicity correction -- "
              f"~2/40 expected by chance))")
        cell_r_pow = np.array([c[6] for c in cell_rows if np.isfinite(c[6])])
        if len(cell_r_pow):
            print(f"    power-controlled: median r = {np.median(cell_r_pow):.3f}  "
                  f"mean r = {cell_r_pow.mean():.3f}  "
                  f"(excl. bottom-decile den per cell)")
        else:
            print("    power-controlled: (no cells with n > 10 after filtering)")
        iqr_keep = [c[3] for c in cell_rows if c[9] > 1e-6]
        n_iqr_keep = len(iqr_keep)
        if n_iqr_keep:
            print(f"    range-filtered: median r = {np.median(iqr_keep):.3f}  "
                  f"(cells with rel-CRLB IQR > 1e-6: {n_iqr_keep}/{n_cells})")
        else:
            print(f"    range-filtered: (no cells with rel-CRLB IQR > 1e-6: "
                  f"0/{n_cells})")
    else:
        print("    (no cells with n > 10 -- cannot aggregate)")
    for fam, exc, dt, r_cell, p_cell, n_cell, r_pow, p_pow, n_pow, iqr in cell_rows:
        pow_str = (f"r_pow={r_pow:+.3f} (p={p_pow:.1e}, n={n_pow})"
                   if np.isfinite(r_pow) else "r_pow=n/a")
        print(f"      {fam:12s} {exc:12s} dt={dt:<5g} "
              f"r={r_cell:+.3f} (p={p_cell:.1e}, n={n_cell})  "
              f"{pow_str}  IQR={iqr:.3g}")

    print("  SECONDARY (pooled across cells -- confounded by between-cell "
          "structure; see within-cell above):")
    r, p = spearmanr(crlb, nmse)
    print(f"    Spearman(max rel-CRLB, nMSE): r={r:.3f} (p={p:.1e})")
    r2, p2 = spearmanr(cond, nmse)
    print(f"    Spearman(log10 FIM cond, nMSE): r={r2:.3f} (p={p2:.1e})")
    for fam in FAMILIES:
        m = fams == fam
        if m.sum() > 10:
            rf, pf = spearmanr(crlb[m], nmse[m])
            print(f"      {fam}: r={rf:.3f} (p={pf:.1e}, n={int(m.sum())})")

    print("  quartiles of max rel-CRLB -> nMSE (mean is heavy-tail-dominated; "
          "median is the defensible summary):")
    for i, (mm, mn, md, n) in enumerate(quartile_table(crlb, nmse), 1):
        if n == 0:
            print(f"    Q{i}: (empty bin, n=0)")
            continue
        print(f"    Q{i}: metric~{mm:.3g}  mean nMSE={mn:.4f}  "
              f"median nMSE={md:.4f}  (n={n})")


if __name__ == "__main__":
    main()
