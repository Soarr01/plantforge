# Audit C — Executable-Truth Audit (2026-07-15)

Repo: /data/nas07_new/PersonalData/phuocthien/plantforge
Branch: main (verified via `git branch --show-current`)
Pre-check: `git status --short` empty at start — tree was clean, proceeded.

## Check 1 — Full offline suite

Command: `python -m plantforge.tests.run_all`

Result: EXIT 0. 33 PASS lines across 6 module sections:
- PLANTFORGE (corpus/ground-truth): 6 PASS
- realbench offline: 12 PASS
- aggregate offline: 4 PASS
- baselines offline: 4 PASS
- ident_exp offline: 4 PASS
- ablation offline: 3 PASS

Total 6+12+4+4+4+3 = 33 PASS, no FAIL, no traceback.

**Verdict: PASS**

## Check 2 — Ablation CLI

Command: `PLANTFORGE_DATA=... CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 python -m plantforge.ablation`

Output:
```
=== architecture ablation: corpus recipe, seed 0, 2 target cells ===
  default  (d=160 L=5 params=1,582,881): reference=0.0210  family_backlash_dt.05=0.2865  (13.6x ref)
  narrow   (d= 80 L=5 params=407,441): reference=0.0261  family_backlash_dt.05=0.2939  (11.2x ref)
  wide     (d=320 L=5 params=6,237,761): reference=0.0180  family_backlash_dt.05=0.2941  (16.3x ref)
  shallow  (d=160 L=2 params=655,041): reference=0.0398  family_backlash_dt.05=0.2835  (7.1x ref)
  deep     (d=160 L=8 params=2,510,721): reference=0.0151  family_backlash_dt.05=0.2901  (19.2x ref)
```
EXIT 0. All 5 variants printed real numbers, no MISSING rows, no traceback.

Cross-check against `docs/superpowers/results/2026-07-15-architecture-ablation-results.md`:
all 5 lines (default, narrow, wide, shallow, deep) match exactly — same param
counts, same reference nMSE, same family_backlash_dt.05 nMSE, same ratios.
**No number mismatches found.**

**Verdict: PASS**

## Check 3 — Aggregate CLI (light, PF_SEEDS=0)

Command: `PLANTFORGE_DATA=... CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 PF_SEEDS=0 python -m plantforge.aggregate`

Output: both `wh_only` and `corpus` transfer matrices printed, every row
`(n=1)` with `± 0.0000`, plus two real-plant lines (Silverbox, Cascaded_Tanks)
under each mode. EXIT 0, no traceback.

**Verdict: PASS** (ran light seed-0 variant only, as instructed — full 5-seed
run not executed, ~5 min cost not incurred)

## Check 4 — Realbench CLI

Command: `PLANTFORGE_DATA=... CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 python -m plantforge.realbench`

Output: Silverbox nMSE for both wh_only (0.8128) and corpus (0.3018) models;
Cascaded_Tanks nMSE for both wh_only (0.3605) and corpus (0.1223) models;
WienerHammerBenchMark SKIPPED with expected message ("record duration (1.54s)
shorter than one context window at any trained rate (min 4.48s)"). EXIT 0,
no traceback.

**Verdict: PASS**

## Check 5 — Ident_exp CLI (light, PF_SEEDS=0 PF_IDENT_N=100)

Command: `PLANTFORGE_DATA=... CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 PF_SEEDS=0 PF_IDENT_N=100 python -m plantforge.ident_exp`

Output: PRIMARY within-cell Spearman block (median r=-0.114, mean r=-0.090,
40 cells, full per-cell breakdown table); SECONDARY pooled block (pooled
Spearman + per-family breakdown); quartile table (Q1-Q4) with both mean and
median nMSE columns showing the mean/median divergence (heavy-tail artifact).
EXIT 0, no traceback.

**Verdict: PASS**

## Check 6 — Figure regeneration

Command (from repo root): `PLANTFORGE_DATA=... CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 python figures/make_figures.py`

Output: 4 png+pdf pairs written (fig1_transfer_matrix, fig2_real_plant,
fig3_within_cell_spearman, fig4_quartile_artifact). EXIT 0.

`git status --short` after: only the 4 `figures/*.pdf` files showed as
modified (binary diff only — matplotlib embeds a creation timestamp in PDF
metadata). All 4 `figures/*.png` files were byte-identical (no diff shown).
`git diff --stat` confirmed: 4 files changed, 0 insertions/deletions (binary),
all under `figures/`, no `.py`/`.md`/`.tex` changes.

Restored with `git checkout -- figures/` — clean after.

**Verdict: PASS**

## Check 7 — Training-script skip paths

Command: `SEEDS=0 CUDA_VISIBLE_DEVICES=6 scripts/train_seeds.sh`
Output:
```
== seed 0 headline: done (10000/10000)
== seed 0 corpus: done (10000/10000)
== ALL SEEDS DONE
```
EXIT 0, instant (no training launched — both checkpoints already at 10000
steps).

Command: `CUDA_VISIBLE_DEVICES=6 bash scripts/train_ablation.sh`
Output:
```
== narrow (d=80 L=5): done (10000/10000)
== wide (d=320 L=5): done (10000/10000)
== shallow (d=160 L=2): done (10000/10000)
== deep (d=160 L=8): done (10000/10000)
== ALL VARIANTS DONE
```
EXIT 0, instant, all 4 variants report done since already trained.

**Verdict: PASS**

## Check 8 — Paper compiles

Command: `cd paper && pdflatex -interaction=nonstopmode main.tex > /tmp/audit_latex.log 2>&1; grep -ciE "^!" /tmp/audit_latex.log`

Result: pdflatex EXIT 0, `grep -ciE "^!"` returned `0` (zero LaTeX errors).
`main.pdf` regenerated (`git status --short paper/` showed `M paper/main.pdf`,
binary/timestamp diff only).

Restored with `git checkout -- paper/` — clean after.

**Verdict: PASS**

## Check 9 — Final state

`git status --short` at the end of all checks: empty except this report file
itself under `.superpowers/final-review/` (the designated output location for
this audit, untracked, not a modification of any tracked file). All tracked
files touched during the audit (`figures/*.pdf`, `paper/main.pdf`, plus
LaTeX aux/log/out byproducts which matched pre-existing tracked content) were
restored via `git checkout -- figures/` and `git checkout -- paper/`.

**Verdict: PASS — final git state clean (only the audit report itself, an
expected untracked deliverable, present).**

## Summary

| # | Check | Verdict |
|---|-------|---------|
| 1 | Full offline suite (33 PASS) | PASS |
| 2 | Ablation CLI (numbers match docs exactly) | PASS |
| 3 | Aggregate CLI (light, PF_SEEDS=0) | PASS |
| 4 | Realbench CLI | PASS |
| 5 | Ident_exp CLI (light) | PASS |
| 6 | Figure regeneration (PDF-only binary diff, restored) | PASS |
| 7 | Training-script skip paths (both instant, exit 0) | PASS |
| 8 | Paper compiles (0 LaTeX errors, restored) | PASS |
| 9 | Final git state clean | PASS |

No discrepancies found. Every CLI/script/test the repo claims to run, does
run, and check 2's printed numbers match the recorded results doc exactly
(same checkpoints, deterministic eval, no drift).
