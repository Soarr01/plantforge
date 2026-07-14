# Paper-strengthening experiments: multi-seed, classical baselines, identifiability

Status: approved design, not yet implemented.

## Purpose

Close the three gaps a Datasets & Benchmarks reviewer would flag first:
(A) all headline numbers are single-seed; (B) no baseline besides the two
in-house transformers; (C) the identifiability annotations — a selling point
of the corpus — are never used in an experiment. Approach: minimal extensions
to existing modules, no new framework.

## Experiment A — multi-seed with error bars

**Code changes (`evaluate.py`, minimal):**
- Read `SEED = int(os.environ.get("PF_SEED", "0"))` at module level.
- `run()` uses `torch.manual_seed(SEED)` (was hardcoded 0) and checkpoint
  path `eval_{mode}_s{SEED}.pt` (was `eval_{mode}.pt`).
- Training-data pool salt incorporates the seed:
  `run_salt = done * 13 + SEED * 1_000_000` — seed 0 reproduces the old
  salt exactly, so the two existing checkpoints ARE seed 0.
- Evaluation seeds (900–905 in `nmse`) stay fixed across all seeds/models —
  paired comparison on identical eval data.

**Checkpoint migration:** rename `eval_wh_only.pt` → `eval_wh_only_s0.pt`
and `eval_corpus.pt` → `eval_corpus_s0.pt` (they were trained with
`torch.manual_seed(0)` and the old salt, i.e. they are exactly seed 0).
`realbench.py`'s `main()` updates its two checkpoint names accordingly.

**Runs:** seeds 0–4 × modes {headline, corpus} = 10 total; seed 0 already
trained, so 8 new runs (~25–40 min each on one RTX 2080 Ti; driver script
loops seed × mode, skips checkpoints already at step 10000, reruns
until done — same pattern as the existing eval driver, with the
budget-accounting fix already in `evaluate.py`).

**Aggregation (`plantforge/aggregate.py`, new):**
- For each mode, load every `eval_{mode}_s{seed}.pt` that reached
  step 10000; recompute the transfer matrix per seed (deterministic given
  the fixed eval seeds); print each cell as `mean±std (n=<seeds>)`.
- Real-plant: for each seed's checkpoint, `nmse_on_windows` on the
  Silverbox / Cascaded_Tanks windows (reusing `realbench`'s loaders);
  report mean±std per dataset per mode.
- CLI: `python -m plantforge.aggregate`. Skips (with a printed note) any
  seed whose checkpoint is missing or unfinished rather than failing.

## Experiment B — classical SysID baselines (`plantforge/baselines.py`, new)

Zero new dependencies: ARX and polynomial NARX (degree 2), both fit by
numpy least squares.

**Protocol (fairness-critical):** identical to the transformer's: per
window, fit on the 192-sample context, then **free-run** simulate the
32-sample query horizon given only u (the transformer's `yprev` is zeroed
in the query region, so it also never sees ground-truth y there).
Normalization: `_norm` per window, same as `nmse_on_windows`. Metric: same
query-horizon nMSE formula.

**Models:**
- ARX(na=nb=k): y_t regressed on k lags of y and k lags of u, lstsq.
- poly-NARX: same lags plus degree-2 products of the lag vector, ridge
  least squares (small λ for conditioning).
- Order selection per window: k ∈ {2, 4, 8} chosen by one-step-ahead error
  on the last 32 samples of the context (never touches the query).

**Evaluation surfaces:**
1. Synthetic: the same cells as `evaluate.report()` (train-like reference,
   held-out family × 3 rates, held-out excitations, held-out rate across
   families), same eval seeds 900–905, B=96 — i.e. the baseline numbers
   drop directly into the existing transfer-matrix rows.
2. Real-plant: the same Silverbox / Cascaded_Tanks windows from
   `realbench` (lazy import — network dependency stays confined).

CLI: `python -m plantforge.baselines` prints both tables.

**Tests (offline, in `tests/test_baselines.py`):** ARX recovers a known ARX
process (near-zero nMSE); poly-NARX is at least as good as ARX on linear
data; order selection returns a member of the candidate set; free-run
does not peek at query y (perturbing query-horizon y leaves predictions
bit-identical).

## Experiment C — identifiability predicts difficulty (`plantforge/ident_exp.py`, new)

**Claim to demonstrate:** the per-instance rel-CRLB / FIM annotations
predict in-context prediction difficulty — i.e. they are informative
dataset metadata, not decoration.

**Method:**
- Add `nmse_per_instance(model, u, y) -> (B,) tensor` (per-instance
  query-horizon nMSE; the batch `nmse_on_windows` is its mean).
- Data: the existing corpus shards (they already carry `rel_crlb` and
  `log10_cond` per instance). Constraint discovered at design time: shards
  hold T_PHYS=12.8 s, so 10 Hz cells have only 128 samples < D=224 —
  **usable cells are dt ∈ {0.05, 0.02} only** (256 / 640 samples; slice
  the first 224). This is a documented limitation, not a blocker.
- Score the corpus-model checkpoints (all available seeds; per-instance
  nMSE averaged over seeds) on every usable cell.
- Identifiability metrics per instance: worst-case parameter
  `max_k rel_crlb[k]` (primary) and `log10_cond` (secondary).
- Report: Spearman correlation (scipy.stats, already a dependency) between
  each metric and per-instance nMSE — overall, and per family; plus a
  quartile table (bin instances by identifiability quartile → mean nMSE
  per bin) for readability.

CLI: `python -m plantforge.ident_exp` (needs corpus shards on disk and at
least one finished corpus checkpoint).

**Tests (offline, in `tests/test_ident_exp.py`):** `nmse_per_instance`'s
mean equals `nmse_on_windows` on the same batch; quartile binning is
correct on hand-built data.

## Execution order

1. Implement + test B and C (fast, CPU/inference-only, use seed-0
   checkpoints and existing corpus shards).
2. In parallel, launch A's 8 training runs in the background.
3. When all seeds finish: run `aggregate`, then re-run B/C reports if the
   extra seeds are wanted in their outputs (C averages over available
   seeds automatically).

## Global constraints

- Do not modify `families.py`, `excitation.py`, `identifiability.py`,
  `corpus.py`. `evaluate.py` gets ONLY the seed changes described in A.
  `realbench.py` gets ONLY the two checkpoint-name updates.
- Reuse `evaluate`'s `_norm`, `T_CTX`, `D`, `DEV`, `CKPT_DIR`, `nmse`
  machinery and `realbench`'s loaders/`nmse_on_windows` — no reimplementing.
- New modules must not import `nonlinear_benchmarks` at top level (lazy
  only, same rule as `realbench.py`).
- All work on a feature branch off `main`; `PLANTFORGE_DATA` env var
  convention unchanged.

## Out of scope

- README rewrite / repo-publishing hygiene (LICENSE, packaging) — separate
  effort after results exist.
- Paper text itself.
- Bouc-Wen loader, WienerHammerBenchMark evaluation (unchanged from prior
  spec's scoping).
