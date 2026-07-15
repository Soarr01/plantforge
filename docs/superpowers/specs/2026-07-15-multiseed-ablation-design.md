# Multi-seed architecture ablation

Status: approved design, not yet implemented.

## Purpose

The single-seed architecture ablation (`docs/superpowers/results/2026-07-15-architecture-ablation-results.md`,
folded into `paper/main.tex` Section 5) showed the family-transfer gap is
capacity-invariant, but is explicitly caveated as single-seed. This closes
that gap: train seeds 1-4 for the four non-default variants (narrow/wide/
shallow/deep) — the default architecture already has 5 seeds (0-4) trained
from the earlier paper-experiments plan — and report mean±std per variant.

## Scope

16 new training runs (4 variants x seeds 1-4, corpus recipe only, 10k steps
each). No code changes to `evaluate.py` — `PF_SEED`+`PF_WIDTH`+`PF_LAYERS`
already produce the correct checkpoint name for any (seed, width, layers)
combination.

## Design

**New script** `scripts/train_ablation_seeds.sh`: analogous to
`scripts/train_ablation.sh`, looping SEEDS (default "1 2 3 4") x the 4
non-default variants, same stall-guard/skip-check pattern.

**Extend `ablation.py`**: `_ckpt_name_for` gains a `seed` parameter (default
0, so existing call sites/tests keep working unless updated); a new
`_finished_variant_models(name, width, layers, seeds)` loads whichever seeds
exist and are finished (mirrors `aggregate._finished_models`'s skip-with-note
behavior); `main()` aggregates mean±std per variant over available seeds for
both target cells (reference, family), reusing `aggregate.mean_std_str`
rather than reimplementing the mean/std formula.

## Global constraints

- Reuse `evaluate.InContextSysID/nmse/HOLD_FAMILY/CKPT_DIR/DEV`,
  `aggregate.mean_std_str` — no reimplementing training or the mean/std
  formula.
- Do not modify `evaluate.py`, `families.py`, `excitation.py`,
  `identifiability.py`, `corpus.py`, `realbench.py`, `aggregate.py`,
  `baselines.py`, `ident_exp.py`, `paper/main.tex` (paper update is a
  separate, later step after numbers exist and are reviewed).
- `ablation.py`'s existing single-arg `load_variant(ckpt_name)` and
  `param_count(width, layers)` interfaces are unchanged; only
  `_ckpt_name_for` and `main()` change.
- Branch off `main`, same subagent-driven-development process (task briefs,
  task-scoped review, final whole-branch review) as prior plans in this repo.

## Out of scope

- Additional seeds for the default architecture (already has 5).
- Updating `paper/main.tex`'s Table 4 / Limitations text with the new
  multi-seed numbers — explicit follow-up after results exist.
- Leave-one-family-out sweep (separate, later plan).
