# Architecture ablation for the paper's headline claim

Status: approved design, not yet implemented.

## Purpose

The paper (`paper/main.tex`) reports headline transfer-gap numbers for one
fixed `InContextSysID` configuration (d=160, 5 layers, 8 heads, 1.58M
params). Section 5's Limitations names "No architecture ablation" as an open
gap. This experiment answers: are the reported gaps (closes rate/excitation,
halves-not-closes family) an artifact of this specific model size, or do they
hold across a reasonable range of capacity?

## Scope

Four variants relative to default, holding everything else (data, optimizer,
schedule, `T_CTX`/`T_QRY`, `heads=8`) fixed:

| name | d | layers | params |
|---|---|---|---|
| narrow | 80 | 5 | 407k |
| wide | 320 | 5 | 6.24M |
| shallow | 160 | 2 | 655k |
| deep | 160 | 8 | 2.51M |
| (default, already trained, seed 0) | 160 | 5 | 1.58M |

Only the `corpus` training recipe (the paper's "fix") is ablated —
`wh_only` is the failure case, not the claim under test. **Single seed per
variant** (seed 0's data-pool salt/eval seeds, for direct comparability with
the already-trained default-architecture seed-0 checkpoint) — explicitly
lighter-weight than the headline's 5-seed claims, reported as such, not
silently presented at the same confidence level.

Evaluation restricted to the two cells that carry the paper's claim: (a)
`reference` (train-like, sanity floor) and (b) `held-out family backlash,
dt=0.05` (the open-challenge cell, and the single most load-bearing number
in the paper). This is deliberately narrower than the full 10-cell transfer
matrix in Table 1 — an ablation's job is "does capacity change the
conclusion," not "reproduce the full result at every capacity."

## Design

**Code change:** `evaluate.py`'s `InContextSysID` and `run()`/`report()`
already parameterize `d`/`layers`/`heads` as constructor args, but `run()`
hardcodes `InContextSysID()` with no size arguments and the checkpoint name
has no size tag. Add environment-variable overrides analogous to `PF_SEED`:
`PF_WIDTH` (default 160), `PF_LAYERS` (default 5), `PF_HEADS` (default 8),
folded into the checkpoint filename only when non-default (so the existing
`eval_corpus_s0.pt` remains the default-architecture seed-0 checkpoint,
untouched, no re-training needed for the baseline row).

**New module** `plantforge/ablation.py`: trains (or resumes) the 4 variants
via the existing `evaluate.run()` with size overrides, then evaluates each
on the two target cells via `evaluate.nmse`, and prints a comparison table
including the already-trained default as the reference row (loaded, not
retrained).

**New script** `scripts/train_ablation.sh`: analogous to
`scripts/train_seeds.sh` — background-friendly, resumable, skips finished
variants, same stall-guard.

## Global constraints

- Reuse `evaluate.InContextSysID`, `evaluate.run`, `evaluate.nmse`,
  `evaluate.HOLD_FAMILY` — no reimplementing training or eval logic.
- `evaluate.py` changes limited to the size-parameterization described above
  (analogous in shape to the existing `PF_SEED` change from the prior plan).
- Do not modify `families.py`, `excitation.py`, `identifiability.py`,
  `corpus.py`, `realbench.py`, `aggregate.py`, `baselines.py`, `ident_exp.py`.
- Checkpoint naming: default architecture keeps today's `eval_{mode}_s{seed}.pt`
  (no size suffix) for backward compatibility with `aggregate.py`/`realbench.py`/
  `ident_exp.py`, which all assume that name; non-default sizes get an
  additional suffix, e.g. `eval_corpus_s0_d80L5.pt`.
- Branch off `main`, same subagent-driven-development process as the prior
  plan (task briefs, task-scoped review, final whole-branch review).

## Out of scope

- Multi-seed ablation (would be the natural follow-up if a reviewer pushes
  back, not needed to answer "is the conclusion capacity-sensitive at all").
- `wh_only` ablation, full transfer matrix per variant, context/query-length
  ablation, head-count ablation.
- Updating `paper/main.tex` with the result — a separate, explicit step
  after the numbers exist, so the paper is never edited to numbers that
  haven't been produced and reviewed yet.
