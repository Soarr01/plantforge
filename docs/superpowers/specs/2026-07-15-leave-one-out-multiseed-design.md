# Leave-one-family-out sweep: multi-seed follow-up

Status: approved design, not yet implemented.

## Purpose

The single-seed leave-one-family-out sweep found `backlash` and `drivetrain`
tied for hardest held-out family (absolute nMSE within 2%), with the other
three families transferring better than in-distribution. Per this project's
established precedent (architecture ablation's single-seed-first ->
multi-seed-confirm pattern), confirm this ranking at 5-seed scale before
folding it into the paper.

## Scope

Train seeds 1-4 (seed 0 already exists) for the 4 non-default held-out
families (`stribeck`, `saturate`, `boucwen`, `drivetrain`), corpus recipe,
10k steps each -- 16 new training runs total. `backlash` already has 5
seeds and is not retrained. Update `leave_one_out.py` to aggregate
mean+-std over however many finished seeds exist per family (reusing
`aggregate.mean_std_str`, same as `ablation.py` already does), rather than
hardcoding "seed 0 only" for the 4 non-default families.

## Design

**New script `scripts/train_leave_one_out_seeds.sh`:** resumable,
stall-guarded, same pattern as `scripts/train_ablation_seeds.sh` crossed
with the family loop already in `scripts/train_leave_one_out.sh`. Loops
`seed in {1,2,3,4} x family in {stribeck,saturate,boucwen,drivetrain}`,
setting `PF_SEED` and `PF_HOLD_FAMILY`, checkpoint path
`eval_corpus_s{seed}_ho{FAMILY_UPPER}.pt` (must match
`leave_one_out._ckpt_name_for` and `evaluate._ckpt_name` exactly -- both
already implement this suffix rule, verified in the original
leave-one-family-out plan; this script only needs to construct the same
path string, not add a new naming rule).

**`leave_one_out.py` changes:**
- Add `_finished_models_for(held_family, seeds)`: mirrors
  `ablation._finished_variant_models` -- for each seed, load
  `_ckpt_name_for(held_family, seed)` if it exists and
  `step >= TOTAL_STEPS`, else print a skip note; return the list of loaded
  models.
- Replace the `HOLD_CHOICES` loop body: instead of loading exactly seed 0,
  call `_finished_models_for(held_family, seeds)` for
  `seeds = range(5)` (same range as `BACKLASH_SEEDS`, so both loops read
  symmetrically), compute `reference_and_heldout` per model, then print
  `mean_std_str(refs)` / `mean_std_str(outs)` / ratio-of-means / `n=len(models)`
  -- same shape as the existing `backlash` block, so unify: both blocks can
  call one shared helper `_report_family(held_family, ckpt_fn, seeds)` that
  does the load-aggregate-print work, eliminating the current
  code duplication between the `backlash` special case and the
  `HOLD_CHOICES` loop.
- `main()` becomes: `_report_family("backlash", seeds=BACKLASH_SEEDS)` then
  `_report_family(f, seeds=range(5))` for `f in HOLD_CHOICES`.
- If a family has zero finished seeds (nothing changes here from today),
  keep printing `MISSING -- run scripts/train_leave_one_out.sh` (extend the
  message to mention the new seeds script too).

## Global constraints

- Reuse `evaluate.nmse`, `evaluate.FAMILIES`, `evaluate.CKPT_DIR`,
  `evaluate.DEV`, `evaluate.InContextSysID`, `aggregate.mean_std_str` -- no
  reimplementing loading, nMSE, or mean/std logic.
- Checkpoint naming must stay byte-identical to the existing
  `_ckpt_name_for` / `evaluate._ckpt_name` suffix rule
  (`eval_corpus_s{seed}_ho{FAMILY_UPPER}.pt`) -- do not introduce a second
  naming convention. Verify the new shell script's constructed path string
  matches `_ckpt_name_for`'s output exactly for at least one (seed, family)
  pair before launching any training.
- Do not modify `evaluate.py`, `families.py`, `ablation.py`, `aggregate.py`,
  `realbench.py`, `baselines.py`, `ident_exp.py`, `paper/main.tex`.
- `backlash`'s existing 5-seed checkpoints and reported numbers must not
  change (same seeds, same metric, same code path -- this is a refactor of
  how the 4 other families are read, not a behavior change for `backlash`).
- Branch off `main`, same subagent-driven-development process (task briefs,
  task-scoped review, final whole-branch review) as prior plans in this repo.

## Out of scope

- Updating `paper/main.tex` -- separate, explicit, later step once these
  numbers exist and are reviewed (same posture as every other results-first
  plan this session).
- Retraining `backlash` or `wh_only`.
