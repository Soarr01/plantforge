# Leave-one-family-out sweep

Status: approved design, not yet implemented.

## Purpose

The paper holds out a single family (`backlash`) throughout, and Limitations
names this explicitly: "we have not verified whether it is representative
of the corpus's hardest held-out-family case or an outlier." This sweeps
the other 4 families (`stribeck`, `saturate`, `boucwen`, `drivetrain`) as
the held-out choice, single-seed each (seed 0), to give a first empirical
answer.

## Scope

4 new training runs (one per non-default held-out family, `corpus` recipe
only, seed 0, 10k steps). `wh_only` training does not depend on
`HOLD_FAMILY` at all (it trains exclusively on the `wh` family via
`build_pool`'s `mode == "wh_only"` branch) so is unaffected and not
retrained. Single-seed, explicitly lighter-weight than the paper's 5-seed
standard — a first empirical answer, not a claim to fold into the paper at
the same confidence level without a follow-up multi-seed pass (same
posture as the architecture ablation's first pass).

## Design

**`evaluate.py` change (minimal, analogous to the existing `PF_SEED`/
`PF_WIDTH`/`PF_LAYERS` pattern):** `HOLD_FAMILY` becomes
`os.environ.get("PF_HOLD_FAMILY", "backlash")`. `build_pool()` already
references `HOLD_FAMILY` generically (`fams = [f for f in FAMILIES if f !=
HOLD_FAMILY]`) — no other change needed there. Checkpoint naming: a new
`_ckpt_name` suffix component, analogous to the width/layers suffix ---
default (`backlash`) keeps the existing unsuffixed-by-family name
(`eval_corpus_s{seed}.pt`, composing correctly with the existing width/layers
suffix when those are also non-default); any other held family adds
`_ho{FAMILY}` (e.g. `eval_corpus_s0_hoSTRIBECK.pt`).

**`evaluate.py`'s `report()` is NOT modified.** It hardcodes `"stribeck"` as
the reference family, which becomes invalid when `stribeck` itself is the
held-out family. Rather than touch `report()` (a function multiple prior
plans have deliberately left untouched), the new module defines its own
metric that doesn't depend on a fixed reference family.

**New module `leave_one_out.py`:**
- `HOLD_CHOICES = ["stribeck", "saturate", "boucwen", "drivetrain"]` (the
  4 non-default choices; `backlash`, the paper's existing default, is
  loaded from its already-trained 5-seed checkpoints for comparison, not
  retrained).
- `_ckpt_name_for(held_family, seed=0)` mirrors `evaluate.py`'s new suffix
  rule.
- For each held-out choice: `reference = mean(nmse(model, fam, "multisine",
  0.05) for fam in FAMILIES if fam != held_family)` (average in-distribution
  performance over whichever 4 families were actually trained — this is the
  metric that generalizes across all 5 possible held-out choices, unlike a
  fixed single reference family) and `held_out = nmse(model, held_family,
  "multisine", 0.05)`.
- For `backlash` (the existing default), compute the SAME two metrics from
  each of its 5 existing seed checkpoints, so the comparison table is
  apples-to-apples (mean-over-4-other-families reference, not the paper's
  `stribeck`-only reference number) — this will differ slightly from the
  paper's reported reference number for `backlash`, which is fine and
  expected, and should be noted in the report rather than confused with it.
- CLI prints a comparison table: held-out family, reference, held-out nMSE,
  ratio, seed count.

**New script `scripts/train_leave_one_out.sh`:** resumable, stall-guarded,
same pattern as `scripts/train_ablation.sh`, training the 4 new
(held_family, seed=0) combinations.

## Global constraints

- Reuse `evaluate.nmse`, `evaluate.FAMILIES`, `evaluate.CKPT_DIR`,
  `evaluate.DEV`, `realbench.load_model` (or equivalent loading via
  `evaluate.InContextSysID` directly, matching `ablation.py`'s pattern) --
  no reimplementing training or eval logic.
- Do not modify `evaluate.report()`, `families.py`, `excitation.py`,
  `identifiability.py`, `corpus.py`, `realbench.py`, `aggregate.py`,
  `baselines.py`, `ablation.py`, `ident_exp.py`, `paper/main.tex`.
- Default-held-family (`backlash`) checkpoint names MUST stay exactly
  unchanged — the already-trained 5-seed `backlash` checkpoints
  (`eval_corpus_s{0..4}.pt`) must remain loadable with zero retraining.
- Branch off `main`, same subagent-driven-development process (task briefs,
  task-scoped review, final whole-branch review) as prior plans in this repo.

## Out of scope

- Multi-seed leave-one-out (natural follow-up if the single-seed pass
  reveals something worth confirming more rigorously — this repo's
  established pattern, per the architecture ablation precedent).
- Updating `paper/main.tex`'s Limitations section — a separate, explicit,
  later step after the numbers exist and are reviewed, so the paper is
  never edited to numbers that haven't been produced and reviewed yet.
- Retraining `wh_only` (unaffected by `HOLD_FAMILY` by construction).
