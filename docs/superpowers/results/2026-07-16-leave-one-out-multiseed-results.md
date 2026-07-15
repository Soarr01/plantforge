# Leave-one-family-out sweep: multi-seed confirmation (2026-07-16)

Follow-up to `docs/superpowers/results/2026-07-15-leave-one-family-out-results.md`
(the single-seed pass). Trained seeds 1-4 for the 4 non-default held-out
families (`stribeck`, `saturate`, `boucwen`, `drivetrain`; seed 0 already
existed) via `scripts/train_leave_one_out_seeds.sh` — 16 new runs, `corpus`
recipe, 10k steps each — and re-ran `python -m plantforge.leave_one_out`,
which now aggregates mean±std over however many finished, non-diverged
seeds exist per family (see "Divergence finding" below for why "non-diverged"
matters).

## A real bug found via this run: one checkpoint diverged to all-NaN

`eval_corpus_s2_hoBOUCWEN.pt` reached step 10000 (so the existing
"finished" check passed) but **every learned parameter in the checkpoint is
NaN** — training diverged during this run and never recovered. `leave_one_out.py`
had no guard against this (only a step-count check), so the first run of
the aggregation script propagated NaN into the entire `boucwen` family's
reported mean/std (`reference: nan ± nan (n=5)`, `held_out: nan ± nan (n=5)`).

Fixed in commit `14f9a3f`: added `_has_finite_weights(model)` (checks
`model.parameters()`, deliberately NOT buffers — the causal attention mask
buffer is `-inf` above the diagonal by design and must not trip this check)
and used it in `_finished_models_for` to skip and report diverged
checkpoints the same way missing/unfinished ones are already skipped, with
a new offline test (`test_finished_models_for_skips_diverged_checkpoint`)
using a synthetic all-NaN checkpoint. `boucwen` now correctly aggregates
over its 4 healthy seeds (n=4) instead of 5 (one of which was garbage).

This divergence appears isolated: it's the only one of the 21 total
leave-one-out checkpoints (5 backlash + 4×4 new) with non-finite weights,
and it's specific to `(seed=2, held_family=boucwen)` — no other
`boucwen`-held-out seed, nor any other family at seed 2, diverged. Not
investigated further (would require re-running seed 2 with e.g. loss
logging to find where in training it diverged); flagged here for honesty,
not chased down, since the other 4 boucwen seeds give a clean n=4 result
and the paper doesn't currently make any claim resting on this specific
checkpoint.

## Comparison table (final, post-fix)

```
=== leave-one-family-out sweep: corpus recipe, reference = mean nMSE over 4 trained families ===
  backlash (existing baseline):
    reference: 0.0600 ± 0.0037 (n=5)
    held_out:  0.2899 ± 0.0095 (n=5)  (4.8x ref)
  stribeck:
    reference: 0.0833 ± 0.0042 (n=5)
    held_out:  0.0388 ± 0.0036 (n=5)  (0.5x ref)
  saturate:
    reference: 0.0823 ± 0.0029 (n=5)
    held_out:  0.0474 ± 0.0028 (n=5)  (0.6x ref)
  boucwen:
    (seed 2: eval_corpus_s2_hoBOUCWEN.pt diverged to non-finite weights -- skipped)
    reference: 0.0776 ± 0.0041 (n=4)
    held_out:  0.0351 ± 0.0016 (n=4)  (0.5x ref)
  drivetrain:
    reference: 0.0345 ± 0.0006 (n=5)
    held_out:  0.2976 ± 0.0110 (n=5)  (8.6x ref)
```

Ranked by absolute held-out nMSE (primary metric — see the single-seed
doc's Reading notes for why ratio is misleading here):

| family | held_out nMSE | reference | ratio | n |
|---|---|---|---|---|
| drivetrain | 0.2976 ± 0.0110 | 0.0345 ± 0.0006 | 8.6x | 5 |
| backlash | 0.2899 ± 0.0095 | 0.0600 ± 0.0037 | 4.8x | 5 |
| saturate | 0.0474 ± 0.0028 | 0.0823 ± 0.0029 | 0.6x | 5 |
| stribeck | 0.0388 ± 0.0036 | 0.0833 ± 0.0042 | 0.5x | 5 |
| boucwen | 0.0351 ± 0.0016 | 0.0776 ± 0.0041 | 0.5x | 4 |

## Reading notes

- **The single-seed finding is confirmed at 5-seed scale: the corpus splits
  into two difficulty tiers, not one hard outlier among four easy
  families.** `{backlash, drivetrain}` sit at held_out nMSE ≈ 0.29-0.30,
  roughly 6-8x higher than `{saturate, stribeck, boucwen}` at ≈ 0.035-0.047.
  The tiers do not overlap at all — even the easiest of the "hard" tier
  (backlash, 0.2899-0.0095=0.2804 at −1σ) is far above the hardest of the
  "easy" tier (saturate, 0.0474+0.0028=0.0502 at +1σ). This is now a
  5-seed-confirmed structural property of the corpus, not a single-seed
  preview.
- **`backlash` and `drivetrain` are statistically indistinguishable from
  each other — a genuine tie, not a close ranking.** At multi-seed scale
  their point estimates swapped order from the single-seed pass
  (drivetrain 0.2976 vs backlash 0.2899 — drivetrain is now nominally
  higher, reversed from single-seed's backlash 0.2899 vs drivetrain 0.2844),
  but their ±1σ bands overlap substantially (backlash
  [0.2804, 0.2994], drivetrain [0.2866, 0.3086]). The correct claim is "tied
  for hardest," not "X is harder than Y" — the single-seed doc's framing of
  this as a tie (not a ranking) holds up and is now the safer, confirmed
  claim rather than a hedge.
- **The single-seed caveat from the prior doc is resolved**: the prior doc
  flagged the 4-non-backlash-family ranking as single-seed and noted the
  risk given the Bouc-Wen real-plant reversal precedent. At 5-seed (4-seed
  for boucwen) scale, every family's point estimate moved only slightly
  from its single-seed value and no family crossed tiers — e.g. `boucwen`
  went from 0.0360 (1 seed) to 0.0351±0.0016 (4 seeds), `drivetrain` from
  0.2844 (1 seed) to 0.2976±0.0110 (5 seeds). No Bouc-Wen-style reversal
  occurred here.

## Disposition

This is now a 5-seed-confirmed (4-seed for boucwen, with a documented and
fixed reason) result, ready to fold into `paper/main.tex`'s "Family
coverage and held-out-family choice" Limitations paragraph, replacing the
current text ("we have not verified whether it is representative of the
corpus's hardest held-out-family case or an outlier, and leave a full
leave-one-family-out sweep to future work") with the actual finding: two
families (backlash, drivetrain) are tied for hardest, the other three
transfer better than in-distribution, confirmed at multi-seed scale. Not
yet applied to the paper — pending discussion with the user on exact
wording and whether/where else in the paper (e.g. Discussion) this should
be mentioned.
