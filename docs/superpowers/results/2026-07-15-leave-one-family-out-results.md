# Leave-one-family-out sweep results (2026-07-15)

Answers the paper's own flagged Limitation: "we have not verified whether
[backlash] is representative of the corpus's hardest held-out-family case
or an outlier." Trained one model per alternative held-out family
(`stribeck`, `saturate`, `boucwen`, `drivetrain`; single seed 0, `corpus`
recipe) via `scripts/train_leave_one_out.sh`, compared via
`python -m plantforge.leave_one_out` against the paper's existing 5-seed
`backlash` baseline.

**Reference metric note:** `leave_one_out.py` deliberately does NOT reuse
`evaluate.report()`'s `stribeck`-only reference (invalid once `stribeck`
itself is held out). Its `reference` is the mean nMSE over whichever 4
families were actually trained for that run — well-defined for any
held-out choice, but numerically different from the paper's Table 1
reference number for `backlash` specifically (0.0600 here vs. 0.0215 in the
paper, because Table 1 uses `stribeck` alone while this uses the mean of
all 4 non-held families). This was independently verified in task review:
0.0600 is exactly the mean of Table 1's four held-out-rate dt=0.05 values
(stribeck 0.0215, saturate 0.0285, boucwen 0.0178, drivetrain 0.1722).

## Comparison table

```
=== leave-one-family-out sweep: corpus recipe, reference = mean nMSE over 4 trained families ===
  backlash (existing 5-seed baseline):
    reference=0.0600  held_out=0.2899  (4.8x ref)  n=5
  stribeck (seed 0):
    reference=0.0797  held_out=0.0342  (0.4x ref)  n=1
  saturate (seed 0):
    reference=0.0842  held_out=0.0500  (0.6x ref)  n=1
  boucwen (seed 0):
    reference=0.0737  held_out=0.0360  (0.5x ref)  n=1
  drivetrain (seed 0):
    reference=0.0337  held_out=0.2844  (8.4x ref)  n=1
```

Ranked by absolute held-out nMSE (the primary metric, per this project's own
established ×ref-ratio-is-misleading caveat — see Reading notes):

| family | held_out nMSE | reference | ratio | seeds |
|---|---|---|---|---|
| backlash | 0.2899 | 0.0600 | 4.8x | 5 |
| drivetrain | 0.2844 | 0.0337 | 8.4x | 1 |
| saturate | 0.0500 | 0.0842 | 0.6x | 1 |
| boucwen | 0.0360 | 0.0737 | 0.5x | 1 |
| stribeck | 0.0342 | 0.0797 | 0.4x | 1 |

## Reading notes

- **`backlash` is not the outlier hardest case the paper's phrasing might
  suggest — it's tied with `drivetrain` for hardest, and the two are the
  only genuinely hard cases.** By absolute held-out nMSE, `backlash`
  (0.2899) and `drivetrain` (0.2844) are within 2% of each other — both
  roughly an order of magnitude worse than the other three families
  (0.0342–0.0500). This corpus's five families cluster into two clear
  difficulty tiers for zero-shot family transfer: {backlash, drivetrain}
  hard, {stribeck, saturate, boucwen} easy — not a single hard outlier
  among four easy ones.
- **The ×ref ratio is misleading here too, for exactly the reason already
  flagged in the architecture-ablation section of the paper.** `drivetrain`
  looks 1.75x "harder" than `backlash` by ratio (8.4x vs 4.8x), but this is
  almost entirely because `drivetrain`'s in-distribution reference is
  unusually low (0.0337, the lowest of all five) rather than because its
  absolute held-out error is higher (it's actually very slightly lower than
  backlash's, 0.2844 vs 0.2899). Absolute nMSE, not the ratio, is the
  number that should drive the "which family is hardest" claim — same
  methodological lesson as the earlier single-vs-multi-seed architecture
  ablation, now recurring in a third experiment.
- **`drivetrain` being the hardest (or tied-hardest) held-out family is
  independently corroborated by an existing paper finding, not a surprise
  from nowhere.** Table 1 (the paper's own headline transfer matrix)
  already reports `drivetrain` as the hardest held-out-RATE family among
  the four trained families (8.0x at dt=0.05, vs. 0.8-1.3x for the other
  three) — the resonant two-inertia dynamics were already flagged as
  resisting rate interpolation. This sweep shows the same family also
  resists family-level holdout, which is a coherent story: `drivetrain`'s
  resonance appears to be qualitatively different from what the other
  families teach the model, in a way robust across two different transfer
  axes (rate and family).
- **The three "easy" families (stribeck, saturate, boucwen) transfer
  BETTER than in-distribution when held out (ratio < 1x).** This is not
  intuitively obvious and is worth a sentence in the paper: it suggests
  these three nonlinearities are close enough to some combination of the
  other four (or simple enough in general) that the model's in-context
  algorithm generalizes to them for free, whereas backlash's discontinuous
  deadzone and drivetrain's resonant two-mass dynamics are structurally
  distinct enough from the rest of the corpus that neither generalizes
  this way.
- **Caveat, stated plainly: this is single-seed for 4 of the 5 rows.**
  Unlike `backlash`'s 5-seed baseline, `stribeck`/`saturate`/`boucwen`/
  `drivetrain` are each one training run. Given the lesson from the
  Bouc-Wen real-plant single-seed preview (which briefly showed the wrong
  direction before a 5-seed rerun corrected it), the *ranking* claimed here
  (two hard, three easy) should be treated as a strong first signal, not a
  final claim, until confirmed at multi-seed scale. That said, the gap
  between the two tiers is large (roughly 6-8x in absolute nMSE, versus
  Bouc-Wen's reversal being within a factor of 1.2x) — a much bigger margin
  than plausible single-seed noise, so this is a lower-risk single-seed
  result than the Bouc-Wen case was.

## Disposition

This directly answers, and complicates in an interesting way, the paper's
own flagged Limitation ("we have not verified whether backlash is
representative... or an outlier"): the honest answer is "neither — it's
tied for hardest, not uniquely so, and the corpus's families split into two
difficulty tiers rather than one outlier among equals." No paper text is
edited in this pass (per the design spec's explicit scope boundary).
Folding this into `paper/main.tex`'s "Family coverage and held-out-family
choice" limitation paragraph is a natural follow-up, ideally after a
multi-seed confirmation pass on the 4 new families given the single-seed
caveat above.
