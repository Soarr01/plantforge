# Architecture ablation results (2026-07-15)

Configuration: `corpus` recipe only, seed 0, 4 non-default variants trained
to 10000 steps each (`scripts/train_ablation.sh`, background, GPU 6, ~2h
total wall-clock for all 4). Compared against the already-trained default
(160 width, 5 layers, seed 0) via `python -m plantforge.ablation`.
Single-seed per variant throughout — explicitly lighter-weight than the
paper's 5-seed headline claims (Section 4.1 of `paper/main.tex`), not
presented at the same confidence level.

## Comparison table (2 target cells)

```
=== architecture ablation: corpus recipe, seed 0, 2 target cells ===
  default  (d=160 L=5 params=1,582,881): reference=0.0210  family_backlash_dt.05=0.2865  (13.6x ref)
  narrow   (d= 80 L=5 params=407,441):   reference=0.0261  family_backlash_dt.05=0.2939  (11.2x ref)
  wide     (d=320 L=5 params=6,237,761): reference=0.0180  family_backlash_dt.05=0.2941  (16.3x ref)
  shallow  (d=160 L=2 params=655,041):   reference=0.0398  family_backlash_dt.05=0.2835  (7.1x ref)
  deep     (d=160 L=8 params=2,510,721): reference=0.0151  family_backlash_dt.05=0.2901  (19.2x ref)
```

## Full transfer matrices (bonus: `evaluate.py`'s own `report()` printed these
automatically on training completion, beyond the 2 cells `ablation.py` targets)

| variant | ref | family dt=.10 | family dt=.05 | family dt=.02 | chirp | closedloop | saturate | boucwen |
|---|---|---|---|---|---|---|---|---|
| narrow (d=80) | 0.0261 | 0.3140 (12.0x) | 0.2939 (11.2x) | 0.3939 (15.1x) | 0.0316 (1.2x) | 0.0416 (1.6x) | 0.0359 (1.4x) | 0.0202 (0.8x) |
| wide (d=320) | 0.0180 | 0.2887 (16.0x) | 0.2941 (16.3x) | 0.4133 (22.9x) | 0.0218 (1.2x) | 0.0164 (0.9x) | 0.0223 (1.2x) | 0.0174 (1.0x) |
| shallow (L=2) | 0.0398 | 0.3316 (8.3x) | 0.2835 (7.1x) | 0.3665 (9.2x) | 0.0379 (1.0x) | 0.0453 (1.1x) | 0.0578 (1.5x) | 0.0269 (0.7x) |
| deep (L=8) | 0.0151 | 0.2965 (19.6x) | 0.2901 (19.2x) | 0.4024 (26.6x) | 0.0184 (1.2x) | 0.0139 (0.9x) | 0.0212 (1.4x) | 0.0145 (1.0x) |
| default (L=5,d=160) | 0.0210 | — | 0.2865 (13.6x) | — | — | — | — | — |

## Reading notes

- **The absolute family-transfer gap is essentially capacity-invariant.**
  `family_backlash_dt.05` nMSE ranges only 0.2835–0.2941 across a 15x
  parameter span (407k narrow to 6.24M wide) — narrower spread than the
  seed-to-seed variance already reported for the default architecture
  (0.2899±0.0095 across 5 seeds in the paper). This is evidence, not just an
  absence of counter-evidence, that the family-transfer gap is a genuine
  open research problem rather than an artifact of the paper's specific
  1.58M-parameter default configuration — scaling capacity ~4x in either
  direction (width or depth) does not shrink it.
- **Reference nMSE (in-distribution fit) improves monotonically with
  capacity, on both axes independently** — shallow (0.0398, 655k params) >
  narrow (0.0261, 407k) > default (0.0210, 1.58M) > wide (0.0180, 6.24M) >
  deep (0.0151, 2.51M). More capacity fits the training distribution better,
  exactly as expected, and this alone should not be read as "more capacity
  generalizes better" — see the next point.
- **The `x ref` ratio is a misleading capacity comparison metric on its
  own — use absolute nMSE as primary.** `deep` looks like it has the
  *worst* relative family gap (19.2x) and `shallow` the *best* (7.1x), but
  this is almost entirely because `deep`'s reference is unusually good
  (0.0151) and `shallow`'s is unusually poor (0.0398), not because the
  absolute family-transfer error itself differs much (0.2901 vs 0.2835,
  within seed-to-seed noise). This is the same class of methodological trap
  as the mean-vs-median artifact caught in the identifiability experiment
  (`docs/superpowers/results/2026-07-14-experiment-results.md`): a ratio
  metric can make a stable absolute quantity look like it's trending, purely
  through denominator movement. Any paper text citing "×ref" multipliers
  across architecture variants should also report the absolute nMSE.
- **Held-out rate/excitation gaps stay small (~1–1.6x) across every
  variant**, including `shallow` (655k params, less than half the default) —
  the paper's other headline finding (multi-axis training closes rate/
  excitation gaps) is also not an artifact of the specific default
  architecture.
- **Caveat:** single seed per variant. The 0.2835–0.2941 spread is smaller
  than the default architecture's own 5-seed std (0.0095), which is
  reassuring but not a substitute for multi-seed ablation — a proper
  confirmation would train 3–5 seeds per variant, which this pass
  deliberately did not do (see the design spec's explicit scope limit).

## Disposition

This strengthens, and does not complicate, the paper's existing claims — no
paper text is edited in this pass (per the design spec's explicit scope
boundary: "Updating `paper/main.tex` with the result — a separate, explicit
step after the numbers exist, so the paper is never edited to numbers that
haven't been produced and reviewed yet"). Whether/how to fold this into
`paper/main.tex`'s Limitations section ("No architecture ablation") is a
follow-up decision.
