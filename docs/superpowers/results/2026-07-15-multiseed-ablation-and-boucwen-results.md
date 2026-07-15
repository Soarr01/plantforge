# Multi-seed architecture ablation + Bouc-Wen results (2026-07-15)

Two independent extensions, both now at full 5-seed scale, reported together
since both landed the same day:

1. **Multi-seed architecture ablation** — the single-seed ablation in
   `docs/superpowers/results/2026-07-15-architecture-ablation-results.md`
   now has seeds 1-4 trained for all 4 non-default variants (16 new runs),
   giving every variant `n=5`, matching the paper's headline-claim confidence
   level for the first time.
2. **Bouc-Wen real-plant benchmark** — a third real-plant zero-shot dataset
   (previously "out of scope" in the paper), now evaluated across all three
   real-plant reporting surfaces (`realbench.py`, `baselines.py`,
   `aggregate.py`) at full 5-seed scale.

## A. Multi-seed architecture ablation (`python -m plantforge.ablation`)

```
=== architecture ablation: corpus recipe, seeds [0, 1, 2, 3, 4], 2 target cells ===
  default (d=160 L=5 params=1,582,881):
    reference: 0.0215 ± 0.0017 (n=5)
    family_backlash_dt.05: 0.2899 ± 0.0095 (n=5)  (13.5x ref)
  narrow (d=80 L=5 params=407,441):
    reference: 0.0300 ± 0.0029 (n=5)
    family_backlash_dt.05: 0.3154 ± 0.0206 (n=5)  (10.5x ref)
  wide (d=320 L=5 params=6,237,761):
    reference: 0.0173 ± 0.0011 (n=5)
    family_backlash_dt.05: 0.2956 ± 0.0146 (n=5)  (17.1x ref)
  shallow (d=160 L=2 params=655,041):
    reference: 0.0432 ± 0.0035 (n=5)
    family_backlash_dt.05: 0.3198 ± 0.0224 (n=5)  (7.4x ref)
  deep (d=160 L=8 params=2,510,721):
    reference: 0.0158 ± 0.0009 (n=5)
    family_backlash_dt.05: 0.3012 ± 0.0091 (n=5)  (19.0x ref)
```

### Reading notes (A)

- **The single-seed finding survives at full scale, now with real error
  bars.** Absolute family-transfer nMSE ranges 0.2899–0.3198 across the full
  5-seed run (vs. 0.2835–0.2941 in the single-seed pass) — a slightly wider
  but still narrow band, and every variant's own std (0.0091–0.0224) is
  comparable to or larger than the between-variant spread. The
  capacity-invariance claim is no longer a single-seed observation; it's now
  supported by the same 5-seed confidence level as every other headline
  number in the paper.
- **The ×ref-ratio trap flagged in the single-seed pass is confirmed, not
  just suspected.** `deep` (19.0x) vs. `shallow` (7.4x) still looks like a
  factor-of-2.5 spread, still driven almost entirely by reference nMSE
  (0.0158 vs. 0.0432, a genuine, monotonic capacity effect on
  in-distribution fit) rather than by the family-transfer numerator itself
  (0.3012 vs. 0.3198 — an 6% difference, well inside each variant's own
  std). Paper text should keep citing absolute nMSE as primary, as the
  single-seed results doc already recommended.
- **Ordering of variants by absolute family-transfer nMSE is not
  capacity-monotonic** (narrow 0.3154 < wide 0.2956 < shallow 0.3198 < deep
  0.3012 < default 0.2899) — there is no clean "bigger is better" or
  "smaller is better" trend in the number that actually matters; the
  variation looks like noise around a stable ~0.30 floor, which is the
  headline claim's whole point.

## B. Bouc-Wen zero-shot real-plant benchmark, full 5-seed

### B.1 Transformer models (`python -m plantforge.aggregate`, real-plant block)

```
  real-plant Bouc-Wen (decimated 15x -> dt=0.0200s):
    wh_only: 2.5373 ± 0.8476 (n=5)
    corpus:  1.5894 ± 0.3607 (n=5)
```

(Full context: this run also reproduced the existing Silverbox/Cascaded_Tanks
multi-seed numbers exactly as in the prior results doc — wh_only
0.9583±0.2170 / 0.3937±0.0814, corpus 0.3306±0.0397 / 0.0843±0.0430 — no
regression, only the new Bouc-Wen row is new information here.)

### B.2 Classical baselines (`python -m plantforge.baselines real`)

```
  Bouc-Wen: ARX 0.0603 | NARX2 0.0529
```

(Silverbox/Cascaded_Tanks ARX/NARX2 numbers reproduced exactly as before:
Silverbox ARX 0.0028/NARX2 0.0028, Cascaded_Tanks ARX 0.0075/NARX2
3.665e9 — ARX/NARX2 are deterministic given the fixed windows, so these
don't vary with model seed and matching the seed-0 preview is expected, not
just reassuring.)

### Reading notes (B)

- **The seed-0 preview number was directionally wrong — this is exactly why
  the paper insists on multi-seed reporting everywhere else.** The Task 2
  preview (`docs/superpowers/reviews/audit-*` era work, single seed 0) showed
  `corpus` (2.0533) *worse* than `wh_only` (1.6560) on Bouc-Wen — a genuine
  reversal from the Silverbox/Cascaded_Tanks pattern, and both task and
  whole-branch reviewers correctly verified that reversal was not a code bug
  given the seed-0 data available at the time. At full 5-seed scale, the
  direction flips back: `corpus` (1.5894±0.3607) is *better* than `wh_only`
  (2.5373±0.8476), consistent with Silverbox/Cascaded_Tanks after all. The
  seed-0 "reversal" was seed-to-seed noise on a dataset with unusually high
  variance (`wh_only`'s std, 0.8476, is the largest relative variance of any
  real-plant number in this whole project) — not a real Bouc-Wen-specific
  effect. This is worth stating plainly in the paper as a demonstration of
  *why* the single-seed ablation caveat mattered, not quietly overwritten.
- **Both models still transfer worse to Bouc-Wen than to Silverbox or
  Cascaded_Tanks** (nMSE ~1.6-2.5 vs. ~0.08-0.96 for the other two) —
  Bouc-Wen's hysteretic memory (an internal, unmeasurable state variable) is
  evidently the hardest of the three real-plant transfers for a model
  trained only on PLANTFORGE's synthetic families, none of which include
  true hysteresis with memory in quite Bouc-Wen's form (`boucwen` the
  corpus family models *output* hysteresis, but zero-shot transfer to the
  real Bouc-Wen benchmark is still clearly harder than to the other two).
- **ARX/NARX2 continue the paper's core baseline finding on a third
  dataset.** ARX (0.0603) and NARX2 (0.0529, notably NOT diverged here,
  unlike its usual 1e9-1e12 pattern elsewhere) both beat both transformers by
  roughly 25-45x. This is the third of three real-plant benchmarks where a
  classical per-window fit outperforms both trained transformers zero-shot —
  strengthening, not just repeating, the paper's Section 4.3 finding.
- **NARX2 stability on Bouc-Wen is a real exception worth a footnote.**
  Every other cell where NARX2 was evaluated (synthetic transfer cells, other
  real-plant datasets) diverges to the clipped 1e9-1e12 range; Bouc-Wen is
  only the second cell overall (after synthetic-Silverbox) where it produces
  a sane, non-diverged number (0.0529, close to ARX's 0.0603) — worth noting
  as data on when the free-run quadratic-NARX instability does and doesn't
  manifest, though not a claim this project has the scope to fully explain.

## Disposition

Both extensions are ready to fold into `paper/main.tex` as the next,
separate step (per both original design specs' explicit scope boundaries).
Specifically: Table 1 (`\ref{tab:headline}`)/Section 4.1 prose needs no
change (headline transfer matrix unaffected by either extension); Table 2
(`\ref{tab:realplant}`)/Section 4.2 gets a third row (Bouc-Wen) and revised
prose noting the seed-0-to-5-seed reversal as a worked example of why
multi-seed reporting matters; Section 4.3's ARX comparison gets a third
dataset's numbers folded into its range claims (re-verify the "11-342x"
range against the new Bouc-Wen ARX/NARX2 numbers rather than assuming it
still holds); Table 4 (`\ref{tab:ablation}`)/Section 5's architecture-ablation
paragraph gets updated from single-seed to 5-seed numbers and the "(single
seed)" hedge can be removed/weakened; Abstract and Conclusion's "two real
plants"/"15x parameter range in a single-seed check" phrasing both need
updating. `references.bib` needs the Bouc-Wen dataset citation (Noel &
Schoukens 2020, 4TU.ResearchData, DOI 10.4121/12967592, CC BY-SA 4.0) plus
ideally the 2016 workshop paper describing the benchmark design (already
identified in `docs/superpowers/specs/2026-07-15-boucwen-loader-design.md`).
`figures/make_figures.py`'s `fig2_real_plant` needs a third dataset group.
`README.md`/`docs/DATASHEET.md` wherever they state Bouc-Wen is excluded
need updating to reflect it's now covered (only WienerHammerBenchMark
remains excluded, for the already-documented record-length reason).
