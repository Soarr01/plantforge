# Audit A — Numeric-accuracy and anonymity audit of paper/main.tex

Sources of truth:
- `docs/superpowers/results/2026-07-14-experiment-results.md` (main experiments)
- `docs/superpowers/results/2026-07-15-architecture-ablation-results.md` (architecture ablation)

Paper checked: `paper/main.tex` (+ `paper/references.bib`, `paper/main.pdf`, `figures/make_figures.py`)

---

## 1. CONFIRMED-ACCURATE (grouped)

**Table 1 (headline synthetic transfer matrix, `tab:headline`).** All 20
mean±std pairs (wh_only and corpus, 10 rows each) and all 20 ×ref ratios are
exact transcriptions of the "Multi-seed transfer matrices" block in the
2026-07-14 doc (reference, backlash dt=0.10/0.05/0.02, chirp, closedloop,
rate-dt=0.05 for stribeck/saturate/boucwen/drivetrain, both wh_only and
corpus). Bold-face on the "winning" model per cell is directionally correct
everywhere it's applied (lower nMSE bolded, including the wh_only-wins
`saturate` exception).

**Table 2 (real-plant, `tab:realplant`).** Silverbox and Cascaded Tanks
wh_only/corpus mean±std round correctly from the source's 4-decimal values
(0.9583±0.2170→0.958±0.217; 0.3306±0.0397→0.331±0.040; 0.3937±0.0814→
0.394±0.081; 0.0843±0.0430→0.084±0.043). ARX values (0.0028 Silverbox,
0.0075 Cascaded Tanks) match the baselines doc exactly.

**Section 4.3 ARX prose.** chirp (0.0039 vs. 0.0383/0.1533), closedloop
(0.0036 vs. 0.0195/0.0545), rate `drivetrain` (0.0253 vs. 0.1722/0.2180),
and the backlash-coarsest-rate near-tie (0.3116 vs. 0.3107) all match the
baselines block exactly.

**Table 3 + Section 4.4 identifiability prose (`tab:ident`).** median/mean r
for base (−0.122/−0.092), power-controlled (−0.117/−0.103), range-filtered
(−0.122, 40/40 cells), the 10/40 positive-cell count, the per-family pooled
values (backlash +0.294, drivetrain +0.091, stribeck −0.081, saturate
−0.148, boucwen −0.050), the pooled r's (−0.088 max rel-CRLB, −0.316 log10
FIM cond), the 39,994/40,000-finite count, the max within-cell r of +0.327
(backlash-closedloop dt=0.05), and the quartile medians (0.037, 0.037,
0.040, 0.021) and the mean-artifact figures (1.5×10³→1.6×10⁹) all match the
ident_exp block exactly. (Note: Q2's 0.0375 median rounds to "0.037" under
IEEE-754 float truncation — verified this is what Python itself produces
for `f"{0.0375:.3f}"`, and it's also what `figures/make_figures.py` would
render, so the paper's rounding is consistent with the actual computation,
not an error.)

**Table 4 (architecture ablation, `tab:ablation`).** All 5 variants × (params,
reference nMSE, family nMSE) — narrow/default/shallow/deep/wide — match the
2026-07-15 ablation doc exactly, including rounded param counts (407k,
1.58M, 655k, 2.51M, 6.24M). Prose numbers 0.2835–0.2941, ±0.0095, 15×,
19.2× vs 7.1×, and 0.2901 vs 0.2835 all match the ablation doc's own text
verbatim.

**Abstract/corpus-level counts.** 240,000 instances (5×4×3×4,000 cells),
93–118× (cross-family wh_only range, correctly rounded from 92.6–117.5),
14–18× (corpus cross-family range, correctly rounded from 13.5–18.3), r=
−0.122, 40 cells, 192/32 context/query split, 5 seeds, seeds 0–4, 10,000
steps, eval seeds 900–905, ≤8 real-plant windows, 224-sample context+query
(192+32), Silverbox 12× decimation → dt≈19.7ms (0.0197s in source),
Cascaded Tanks 40× coarser (4.0s/0.10s) — all check out against the source
docs (some, like native Silverbox dt and WH-Process-Noise record length, are
external facts not present in the two results docs and were only checked
for internal arithmetic consistency, which holds).

## 2. DISCREPANCIES

**(1) "30–300×" / "two to three orders of magnitude" real-plant ARX
margin — overstated at the low end. [Locations: Abstract, Intro summary
bullet, Section 4.2 prose ("two to three orders of magnitude"), Section 4.3
"Result" paragraph ("30–300×"), Limitations ("the 30–300× margins").]
Severity: HIGH (repeated headline claim).**
Computing the actual per-cell ARX-vs-transformer ratios from Table 2's own
values: Silverbox wh_only/ARX = 0.9583/0.0028 ≈ 342×; Silverbox corpus/ARX =
0.3306/0.0028 ≈ 118×; Cascaded Tanks wh_only/ARX = 0.3937/0.0075 ≈ 52×;
Cascaded Tanks corpus/ARX = 0.0843/0.0075 ≈ **11.2×**. The claimed floor of
"30×" (and "two orders of magnitude" ≈100×) is violated by the Cascaded
Tanks corpus-vs-ARX ratio (≈11.2×, only ~1 order of magnitude), and the
claimed ceiling of "300×" is violated by the Silverbox wh_only-vs-ARX ratio
(≈342×). The true range spanned by the four comparisons the sentence claims
to cover ("both trained transformers ... on both real-plant benchmarks") is
approximately 11×–342×, not 30×–300×.

**(2) "8–39× → ~1–2×" rate/excitation-gap-closing claim doesn't account for
the `drivetrain` rate cell shown two lines above it in Table 1. [Locations:
Section 4.1 prose, echoed by Abstract's "~1–2×".] Severity: MEDIUM.**
Table 1's held-out-rate-dt=0.05 row for `drivetrain` shows wh_only 54.9×
and corpus **8.0×** — both far outside the stated "8–39×" (wh_only) and
"~1–2×" (corpus) ranges. The `saturate` rate cell (wh_only 3.5×) is also
below the stated "8×" floor. The text is partially hedged by "solves the
rate and excitation axes for most families," but neither exception
(`drivetrain`, `saturate`) is named alongside this specific numeric range,
so a reader checking the immediately-preceding table finds two cells the
prose's own bracketed range doesn't cover.

**(3) Degree-2 NARX "10¹⁰–10¹¹" divergence-range claim is violated by 2 of
the cells it should cover. [Location: Section 4.3, "Degree-2 NARX diverges
broadly" paragraph.] Severity: MEDIUM.**
Text: "clipped-but-huge nMSE, 10¹⁰–10¹¹, on all but the real-plant Silverbox
cell." From the baselines block: held-out excitation `chirp` NARX2 =
450,212,432,381.90 ≈ **4.50×10¹¹** (exceeds the stated 10¹¹ ceiling by
~4.5×), and real-plant `Cascaded_Tanks` NARX2 = 3,665,027,259.18 ≈
**3.67×10⁹** (below the stated 10¹⁰ floor) — and Cascaded Tanks is not
named as an exception (only Silverbox is). So the stated range fails to
cover 2 of the 12 NARX2 cells in the source data, one of which
(Cascaded Tanks) the text explicitly claims is *included* in the range.

**(4) "153 quadratic features vs. 152 fit samples" — not present in either
source-of-truth results doc; unable to verify against provided sources.
Severity: INFORMATIONAL, not a confirmed discrepancy.**
Neither results doc states a feature/sample count for the ARX order-k=8
selection step. Independently sanity-checked: 153 = C(18,2), the number of
degree-≤2 monomials over 16 lagged regressors (8 lags of u + 8 lags of y),
which is at least internally plausible for a degree-2 NARX with order 8 —
but this cannot be confirmed or refuted from the two provided
source-of-truth documents, so flag it as unverified rather than wrong.

**(5) Table 1 bold-face emphasis is inconsistent across the three
`backlash` rows. Severity: COSMETIC, not a numeric error.**
The `backlash dt=0.05` row bolds corpus's win (0.2899±0.0095, 13.5×), but
the `dt=0.10` and `dt=0.02` backlash rows — where corpus also wins on both
raw nMSE and ×ref — are left unbolded. Not a transcription error (all
values themselves are correct), but a presentational inconsistency worth a
copyedit pass, since selective bolding can visually imply a comparison
result is confined to one rate when the underlying numbers show it holding
across all three.

## 3. ANONYMITY

**Verdict: CLEAN.** No author-identifying strings found.

- `paper/main.tex` author block: `Anonymous Author(s)` / `Anonymous
  Institution` / `anonymous@example.com` — matches expected NeurIPS
  anonymous placeholder.
- `paper/references.bib`: `plantforgedataset` entry uses
  `author={{Anonymous}}`, `howpublished={link omitted for anonymous review;
  will be included in the camera-ready version}` — matches expected state.
  No huggingface.co/datasets/<user> or similar dataset URL present anywhere.
- `paper/main.pdf`: `pdftotext` full-text scan for
  `phuocthien|nas07|huggingface|github\.com/|@gmail|anonymous@example|
  stark4062|/data/|Author|Institution` returns only the intended
  "Anonymous Author(s)" string (page 1 author block). No other hits.
- PDF metadata (`pdfinfo` / `strings`): `Author` field empty, `Creator:
  LaTeX with hyperref`, `Producer: pdfTeX-1.40.25` — no username, host, or
  path embedded in document info dictionary.
- `main.aux` / `main.log` / `main.out` / `main.bbl`: scanned for
  `phuocthien|nas07|/home/|/data/` — no hits (these are build artifacts,
  not shipped, but checked for completeness since they live in `paper/`).
- Figure paths: all four `\includegraphics` targets
  (`../figures/fig{1,2,3,4}_*.pdf`) resolve to files that exist relative to
  `paper/`.
- `figures/make_figures.py`: reviewed in full. All strings are generic
  (color hex codes, axis labels, family/excitation names, numeric literals
  transcribed from the results doc). No embedded file paths, usernames, or
  identifying comments — the module docstring cites only the repo-relative
  results-doc path (`docs/superpowers/results/2026-07-14-experiment-
  results.md`), which is not identifying on its own.
