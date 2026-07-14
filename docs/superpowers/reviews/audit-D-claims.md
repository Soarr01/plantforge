# Audit D — Cross-document claims/logic/framing consistency

Scope: LOGIC and FRAMING only (no arithmetic re-verification). Documents read
in full: `paper/main.tex`, `README.md`,
`docs/superpowers/results/2026-07-14-experiment-results.md`,
`docs/superpowers/results/2026-07-15-architecture-ablation-results.md`,
`docs/DATASHEET.md`, `/data/nas07_new/PersonalData/phuocthien/plantforge_data/README.md` (HF dataset card).

---

## (a) "Solves the rate and excitation axes" hedging

**Verdict: TENSION.**

Section 4.1 hedges correctly and completely:

> "Multi-axis training \textbf{solves the rate and excitation axes for most
> families}... One cell, \texttt{saturate} at the held-out rate, is won by
> \texttt{wh\_only}... 'corpus training helps' is not a universal law across
> every cell, and we report the exception rather than omit it."

README's headline section also hedges correctly ("solves the rate and
excitation axes **for most families**", plus the saturate exception called
out explicitly in the table footnote).

But the **Abstract** and the **Introduction's "Summary of findings" bullet**
both drop the hedge and drop the exception:

- Abstract: "training on PLANTFORGE's multi-axis corpus **closes the rate
  and excitation gaps (to ∼1–2×)**" — stated as a flat, unqualified range.
- Intro bullet: "Multi-axis training on PLANTFORGE **closes the rate and
  excitation transfer gaps** that Wiener-Hammerstein-only training leaves
  open, but only halves the harder cross-family gap" — again unqualified,
  no "for most families," no mention of an exception.

Table 1 itself contradicts the "∼1–2×" abstract parenthetical on two counts:
(1) the held-out-rate `drivetrain` cell is **8.0×**, not 1–2×; (2) the
held-out-rate `saturate` cell is a case where `corpus` is *worse* than
`wh_only` (1.3× vs 3.5×) — not "closed," but *lost*. A referee reading only
the abstract would form a materially rosier picture than Table 1 supports.

**Suggested minimal fix:** In the abstract, change "closes the rate and
excitation gaps (to ∼1–2×)" to "closes the rate and excitation gaps for most
cells (to ∼1–2×, with one rate cell — `drivetrain` — at 8.0× and one cell
where `wh\_only` wins outright)," or point to Section 4.1's hedge. In the
Intro bullet, add "for most families/cells" to match Section 4.1 and README.

**Severity:** would-question. Not a fabrication (the underlying data is
reported honestly in Table 1 and Section 4.1), but the two most-read,
least-detailed passages (abstract, intro bullets) are the least hedged —
exactly backwards from where hedging matters most.

---

## (b) ARX result vs. the corpus's value proposition

**Verdict: COHERENT.**

Section 5's "What this paper does and does not claim" paragraph is
internally consistent and matches the abstract, README, and Conclusion:

> "We do \emph{not} claim in-context transformers are the strongest available
> method for real-plant system identification... We \emph{do} claim that (a)
> training-distribution breadth matters... and (b) a released,
> parameter-annotated, three-axis corpus is a more useful research artifact
> for studying \emph{why}... independent of whether the transformer
> architecture itself is currently the best tool for the job."

This is a coherent decoupling: the corpus's value is argued to lie in
*enabling* the comparison/diagnosis (breadth of axes, annotations,
reproducibility), not in *transformers being state-of-the-art*. The
abstract states the ARX result plainly ("tempering any claim of transformer
superiority") and the README mirrors this ("It tempers, rather than
refutes, the transformer's value proposition" / "'turns the incumbent into
the validator' is not yet a settled claim"). No passage found that still
implies unqualified transformer superiority after Section 4.3 — Related
Work's framing of in-context SysID as "an increasingly popular paradigm" is
a field description, not a superiority claim, and doesn't conflict.

**Severity:** ignore.

---

## (c) Identifiability framing

**Verdict: TENSION (one document), otherwise coherent.**

Abstract, Section 4.4, README, and DATASHEET's Uses section are all
consistent and correctly hedged:
- Abstract: "do \emph{not} positively predict in-context prediction
  difficulty... the opposite of our working hypothesis."
- Section 4.4: "weak, robustly negative, not the hypothesized direction,"
  and explicitly: "not, and should not be marketed as, a
  prediction-difficulty predictor."
- README: "not a prediction-difficulty predictor."
- DATASHEET Uses: "(4) testing whether identifiability annotations predict
  in-context prediction difficulty (`ident_exp.py` — result: they largely do
  not; ...)" — states the outcome, consistent with the others.

However, the **HF dataset card** (`plantforge_data/README.md`, the
world-facing artifact most likely to be read in isolation from the paper)
presents the identifiability annotations as an unqualified headline feature
with **no mention of the null result anywhere**:

> "with per-instance **Fisher-information identifiability annotations**"
> (top-line description) ... "**Identifiability annotations**: per-parameter
> relative Cramér-Rao lower bound and FIM log10-condition-number, per
> (instance, excitation, rate)." (Quick facts)

Nothing in the card says these annotations don't predict prediction
difficulty, nor does it point to the datasheet's Uses section for that
caveat (it only points to `docs/DATASHEET.md` "in the code repo" for the
*general* writeup, not specifically for this finding). This isn't a false
claim — the card never explicitly claims the annotations predict
difficulty — but by omission it lets the strongest-looking selling point
stand un-caveated exactly where a downstream dataset user (who may never
read the paper) would consume it.

**Suggested minimal fix:** add one sentence to the HF card, e.g. "Note: our
own experiments find these annotations do *not* positively predict
in-context prediction difficulty (see paper/datasheet); they are intended
for excitation-design and parameter-recovery studies."

**Severity:** would-question. Not a contradiction, but an inconsistency in
*where* the caveat is surfaced — present in every in-repo document, absent
from the one most likely to travel alone.

---

## (d) Capacity-invariance claim strength

**Verdict: TENSION (mild).**

The Discussion/Limitations section is properly and thoroughly hedged, with
"(single-seed check)" in the paragraph header itself, and: "This check
remains single-seed and does not cover context/query-length variation; a
full multi-seed architecture sweep is left to future work." The ablation
results doc mirrors this exactly ("Caveat: single seed per variant... not a
substitute for multi-seed ablation").

The **Conclusion** states, in the same paragraph that lists "a single-seed
architecture ablation" among the checks performed:

> "the held-out-family gap specifically **holds up as capacity-invariant**
> across a 15$\times$ parameter range, not an artifact of one model size."

The word "single-seed" does appear earlier in the same sentence's list of
checks, so a careful reader has the caveat nearby — but the specific
capacity-invariance clause itself is phrased as a flat, confirmed finding
("holds up as capacity-invariant," "not an artifact") rather than "is
consistent with capacity-invariance in this single-seed check." A reader
skimming just the Conclusion's punchline sentence (common referee behavior)
could walk away with a stronger claim than the Limitations section
licenses.

**Suggested minimal fix:** "...the held-out-family gap specifically holds up
as capacity-invariant, in this single-seed check, across a 15× parameter
range..." — repeat the qualifier at the point of the claim, not just
earlier in the sentence.

**Severity:** would-question. Same substance in both places, but rhetorical
strength diverges at the exact clause a reader is likely to quote.

---

## (e) Statistical hedging symmetry ("10 of 11 cells")

**Verdict: COHERENT.**

Section 4.1 carries essentially the same multiplicity caveat as the
internal results doc, near-verbatim:

> "Non-overlapping $\pm1$ standard deviation across 11 simultaneous cell
> comparisons is a descriptive heuristic, not a multiplicity-corrected test;
> at $n=5$ seeds this is a conservative bar relative to the standard error
> of the mean, and the largest-margin cells (below) clear it comfortably."

vs. results doc: "non-overlapping ±1std is a descriptive heuristic across 11
simultaneous cell comparisons, not a multiplicity-corrected test... the
load-bearing cells... clear it by a wide margin." Equivalent hedging,
present in the paper body.

The abstract does not use "significant" or imply a formal hypothesis test
anywhere for the headline transfer-matrix result — it reports only
mean±std nMSE ratios. The one place the paper does use "statistically
significant" is Section 4.4's description of the **discarded, confounded,
pooled** identifiability correlation (r=-0.088, r=-0.316) — which is
accurately labeled as significant-but-confounded and explicitly superseded
by the within-cell primary analysis, so this is not an overreach; it is
honest reporting of a result the paper then discards.

**Severity:** ignore.

---

## (f) "Released" claims vs. reality — licensed vs. published, and an anonymity leak

**Verdict: TENSION — two distinct issues, one of them severe.**

### f1. "Released under MIT" conflates license with publication

Paper, Section 3 "Release" paragraph: "All generation, training, and
evaluation code is released under MIT." README's License section says the
same. Neither the paper nor the README gives a code-repository URL
anywhere. The HF dataset card is explicit that the code location is
**not yet available**:

> "Code, generator, training/evaluation pipeline, and full documentation:
> `<code repo URL — TBD>`"

"Released under MIT" is true as a *licensing* statement (the repo is
licensed MIT) but reads as a *publication* statement ("released" implies
made available), which is not yet true — there is no public URL. This is
exactly the licensed-vs-published conflation the review brief flags.
DATASHEET.md is actually more careful here: "Issues/questions: via the
repository's issue tracker (see repository README for the canonical URL
**once published**)" — correctly forward-looking. The paper/README text
should match DATASHEET's more careful phrasing.

**Suggested minimal fix:** "All generation, training, and evaluation code is
licensed under MIT (repository URL to be added on publication)" instead of
"released under MIT."

**Severity:** would-question — a referee doing exactly this kind of
consistency pass would flag it, though it's a common and low-stakes
overstatement in anonymized submissions.

### f2. Anonymity: the HF handle deanonymizes the (nominally anonymous) submission

The paper's author block is `Anonymous Author(s)` / `Anonymous Institution`
— i.e., this is framed as a double-blind submission. But README.md and
DATASHEET.md both point to a **live, named Hugging Face dataset URL**:

> `https://huggingface.co/datasets/stark4062/plantforge`

The HF namespace `stark4062` is a personally identifying handle (matches
the submitting user's own account), not an anonymized placeholder like the
paper's `\citep{plantforgedataset}` bibliography entry. A referee (or
anyone) who clicks the README's Hugging Face link, or who searches for the
corpus name, immediately deanonymizes the submission — directly undermining
the paper's own "Anonymous Author(s)" framing. This is a genuine
**logical inconsistency between documents**: the paper-facing artifact
claims anonymity; the release-facing artifact (which the paper's own
"released" claim points a reader toward) does not preserve it.

**Suggested minimal fix:** host the dataset under an anonymized HF
namespace/handle for the review period (many venues support this, e.g. via
an anonymous HF org or a mirror), and keep the real `stark4062` handle for
the camera-ready release only.

**Severity:** would-reject-or-flag-desk-issue. This is the single highest-severity
finding in this audit — not a wording nit but a policy-relevant
inconsistency that a referee (or program chair) would very likely act on if
double-blind review is in effect for this venue.

---

## (g) Datasheet forward-looking promises vs. current state

**Verdict: TENSION (minor) — internal staleness, not incoherence with reality.**

DATASHEET.md's Distribution section hedges the dataset URL prospectively:

> "Will the dataset be distributed to third parties? Yes, via a public
> Hugging Face Datasets repository (see repository README for the link
> **once published**)."

But README.md, at the same release, already gives the **live, concrete**
HF URL (`stark4062/plantforge`) rather than treating it as pending. So the
datasheet's "once published" phrasing is stale relative to the README it
points to — a reader following the datasheet's instruction ("see README for
the link once published") would find the link is already there, which is
harmless but reads as out of sync (as if the datasheet was drafted before
the HF push and not updated after). This is a much lower-severity version
of the same drift as (f1)/(f2): documents disagree about whether
publication has already happened.

The Maintenance section's issue-tracker language ("see repository README
for the canonical URL once published") is, by contrast, **accurate** —
there genuinely is no code repo URL yet anywhere in the release, so this
promise is correctly hedged and not incoherent with current state.

**Suggested minimal fix:** Update DATASHEET's Distribution section to state
the HF URL directly (matching README), and leave the Maintenance section's
"once published" language for the code repo only, since that one really is
still pending.

**Severity:** ignore-to-would-question — cosmetic staleness, not a claim of
something false.

---

## Additional tensions found (not in the a–g list)

1. **Abstract's cross-family point estimates vs. Table 1's range.** Abstract
   states "93–118× → ... 93× → 14×" is compressed to a single "93× → 14×" in
   the Intro bullet ("93–118×→14–18×" per the body). This is a rounding/
   compression choice, not a logic error, and is flagged here only because
   it borders on arithmetic-transcription territory — leaving it to the
   other auditor, noting it here for completeness since the ranges quoted
   differ slightly across the abstract, intro bullets, and Section 4.1
   ("93–118× → 14–18×" in 4.1's prose vs. the abstract's single-point "93×→
   14×"). Not flagged as a logic/framing issue on its own.

2. **DATASHEET "Uses" section value judgment ("largely do not") is slightly
   softer than the paper's blunter "not the hypothesized direction" /
   README's "not a prediction-difficulty predictor."** All three are
   directionally consistent (none oversell), so this is not a tension, just
   noted for completeness — "largely do not" is arguably the most accurate
   phrasing of the three, since 10/40 cells are in fact positive.

---

## Summary table

| Point | Verdict | Severity |
|---|---|---|
| a. rate/excitation "solves" hedge | TENSION | would-question |
| b. ARX vs. corpus value prop | COHERENT | ignore |
| c. identifiability framing | TENSION (HF card omission) | would-question |
| d. capacity-invariance strength | TENSION (mild) | would-question |
| e. statistical hedging symmetry | COHERENT | ignore |
| f1. "released" = licensed vs. published | TENSION | would-question |
| f2. HF handle deanonymizes submission | TENSION | **would-reject / desk-flag** |
| g. datasheet forward-looking promises | TENSION (minor, stale) | ignore-to-question |

**Overall assessment:** The release is substantively honest and
well-self-critiqued — the paper actively reports its own two "inconvenient"
results (ARX beats transformers; identifiability doesn't predict
difficulty) with real hedging in the body text. The problems found are
concentrated in (1) headline surfaces (abstract, intro bullets, Conclusion)
occasionally stating claims more crisply/confidently than the body text
they summarize, and (2) one process/policy-level issue — the Hugging Face
handle `stark4062` in README.md/DATASHEET.md deanonymizes an ostensibly
double-blind submission — that is more consequential than any pure wording
issue in this audit and should be fixed before submission, not just noted.
