# PLANTFORGE — NeurIPS D&B/Evaluations & Datasets submission checklist

Compiled 2026-07-16, after the 4-way final-audit pass
(`docs/superpowers/reviews/`) and its fix commit (`3ca6726`). This is a
pre-submission checklist, not a review — it covers what the 4-way audit
does not: venue/timeline reality, formal-checklist compliance, and
mechanical submission logistics. Sourced against the live NeurIPS
2026 Evaluations & Datasets (formerly "Datasets and Benchmarks") track Call
for Papers and the official NeurIPS Paper Checklist guide (both fetched
2026-07-16 — re-verify against whatever cycle you actually target, since
these requirements change year to year).

**Update 2026-07-16 (a)**: single-blind mode chosen — the paper now uses
`\usepackage[preprint]{neurips_2024}` with real author info (Huynh Phuoc
Thien Nguyen, National Central University) and the self-citation in
`references.bib` links the real HuggingFace dataset URL. §1 item 5 and §2
below are resolved.

**Update 2026-07-16 (b)**: the formal NeurIPS Paper Checklist (§1, all 16
items) is now written into `paper/main.tex` as a real section (after the
bibliography, using the template's `\answerYes`/`\answerNo`/`\answerNA`
macros), using the draft answers from §1's table below, finalized. Items 8
(compute resources) and 9/11 (ethics/safeguards) used the draft answers
as-is; item 16 (LLM usage) went with the "N/A for core methods, disclosed
transparently anyway" framing rather than asking for further sign-off,
consistent with this project's established transparency norm — revisit
its exact wording before final submission if a stricter framing is
preferred. Recompiled clean (16 pages total; checklist adds ~6 pages,
excluded from the venue's main-text page count). Still open: Croissant/RAI
metadata (§5) and the main-text page-limit risk (§4, unaffected by the
checklist addition since it's excluded from the count either way).

## 0. Timeline reality check — read this first

**The NeurIPS 2026 Evaluations & Datasets track deadline has already
passed**: abstract deadline was 2026-05-04, full paper 2026-05-06 (AoE).
Today is 2026-07-16. This paper cannot be submitted to that cycle. Options:

- Target the next cycle (NeurIPS 2027's D&B-equivalent track — not yet
  announced; historically these open their CFP in the preceding winter/
  spring, so watch for it around late 2026/early 2027).
- Target a different, currently-open venue (a journal, a different
  conference's D&B-style track, or an arXiv-only preprint release without
  a venue).
- Treat this as an arXiv preprint for now (drop the "Anonymous Author(s)"
  submission mode, add real authors, and release without conference review)
  and decide on a venue later.

Nothing else in this checklist is blocking if the near-term plan is "arXiv
preprint, decide on a venue later" — items 1-4 below become "do before
whichever submission actually happens," not "do this week."

## 1. The formal NeurIPS Paper Checklist — currently MISSING, hard blocker

Confirmed: `paper/main.tex` has no "NeurIPS Paper Checklist" section. This
is a **mandatory, separate section** (not counted against the main-text
page limit) — per the official guidance, **submissions without it are
desk-rejected**, no exceptions. It must be added before any real
submission. The checklist is 16 questions, each answered Yes/No/NA with a
short justification. Draft answers below, grounded in what's actually in
this paper — review and adjust before pasting into the paper, especially
items 5, 8, 9, 10, 11 which need a real decision, not just a description
of current state.

| # | Question (topic) | Draft answer | Draft justification |
|---|---|---|---|
| 1 | Claims: do abstract/intro accurately reflect contributions/scope? | Yes | Every headline claim (multi-axis training narrows-but-doesn't-close the rate/excitation/family gaps; ARX beats both transformers on real-plant data; identifiability annotations don't positively predict difficulty; the `backlash` architecture-invariance result; the leave-one-family-out finding) maps to a dedicated experiment section and is scoped by a matching Limitations paragraph. The recent 4-way audit specifically caught and fixed one claim (the cross-family-gap figure) that had drifted into overgeneralization — see Section 5 and 4.1. |
| 2 | Limitations discussed? | Yes | Section 5 has 6 dedicated paragraphs: what the paper does/doesn't claim, real-plant validation scope, family coverage/held-out choice, architecture ablation scope, and negative/inconvenient results named explicitly. |
| 3 | Theory assumptions/proofs complete? | N/A | No theoretical results — this is an empirical dataset+benchmark paper. |
| 4 | Reproducibility steps for dataset/model? | Yes | Full corpus-generation code, exact family/excitation/rate definitions, fixed evaluation seeds (900-905) shared across all paired comparisons, checkpoint-resumable training with disclosed hyperparameters, 5-seed training-seed disclosure (0-4) for every headline number. |
| 5 | Open access to code/data (or URL)? | **Resolved: Yes** | Public GitHub (`Soarr01/plantforge`, MIT) and HuggingFace (`stark4062/plantforge`, CC BY 4.0) releases already exist; the paper's self-citation now links the real HuggingFace URL directly (single-blind, see §2). |
| 6 | All training details specified (splits, hyperparameters, selection)? | Yes | Training-seed counts, evaluation-seed fixation, context/query window lengths (192/32), decimation factors per real dataset, ARX order-selection protocol (last-32-context one-step-ahead error), and all 4 ablation variants' exact width/layers are stated in-text. |
| 7 | Error bars / statistical significance reported correctly? | Yes | Nearly every number is mean±std over 5 (4 for one flagged exception) independently-seeded runs; the paper explicitly discusses the limits of the ±1σ-overlap heuristic (not multiplicity-corrected) where it's used as a comparison bar. |
| 8 | Compute resources disclosed per experiment? | **Currently No — needs a new paragraph, see §3 below** | Not yet stated anywhere in `main.tex`. |
| 9 | Conforms to the NeurIPS Code of Ethics? | Needs your read of the actual Code of Ethics text, but likely Yes | Synthetic corpus, no human subjects, no PII, no scraped personal data; real-plant benchmarks used are public academic system-ID datasets (Silverbox, Cascaded Tanks, Bouc-Wen) under their own open licenses. |
| 10 | Broader societal impacts discussed (if applicable)? | Yes | Section 6, Broader Impact: positive impact (reduces duplicated private data-generation effort, supports excitation-design research); states no misuse vectors beyond those generic to open ML tooling. |
| 11 | Safeguards for high-misuse-risk model release? | N/A / No safeguards needed | The released model is a small (1.58M-2.51M param) system-ID transformer trained on synthetic control-plant data — not a generative/language model with a plausible misuse vector; no safeguards beyond the standard open license are applicable. State this explicitly rather than leaving it blank. |
| 12 | Existing-asset licenses respected/cited? | Yes | Silverbox/Cascaded Tanks/Bouc-Wen are cited with their DOIs (`references.bib`); Bouc-Wen specifically has its CC BY-SA 4.0 license and DOI (10.4121/12967592) noted in-repo (`realbench.py` docstring, DATASHEET). NeurIPS 2024 D&B template cited. |
| 13 | New assets (the corpus, the code) documented? | Yes | `docs/DATASHEET.md` (Gebru et al. template: motivation, composition, collection, uses, distribution, maintenance), `LICENSE` (MIT, code) / `LICENSE-DATA` (CC BY 4.0, corpus), README. |
| 14 | Crowdsourcing / human-subjects instructions? | N/A | No crowdsourcing, no human-subjects research. |
| 15 | IRB approval? | N/A | No human subjects — no IRB was needed. |
| 16 | LLM usage declared, if a core/non-standard component? | **Needs your decision on exact wording** | If you want to disclose that this corpus/paper/experiment pipeline was built with heavy LLM (Claude) assistance throughout — code, experiment design, paper drafting — this question is where that goes. LLM use here isn't a *methods* component (the corpus generator and transformer training are ordinary code/ML, not LLM-based), so a defensible answer is "N/A — LLMs were not used as a methodological component of the corpus generation or the trained models," while optionally still noting LLM-assisted authorship in the Acknowledgments/checklist justification. This is a real editorial decision, not something I should pick for you. |

Items 5, 8, 9, 11, 16 need your input before this section can be pasted into
the paper — everything else has a defensible draft answer above grounded in
what's actually in the repo.

## 2. Item 5 in depth: the anonymity-vs-accessibility tension — RESOLVED

**Decision (2026-07-16): single-blind.** The live NeurIPS 2026 Evaluations &
Datasets CFP states data/code "must be available and accessible to all
reviewers... without personal requests at submission time," and separately
notes that D&B-style tracks often can't be reviewed fully double-blind, so
single-blind submission is explicitly sanctioned for exactly this situation
(a released dataset). Applied to `paper/main.tex` and `paper/references.bib`:

- `\usepackage{neurips_2024}` → `\usepackage[preprint]{neurips_2024}`
  (real author names visible; footer reads "Preprint. Under review.",
  distinct from the `final`/camera-ready option which would print an
  accepted-venue notice instead).
- Author block: real name (Huynh Phuoc Thien Nguyen), affiliation
  (National Central University), email.
- `references.bib`'s `plantforgedataset` self-citation: real author name,
  direct link to `https://huggingface.co/datasets/stark4062/plantforge`
  (previously `{{Anonymous}}` / "link omitted for anonymous review").
- Recompiled clean (`pdflatex` + `bibtex` + 2x `pdflatex`, 12 pages, no
  undefined refs), title page and reference-list entry visually verified.

Not pursued: the double-blind-with-anonymized-mirror alternative
(`anonymous.4open.science` + a fresh anonymous HF org) — single-blind was
simpler and is explicitly allowed for this track/situation.

Current state (link fully omitted, no anonymized mirror) satisfies neither
cleanly — it's stricter than double-blind requires (no anonymized link at
all) while also not meeting the track's explicit accessibility requirement.
Pick one of the two paths above once you know the target venue's exact
policy for that cycle.

## 3. Item 8 in depth: compute-resources — DONE (2026-07-16, via the checklist itself)

Resolved differently than originally planned: rather than a new main-text
paragraph (which would have worsened the page-limit pressure in §4), the
compute numbers below were written directly into the formal NeurIPS Paper
Checklist's item 8 justification (`paper/main.tex`, excluded from the
9-page main-text count) when that section was authored. Kept here for
reference:

- **Hardware**: single NVIDIA GeForce RTX 2080 Ti (11 GB), one GPU per
  training run — the model (1.58M params at default width/depth, up to
  6.24M for the `wide` ablation variant) does not need multi-GPU.
- **Per-run wall-clock**: measured from the 16-run leave-one-out-seeds
  training job (2026-07-15/16, dedicated GPU, back-to-back runs): steady-
  state gap between consecutive 10,000-step training-run completions was
  **~14-16 minutes** per run for the default (1.58M param) architecture.
  One outlier gap (~75 min, at a seed boundary) was observed and not fully
  explained — flag as "typically ~15 min, occasionally longer on a shared
  machine" rather than asserting a tight bound.
- **Total training runs behind this paper's numbers**: 5 seeds × 2 recipes
  (headline, Table 1) + 4 architecture variants × 5 seeds (ablation,
  Table 5, `default` reused from headline) + 4 held-out families × 5 seeds
  (leave-one-out, Table 4, `backlash` reused from headline) = order of
  50 total 10,000-step training runs, each ~15 min on one RTX 2080 Ti →
  roughly **12-13 GPU-hours** total for all trained-model results in the
  paper (rough arithmetic from the above, not independently re-measured
  end-to-end — sanity-check before quoting a precise number in print).
- **Corpus generation** (separate, CPU-bound, not GPU time): ~14 hours
  wall-clock on a heavily-shared/CPU-contended machine, measured from shard
  mtimes (already documented in `docs/DATASHEET.md`).

Suggest adding a short paragraph (Section 3 "The PLANTFORGE Corpus" or a
new subsection under Section 4/Experiments preamble) with the above,
trimmed to what fits the page budget (see §4).

## 4. Page limit — DONE (2026-07-16)

Confirmed from the primary source (NeurIPS 2026 Main Track Handbook,
fetched live): "The main text of a submitted paper is limited to nine
content pages, including all figures and tables." References, appendix,
and the mandatory checklist do not count.

Measured before trimming: main content (Abstract through Conclusion)
spilled ~0.6-0.7 pages onto page 10, with References starting partway down
that page. Trimmed prose across ~12 paragraphs (Related Work's 6
subsections, all 5 Discussion/Limitations paragraphs, Broader Impact, and
the Conclusion) — tightening connective language and removing one genuinely
redundant explanation (the $\times$ref-ratio-is-misleading argument was
fully spelled out twice; the second occurrence, in the architecture-ablation
paragraph, now cross-references the first instead of re-deriving it).

**No numbers, citations, or claims were removed or altered** — verified by
grepping every key figure that appears in the trimmed passages (93--118x,
14--18x, 11--342x, 26--42x, 19.1x/7.4x, 0.29--0.30, 0.035--0.047, 6--8x)
against the post-trim source, all present and unchanged. Recompiled clean
(pdflatex x2, no undefined references), and verified visually: page 9 now
ends exactly at the end of the Conclusion, page 10 starts directly with
References, no main-text spillover. Total PDF is 16 pages (9 main + ~2
references + the new NeurIPS checklist section, which is correctly excluded
from the content-page count).

## 5. Croissant metadata — DONE (2026-07-16)

The live 2026 CFP requires datasets to ship **Croissant machine-readable
metadata, including both core AND Responsible AI (RAI) fields** — stated
as a compliance item whose absence "justifies desk rejection." HuggingFace's
automatic Croissant generation doesn't apply here (the corpus is 60 raw
`.pt` PyTorch shard files, not Parquet/ImageFolder), so a Croissant file was
hand-authored: [`croissant.json`](../croissant.json) at the repo root.

**How it was verified, not just written:** installed the official
`mlcroissant` Python library (v1.1.0) and validated with
`mlc.Dataset(jsonld='croissant.json')` — iterated through 3 real validation
error rounds (wrong `@type` namespace for FileObject/FileSet, missing
`conformsTo` CroissantVersion, missing checksums, fields without a `source`)
until it loads with **zero errors and zero warnings**. Real facts used
throughout rather than placeholders: exact shard filenames/glob pattern and
tensor schema (confirmed by loading `corpus/backlash_prbs_dt10hz.pt`
directly), a real SHA-256 checksum of `corpus/registry.json`, and the
dataset repo's actual creation date and git commit (queried live via
`huggingface_hub.HfApi().dataset_info(...)`).

**One honest limitation, not hidden**: full end-to-end record
materialization via `ds.records('shards')` did not successfully enumerate
files in this environment (the FileSet's git+https-hosted container didn't
populate a local download cache during testing) — a known rough edge for
git/LFS-hosted non-Parquet datasets in the mlcroissant tooling, not a defect
in the file's structure (which independently validates clean). Documented
in the file's own `rai:dataLimitations` field rather than silently claimed
as fully working.

Also referenced from `docs/DATASHEET.md` and `README.md`, and uploaded
alongside the data itself to the HuggingFace dataset repo
(`stark4062/plantforge`), satisfying "ship it alongside the dataset."

This is real, unstarted work — worth scoping as its own small task before
whatever submission actually happens, not something to rush the week of a
deadline.

## 6. Mechanical pre-submission checklist (do these regardless of venue specifics)

- [ ] Formal NeurIPS Paper Checklist section added to `main.tex` (§1 above)
- [ ] Compute-resources paragraph added (§3 above)
- [ ] Page-limit compliance re-verified against the exact target-year CFP (§4)
- [ ] Anonymity mode decided and applied consistently (§2 above) — if
      switching to single-blind, also un-anonymize the author block
      (currently "Anonymous Author(s)") and remove the anonymized-citation
      placeholder text in `references.bib`
- [ ] Croissant metadata authored and attached to the HF dataset repo (§5)
- [ ] OpenReview account(s) created for all authors before the abstract
      deadline (accounts must exist by that date per the CFP)
- [ ] Supplementary material assembled per whatever the target venue's
      portal expects (a zip of code, or a link — re-check the exact
      current mechanism, this has changed across NeurIPS cycles)
- [ ] Final anonymity re-scan of the submission PDF (already clean as of
      the 4-way audit — re-run if `main.tex` changes again before
      submission: `pdftotext main.pdf - | grep -iE "stark4062|Soarr01|phuocthien|nas07"`)
- [ ] Re-run `python -m plantforge.tests.run_all` and recompile the paper
      one final time immediately before upload, so the submitted PDF
      matches a green test suite

## 7. Already done — no action needed (from the 4-way audit, 2026-07-16)

- Anonymity: clean (author block, bib, PDF text/metadata, figure PDFs all
  scanned, no leaks).
- Numeric accuracy: every number in the paper cross-checked against live
  code output or source results docs; 2 real errors found and fixed.
- Executable truth: 43/43 offline tests pass, every README-advertised CLI
  reproduces its claimed numbers exactly, both leave-one-out training
  scripts confirmed idempotent/resumable.
- Adversarial claims coherence: the one real narrative gap found (headline
  cross-family-gap claim needing reconnection to the new leave-one-out
  finding) has been fixed and re-verified by recompiling and visually
  reviewing the changed pages.
