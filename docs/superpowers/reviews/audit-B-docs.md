# Audit B — Documentation Consistency Review

Scope: README.md, docs/DATASHEET.md, plantforge_data/README.md (HF card),
LICENSE, LICENSE-DATA, requirements.txt, figures/README.md, paper/README.md,
cross-checked against docs/superpowers/results/2026-07-14-experiment-results.md,
docs/superpowers/results/2026-07-15-architecture-ablation-results.md, and the
actual code (families.py, corpus.py, evaluate.py, baselines.py, realbench.py,
aggregate.py, ident_exp.py, scripts/*.sh, figures/make_figures.py, tests/*).

## 1. CONSISTENT

- **Transfer-matrix numbers**: every value/std/×ref multiplier in README's
  headline table (family backlash dt=0.05, rate dt=0.05 stribeck/saturate,
  excitation chirp/closedloop) matches
  `2026-07-14-experiment-results.md` section A exactly, and also matches
  `paper/main.tex` (spot-checked lines 216, 280–281, 50, 382/384).
- **Real-plant numbers** (Silverbox 0.958±0.217 / 0.331±0.040, Cascaded_Tanks
  0.394±0.081 / 0.084±0.043) and **ARX baseline numbers** (0.0028, 0.0075,
  drivetrain 0.0253) match the results doc and README's "Classical baselines"
  section, including the ARX-loses-on-`saturate` explosion footnote being
  consistent with the reading notes.
- **Identifiability r-values** (median r = −0.122 / −0.117 / −0.122, 40
  cells, 10/40 positive) match exactly between README, results doc, and
  `paper/main.tex`.
- **Corpus size arithmetic**: 5 families × 4 excitations × 3 rates = 60
  cells × 4000/cell = 240,000 instances — consistent across README,
  DATASHEET, and the HF card. `FAMILIES`/`EXCITATIONS`/`RATES` in
  families.py/excitation.py/corpus.py match all three docs' lists
  (stribeck, backlash, saturate, boucwen, drivetrain; prbs, multisine,
  chirp, closedloop; 10/20/50 Hz ↔ dt 0.10/0.05/0.02).
- **Shard format**: HF card's Format section (`u`,`y`,`theta`,`keys`,
  `rel_crlb`,`log10_cond`,`dt`,`family`,`excitation`) matches DATASHEET's
  Composition section field-for-field.
- **License story**: README, DATASHEET (Distribution), and HF card's
  `license: cc-by-4.0` YAML + License section all agree code=MIT/data=CC BY
  4.0. LICENSE and LICENSE-DATA text is internally consistent and
  cross-references each other correctly. `plantforge_data/LICENSE-DATA` is
  **byte-identical** to the repo's `LICENSE-DATA` (`diff` confirms no
  difference).
- **TBD placeholders**: exactly the two expected/intentional ones in the HF
  card (code-repo URL, BibTeX) — no other bare placeholders found via grep
  across all seven audited files.
- **Test-suite claims**: no doc claims a stale test count (e.g. "6 invariant
  tests"); actual suite is 33 `def test_` functions across 6
  `tests/test_*.py` modules (ablation 3, aggregate 4, baselines 4, ident_exp
  4, plantforge 6, realbench 12) — nothing contradicts this.
- **Checkpoint naming**: README's `eval_{mode}_s{seed}.pt` claim matches
  `evaluate.py`'s `_ckpt_path` logic and `scripts/train_seeds.sh`'s
  `eval_${name}_s${seed}.pt`; no stale unsuffixed `eval_corpus.pt` reference
  found anywhere in the audited docs.
- **Use-section commands**: `python -m plantforge.corpus --instances 4000`
  (argparse `--instances`, default 200), `python -m plantforge.evaluate
  headline|corpus` (sys.argv[1] mapped to wh_only/corpus), `python -m
  plantforge.baselines real` (sys.argv[1]=="real" gate), `scripts/
  train_seeds.sh` (exists, does what its README line says: iterates SEEDS="0
  1 2 3 4" × {headline,corpus}, skips finished checkpoints via a `step`
  read) — all verified against actual code behavior, no argv mismatches.
- **Layout table**: every file path listed (families.py, excitation.py,
  identifiability.py, corpus.py, evaluate.py, realbench.py, aggregate.py,
  baselines.py, ident_exp.py, scripts/train_seeds.sh,
  figures/make_figures.py) exists at the stated path.
- **All relative links in README.md resolve**: figures/fig{1,2,3}.png
  (embedded), figures/fig4_quartile_artifact.png (text ref),
  figures/README.md, docs/DATASHEET.md,
  docs/superpowers/results/2026-07-14-experiment-results.md, LICENSE,
  LICENSE-DATA — all present on disk.
- **Function cross-references** in DATASHEET (`families.py:sample`,
  `corpus.gen_cell`, `evaluate._norm`,
  `tests/test_plantforge.py:test_physical_consistency_of_pairs`) all exist
  in code as named.
- Architecture-ablation results doc is internally consistent with its own
  disposition note ("no paper text edited in this pass") — `paper/main.tex`
  does not currently cite ablation numbers, so no ablation-vs-paper conflict.

## 2. DISCREPANCIES

1. **Corpus generation wall-clock time contradicts itself across two docs
   describing the identical command/run.** README.md line 152:
   `python -m plantforge.corpus --instances 4000  # generate corpus shards
   (CPU, ~2h)`. docs/DATASHEET.md lines 92–94 (Collection process): "this
   repository's reference corpus was generated 2026-07-14 with `python -m
   plantforge.corpus --instances 4000` (**~20 minutes** wall-clock on the
   reference machine, CPU-bound)." Same command, same reference corpus, 6×
   discrepancy (20 min vs 2h). One of these is wrong.
   **Severity: HIGH** (reproducibility-facing numeric claim, directly
   contradictory, likely to mislead a reader deciding whether to
   regenerate locally vs. download from HF).

2. **figures/README.md's dependency claim is false — matplotlib is not in
   requirements.txt.** figures/README.md line 3: "Regenerate all four from
   source: `python figures/make_figures.py` (from the repo root; needs
   `matplotlib`, already implied by `requirements.txt`'s deps)." Actual
   requirements.txt (4 lines of content) lists only `numpy`, `scipy`,
   `torch`, `nonlinear_benchmarks` — no matplotlib, direct or transitive.
   `figures/make_figures.py` does `import matplotlib`, `import
   matplotlib.pyplot as plt`, `import matplotlib.ticker as mticker`. A
   reader who does exactly `pip install -r requirements.txt` (as the
   README's Requirements section instructs) then runs
   `figures/make_figures.py` per figures/README's instructions will hit
   `ModuleNotFoundError: matplotlib`.
   **Severity: MEDIUM-HIGH** (broken instructions as literally written;
   easy fix is either adding matplotlib to requirements.txt or correcting
   the claim).

3. **DATASHEET.md's Distribution section is stale relative to the
   HF-published state described elsewhere.** docs/DATASHEET.md lines
   142–144: "Will the dataset be distributed to third parties? Yes, via a
   public Hugging Face Datasets repository (see repository README for the
   link **once published**)." But README.md line 218–219 already contains a
   live, concrete link
   (`https://huggingface.co/datasets/stark4062/plantforge`), and per the
   audit brief the dataset is already uploaded there. "once published"
   reads as still-pending when it isn't.
   **Severity: LOW-MEDIUM** (stale wording, not a factual numeric error,
   but inconsistent tense/state vs. README and the HF card that already
   exist).

4. **DATASHEET.md's Maintenance section points to a URL that README.md
   does not contain.** docs/DATASHEET.md lines 156–158: "Who will be
   supporting/hosting/maintaining the dataset? ... Issues/questions: via
   the repository's issue tracker (**see repository README for the
   canonical URL once published**)." README.md has **no** GitHub URL, no
   issue-tracker link, and no repository URL anywhere in its text (grepped
   for "github"/"issue" — zero hits). The forward-reference is dangling:
   following DATASHEET's pointer to README yields nothing.
   **Severity: LOW-MEDIUM** (same class as #3 — a doc promises information
   that its cross-referenced target doesn't actually have).

## 3. BROKEN LINKS / STALE CONTENT

- No broken relative links found in README.md, DATASHEET.md, or
  figures/README.md — every path/anchor checked resolves to a real file.
- No stray/unintentional `TBD` or placeholder tokens found outside the two
  expected HF-card spots (code-repo URL, BibTeX).
- No stale `eval_corpus.pt` (unsuffixed) references in any audited doc.
- No stale `/home/coder` path instructions in any of the seven audited
  docs. (Note, outside the audited file set: `tests/run_all.py`'s own
  docstring comment still says "(from /home/coder)" — this is a code
  comment, not one of the seven files under audit, and README.md's Use
  section already gives the correct/current cwd instruction ("cwd one
  level above this package") without referencing `/home/coder`, so this
  does not create a user-facing doc inconsistency. Flagged here only for
  completeness/awareness, not counted in the discrepancy total.)
- No stale test-count claims (e.g. "6 invariant tests") found; actual count
  (33 tests / 6 modules) is not contradicted anywhere.
- Two "dangling forward-reference" issues counted above (items 3 and 4) —
  both point to README.md for information README.md doesn't have (or no
  longer needs the hedge for, in item 3's case).

## Summary

- Numeric/factual discrepancies: **2** (corpus generation timing 2h-vs-20min;
  figures/README's false matplotlib-in-requirements.txt claim).
- Stale/dangling cross-reference issues: **2** (DATASHEET Distribution
  "once published" hedge is stale; DATASHEET Maintenance points to a
  README issue-tracker URL that doesn't exist).
- Broken links: **0**.
- All headline numeric results (transfer matrix, real-plant nMSE,
  identifiability r-values, corpus size/instance counts, license story,
  function/file cross-references) check out consistent across README,
  DATASHEET, HF card, results docs, and the paper draft.
