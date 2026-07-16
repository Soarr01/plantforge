# Datasheet: PLANTFORGE corpus

Following the *Datasheets for Datasets* template (Gebru et al., 2018,
[arXiv:1803.09010](https://arxiv.org/abs/1803.09010)).

## Motivation

**For what purpose was the dataset created?**
No existing dynamics/system-identification corpus jointly varies
**nonlinearity family × excitation class × sampling rate**, and none ships
per-instance identifiability annotations. The incumbent recipe in the
in-context SysID literature (the "forgi86 lineage") privately regenerates
Wiener-Hammerstein-only white-noise data per paper. PLANTFORGE was created
to (1) make that training distribution's narrowness measurable — a model
trained WH-only collapses off-distribution (see `README.md`) — and (2)
provide a released, static, reproducible corpus broad enough to fix it.

**Who created this dataset and on whose behalf?**
Generated procedurally by the code in this repository (`families.py`,
`excitation.py`, `identifiability.py`, `corpus.py`). No human or organization
is a data subject — see Composition below.

**Who funded the creation of the dataset?**
N/A — procedurally generated, no data-collection cost beyond compute.

## Composition

**What do the instances that comprise the dataset represent?**
Each instance is one draw of a control/mechatronics plant: a named
physical-parameter vector θ for one of 5 nonlinearity families, simulated
under one excitation class at one sample rate, producing an (u, y) input/output
trajectory pair plus Fisher-information identifiability annotations for that
specific (θ, excitation, rate) combination.

**How many instances are there in total?**
5 families × 4 excitations × 3 rates = 60 "cells," 4000 instances per cell =
**240,000 instances total** (default `--instances 4000` generation; the
corpus generator is resumable and re-runnable at any instance count).
On-disk size at this scale: **583 MB** across 60 `.pt` shard files
(4.3–20.7 MB each, larger at higher sample rates).

**What data does each instance consist of?**
Per shard (one family × excitation × rate cell), a dict of stacked tensors:
- `u`, `y`: `(T, B)` float32 — input/output trajectories, `T` = `round(12.8s / dt)`
  samples, `B` = instances in the shard
- `theta`: `(B, K)` — named physical parameters (`keys` gives the K names)
- `rel_crlb`: `(B, K)` — per-parameter relative Cramér-Rao lower bound
  (identifiability annotation)
- `log10_cond`: `(B,)` — log10 FIM condition number (identifiability annotation)
- `dt`, `family`, `excitation`: shard-level metadata

**Is there a label or target associated with each instance?**
No single "label" — this is a system-identification corpus. The (u, y) pair
IS the data; `theta` is the ground-truth generating parameter (usable as a
regression target for parameter-recovery tasks, distinct from the
next-step/in-context-prediction task this repo's own experiments use it for).

**Is any information missing from individual instances?**
No — every instance has a complete (u, y, theta, rel_crlb, log10_cond) tuple.
Some (instance, excitation) combinations under closed-loop excitation can
diverge (e.g. a PI loop against a resonant `drivetrain` instance going
unstable); such draws are dropped during generation, not retained with
missing fields (see `corpus.gen_cell` / evaluation code's finite-check
filtering).

**Does the dataset contain data that might be considered confidential, or
that, if viewed directly, might be offensive, insulting, threatening?**
No. All data is synthetic simulation output; no personal, sensitive, or
human-subject data of any kind.

## Collection process

**How was the data associated with each instance acquired?**
Not "collected" — procedurally generated. Physical parameters θ are sampled
per-family from documented prior ranges (`families.py:sample`); excitation
signals are generated in physical time (`excitation.py`); the plant is
simulated forward via family-specific `Stepper` objects with **exact
zero-order-hold** at the discretization rate and internal substepping
(≤2ms) for state-nonlinear families, so the same continuous-time plant is
consistent across all three sampling rates by construction (design invariant
2, tested in `tests/test_plantforge.py`). Fisher-information identifiability
annotations are computed analytically/numerically per instance
(`identifiability.py`) from the same simulator.

**What mechanisms or procedures were used to collect the data?**
Deterministic PyTorch simulation code, seeded (`corpus.gen_cell`'s per-family,
per-excitation seed scheme ensures the SAME parameter draws recur across
every (excitation, rate) cell of a family, making cross-excitation and
cross-rate comparisons instance-aligned, not resampled).

**Over what timeframe was the data collected?**
Generated on demand; this repository's reference corpus was generated
2026-07-13/14 with `python -m plantforge.corpus --instances 4000` (~14 hours
wall-clock, measured from shard mtimes, on a heavily-shared CPU-contended
machine; the generator is resumable, so an interrupted run continues where it
left off).

## Preprocessing / cleaning / labeling

**Was any preprocessing/cleaning/labeling of the data done?**
No — ground-truth (u, y) pairs carry no hidden normalization; this is a
tested design invariant (re-simulation from θ reproduces y to 1e-6,
`tests/test_plantforge.py:test_physical_consistency_of_pairs`).
Per-series normalization is applied only downstream, at evaluation time
(`evaluate._norm`), as an explicit, documented method choice — never baked
into the released data.

**Is the software used to preprocess/clean/label the instances available?**
Yes — this entire repository. Corpus generation (`corpus.py`), simulation
(`families.py`), excitation generation (`excitation.py`), and identifiability
computation (`identifiability.py`) are all released alongside the data.

## Uses

**Has the dataset been used for any tasks already?**
Yes, within this repository: (1) training/evaluating in-context
system-identification transformers across held-out family / rate / excitation
splits (`evaluate.py`); (2) zero-shot transfer to real measured plants
(`realbench.py`); (3) classical baseline comparison (`baselines.py`);
(4) testing whether identifiability annotations predict in-context
prediction difficulty (`ident_exp.py` — result: they largely do not; see
`docs/superpowers/results/2026-07-14-experiment-results.md`).

**What (other) tasks could the dataset be used for?**
Parameter-recovery / classical system identification benchmarking; transfer
learning across nonlinearity families; excitation-design studies (which
excitation classes best identify which parameters, using the rel-CRLB
annotations directly); sample-rate robustness studies for SysID methods
generally, not just in-context transformers.

**Are there tasks for which the dataset should not be used?**
The corpus is entirely synthetic and its family set (5 nonlinearity types
plus a Wiener-Hammerstein baseline family) does not cover all real-world
plant nonlinearities (e.g. no multi-input/multi-output plants, no
time-varying parameters within a trajectory). It should not be presented as
a substitute for real-plant validation — this repository's own zero-shot
real-plant results (`realbench.py`) show synthetic-only training transfers
imperfectly, and a classical ARX baseline outperforms both trained
transformers on several real-plant and synthetic cells (see results doc) —
the corpus supports studying that gap, not concluding it away.

## Distribution

**Will the dataset be distributed to third parties?**
Yes — published on Hugging Face Datasets:
https://huggingface.co/datasets/stark4062/plantforge (also linked from the
repository README).

**How will the dataset be distributed?**
As the same `.pt` shard files this repository's `corpus.py` generates
(PyTorch tensor dicts), plus this datasheet, a Hugging Face dataset card, and
machine-readable Croissant metadata (core + Responsible AI fields) at
[`croissant.json`](../croissant.json) in the repository root — validated
against the official `mlcroissant` Python library (v1.1.0), zero structural
errors or warnings. Regenerable from source at any instance count via
`python -m plantforge.corpus`.

**License:** CC BY 4.0 — see `LICENSE-DATA`. (The code in this repository is
separately licensed under MIT — see `LICENSE`.)

## Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
The maintainers of this repository. Issues/questions: via the code
repository's issue tracker (https://github.com/Soarr01/plantforge/issues) or
the Hugging Face dataset's community tab.

**Will the dataset be updated?**
The corpus generator is deterministic given a seed and instance count; the
released reference corpus (4000 instances/cell) is a static snapshot. Future
updates (e.g. additional families, additional real-plant validation) would
be released as a new versioned corpus, not a silent overwrite.
