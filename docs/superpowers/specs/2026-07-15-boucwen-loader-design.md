# Bouc-Wen real-plant loader

Status: approved design, not yet implemented.

## Purpose

The paper currently states Bouc-Wen is "out of scope: not exposed by
`nonlinear_benchmarks`'s public API" (confirmed: `dir(nonlinear_benchmarks)`
in the installed v1.0.1 has no Bouc-Wen entry, and the package's own source
has no mention of it). This closes that gap with a hand-rolled loader
against the benchmark's official data source, covering the 3rd of the 4
real-plant benchmarks named in the original corpus motivation.

## Data source (verified by direct download in this session)

- Official dataset: "Hysteretic Benchmark with a Dynamic Nonlinearity",
  Jean-Philippe Noël & Maarten Schoukens, 4TU.ResearchData, 2020.
  DOI: 10.4121/12967592. **License: CC BY-SA 4.0** (confirmed via the
  DataCite API — permits use/citation; we do not redistribute the data,
  only fetch-on-demand and cite, matching how `nonlinear_benchmarks` itself
  handles its datasets).
- Direct download URL (resolved from the DOI, stable ndownloader endpoint):
  `https://data.4tu.nl/ndownloader/items/7060f9bc-8289-411e-8d32-57bef2740d32/versions/1`
  — a ~5.6MB zip.
- Benchmark design described in: J.P. Noël, M. Schoukens, "Hysteretic
  benchmark with a dynamic nonlinearity," Workshop on Nonlinear System
  Identification Benchmarks, Brussels, 2016 (PDF included in the download).

**Archive structure** (verified by extraction): the outer zip contains
`BoucWenFiles.zip` (+ a duplicate PDF); `BoucWenFiles.zip` contains, among
other things (a MATLAB-only, license-protected `.p` simulator we do NOT
need), `Test signals/Validation signals/{uval,yval}_{multisine,sinesweep}.mat`
— plain MATLAB v5 `.mat` files, loadable directly via `scipy.io.loadmat`
(scipy is already a hard dependency; no new dependency needed). No training
data is distributed (generating it requires MATLAB + the protected `.p`
file) — irrelevant for PLANTFORGE, which only ever uses real datasets'
*test* splits for zero-shot evaluation, never their training splits.

**Signal facts** (from the benchmark's own spec PDF, cross-checked against
the actual `.mat` array shapes):
- Native sampling rate: 750 Hz exactly (`dt = 1/750 s`).
- `multisine`: 8192 samples, noiseless, steady-state, 5-150 Hz band.
- `sinesweep`: 153000 samples, noiseless, non-steady-state chirp, 20-50 Hz
  band swept at 10 Hz/min.
- Both verified to load via `scipy.io.loadmat` with shapes `(1, 8192)` /
  `(1, 153000)` respectively.

## Rate handling

`750 Hz / 15 = 50 Hz` exactly — decimating by q=15 lands on `dt = 0.02s`
**exactly** (no rounding error at all, the cleanest match of any real-plant
dataset so far; Silverbox's 12x decimation only approximates 50Hz). This
uses the existing `realbench.best_decimation_factor`/`decimate_to_factor`
unchanged — `best_decimation_factor(1/750, RATES)` should return `(15,
0.02)` deterministically, which becomes an offline-testable assertion (no
network needed to test the decimation math itself).

Window counts at q=15: `multisine` decimates to ~546 samples (2 windows of
D=224); `sinesweep` decimates to ~10200 samples (45 windows, far more than
`WINDOW_CAP=8`). Pooled across both records the same way Silverbox pools its
3 test records, capped at 8.

## Design

**Extend `realbench.py`** (the file that already owns Silverbox/Cascaded_Tanks/
WienerHammerBenchMark loading, decimation, and windowing):
- `BOUCWEN_URL`, `BOUCWEN_CACHE` (under `~/.nonlinear_benchmarks/BoucWen/`,
  matching the cache-directory convention the `nonlinear_benchmarks` package
  itself uses for its own datasets, even though this loader doesn't go
  through that package), `BOUCWEN_DT = 1/750` constants.
- `_boucwen_fetch()`: download+extract (stdlib `urllib.request` + `zipfile`
  only, no new dependency) into the cache dir if not already present;
  returns the directory containing the 4 `.mat` files.
- `boucwen_windows()`: mirrors `silverbox_windows()`'s shape exactly —
  decimates both test records, pools windows, returns
  `(windows_or_None, achieved_dt, decimation_factor)`.
- `main()`'s report gains a third real-plant block.

**Extend `baselines.py`**'s `real_report()`: add Bouc-Wen alongside
Silverbox/Cascaded_Tanks (same pattern, same `_norm` + ARX/NARX2 comparison).

**Extend `aggregate.py`**'s `main()`: add Bouc-Wen to the `realplant` dict
so the multi-seed mean±std report covers it too, same pattern as the other
two real datasets.

**After the code lands and produces real numbers** (a later, explicit step,
not part of the code tasks): update `paper/main.tex` --- abstract ("two" ->
"three" real plants), Table 2 (+1 row), Section 4.2 prose (+rate-handling
bullet), Section 4.3 prose (ARX comparison table/range, recomputed exactly
from real output, not approximated --- learn from the earlier "30-300x"
mistake), Section 5 Limitations ("2 of the 4" -> "3 of the 4", only
WienerHammerBenchMark still excluded), Related Work's "Real-measured
benchmarks" paragraph (Bouc-Wen moves from "not evaluated" to "evaluated"),
`references.bib` (Noël & Schoukens 2020 dataset + the 2016 workshop paper),
`figures/make_figures.py`'s `fig2_real_plant` (3rd dataset group),
`README.md`/`docs/DATASHEET.md` wherever they currently say Bouc-Wen is
excluded. This step only happens after the real numbers exist and are
reviewed --- no paper edits with placeholder/guessed numbers.

## Global constraints

- Reuse `realbench.decimate_to_factor`, `realbench.best_decimation_factor`,
  `realbench.make_windows`, `realbench.pooled_windows`, `realbench.RATES`
  (imported from `corpus`) --- no reimplementing decimation/windowing.
- Zero new pip dependencies (`urllib.request`, `zipfile` are stdlib;
  `scipy.io` is already available via the existing `scipy` dependency).
- Do not modify `families.py`, `excitation.py`, `identifiability.py`,
  `corpus.py`, `evaluate.py`, `ablation.py`, `ident_exp.py`.
- `realbench.py`'s existing functions (`silverbox_windows`,
  `cascaded_tanks_windows`, `wienerhammer_status`, `load_model`,
  `nmse_on_windows`, `main`'s existing two real-plant blocks) are extended,
  not rewritten --- their current behavior/output for Silverbox/Cascaded_Tanks/
  WienerHammerBenchMark must not change.
- Branch off `main`, same subagent-driven-development process (task briefs,
  task-scoped review, final whole-branch review) as prior plans in this repo.

## Out of scope

- Generating Bouc-Wen *training* data (would require MATLAB + a
  license-protected `.p` file we don't have and don't need for zero-shot eval).
- Any change to the trained model checkpoints --- this uses the existing,
  already-trained `eval_{wh_only,corpus}_s{seed}.pt` checkpoints, no retraining.
- Editing `paper/main.tex` with numbers before they exist and are reviewed
  (explicit two-phase split: code first, paper update after, as a separate
  reviewed step).
