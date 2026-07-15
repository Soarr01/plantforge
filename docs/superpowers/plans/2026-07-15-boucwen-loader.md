# Bouc-Wen loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add zero-shot Bouc-Wen evaluation, closing the paper's stated gap
("2 of the 4 real benchmarks" -> "3 of the 4"), reusing the existing
decimation/windowing infrastructure and the already-trained checkpoints.

**Architecture:** Extend `realbench.py` with a hand-rolled fetch+cache
loader (stdlib only) plus a `boucwen_windows()` function shaped identically
to the existing `silverbox_windows()`; wire it into `baselines.py` and
`aggregate.py`'s existing real-plant reporting blocks; run everything and
record results; update the paper as an explicit final step once real
numbers exist.

**Tech Stack:** Python, `urllib.request`+`zipfile` (stdlib), `scipy.io`
(already a dependency). Zero new pip dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-15-boucwen-loader-design.md` — binding.
- Reuse `realbench.decimate_to_factor`, `realbench.best_decimation_factor`,
  `realbench.make_windows`, `realbench.pooled_windows`, `corpus.RATES` — no
  reimplementing decimation/windowing.
- Zero new pip dependencies.
- Do not modify `families.py`, `excitation.py`, `identifiability.py`,
  `corpus.py`, `evaluate.py`, `ablation.py`, `ident_exp.py`.
- `realbench.py`'s existing Silverbox/Cascaded_Tanks/WienerHammerBenchMark
  functions and behavior must not change.
- Data source: `https://data.4tu.nl/ndownloader/items/7060f9bc-8289-411e-8d32-57bef2740d32/versions/1`
  (verified working, ~5.6MB zip, CC BY-SA 4.0). Native rate `1/750` s.
  `best_decimation_factor(1/750, RATES)` must equal exactly `(15, 0.02)`.
- Git repo root `/data/nas07_new/PersonalData/phuocthien/plantforge`, branch
  `boucwen-loader` off `main`. `git` commands run with cwd at the repo root;
  `python -m plantforge...` commands run with cwd
  `/data/nas07_new/PersonalData/phuocthien`.
- **Do not edit `paper/main.tex`, `paper/references.bib`,
  `figures/make_figures.py`, `README.md`, or `docs/DATASHEET.md` in any task
  of this plan.** Those are explicit follow-up work after real numbers exist
  (see the plan's final controller note).

---

### Task 1: Bouc-Wen fetch/cache/decimate loader in `realbench.py`

**Files:**
- Modify: `plantforge/realbench.py` (append)

**Interfaces:**
- Consumes: `decimate_to_factor`, `best_decimation_factor`, `pooled_windows`,
  `RATES` (all already in this file/its imports).
- Produces: `BOUCWEN_URL`, `BOUCWEN_CACHE`, `BOUCWEN_DT` constants;
  `_boucwen_fetch() -> pathlib.Path`; `boucwen_windows() -> tuple[tuple[torch.Tensor, torch.Tensor] | None, float, int]`
  (same shape as `silverbox_windows()`'s return).

This task's network-touching code is verified manually (Step 2), not by an
automated test — same pattern as `silverbox_windows`/`cascaded_tanks_windows`
in this file.

- [ ] **Step 1: Append to `plantforge/realbench.py`**

First, add `pathlib` to the existing import block. Change:

```python
import numpy as np
import scipy.signal
import torch
```

to:

```python
import pathlib

import numpy as np
import scipy.signal
import torch
```

Then append these constants near the top of the file, directly below the
existing `WINDOW_CAP = 8` line:

```python
BOUCWEN_URL = "https://data.4tu.nl/ndownloader/items/7060f9bc-8289-411e-8d32-57bef2740d32/versions/1"
BOUCWEN_CACHE = pathlib.Path.home() / ".nonlinear_benchmarks" / "BoucWen"
BOUCWEN_DT = 1.0 / 750.0   # native sampling rate, 750 Hz exactly (benchmark spec PDF)
```

Then append these two functions at the end of the file (after the existing
`wienerhammer_status` function, before `_report_dataset`/`main`):

```python
def _boucwen_fetch() -> pathlib.Path:
    """Download+cache the official Bouc-Wen test signals (CC BY-SA 4.0,
    Noel & Schoukens 2020, data.4tu.nl DOI 10.4121/12967592) if not already
    cached. Returns the directory containing the four .mat test-signal
    files. Uses only the standard library (urllib, zipfile) -- no new
    dependency for a one-off fetch of a single small dataset."""
    inner = BOUCWEN_CACHE / "BoucWenFiles" / "Test signals" / "Validation signals"
    if (inner / "uval_multisine.mat").exists():
        return inner
    import io
    import urllib.request
    import zipfile
    print("dataset not found, downloading Bouc-Wen from data.4tu.nl ...")
    BOUCWEN_CACHE.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(BOUCWEN_URL) as resp:
        outer_bytes = resp.read()
    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer_zip:
        outer_zip.extract("BoucWenFiles.zip", BOUCWEN_CACHE)
    inner_zip_path = BOUCWEN_CACHE / "BoucWenFiles.zip"
    with zipfile.ZipFile(inner_zip_path) as inner_zip:
        inner_zip.extractall(BOUCWEN_CACHE / "BoucWenFiles")
    return inner


def boucwen_windows():
    """Load Bouc-Wen's multisine + sinesweep test records (noiseless,
    750 Hz native), decimate to the nearest trained rate, and pool windows
    across both. Returns (windows_or_None, achieved_dt, decimation_factor)."""
    import scipy.io as sio
    d = _boucwen_fetch()
    q, achieved_dt = best_decimation_factor(BOUCWEN_DT, RATES)
    records = []
    for name in ("multisine", "sinesweep"):
        u = sio.loadmat(d / f"uval_{name}.mat")[f"uval_{name}"].ravel().astype(np.float64)
        y = sio.loadmat(d / f"yval_{name}.mat")[f"yval_{name}"].ravel().astype(np.float64)
        u = decimate_to_factor(u, q)
        y = decimate_to_factor(y, q)
        records.append((u, y))
    return pooled_windows(records), achieved_dt, q
```

- [ ] **Step 2: Manually verify against the live network**

Run:
```bash
cd /data/nas07_new/PersonalData/phuocthien && python -c "
from plantforge.realbench import boucwen_windows
w, dt, q = boucwen_windows()
print('boucwen', None if w is None else w[0].shape, dt, q)
"
```
Expected: `boucwen (8, 224) 0.02 15` (exactly `q=15`, `dt=0.02` with no
rounding error — this is the load-bearing check for this task). Run it a
second time to confirm the cache path is used (should print no "downloading"
message and return the same shape near-instantly):
```bash
cd /data/nas07_new/PersonalData/phuocthien && time python -c "
from plantforge.realbench import boucwen_windows
boucwen_windows()
print('cached run OK')
"
```

- [ ] **Step 3: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add realbench.py
git commit -m "Add Bouc-Wen fetch/cache/decimate loader to realbench.py"
```

---

### Task 2: Offline test for the decimation math + wire into `realbench.main()`

**Files:**
- Modify: `plantforge/realbench.py` (`main()` only)
- Modify: `plantforge/tests/test_realbench.py`

**Interfaces:**
- Consumes: `boucwen_windows`, `_report_dataset` (existing helper).
- Produces: no new public interfaces — extends `main()`'s printed report.

- [ ] **Step 1: Write the failing test**

Add to `plantforge/tests/test_realbench.py` (add to the existing import
line from `plantforge.realbench` and add one new test function + its call
in `_run_all()`):

```python
def test_best_decimation_factor_matches_boucwen():
    native_dt = 1.0 / 750.0
    result = best_decimation_factor(native_dt, RATES)
    assert result is not None
    q, achieved_dt = result
    assert q == 15, q
    assert abs(achieved_dt - 0.02) < 1e-9, achieved_dt   # exact match, no rounding error
    print("  PASS  test_best_decimation_factor_matches_boucwen")
```

Add the call `test_best_decimation_factor_matches_boucwen()` inside
`_run_all()`, alongside the other `test_best_decimation_factor_*` calls.
(`best_decimation_factor` and `RATES` are already imported at the top of
this test file from earlier tasks — no import changes needed for this step.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_realbench`
Expected: `AssertionError` (or similar) — the test isn't in `_run_all()`'s
call list yet until this edit lands, so verify by temporarily confirming
the assertion logic is sound: `q=15` for `native_dt=1/750` should already
hold given `best_decimation_factor`'s existing implementation (unchanged by
this plan) — this step is confirming the test is correctly written and
would fail if the constant were wrong, not that the underlying function is
broken. If it unexpectedly fails, stop and report — that would mean
`best_decimation_factor`'s existing logic doesn't handle this cleanly, which
would be a real finding.

- [ ] **Step 3: Edit `plantforge/realbench.py`'s `main()`**

Find the existing block in `main()` that reports Cascaded_Tanks (calls
`_report_dataset` for `ct_windows`), and insert a new Bouc-Wen block
directly after it, before the `wienerhammer_status()` block:

```python
    bw_windows, bw_dt, bw_q = boucwen_windows()
    _report_dataset(
        f"  Bouc-Wen (decimated {bw_q}x -> dt={bw_dt:.4f}s, exact 50Hz match):",
        bw_windows, "no full window available after decimation", models)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_realbench`
Expected: 13 PASS lines (12 existing + 1 new), exit 0.

- [ ] **Step 5: Manually verify the full report (network + checkpoints)**

```bash
cd /data/nas07_new/PersonalData/phuocthien && \
PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data python -m plantforge.realbench
```
Expected: the existing Silverbox/Cascaded_Tanks/WienerHammerBenchMark
sections print as before, PLUS a new Bouc-Wen section with real nMSE
numbers for both `wh_only` and `corpus` models (or a SKIPPED line if
windows is somehow `None`, which given Step 2's manual check succeeding is
not expected). No traceback.

- [ ] **Step 6: Run the full offline suite for import safety**

```bash
cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.run_all
```
Expected: 35 PASS lines total (34 prior + 1 new).

- [ ] **Step 7: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add realbench.py tests/test_realbench.py
git commit -m "Wire Bouc-Wen into realbench.main()'s report; add offline decimation-math test"
```

---

### Task 3: Wire Bouc-Wen into `baselines.py` and `aggregate.py`

**Files:**
- Modify: `plantforge/baselines.py` (`real_report` only)
- Modify: `plantforge/aggregate.py` (`main` only)

**Interfaces:**
- Consumes: `realbench.boucwen_windows`.
- Produces: no new public interfaces — extends both files' printed reports.

- [ ] **Step 1: Edit `plantforge/baselines.py`'s `real_report()`**

Replace:

```python
def real_report():
    from .realbench import silverbox_windows, cascaded_tanks_windows
    print("=== classical baselines on real-plant windows (nMSE) ===")
    sb, sb_dt, sb_q = silverbox_windows()
    ct, ct_dt = cascaded_tanks_windows()
    for name, windows in (("Silverbox", sb), ("Cascaded_Tanks", ct)):
        if windows is None:
            print(f"  {name}: SKIPPED (no windows)")
            continue
        u_n, y_n = _norm(*windows)                    # raw -> same norm as model eval
        arx = baseline_nmse_batch(u_n, y_n, poly=False)
        narx = baseline_nmse_batch(u_n, y_n, poly=True)
        print(f"  {name}: ARX {arx:.4f} | NARX2 {narx:.4f}")
```

with:

```python
def real_report():
    from .realbench import silverbox_windows, cascaded_tanks_windows, boucwen_windows
    print("=== classical baselines on real-plant windows (nMSE) ===")
    sb, sb_dt, sb_q = silverbox_windows()
    ct, ct_dt = cascaded_tanks_windows()
    bw, bw_dt, bw_q = boucwen_windows()
    for name, windows in (("Silverbox", sb), ("Cascaded_Tanks", ct), ("Bouc-Wen", bw)):
        if windows is None:
            print(f"  {name}: SKIPPED (no windows)")
            continue
        u_n, y_n = _norm(*windows)                    # raw -> same norm as model eval
        arx = baseline_nmse_batch(u_n, y_n, poly=False)
        narx = baseline_nmse_batch(u_n, y_n, poly=True)
        print(f"  {name}: ARX {arx:.4f} | NARX2 {narx:.4f}")
```

- [ ] **Step 2: Edit `plantforge/aggregate.py`'s `main()`**

Replace:

```python
    try:
        from .realbench import silverbox_windows, cascaded_tanks_windows
        sb, sb_dt, sb_q = silverbox_windows()
        ct, ct_dt = cascaded_tanks_windows()
        realplant = {f"Silverbox (decimated {sb_q}x -> dt={sb_dt:.4f}s)": sb,
                     f"Cascaded_Tanks (native dt={ct_dt:.2f}s, extrapolation)": ct}
    except Exception as e:
```

with:

```python
    try:
        from .realbench import silverbox_windows, cascaded_tanks_windows, boucwen_windows
        sb, sb_dt, sb_q = silverbox_windows()
        ct, ct_dt = cascaded_tanks_windows()
        bw, bw_dt, bw_q = boucwen_windows()
        realplant = {f"Silverbox (decimated {sb_q}x -> dt={sb_dt:.4f}s)": sb,
                     f"Cascaded_Tanks (native dt={ct_dt:.2f}s, extrapolation)": ct,
                     f"Bouc-Wen (decimated {bw_q}x -> dt={bw_dt:.4f}s)": bw}
    except Exception as e:
```

- [ ] **Step 3: Manually verify both CLIs (network + checkpoints)**

```bash
cd /data/nas07_new/PersonalData/phuocthien && \
PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data python -m plantforge.baselines real 2>&1 | tail -6
cd /data/nas07_new/PersonalData/phuocthien && \
PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data PF_SEEDS=0 python -m plantforge.aggregate 2>&1 | grep -A1 "Bouc-Wen"
```
Expected: `baselines real`'s tail shows a `Bouc-Wen: ARX ... | NARX2 ...`
line; `aggregate`'s output includes a `real-plant Bouc-Wen (...)` line with
a mean±std (n=1 since only `PF_SEEDS=0` was requested here). No traceback.

- [ ] **Step 4: Run the full offline suite for import safety**

```bash
cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.run_all
```
Expected: still 35 PASS lines (this task adds no new tests, only extends
two `main()`-style functions that have no dedicated offline tests of their
own, matching the existing pattern for `real_report`/`aggregate.main`).

- [ ] **Step 5: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add baselines.py aggregate.py
git commit -m "Wire Bouc-Wen into baselines.real_report and aggregate.main"
```

---

**Execution note for the controller:** after all 3 tasks' reviews pass and
the final whole-branch review approves, merge, then run the full-scale
reports (`python -m plantforge.aggregate` with default `PF_SEEDS`,
`python -m plantforge.baselines real`, `python -m plantforge.realbench`) to
get final Bouc-Wen numbers across all 5 trained seeds, record them in a
results note, THEN (as a separate, later piece of work — not part of this
plan) update `paper/main.tex`, `references.bib`, `figures/make_figures.py`,
`README.md`, and `docs/DATASHEET.md` with the real numbers.

---

## Self-Review Notes

- **Spec coverage:** Task 1 = loader; Task 2 = offline test + realbench.main
  wiring; Task 3 = baselines/aggregate wiring. Spec's "no paper edits in
  this plan" constraint reflected in the Global Constraints and the
  controller execution note. Spec's exact decimation math (q=15, dt=0.02
  exactly) is directly asserted in Task 2's test.
- **Placeholder scan:** none — every step has complete, concrete code.
- **Type consistency:** `boucwen_windows()` returns the same
  `(windows_or_None, achieved_dt, decimation_factor)` 3-tuple shape as
  `silverbox_windows()`, consumed identically by `realbench.main()`,
  `baselines.real_report()`, and `aggregate.main()` — verified consistent
  across all three call sites in this plan.
