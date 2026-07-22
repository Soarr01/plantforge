# Reference-Rate Fix (aggregate/ablation/leave_one_out) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the reference-rate bug (baseline measured at the held-out rate
`dt=0.05` instead of the trained rates `TRAIN_RATES=[0.10, 0.02]`) that was
already fixed in `evaluate.py`'s `report()` but is duplicated, unfixed, in
`aggregate.py`, `ablation.py`, and `leave_one_out.py`.

**Architecture:** Three independent, mechanically similar changes, one per
file. In each, replace a `nmse(..., 0.05)` call used as a "reference"/baseline
value with an average of `nmse(..., dt)` over `TRAIN_RATES`. Every other value
(held-out family/excitation/rate probes) is untouched. Each change gets
offline tests that monkeypatch the module's own imported `nmse` binding to a
deterministic stub (no GPU, no checkpoints, no real model inference).

**Tech Stack:** Python 3, PyTorch (only used for pre-existing model-creation
paths, not touched here). No new dependencies.

## Global Constraints

- Modify ONLY `aggregate.py`, `ablation.py`, `leave_one_out.py`, and their
  three test files (`tests/test_aggregate.py`, `tests/test_ablation.py`,
  `tests/test_leave_one_out.py`). Do NOT touch `evaluate.py`, `baselines.py`,
  `corpus.py`, or `ident_exp.py`.
- No `nmse(...)` VALUE changes anywhere — only which cell(s) get averaged to
  form each "reference" denominator. All held-out-condition probe values
  (`fams` in ablation.py, `held_out` in leave_one_out.py, every non-reference
  row in aggregate.py's `matrix()`) are computed exactly as before.
- `aggregate.py`'s `transfer_cells()` return structure/signature is unchanged
  — the averaging is added in `matrix()`, not `transfer_cells()`.
- Tests monkeypatch each module's own imported `nmse` name (e.g.
  `plantforge.aggregate.nmse`, not `plantforge.evaluate.nmse`) to a
  deterministic stub — no real model forward pass, no GPU, no checkpoints.
  Always restore the original in a `finally` block.
- New behavior gets offline tests, runnable via
  `python -m plantforge.tests.test_aggregate` /
  `python -m plantforge.tests.test_ablation` /
  `python -m plantforge.tests.test_leave_one_out` from
  `/data/nas07_new/PersonalData/phuocthien`.
- Branch off `main`.

---

### Task 1: Fix `aggregate.py`'s reference cell

**Files:**
- Modify: `aggregate.py` (`matrix()` function; add `TRAIN_RATES` import)
- Test: `tests/test_aggregate.py` (extend)

**Interfaces:**
- Consumes: `TRAIN_RATES` from `.evaluate` (already exists there:
  `TRAIN_RATES = [0.10, 0.02]`).
- Produces: `matrix()`'s external signature/behavior for every row except
  `"reference (train-like)"` under corpus mode is unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_aggregate.py` (add `import` line for `matrix` alongside
the existing `transfer_cells, aggregate_matrices, mean_std_str` import, and
add `TRAIN_RATES` to the `plantforge.evaluate` import line):

```python
from plantforge.aggregate import (
    transfer_cells, aggregate_matrices, mean_std_str, matrix,
)
from plantforge.evaluate import FAMILIES, HOLD_FAMILY, TRAIN_RATES
```

Add these two test functions:

```python
def test_matrix_averages_reference_over_train_rates_for_corpus_mode():
    """The bug this task fixes: corpus mode's 'reference (train-like)' cell
    used to be nmse(model, 'stribeck', 'multisine', 0.05) -- dt=0.05 is a
    held-out rate, not a trained one. matrix() must now average over
    TRAIN_RATES (0.10, 0.02) instead, mirroring the evaluate.report() fix."""
    import plantforge.aggregate as agg
    calls = []

    def fake_nmse(model, fam, exc, dt):
        calls.append((fam, exc, dt))
        return dt   # deterministic: returned value IS dt, so the mean is checkable

    original = agg.nmse
    agg.nmse = fake_nmse
    try:
        result = agg.matrix(None, "corpus")
    finally:
        agg.nmse = original

    assert abs(result["reference (train-like)"] - 0.06) < 1e-9, \
        f"expected mean of TRAIN_RATES (0.06), got {result['reference (train-like)']}"
    ref_calls = [c for c in calls if c[0] == "stribeck" and c[1] == "multisine"
                 and c[2] in TRAIN_RATES]
    assert len(ref_calls) == 2, \
        f"expected exactly 2 calls (one per TRAIN_RATES entry) for the reference cell, got {ref_calls}"
    assert all(c[2] != 0.05 for c in ref_calls), "reference calls must not use the held-out rate 0.05"
    print("  PASS  test_matrix_averages_reference_over_train_rates_for_corpus_mode")


def test_matrix_wh_only_reference_unchanged_at_dt05():
    """wh_only mode's reference (wh/multisine/dt=0.05) IS wh_only's only
    trained combination -- it was already correct and must stay a single
    dt=0.05 call, not an average."""
    import plantforge.aggregate as agg
    calls = []

    def fake_nmse(model, fam, exc, dt):
        calls.append((fam, exc, dt))
        return 1.0

    original = agg.nmse
    agg.nmse = fake_nmse
    try:
        result = agg.matrix(None, "wh_only")
    finally:
        agg.nmse = original

    assert result["reference (train-like)"] == 1.0
    assert calls[0] == ("wh", "multisine", 0.05)
    print("  PASS  test_matrix_wh_only_reference_unchanged_at_dt05")
```

Update `_run_all()` to call both new tests:

```python
def _run_all():
    test_transfer_cells_structure()
    test_aggregate_matrices_mean_std_n()
    test_aggregate_matrices_single_seed_no_nan_std()
    test_mean_std_str_format()
    test_matrix_averages_reference_over_train_rates_for_corpus_mode()
    test_matrix_wh_only_reference_unchanged_at_dt05()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_aggregate`
Expected: FAIL on `test_matrix_averages_reference_over_train_rates_for_corpus_mode`
— `matrix()`'s current code calls `nmse(model, "stribeck", "multisine", 0.05)`
for the reference cell (a single call at `dt=0.05`, not averaged), so
`result["reference (train-like)"]` will be `0.05` (the stub returns `dt`), not
`0.06`, and `ref_calls` will be empty (no call used `dt` in `TRAIN_RATES`).

- [ ] **Step 3: Implement the fix**

In `aggregate.py`, change the import line:

```python
from .evaluate import (FAMILIES, HOLD_FAMILY, TRAIN_RATES, CKPT_DIR, nmse)
```

Replace the `matrix()` function:

```python
def matrix(model, mode: str) -> dict:
    return {label: nmse(model, fam, exc, dt)
            for label, fam, exc, dt in transfer_cells(mode)}
```

with:

```python
def matrix(model, mode: str) -> dict:
    out = {}
    for label, fam, exc, dt in transfer_cells(mode):
        if label == "reference (train-like)" and mode != "wh_only":
            # dt=0.05 is a held-out rate (TRAIN_RATES = [0.10, 0.02]) --
            # average over the actually-trained rates instead, mirroring the
            # fix in evaluate.report() (2026-07-20 adversarial-review-fixes,
            # Finding 1). wh_only mode is unaffected: dt=0.05 IS wh_only's
            # only trained rate, so it was already a valid reference.
            out[label] = sum(nmse(model, fam, exc, r) for r in TRAIN_RATES) / len(TRAIN_RATES)
        else:
            out[label] = nmse(model, fam, exc, dt)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_aggregate`
Expected: all 6 tests PASS (4 pre-existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add aggregate.py tests/test_aggregate.py
git commit -m "Fix aggregate.py's reference cell to average over TRAIN_RATES"
```

---

### Task 2: Fix `ablation.py`'s reference computation

**Files:**
- Modify: `ablation.py` (`main()`'s `refs` computation; add `TRAIN_RATES` import)
- Test: `tests/test_ablation.py` (extend)

**Interfaces:**
- Consumes: `TRAIN_RATES` from `.evaluate`.
- Produces: no external interface change -- `main()`'s printed output values
  change (the reference number), but no function signature changes.

- [ ] **Step 1: Write the failing test**

Since `refs` is computed inline inside `main()` (not a separately importable
function), test the averaging arithmetic directly by monkeypatching
`plantforge.ablation.nmse` and calling `main()`, capturing stdout. Append to
`tests/test_ablation.py` (add `io`/`contextlib` are already imported; add
`TRAIN_RATES` to the evaluate import, and add an `os` import for `PF_SEEDS`):

```python
def test_main_reference_averages_over_train_rates():
    """The bug this task fixes: ablation.py's 'reference' used to be
    nmse(m, 'stribeck', 'multisine', 0.05) -- dt=0.05 is a held-out rate.
    main() must now average over TRAIN_RATES (0.10, 0.02) for the reference,
    which changes the printed 'reference:' value and the 'x ref' ratio."""
    import plantforge.ablation as abl
    from plantforge.evaluate import TRAIN_RATES

    calls = []

    def fake_nmse(model, fam, exc, dt):
        calls.append((fam, exc, dt))
        return dt  # deterministic: returned value IS dt

    def fake_finished_variant_models(name, width, layers, seeds):
        return [object()] if name == "default" else []  # one fake "model" for default only

    original_nmse = abl.nmse
    original_finished = abl._finished_variant_models
    abl.nmse = fake_nmse
    abl._finished_variant_models = fake_finished_variant_models
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            abl.main()
    finally:
        abl.nmse = original_nmse
        abl._finished_variant_models = original_finished

    ref_calls = [c for c in calls if c[0] == "stribeck" and c[1] == "multisine"
                 and c[2] in TRAIN_RATES]
    assert len(ref_calls) >= 2, f"expected reference calls at TRAIN_RATES, got {calls}"
    assert not any(c[0] == "stribeck" and c[2] == 0.05 for c in calls), \
        "reference must not be computed at the held-out rate 0.05"
    print("  PASS  test_main_reference_averages_over_train_rates")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_ablation`
Expected: FAIL on `test_main_reference_averages_over_train_rates` -- the
current `refs = [nmse(m, "stribeck", "multisine", 0.05) for m in models]`
calls the stub only at `dt=0.05`, so no call has `dt` in `TRAIN_RATES` and the
assertion `len(ref_calls) >= 2` fails.

- [ ] **Step 3: Implement the fix**

In `ablation.py`, change the import line:

```python
from .evaluate import InContextSysID, nmse, HOLD_FAMILY, TRAIN_RATES, CKPT_DIR, DEV
```

Replace this line in `main()`:

```python
        refs = [nmse(m, "stribeck", "multisine", 0.05) for m in models]
```

with:

```python
        refs = [sum(nmse(m, "stribeck", "multisine", dt) for dt in TRAIN_RATES) / len(TRAIN_RATES)
                for m in models]
```

Leave the `fams = [nmse(m, HOLD_FAMILY, "multisine", 0.05) for m in models]`
line exactly as-is (it deliberately probes the held-out family at the
held-out rate).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_ablation`
Expected: all 8 tests PASS (7 pre-existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add ablation.py tests/test_ablation.py
git commit -m "Fix ablation.py's reference to average over TRAIN_RATES"
```

---

### Task 3: Fix `leave_one_out.py`'s `reference_and_heldout`

**Files:**
- Modify: `leave_one_out.py` (`reference_and_heldout()`; add `TRAIN_RATES` import)
- Test: `tests/test_leave_one_out.py` (extend)

**Interfaces:**
- Consumes: `TRAIN_RATES` from `.evaluate`.
- Produces: `reference_and_heldout(model, held_family) -> (reference, held_out)`
  -- same signature and return shape, `reference`'s value now averaged over
  4 families × `TRAIN_RATES`, `held_out` unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_leave_one_out.py` (add `reference_and_heldout` to the
existing import from `plantforge.leave_one_out`, and add `TRAIN_RATES` to the
`plantforge.evaluate` import):

```python
def test_reference_and_heldout_averages_reference_over_train_rates():
    """The bug this task fixes: reference_and_heldout()'s reference used to
    be the mean of nmse(model, fam, 'multisine', 0.05) over the 4 non-held
    families -- dt=0.05 is a held-out rate. reference must now average over
    the 4 families x TRAIN_RATES (8 values); held_out must stay a single
    dt=0.05 call (it deliberately probes the held-out family at the held-out
    rate, matching how held-out family generalization is reported
    elsewhere)."""
    import plantforge.leave_one_out as loo
    from plantforge.evaluate import FAMILIES, TRAIN_RATES

    calls = []

    def fake_nmse(model, fam, exc, dt):
        calls.append((fam, exc, dt))
        return dt  # deterministic: returned value IS dt

    original = loo.nmse
    loo.nmse = fake_nmse
    try:
        reference, held_out = loo.reference_and_heldout(None, "backlash")
    finally:
        loo.nmse = original

    others = [f for f in FAMILIES if f != "backlash"]
    expected_ref_calls = len(others) * len(TRAIN_RATES)
    ref_calls = [c for c in calls if c[0] in others and c[2] in TRAIN_RATES]
    assert len(ref_calls) == expected_ref_calls, \
        f"expected {expected_ref_calls} reference calls (families x TRAIN_RATES), got {len(ref_calls)}"
    assert not any(c[0] in others and c[2] == 0.05 for c in calls), \
        "reference must not be computed at the held-out rate 0.05"
    assert held_out == 0.05, \
        f"held_out must still be the single dt=0.05 call (stub returns dt), got {held_out}"
    held_out_calls = [c for c in calls if c[0] == "backlash"]
    assert held_out_calls == [("backlash", "multisine", 0.05)], \
        f"held_out must be exactly one call at dt=0.05, got {held_out_calls}"
    print("  PASS  test_reference_and_heldout_averages_reference_over_train_rates")
```

Update `_run_all()` (find it near the bottom of the file) to also call
`test_reference_and_heldout_averages_reference_over_train_rates()`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_leave_one_out`
Expected: FAIL -- the current `reference_and_heldout()` calls `nmse` only at
`dt=0.05` for every family (including the reference families), so
`ref_calls` will be empty (`len(ref_calls) == 0 != expected_ref_calls`).

- [ ] **Step 3: Implement the fix**

In `leave_one_out.py`, change the import line:

```python
from .evaluate import InContextSysID, nmse, FAMILIES, TRAIN_RATES, CKPT_DIR, DEV
```

Replace `reference_and_heldout()`:

```python
def reference_and_heldout(model, held_family: str):
    """mean nMSE over the 4 trained (non-held) families, and nMSE on the
    held-out family itself, both at dt=0.05, multisine."""
    others = [f for f in FAMILIES if f != held_family]
    ref_vals = [nmse(model, fam, "multisine", 0.05) for fam in others]
    reference = sum(ref_vals) / len(ref_vals)
    held_out = nmse(model, held_family, "multisine", 0.05)
    return reference, held_out
```

with:

```python
def reference_and_heldout(model, held_family: str):
    """mean nMSE over the 4 trained (non-held) families, averaged over the
    actually-trained rates (TRAIN_RATES) so the reference is genuinely
    in-distribution; and nMSE on the held-out family itself at dt=0.05 (the
    held-out rate), matching how held-out family generalization is reported
    elsewhere (evaluate.report())."""
    others = [f for f in FAMILIES if f != held_family]
    ref_vals = [nmse(model, fam, "multisine", dt) for fam in others for dt in TRAIN_RATES]
    reference = sum(ref_vals) / len(ref_vals)
    held_out = nmse(model, held_family, "multisine", 0.05)
    return reference, held_out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_leave_one_out`
Expected: all 8 tests PASS (7 pre-existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add leave_one_out.py tests/test_leave_one_out.py
git commit -m "Fix leave_one_out.py's reference to average over TRAIN_RATES"
```

---

## After all tasks: controller-run result regeneration (not a task step)

Per this repo's established convention, regenerating and publishing numbers
is a controller step after the code changes are reviewed, merged, and the
full offline suite passes -- NOT part of the tasks' TDD steps:

1. After merge to `main` and push, run `aggregate.py`, `ablation.py`,
   `leave_one_out.py`, `baselines.py`, and `ident_exp.py` against the
   2026-07-23 retrained (50/50 finite) checkpoints and save fresh output to
   new dated files under `docs/superpowers/results/`.
2. Fold the corrected numbers into `paper/main.tex` / `README.md` /
   `docs/DATASHEET.md`, together with the still-pending text-only fixes from
   the original adversarial-review-fixes work (Findings 2, 5, 8: white-noise
   wording, "halves" arithmetic claim, capacity-invariance claim softening;
   Finding 4 was already addressed as a claim-only fix in that plan).
3. Re-verify the arXiv submission bundle (`paper/arxiv_submission/`,
   `paper/plantforge_arxiv_submission.tar.gz`) once the paper's numbers are
   finalized -- these were flagged as present-but-untracked by an earlier
   whole-branch review and reflect pre-fix numbers.
