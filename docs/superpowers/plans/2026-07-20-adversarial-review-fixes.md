# Adversarial Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix six independently-verified adversarial-review findings against
this repo's evaluation/corpus-generation code: a normalization leak that lets
the model see query-horizon scale, a mislabeled "in-distribution reference"
cell that's actually a held-out rate, non-deterministic corpus-generation
seeds, an undocumented gap between the datasheet's divergence-filtering claim
and the actual code, and an identifiability-annotation/prediction-window
length mismatch.

**Architecture:** Three independent code-change groups. Group A
(`evaluate._norm`) and Group B (`evaluate.report`, `corpus.gen_cell`'s seeding
and divergence filter) both touch functions whose outputs feed model training
and are combined into one retraining campaign after merge (a controller step,
not part of this plan's tasks). Group C (`ident_exp.py`'s per-cell
identifiability recomputation) is independent of model checkpoints and
requires no retraining.

**Tech Stack:** Python 3, PyTorch, NumPy, SciPy (`spearmanr`). No new
dependencies.

## Global Constraints

- Do not modify `families.py`'s physical simulation logic (`_core2`, `_zoh`,
  `Stepper.step`) or `excitation.py`'s signal-generation logic — this plan
  fixes measurement/protocol bugs, not physics.
- `_norm`'s signature stays `(u, y) -> (u, y)` — only its numeric output
  changes, not its interface. Every caller (`make_batch` in `evaluate.py`,
  `realbench.nmse_on_windows`, `baselines.real_report`, `ident_exp.qry_stats`
  and `ident_exp.main`) imports it directly from `evaluate` and needs zero
  code changes — verify this by grep, don't just assume it.
- `corpus.gen_cell`'s return shape (the `shard` dict's keys: `u`, `y`,
  `theta`, `keys`, `dt`, `family`, `excitation`, optionally `rel_crlb`/
  `log10_cond`) must stay byte-identical — only the internal seeding scheme
  and the addition of a finite-value filter change.
- `evaluate.report()`'s printed output format (line structure, `x ref` label
  text) stays the same — only the reference value computation changes for
  `corpus` mode. The `wh_only` mode's reference computation is UNCHANGED
  (`nmse(model, "wh", "multisine", 0.05)` is already a valid in-distribution
  cell for that mode — do not touch it).
- Every new/changed function gets an offline test (no GPU, no network,
  runnable via `python -m plantforge.tests.test_X`), following this repo's
  established test-module pattern.
- Do not touch `identifiability.py`'s FIM computation itself (`SIGMA_REF`,
  `REL_STEP`, the finite-difference method) — Group C changes what window of
  data is passed into it, not how it computes.
- No GPU training is launched as part of this plan's tasks — retraining is a
  controller-run step after merge, per this repo's established convention for
  any plan that changes a training-affecting invariant.
- Branch off `main`.

---

### Task 1: Fix the normalization leak in `evaluate._norm`

**Files:**
- Modify: `evaluate.py` (the `_norm` function, lines 56-58)
- Test: `tests/test_evaluate.py` (new file)
- Modify: `tests/run_all.py` is NOT touched by this task — new test-module
  wiring happens only at the final whole-branch review stage, per this
  repo's established convention (see `.superpowers/sdd/progress.md`'s
  history of prior plans making this exact choice).

**Interfaces:**
- Consumes: `evaluate.T_CTX` (already defined, `= 192`), `torch` (already
  imported in `evaluate.py`).
- Produces: `_norm(u, y) -> (u, y)` — SAME signature and SAME shapes as
  before ((T, B) or (B, T) — whichever `u`/`y` layout the caller passes in
  is preserved unchanged; `_norm` never transposes). Only the numeric
  values returned change. All 4 existing call sites (`evaluate.make_batch`,
  `realbench.nmse_on_windows`, `baselines.real_report`,
  `ident_exp.qry_stats`/`ident_exp.main`) need zero code changes — they
  already just call `_norm(u, y)` and use the results.

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluate.py`:

```python
"""Offline tests for plantforge.evaluate's pure helpers -- no GPU training,
no checkpoints, no network.

    python -m plantforge.tests.test_evaluate     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from plantforge.evaluate import _norm, T_CTX


def test_norm_uses_context_only_std():
    """_norm's scale factor must come from the first T_CTX samples only --
    changing the query-horizon (last T_QRY) samples must not change the
    normalization applied to the context portion."""
    torch.manual_seed(0)
    u = torch.randn(4, 224)
    y = torch.randn(4, 224)
    u1, y1 = _norm(u.clone(), y.clone())

    u_altered_query = u.clone()
    y_altered_query = y.clone()
    u_altered_query[:, T_CTX:] *= 1000.0
    y_altered_query[:, T_CTX:] *= 1000.0
    u2, y2 = _norm(u_altered_query, y_altered_query)

    # Context portion (same raw values in both calls) must normalize
    # identically -- only reachable if the scale factor ignores the query.
    assert torch.allclose(u1[:, :T_CTX], u2[:, :T_CTX], atol=1e-5), \
        "context normalization changed when only query values changed -- leak"
    assert torch.allclose(y1[:, :T_CTX], y2[:, :T_CTX], atol=1e-5), \
        "context normalization changed when only query values changed -- leak"
    print("  PASS  test_norm_uses_context_only_std")


def test_norm_context_std_is_approximately_one():
    """Sanity check: normalizing by the context's own std should leave the
    context portion with std close to 1 (not the full-sequence std, which
    would differ if the query portion has different scale)."""
    torch.manual_seed(1)
    u = torch.randn(8, 192) * 3.0
    u_query = torch.randn(8, 32) * 0.01   # very different scale from context
    u_full = torch.cat([u, u_query], dim=1)
    y_full = torch.randn(8, 224)

    u_n, _ = _norm(u_full, y_full)
    ctx_std = u_n[:, :T_CTX].std(dim=1)
    assert torch.allclose(ctx_std, torch.ones(8), atol=0.15), \
        f"context std after norm should be ~1, got {ctx_std}"
    print("  PASS  test_norm_context_std_is_approximately_one")


def _run_all():
    test_norm_uses_context_only_std()
    test_norm_context_std_is_approximately_one()


if __name__ == "__main__":
    print("PLANTFORGE evaluate -- offline tests:")
    _run_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_evaluate`
Expected: FAIL on `test_norm_uses_context_only_std` — the current `_norm`
computes `std` over the full 224-sample sequence, so altering the query
portion changes the scale factor applied to the context portion too.

- [ ] **Step 3: Fix `_norm` in `evaluate.py`**

Replace the current `_norm` (evaluate.py:56-58):

```python
def _norm(u, y):
    """Per-series normalization — a METHOD choice, applied identically everywhere."""
    return u / (u.std(1, keepdim=True) + 1e-6), y / (y.std(1, keepdim=True) + 1e-6)
```

with:

```python
def _norm(u, y):
    """Per-series normalization using CONTEXT-ONLY statistics (first T_CTX
    samples) -- avoids leaking query-horizon scale into the model's input.
    The prior version computed std over the full sequence (context + query),
    which let the model's context tokens carry information about the
    query-horizon's own magnitude before that portion is ever revealed."""
    u_ctx_std = u[:, :T_CTX].std(1, keepdim=True) + 1e-6
    y_ctx_std = y[:, :T_CTX].std(1, keepdim=True) + 1e-6
    return u / u_ctx_std, y / y_ctx_std
```

Note: `u`/`y` are always `(B, D)` at every call site (`make_batch`
transposes to `(B, D)` before calling `_norm`; `realbench`/`baselines`/
`ident_exp` all follow the same `(B, D)` convention) — so `u[:, :T_CTX]`
correctly slices the context portion along the time axis in every case.
Verify this assumption by re-reading each call site before implementing,
not just trusting this note.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_evaluate`
Expected: both tests PASS.

- [ ] **Step 5: Run the existing offline suite to confirm no regressions**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.run_all`
Expected: all existing tests still PASS (this task does not wire
`test_evaluate` into `run_all.py` yet, per Global Constraints — this run
just confirms nothing else broke).

- [ ] **Step 6: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "Fix normalization leak: _norm now uses context-only std, not full-sequence"
```

---

### Task 2: Fix the reference-rate mislabeling in `evaluate.report`

**Files:**
- Modify: `evaluate.py` (the `report` function, lines 166-181)
- Test: `tests/test_evaluate.py` (extend, from Task 1)

**Interfaces:**
- Consumes: `evaluate.nmse` (unchanged), `evaluate.TRAIN_RATES` (already
  defined, `= [0.10, 0.02]`), `evaluate.HOLD_FAMILY` (already defined).
- Produces: `report(model, mode)` — same signature, same printed-line
  structure (this task changes what number is computed as `ref` for
  `mode="corpus"`, not the print statements' format).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evaluate.py`. This test does not require a trained
model — it verifies the *reference cell selection logic* in isolation by
checking that `TRAIN_RATES` (the rates the corpus recipe actually trains
on) does not include `0.05` (the rate the old code used as "reference"),
which is the root fact the bug hinges on:

```python
from plantforge.evaluate import TRAIN_RATES


def test_train_rates_excludes_the_old_reference_rate():
    """Documents the bug this task fixes: 0.05 (the rate report() used to
    call 'reference (train-like)' for corpus mode) is NOT a trained rate --
    it's the held-out rate. TRAIN_RATES must be {0.10, 0.02} for the
    report() fix (using TRAIN_RATES as the reference rates) to be correct."""
    assert 0.05 not in TRAIN_RATES, \
        "0.05 must stay the held-out rate for this fix to be meaningful"
    assert set(TRAIN_RATES) == {0.10, 0.02}
    print("  PASS  test_train_rates_excludes_the_old_reference_rate")
```

Add the new test's call to `_run_all()`:

```python
def _run_all():
    test_norm_uses_context_only_std()
    test_norm_context_std_is_approximately_one()
    test_train_rates_excludes_the_old_reference_rate()
```

This test passes trivially against the CURRENT code (it documents a fact
about `TRAIN_RATES`, not `report()`'s behavior) — it exists to make the
bug's root cause explicit and regression-proof, not to fail-then-pass.
The actual behavior change is verified in Step 3 by direct inspection
since `report()` requires a trained model checkpoint and GPU to exercise
end-to-end; this task fixes the source computation, and Step 3's manual
trace is the verification a full integration test would otherwise provide.

- [ ] **Step 2: Run test to verify it passes as expected**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_evaluate`
Expected: all 3 tests PASS (this new test passes immediately since it's
checking a pre-existing fact, not the bug itself).

- [ ] **Step 3: Fix `report()` in `evaluate.py`**

Replace the current `report()` (evaluate.py:166-181):

```python
def report(model, mode):
    print(f"\n=== {mode}: transfer matrix (in-context nMSE) ===")
    ref = nmse(model, "wh" if mode == "wh_only" else "stribeck", "multisine", 0.05)
    print(f"  reference (train-like): {ref:.4f}")
    ...
```

with:

```python
def report(model, mode):
    print(f"\n=== {mode}: transfer matrix (in-context nMSE) ===")
    if mode == "wh_only":
        # wh/multisine/dt=0.05 IS wh_only's only trained combination --
        # already a valid in-distribution reference, unchanged.
        ref = nmse(model, "wh", "multisine", 0.05)
    else:
        # stribeck/multisine/dt=0.05 is NOT in-distribution for corpus mode
        # (dt=0.05 is the held-out rate -- TRAIN_RATES = [0.10, 0.02]).
        # The correct in-distribution reference averages stribeck's nMSE
        # over the two rates corpus mode actually trains on.
        ref = sum(nmse(model, "stribeck", "multisine", dt) for dt in TRAIN_RATES) / len(TRAIN_RATES)
    print(f"  reference (train-like): {ref:.4f}")
    ...
```

Leave every line below `print(f"  reference (train-like): {ref:.4f}")`
unchanged — this task only changes how `ref` is computed for the `else`
(corpus-mode) branch.

- [ ] **Step 4: Manually trace the fix against a real checkpoint**

Run (requires an existing `eval_corpus_s0.pt` checkpoint and
`PLANTFORGE_DATA` set — this repo already has these from prior plans):

```bash
cd /data/nas07_new/PersonalData/phuocthien && PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data python3 -c "
import torch
from plantforge.evaluate import InContextSysID, nmse, CKPT_DIR, DEV, TRAIN_RATES

ck = torch.load(CKPT_DIR / 'eval_corpus_s0.pt', map_location=DEV)
model = InContextSysID().to(DEV)
model.load_state_dict(ck['model'])
model.eval()

old_ref = nmse(model, 'stribeck', 'multisine', 0.05)
new_ref = sum(nmse(model, 'stribeck', 'multisine', dt) for dt in TRAIN_RATES) / len(TRAIN_RATES)
print(f'old (held-out-rate) reference: {old_ref:.4f}')
print(f'new (trained-rate-average) reference: {new_ref:.4f}')
assert abs(old_ref - new_ref) > 1e-6, 'references should differ -- they are different cells'
print('OK -- fix changes the reference value as expected')
"
```

Expected: prints two different numbers and `OK`. (Exact values will differ
from any numbers already published in `paper/main.tex` or results docs --
this checkpoint was trained under the OLD `_norm`, before Task 1's fix, so
these numbers are for verifying the CODE CHANGE's mechanics only, not for
reporting anywhere. The real numbers come from the post-merge retraining
campaign.)

- [ ] **Step 5: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "Fix reference-rate mislabeling: corpus-mode reference now averages the two actually-trained rates, not the held-out rate"
```

---

### Task 3: Fix corpus-generation seed determinism and add a divergence filter

**Files:**
- Modify: `corpus.py` (the `gen_cell` function, lines 30-53)
- Test: `tests/test_plantforge.py` (extend — this file already tests
  `corpus`-adjacent invariants per its existing structure)

**Interfaces:**
- Consumes: `families.FAMILIES` (already imported), `excitation.EXCITATIONS`
  (already imported).
- Produces: `gen_cell(family, exc, dt, n_inst, seed=0, ident=True, chunk=64)
  -> dict` — SAME return shape (keys: `u`, `y`, `theta`, `keys`, `dt`,
  `family`, `excitation`, and if `ident`, `rel_crlb`/`log10_cond`) and SAME
  guarantee (the shard has exactly `n_inst` instances) as before. Internal
  seeding scheme and divergence handling change; external contract does
  not.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plantforge.py`. Import `gen_cell` at the top of the
file alongside the existing imports:

```python
from plantforge.corpus import gen_cell
```

Add these two tests:

```python
def test_gen_cell_seeds_are_deterministic_across_hash_randomization():
    """gen_cell's per-instance draws must not depend on Python's
    per-process-randomized hash() -- two calls with the same explicit seed
    must produce identical output regardless of hash seed randomization
    (simulated here by calling gen_cell twice in the same process, which
    would already differ under PYTHONHASHSEED=random if hash() were still
    used, since hash('stribeck') is stable WITHIN one process but not
    ACROSS processes -- the real regression this guards is a fixed,
    order-stable index lookup, not process-level hash volatility, which a
    single-process test can only partially exercise. The stronger
    guarantee -- byte-identical shards from two separate `python -m
    plantforge.corpus` invocations -- is verified operationally, not by
    this unit test)."""
    shard1 = gen_cell("stribeck", "multisine", 0.05, n_inst=16, seed=42, ident=False)
    shard2 = gen_cell("stribeck", "multisine", 0.05, n_inst=16, seed=42, ident=False)
    assert torch.equal(shard1["u"], shard2["u"]), \
        "same explicit seed must give identical output within the same process"
    assert torch.equal(shard1["theta"], shard2["theta"])
    print("  PASS  test_gen_cell_seeds_are_deterministic_across_hash_randomization")


def test_gen_cell_different_families_get_different_seeds():
    """Two different families with the same seed/lo must NOT draw the same
    parameter values (would happen if the family-index lookup collided)."""
    shard_a = gen_cell("stribeck", "prbs", 0.05, n_inst=16, seed=0, ident=False)
    shard_b = gen_cell("saturate", "prbs", 0.05, n_inst=16, seed=0, ident=False)
    assert not torch.equal(shard_a["u"], shard_b["u"]), \
        "different families must not produce identical trajectories"
    print("  PASS  test_gen_cell_different_families_get_different_seeds")


def test_gen_cell_returns_exactly_n_inst_even_with_divergence():
    """gen_cell must always return exactly n_inst instances, even if some
    draws diverge to non-finite values and must be retried."""
    for fam in FAMILIES:
        shard = gen_cell(fam, "closedloop", 0.05, n_inst=32, seed=0, ident=False)
        assert shard["u"].shape[1] == 32, (fam, shard["u"].shape)
        assert torch.isfinite(shard["u"]).all(), f"{fam}: non-finite u in returned shard"
        assert torch.isfinite(shard["y"]).all(), f"{fam}: non-finite y in returned shard"
    print("  PASS  test_gen_cell_returns_exactly_n_inst_even_with_divergence")
```

Add all three calls to this file's existing `_run_all()` function (find it
near the bottom of `tests/test_plantforge.py` and add the three new calls
alongside the existing ones, preserving whatever ordering convention is
already there).

- [ ] **Step 2: Run tests to verify current behavior**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_plantforge`
Expected: `test_gen_cell_seeds_are_deterministic_across_hash_randomization`
and `test_gen_cell_different_families_get_different_seeds` likely PASS
already against the CURRENT code (within a single process, `hash()` is
stable — the bug is cross-process, not intra-process, so a naive
same-process test won't catch it before the fix). This is expected and
consistent with the test's own docstring caveat above.
`test_gen_cell_returns_exactly_n_inst_even_with_divergence` should also
PASS already if `closedloop` at these families/seed doesn't happen to
diverge — if it does, the CURRENT code returns FEWER than `n_inst`
instances (no retry logic exists yet), which would FAIL this test and
independently confirm finding 10's core claim.

Run this step and record which tests actually failed before proceeding --
don't assume; check the real output.

- [ ] **Step 3: Fix `gen_cell` in `corpus.py`**

Replace the current `gen_cell` (corpus.py:30-53):

```python
def gen_cell(family, exc, dt, n_inst, seed=0, ident=True, chunk=64):
    """One shard. Instances are drawn with a per-family seed so the SAME instances
    appear across all (exc, rate) cells of that family."""
    T = round(T_PHYS / dt)
    us, ys, thetas, crlbs, conds = [], [], [], [], []
    keys = None
    for lo in range(0, n_inst, chunk):
        B = min(chunk, n_inst - lo)
        gen_p = torch.Generator().manual_seed(seed * 7919 + hash(family) % 10007 + lo)
        p = sample(family, B, gen_p)
        gen_u = torch.Generator().manual_seed(seed * 104729 + hash(exc) % 3571 + lo)
        u, y = generate(family, p, exc, T, B, dt, gen_u)
        us.append(u); ys.append(y)
        from .families import param_vector
        th, keys = param_vector(family, p)
        thetas.append(th)
        if ident:
            idn = identifiability(family, p, u, dt)
            crlbs.append(idn["rel_crlb"]); conds.append(idn["log10_cond"])
    shard = {"u": torch.cat(us, dim=1), "y": torch.cat(ys, dim=1),
             "theta": torch.cat(thetas), "keys": keys, "dt": dt,
             "family": family, "excitation": exc}
    if ident:
        shard["rel_crlb"] = torch.cat(crlbs)
        shard["log10_cond"] = torch.cat(conds)
    return shard
```

with:

```python
_FAMILY_IDX = {f: i for i, f in enumerate(FAMILIES)}
_EXC_IDX = {e: i for i, e in enumerate(EXCITATIONS)}


def gen_cell(family, exc, dt, n_inst, seed=0, ident=True, chunk=64):
    """One shard. Instances are drawn with a per-family seed so the SAME instances
    appear across all (exc, rate) cells of that family.

    Uses a fixed FAMILIES/EXCITATIONS index lookup for seeding (not
    hash(family)/hash(exc), which are randomized per Python process unless
    PYTHONHASHSEED is pinned -- the prior version was not actually
    deterministic across separate `python -m plantforge.corpus` runs).

    Non-finite (diverged) draws are filtered out and the chunk is topped up
    by re-drawing with a fresh seed, so gen_cell always returns exactly
    n_inst instances with finite u/y -- matching the datasheet's documented
    behavior (previously the datasheet's claim did not match this
    function's actual, filter-less behavior)."""
    T = round(T_PHYS / dt)
    us, ys, thetas, crlbs, conds = [], [], [], [], []
    keys = None
    collected = 0
    lo = 0
    while collected < n_inst:
        B = min(chunk, n_inst - collected)
        attempt = 0
        got = 0
        while got < B:
            attempt += 1
            gen_p = torch.Generator().manual_seed(
                seed * 7919 + _FAMILY_IDX[family] * 10007 + lo * 1000 + attempt)
            p = sample(family, B - got, gen_p)
            gen_u = torch.Generator().manual_seed(
                seed * 104729 + _EXC_IDX[exc] * 3571 + lo * 1000 + attempt)
            u, y = generate(family, p, exc, T, B - got, dt, gen_u)
            finite = torch.isfinite(u).all(dim=0) & torch.isfinite(y).all(dim=0)
            if not finite.all():
                u, y = u[:, finite], y[:, finite]
                p = {k: v[finite] for k, v in p.items()}
            if u.shape[1] == 0:
                continue   # whole sub-chunk diverged, retry with a fresh seed
            us.append(u); ys.append(y)
            from .families import param_vector
            th, keys = param_vector(family, p)
            thetas.append(th)
            if ident:
                idn = identifiability(family, p, u, dt)
                crlbs.append(idn["rel_crlb"]); conds.append(idn["log10_cond"])
            got += u.shape[1]
        collected += B
        lo += chunk
    shard = {"u": torch.cat(us, dim=1), "y": torch.cat(ys, dim=1),
             "theta": torch.cat(thetas), "keys": keys, "dt": dt,
             "family": family, "excitation": exc}
    if ident:
        shard["rel_crlb"] = torch.cat(crlbs)
        shard["log10_cond"] = torch.cat(conds)
    return shard
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_plantforge`
Expected: all tests PASS, including the 3 new ones.

- [ ] **Step 5: Verify shard instance count exactness under heavier divergence**

Run this smoke check (drivetrain + closedloop is the combination flagged
in the datasheet as most divergence-prone):

```bash
cd /data/nas07_new/PersonalData/phuocthien && python3 -c "
import torch
from plantforge.corpus import gen_cell
shard = gen_cell('drivetrain', 'closedloop', 0.05, n_inst=200, seed=0, ident=False)
assert shard['u'].shape[1] == 200, shard['u'].shape
assert torch.isfinite(shard['u']).all() and torch.isfinite(shard['y']).all()
print('OK -- exactly 200 finite instances returned')
"
```
Expected: `OK -- exactly 200 finite instances returned`, no traceback, no
infinite loop (if this hangs for more than ~30s, the retry logic has a
bug — investigate before proceeding, don't just wait longer).

- [ ] **Step 6: Commit**

```bash
git add corpus.py tests/test_plantforge.py
git commit -m "Fix corpus seed determinism (drop hash()) and add missing divergence filter to gen_cell"
```

---

### Task 4: Fix the identifiability/prediction window mismatch in `ident_exp.py`

**Files:**
- Modify: `ident_exp.py` (the per-cell loop inside `main()`, lines 140-158)
- Test: `tests/test_ident_exp.py` (extend — this file already exists)

**Interfaces:**
- Consumes: `identifiability.identifiability(family, p, u, dt) -> dict`
  (unchanged signature: `p` is a dict of per-instance parameter tensors,
  `u` is `(T, B)`), `corpus.OUT` (unchanged).
- Produces: `main()`'s per-cell `crlb`/`cond` values now come from a
  224-sample recomputation instead of the shard's stored (full-trajectory)
  annotation. No new public function — this is an internal change to
  `main()`'s loop body. `main()`'s printed output format is unchanged.

- [ ] **Step 1: Read the existing test file to match its conventions**

Run: `cat /data/nas07_new/PersonalData/phuocthien/plantforge/tests/test_ident_exp.py`
before writing anything, to match its existing style (this file already
tests `qry_stats`, `quartile_table`, `filter_low_power` — follow the same
patterns for the new test).

- [ ] **Step 2: Write the failing test**

Add this test to `tests/test_ident_exp.py` (adjust the import line at the
top of the file to add `identifiability` from `plantforge.identifiability`
and `param_vector` from `plantforge.families` alongside whatever's already
imported there):

```python
from plantforge.identifiability import identifiability
from plantforge.families import sample, param_vector


def test_identifiability_recomputed_on_224_window_differs_from_full_trajectory():
    """The identifiability annotation computed over a 224-sample window
    must generally differ from one computed over a longer trajectory at
    the same rate -- if they were identical, recomputing on the shorter
    window would be pointless (the whole point of Group C's fix)."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    p = sample("stribeck", 8, gen)
    dt = 0.05
    T_full = round(12.8 / dt)   # 256 samples, the full corpus trajectory length
    u_full = torch.randn(T_full, 8)
    u_224 = u_full[:224]

    idn_full = identifiability("stribeck", p, u_full, dt)
    idn_224 = identifiability("stribeck", p, u_224, dt)

    assert not torch.allclose(idn_full["rel_crlb"], idn_224["rel_crlb"]), \
        "224-sample and full-trajectory rel_crlb should generally differ " \
        "(same excitation realization, different record length -> different FIM)"
    print("  PASS  test_identifiability_recomputed_on_224_window_differs_from_full_trajectory")


def test_theta_keys_roundtrip_reconstructs_p_dict():
    """param_vector's forward direction (p -> theta, keys) must invert
    losslessly: theta[:, i] alongside keys[i] must reconstruct exactly the
    values sample() produced, in the same dict-key layout identifiability()
    expects."""
    gen = torch.Generator().manual_seed(1)
    p = sample("boucwen", 5, gen)
    theta, keys = param_vector("boucwen", p)
    p_reconstructed = {k: theta[:, i] for i, k in enumerate(keys)}
    for k in p:
        assert torch.equal(p[k], p_reconstructed[k]), k
    print("  PASS  test_theta_keys_roundtrip_reconstructs_p_dict")
```

Add both new test calls to this file's existing `_run_all()`.

- [ ] **Step 3: Run tests to verify current state**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_ident_exp`
Expected: both new tests PASS already (they test `identifiability()` and
`param_vector()` directly, which are not modified by this task — they
verify the PRECONDITIONS the `main()` fix in Step 4 relies on). This
confirms the building blocks work correctly before wiring them into
`main()`.

- [ ] **Step 4: Fix the per-cell loop in `ident_exp.main()`**

Add this import near the top of `ident_exp.py`, alongside the existing
`from .corpus import OUT` line:

```python
from .identifiability import identifiability
```

Replace this block inside `main()`'s per-cell loop (ident_exp.py:149-158):

```python
                shard = torch.load(path, map_location="cpu")
                u = shard["u"].t()[:n_per_cell, :D].to(DEV)   # (B, D)
                y = shard["y"].t()[:n_per_cell, :D].to(DEV)
                v = _score_cell(models, u, y)
                crlb = shard["rel_crlb"][:n_per_cell].max(dim=1).values.numpy()
                cond = shard["log10_cond"][:n_per_cell].numpy()
                # Query-horizon power, model-independent (no forward pass needed):
                # same normalization qry_stats uses for `den`, computed once per cell.
                with torch.no_grad():
                    _, y_n = _norm(u, y)
                den = (y_n[:, T_CTX:] ** 2).mean(dim=1).cpu().numpy()
```

with:

```python
                shard = torch.load(path, map_location="cpu")
                u = shard["u"].t()[:n_per_cell, :D].to(DEV)   # (B, D)
                y = shard["y"].t()[:n_per_cell, :D].to(DEV)
                v = _score_cell(models, u, y)
                # Recompute identifiability on the SAME 224-sample window the
                # prediction task uses, rather than reusing the shard's
                # full-trajectory annotation (128-640 samples depending on
                # rate) -- the two must cover the same record length for the
                # comparison to be meaningful. theta/keys in the shard fully
                # capture the sampled p dict (param_vector's forward
                # direction), so this reconstruction is exact and lossless.
                theta_window = shard["theta"][:n_per_cell]           # (n_per_cell, K)
                keys = shard["keys"]
                p_window = {k: theta_window[:, i] for i, k in enumerate(keys)}
                # identifiability() expects u as (T, B); shard["u"] is
                # already stored (T_full, B_total) uncut, so slice directly
                # without transposing -- this is a DIFFERENT slice of the
                # same underlying tensor than the (B, D)-shaped `u` above,
                # which is used for the model's forward pass.
                u_for_ident = shard["u"][:D, :n_per_cell]            # (D, n_per_cell) = (T, B)
                idn = identifiability(fam, p_window, u_for_ident, dt)
                crlb = idn["rel_crlb"].max(dim=1).values.numpy()
                cond = idn["log10_cond"].numpy()
                # Query-horizon power, model-independent (no forward pass needed):
                # same normalization qry_stats uses for `den`, computed once per cell.
                with torch.no_grad():
                    _, y_n = _norm(u, y)
                den = (y_n[:, T_CTX:] ** 2).mean(dim=1).cpu().numpy()
```

- [ ] **Step 5: Manually verify the fixed loop runs without shape errors**

This requires at least one real corpus shard on disk. Run (against
whatever shards already exist in `PLANTFORGE_DATA`):

```bash
cd /data/nas07_new/PersonalData/phuocthien && PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data python3 -c "
import torch
from plantforge.corpus import OUT
from plantforge.identifiability import identifiability

path = OUT / 'stribeck_multisine_dt20hz.pt'
shard = torch.load(path, map_location='cpu')
n_per_cell = 50
theta_window = shard['theta'][:n_per_cell]
keys = shard['keys']
p_window = {k: theta_window[:, i] for i, k in enumerate(keys)}
D = 224
u_for_ident = shard['u'][:D, :n_per_cell]
idn = identifiability('stribeck', p_window, u_for_ident, 0.05)
print('rel_crlb shape:', idn['rel_crlb'].shape, '(expect', (n_per_cell, len(keys)), ')')
print('log10_cond shape:', idn['log10_cond'].shape, '(expect', (n_per_cell,), ')')
assert idn['rel_crlb'].shape == (n_per_cell, len(keys))
assert idn['log10_cond'].shape == (n_per_cell,)
print('OK')
"
```
Expected: `OK`, matching shapes printed, no traceback. If
`stribeck_multisine_dt20hz.pt` doesn't exist in this environment's
`PLANTFORGE_DATA`, substitute any shard filename that does exist (check
with `ls $PLANTFORGE_DATA/corpus/*.pt`) — the point is to verify the shape
mechanics against one real shard, not a specific family.

- [ ] **Step 6: Run the full offline suite**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.run_all`
Expected: all tests PASS (still not wired to include `test_evaluate` yet —
that happens at final whole-branch review).

- [ ] **Step 7: Commit**

```bash
git add ident_exp.py tests/test_ident_exp.py
git commit -m "Fix identifiability/prediction window mismatch: recompute rel-CRLB/log10-cond on the 224-sample prediction window instead of the shard's full-trajectory annotation"
```

---

## After all tasks: controller steps (not part of task-by-task execution)

These happen after all 4 tasks pass review and the branch is merged, per
this repo's established pattern (training-affecting code changes are
followed by a controller-run retraining campaign, never launched inside an
implementer task):

1. Wire `test_evaluate` into `tests/run_all.py` (the one new test module
   this plan adds; `test_plantforge.py` and `test_ident_exp.py` were
   extended, not newly created, so they're already wired).
2. Delete all existing checkpoints that were trained under the old (leaking)
   `_norm` — every `eval_{headline,corpus}_s*.pt`, every architecture-ablation
   variant checkpoint, every leave-one-out held-family variant checkpoint.
   Confirm the exact file list with the user before deleting anything (this
   is a destructive, hard-to-reverse action against ~50 files representing
   real GPU-hours of prior work).
3. Regenerate the released corpus shards under the new deterministic seeding
   scheme (Task 3's fix) if the corpus is to be re-released — decide with
   the user whether this is in scope for this retraining pass or deferred.
4. Launch the retraining campaign (same `scripts/train_seeds.sh`,
   `scripts/train_ablation.sh` + `scripts/train_ablation_seeds.sh`,
   `scripts/train_leave_one_out.sh` + `scripts/train_leave_one_out_seeds.sh`
   drivers already in this repo — no new training scripts needed, since
   Task 1/2's fixes don't change any script's checkpoint-naming or
   resumability contract).
5. Re-run `ident_exp.py`, `baselines.py`, `aggregate.py`, `ablation.py`,
   `leave_one_out.py` against the new checkpoints and record fresh results
   docs following this repo's established `docs/superpowers/results/`
   pattern.
6. Only after fresh, reviewed numbers exist: update `paper/main.tex`,
   `README.md`, `docs/DATASHEET.md` for both the code-driven number changes
   AND the deferred text-only fixes (findings 2, 4, 5, 8) in the same pass,
   since by then the exact new numbers for finding 5's "halves" claim will
   be known.
