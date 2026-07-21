# Training Divergence Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the training loop from silently corrupting weights and saving
all-NaN checkpoints when a rare batch produces non-finite gradients under the
context-only normalization, by skipping the optimizer step on non-finite
gradients and refusing to persist an all-NaN checkpoint.

**Architecture:** Two additions to `evaluate.py`'s `run()` training loop: (1)
capture `clip_grad_norm_`'s returned total norm and `continue` past
`opt.step()` when it is non-finite; (2) before `torch.save`, raise
`RuntimeError` if any model weight is non-finite. A small `_has_finite_weights`
helper (mirroring the copies already in `ablation.py`/`leave_one_out.py`) makes
the weight check testable. The context-only normalization, model, and data
pipeline are unchanged.

**Tech Stack:** Python 3, PyTorch. No new dependencies.

## Global Constraints

- Modify ONLY `evaluate.py` and `tests/test_evaluate.py`. Do not touch `_norm`,
  `InContextSysID`, `make_batch`, `build_pool`, or any other module.
- `run()`'s signature, its return value (`(model, finished_bool)`), the
  checkpoint dict keys (`model`, `opt`, `step`), the resumability contract, and
  the printed-line format are all unchanged except for the two additions
  specified below. The `scripts/train_*.sh` drivers must need zero changes.
- The existing `if not torch.isfinite(loss)` loss-skip path (which replaces the
  poisoned pooled batch) stays EXACTLY as-is — the new guard is a separate case
  (finite loss, non-finite gradient).
- New behavior gets offline tests (no GPU, no network, no real training,
  runnable via `python -m plantforge.tests.test_evaluate` from
  `/data/nas07_new/PersonalData/phuocthien`).
- Branch off `main`.

---

### Task 1: Add non-finite-gradient skip and all-NaN checkpoint refusal to `run()`

**Files:**
- Modify: `evaluate.py` (add `_has_finite_weights` helper; modify `run()`'s
  loop body and its checkpoint-save)
- Test: `tests/test_evaluate.py` (extend)

**Interfaces:**
- Consumes: `torch`, `torch.nn as nn` (both already imported in `evaluate.py`).
- Produces: new module-level `_has_finite_weights(model) -> bool`. `run()`'s
  external contract is otherwise unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evaluate.py`. Add `nn` and `_has_finite_weights` /
`InContextSysID` to the imports at the top (change the existing import line):

```python
import torch
import torch.nn as nn

from plantforge.evaluate import _norm, T_CTX, TRAIN_RATES, _has_finite_weights, InContextSysID
```

Add these three test functions:

```python
def test_clip_grad_norm_returns_nonfinite_when_grads_are_nan():
    """The guard in run() keys on clip_grad_norm_'s RETURN value (the total
    grad norm before clipping) being non-finite. This documents/locks that
    assumption: nan gradients -> non-finite returned norm."""
    m = nn.Linear(4, 2)
    for p in m.parameters():
        p.grad = torch.full_like(p, float("nan"))
    gnorm = nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    assert not torch.isfinite(gnorm), \
        "clip_grad_norm_ must return a non-finite norm when grads contain nan"
    # and the happy path stays finite
    for p in m.parameters():
        p.grad = torch.ones_like(p)
    gnorm_ok = nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    assert torch.isfinite(gnorm_ok), \
        "clip_grad_norm_ must return a finite norm for finite grads"
    print("  PASS  test_clip_grad_norm_returns_nonfinite_when_grads_are_nan")


def test_has_finite_weights_true_for_fresh_model():
    m = InContextSysID()
    assert _has_finite_weights(m) is True
    print("  PASS  test_has_finite_weights_true_for_fresh_model")


def test_has_finite_weights_false_when_a_parameter_is_nan():
    m = InContextSysID()
    with torch.no_grad():
        next(m.parameters()).fill_(float("nan"))
    assert _has_finite_weights(m) is False
    print("  PASS  test_has_finite_weights_false_when_a_parameter_is_nan")
```

Update `_run_all()` to call the three new tests:

```python
def _run_all():
    test_norm_uses_context_only_std()
    test_norm_context_std_is_approximately_one()
    test_train_rates_excludes_the_old_reference_rate()
    test_clip_grad_norm_returns_nonfinite_when_grads_are_nan()
    test_has_finite_weights_true_for_fresh_model()
    test_has_finite_weights_false_when_a_parameter_is_nan()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_evaluate`
Expected: FAIL at import time — `cannot import name '_has_finite_weights' from
'plantforge.evaluate'` (the helper doesn't exist yet). The
`test_clip_grad_norm_...` test would pass on its own (it tests PyTorch
behavior, not our code) but the module won't import until Step 3 adds the
helper, so the whole file errors first.

- [ ] **Step 3: Add the `_has_finite_weights` helper to `evaluate.py`**

Insert this helper into `evaluate.py` immediately AFTER the `_norm` function
(around line 58, before `make_batch`):

```python
def _has_finite_weights(model) -> bool:
    """True iff every learned parameter (not buffers -- the model's causal
    mask buffer is legitimately -inf by design) is finite. Mirrors the
    same-named helpers in ablation.py / leave_one_out.py, used here to refuse
    to persist a diverged (all-NaN) checkpoint at train time."""
    return all(torch.isfinite(p).all() for p in model.parameters())
```

- [ ] **Step 4: Add the non-finite-gradient skip in `run()`'s loop**

Replace this block in `run()`:

```python
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if i % 1000 == 0:
            print(f"  [{mode}] step {i:5d} mse {loss.item():.4f}")
        i += 1
```

with:

```python
        opt.zero_grad(); loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            # A rare batch can produce a finite forward loss but non-finite
            # backward gradients under context-only normalization; clip_grad_norm_
            # then returns a non-finite total norm and has already scaled every
            # gradient by a non-finite clip coefficient. Skipping opt.step() here
            # (the grads are cleared by the next opt.zero_grad()) prevents nan from
            # being written into every weight -- the confirmed root cause of the
            # 2026-07-21 all-NaN-checkpoint divergences. Bad batches are rare, so
            # skipping them has negligible training impact.
            print(f"  [{mode}] step {i}: non-finite gradient (norm={gnorm}), skipping step")
            i += 1
            continue
        opt.step()
        if i % 1000 == 0:
            print(f"  [{mode}] step {i:5d} mse {loss.item():.4f}")
        i += 1
```

Leave the `if not torch.isfinite(loss)` block above it exactly as-is.

- [ ] **Step 5: Add the all-NaN checkpoint refusal before the save**

Replace this line in `run()`:

```python
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": i}, ck_path)
```

with:

```python
    if not _has_finite_weights(model):
        raise RuntimeError(
            f"[{mode}] refusing to save checkpoint at step {i}: model weights are "
            f"non-finite (training diverged). This should be unreachable given the "
            f"non-finite-gradient skip guard above; if it fires, a new divergence "
            f"path exists and must be investigated, not silently persisted.")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": i}, ck_path)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_evaluate`
Expected: all 6 tests PASS (3 pre-existing + 3 new).

- [ ] **Step 7: Run the full offline suite to confirm no regressions**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.run_all`
Expected: all tests PASS, no failures/tracebacks, exit 0.

- [ ] **Step 8: Sanity-check that `run()` still imports and the guards are wired**

Run this quick structural check (no training — just confirms the module
imports cleanly and the two guard strings are present in `run`'s source):

```bash
cd /data/nas07_new/PersonalData/phuocthien && python3 -c "
import inspect, plantforge.evaluate as ev
src = inspect.getsource(ev.run)
assert 'non-finite gradient' in src, 'grad-skip guard missing from run()'
assert '_has_finite_weights(model)' in src, 'all-NaN checkpoint refusal missing from run()'
assert callable(ev._has_finite_weights)
print('OK -- both guards present in run(); _has_finite_weights importable')
"
```
Expected: `OK -- both guards present in run(); _has_finite_weights importable`

- [ ] **Step 9: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "Guard training loop against non-finite gradients and all-NaN checkpoints"
```

---

## After the task: controller-run retraining campaign (not a task step)

Per this repo's established convention, the retraining is a controller step
after the code change is reviewed and merged, NOT part of the task's TDD steps:

1. After merge to `main` and push, delete all 50 existing checkpoints
   (`eval_corpus_s*.pt`, `eval_wh_only_s*.pt` under `$PLANTFORGE_DATA`) — the
   30 healthy ones are retrained too, so every published number comes from the
   same corrected training loop (user-approved: retrain all 50). Confirm the
   deletion list with the user before removing, as before.
2. Relaunch the same balanced two-GPU campaign
   (`scripts/train_seeds.sh` + `train_ablation.sh`/`train_ablation_seeds.sh` +
   `train_leave_one_out.sh`/`train_leave_one_out_seeds.sh`) on the free GPUs.
   No script changes are needed (the guards are internal to `run()`).
3. When all 50 finish, re-scan every checkpoint for NaN weights (the same scan
   that found 20/50 bad this round) and confirm **0/50 diverged**. If any run
   now raises the new `RuntimeError` (all-NaN at save), that is a NEW
   divergence path and must be investigated before proceeding — do not work
   around it.
4. Only after a clean 0/50 scan: re-run `aggregate.py`, `ablation.py`,
   `leave_one_out.py`, `baselines.py`, `ident_exp.py` and record fresh results
   docs, then fold the corrected numbers into `paper/main.tex` / `README.md` /
   `docs/DATASHEET.md` (together with the still-pending text-only fixes from the
   adversarial-review-fixes work).
