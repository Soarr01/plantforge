# Zero-shot real-plant evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `plantforge/realbench.py`, evaluating the already-trained `eval_corpus.pt` / `eval_wh_only.pt` checkpoints zero-shot against real measured plants (Silverbox, Cascaded_Tanks, WienerHammerBenchMark) from `nonlinear_benchmarks`.

**Architecture:** One new standalone module reusing `InContextSysID`, `_norm`, `T_CTX`, `D`, `DEV`, `CKPT_DIR` from `evaluate.py` and `RATES` from `corpus.py`. Pure decimation/windowing helpers are unit-tested offline; dataset loaders hit the network (verified manually, not part of the automated suite); `main()` ties it together into a printed report.

**Tech Stack:** Python, PyTorch, `scipy.signal` (already installed), `nonlinear_benchmarks` pip package (already installed and verified working in this environment).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-14-realbench-zero-shot-design.md` — every requirement there is binding.
- `nonlinear_benchmarks` is a soft dependency: import it lazily, only inside the functions in `plantforge/realbench.py` that need it. Never import it at module top-level or from any other plantforge file.
- Do not modify `families.py`, `excitation.py`, `identifiability.py`, `corpus.py`, or `evaluate.py`.
- Trained-rate targets are `corpus.RATES = [0.10, 0.05, 0.02]` — reuse this constant, do not redefine it.
- Context window length is `evaluate.D = evaluate.T_CTX + evaluate.T_QRY = 224` — reuse this constant, do not redefine it.
- `WINDOW_CAP = 8` windows per dataset (pooled across records where a dataset has multiple test records).
- Git repo root is `/data/nas07_new/PersonalData/phuocthien/plantforge` (root commit `00a02c6`, baseline = pre-existing extracted+fixed project). Work happens on branch `realbench-zero-shot`, never on `main`. Commit at the end of each task as normal.
- Two different working directories matter, because `plantforge` is both the git repo root and a Python package: run `git` commands with cwd `/data/nas07_new/PersonalData/phuocthien/plantforge`; run `python -m plantforge...` commands with cwd `/data/nas07_new/PersonalData/phuocthien` (the package's parent — required for `python -m` to resolve the `plantforge` package).
- Set `PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data` for any command that touches `evaluate.CKPT_DIR` (Tasks 2-4) — same convention as the existing `corpus`/`evaluate` runs.

---

### Task 1: Decimation + windowing helpers

**Files:**
- Create: `plantforge/realbench.py`
- Test: `plantforge/tests/test_realbench.py`

**Interfaces:**
- Produces: `decimate_to_factor(x: np.ndarray, q: int) -> np.ndarray`, `best_decimation_factor(native_dt: float, target_dts=RATES) -> tuple[int, float] | None`, `make_windows(u: np.ndarray, y: np.ndarray, cap: int = WINDOW_CAP) -> tuple[torch.Tensor, torch.Tensor] | None`, `pooled_windows(records: list[tuple[np.ndarray, np.ndarray]], cap: int = WINDOW_CAP) -> tuple[torch.Tensor, torch.Tensor] | None`, module constant `WINDOW_CAP = 8`, re-exports `D`, `DEV` (imported from `evaluate`).

- [ ] **Step 1: Write the failing tests**

Create `plantforge/tests/test_realbench.py`:

```python
"""Offline-safe tests for plantforge.realbench's decimation/windowing
helpers -- no network access, no trained checkpoints required.

    python -m plantforge.tests.test_realbench     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from plantforge.realbench import (
    decimate_to_factor, best_decimation_factor, make_windows,
    pooled_windows, D, DEV,
)
from plantforge.corpus import RATES


def test_make_windows_shapes_and_cap():
    n = D * 5 + 30          # 5 full windows + a remainder
    u = np.arange(n, dtype=np.float64)
    y = np.arange(n, dtype=np.float64) * 2
    windows = make_windows(u, y, cap=3)
    assert windows is not None
    u_win, y_win = windows
    assert u_win.shape == (3, D)
    assert y_win.shape == (3, D)
    assert torch.equal(u_win[0], torch.tensor(u[:D], dtype=torch.float32, device=DEV))
    print("  PASS  test_make_windows_shapes_and_cap")


def test_make_windows_too_short_returns_none():
    u = np.zeros(D - 1, dtype=np.float64)
    y = np.zeros(D - 1, dtype=np.float64)
    assert make_windows(u, y) is None
    print("  PASS  test_make_windows_too_short_returns_none")


def test_decimate_to_factor_single_call():
    t = np.linspace(0, 10, 20000)
    x = np.sin(2 * np.pi * 0.5 * t)
    out = decimate_to_factor(x, 12)
    assert np.isfinite(out).all()
    assert abs(len(out) - len(x) // 12) <= 2
    assert out.std() > 0.1          # signal survives decimation, not degenerate
    print("  PASS  test_decimate_to_factor_single_call")


def test_decimate_to_factor_chained():
    t = np.linspace(0, 10, 200000)
    x = np.sin(2 * np.pi * 0.1 * t)
    out = decimate_to_factor(x, 100)   # forces chaining (10 * 10)
    assert np.isfinite(out).all()
    assert abs(len(out) - len(x) // 100) <= 5
    print("  PASS  test_decimate_to_factor_chained")


def test_best_decimation_factor_matches_silverbox():
    native_dt = 0.0016384041943147375     # measured Silverbox sampling_time
    result = best_decimation_factor(native_dt, RATES)
    assert result is not None
    q, achieved_dt = result
    assert q == 12
    assert abs(achieved_dt - 0.02) < 0.001
    print("  PASS  test_best_decimation_factor_matches_silverbox")


def test_best_decimation_factor_none_when_native_already_coarser():
    result = best_decimation_factor(4.0, RATES)   # Cascaded_Tanks case
    assert result is None
    print("  PASS  test_best_decimation_factor_none_when_native_already_coarser")


def test_pooled_windows_caps_across_records():
    rec1 = (np.zeros(D * 2, dtype=np.float64), np.zeros(D * 2, dtype=np.float64))
    rec2 = (np.ones(D * 5, dtype=np.float64), np.ones(D * 5, dtype=np.float64))
    windows = pooled_windows([rec1, rec2], cap=5)
    assert windows is not None
    u_win, y_win = windows
    assert u_win.shape == (5, D)
    assert (u_win[:2] == 0).all() and (u_win[2:] == 1).all()
    print("  PASS  test_pooled_windows_caps_across_records")


def _run_all():
    test_make_windows_shapes_and_cap()
    test_make_windows_too_short_returns_none()
    test_decimate_to_factor_single_call()
    test_decimate_to_factor_chained()
    test_best_decimation_factor_matches_silverbox()
    test_best_decimation_factor_none_when_native_already_coarser()
    test_pooled_windows_caps_across_records()


if __name__ == "__main__":
    print("PLANTFORGE realbench -- offline decimation/windowing tests:")
    _run_all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_realbench`
Expected: `ModuleNotFoundError: No module named 'plantforge.realbench'`

- [ ] **Step 3: Write `plantforge/realbench.py` (helpers only for this task)**

```python
"""Zero-shot evaluation of trained in-context SysID models on real measured
plants from nonlinearbenchmark.org, via the `nonlinear_benchmarks` package.
`nonlinear_benchmarks` is imported lazily (inside functions) so this module
only requires it when actually run, not merely imported.

    python -m plantforge.realbench
"""
from __future__ import annotations

import numpy as np
import scipy.signal
import torch

from .corpus import RATES
from .evaluate import InContextSysID, _norm, T_CTX, D, DEV, CKPT_DIR

WINDOW_CAP = 8


def decimate_to_factor(x: np.ndarray, q: int) -> np.ndarray:
    """Anti-aliased decimation by integer factor q, chaining calls in steps
    of at most 10 (scipy.signal.decimate recommends q<=13 per call)."""
    out = x
    remaining = q
    while remaining > 1:
        step = min(remaining, 10)
        out = scipy.signal.decimate(out, step, ftype="iir", zero_phase=True)
        remaining //= step
    return out


def best_decimation_factor(native_dt: float, target_dts=RATES):
    """Integer decimation factor q>=1 that brings native_dt closest to one of
    target_dts. Only considers decimating DOWN in rate (q>=1, native_dt*q ~
    target); returns None if native_dt is already coarser than every target
    (nothing to decimate to -- the Cascaded_Tanks case)."""
    if native_dt >= max(target_dts):
        return None
    best = None
    for target in target_dts:
        q = max(1, round(target / native_dt))
        achieved = native_dt * q
        err = abs(achieved - target)
        if best is None or err < best[2]:
            best = (q, achieved, err)
    return best[0], best[1]


def make_windows(u: np.ndarray, y: np.ndarray, cap: int = WINDOW_CAP):
    """Slice 1-D (u, y) arrays into non-overlapping length-D windows, stacked
    as (B, D) float32 tensors on DEV. Returns None if fewer than 1 full
    window is available."""
    n_windows = min(len(u), len(y)) // D
    if n_windows < 1:
        return None
    n_windows = min(n_windows, cap)
    u_win = np.stack([u[i * D:(i + 1) * D] for i in range(n_windows)])
    y_win = np.stack([y[i * D:(i + 1) * D] for i in range(n_windows)])
    return (torch.tensor(u_win, dtype=torch.float32, device=DEV),
            torch.tensor(y_win, dtype=torch.float32, device=DEV))


def pooled_windows(records, cap: int = WINDOW_CAP):
    """Slice windows from each (u, y) record and concatenate across records,
    stopping once `cap` total windows are collected."""
    u_all, y_all = [], []
    remaining = cap
    for u, y in records:
        if remaining <= 0:
            break
        w = make_windows(u, y, cap=remaining)
        if w is None:
            continue
        u_all.append(w[0]); y_all.append(w[1])
        remaining -= w[0].shape[0]
    if not u_all:
        return None
    return torch.cat(u_all, dim=0), torch.cat(y_all, dim=0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_realbench`
Expected: all 7 `PASS` lines printed, exit code 0.

---

### Task 2: Model loading + nMSE-on-windows

**Files:**
- Modify: `plantforge/realbench.py` (append)
- Modify: `plantforge/tests/test_realbench.py` (append)

**Interfaces:**
- Consumes: `InContextSysID`, `_norm`, `T_CTX`, `DEV`, `CKPT_DIR` (from Task 1's imports).
- Produces: `load_model(ckpt_name: str) -> InContextSysID | None`, `nmse_on_windows(model, u: torch.Tensor, y: torch.Tensor) -> float`.

- [ ] **Step 1: Write the failing test**

Append to `plantforge/tests/test_realbench.py` (add to the imports line and add the new test + call in `_run_all`):

```python
# change the existing import line to:
from plantforge.realbench import (
    decimate_to_factor, best_decimation_factor, make_windows,
    pooled_windows, nmse_on_windows, load_model, D, DEV,
)
```

```python
def test_nmse_on_windows_finite_untrained_model():
    from plantforge.evaluate import InContextSysID
    model = InContextSysID().to(DEV)
    model.eval()
    u = torch.randn(4, D, device=DEV)
    y = torch.randn(4, D, device=DEV)
    v = nmse_on_windows(model, u, y)
    assert np.isfinite(v)
    print("  PASS  test_nmse_on_windows_finite_untrained_model")


def test_load_model_missing_checkpoint_returns_none():
    assert load_model("eval_this_checkpoint_does_not_exist.pt") is None
    print("  PASS  test_load_model_missing_checkpoint_returns_none")
```

Add both calls inside `_run_all()`, after the existing calls:

```python
    test_nmse_on_windows_finite_untrained_model()
    test_load_model_missing_checkpoint_returns_none()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_realbench`
Expected: `ImportError: cannot import name 'nmse_on_windows' from 'plantforge.realbench'`

- [ ] **Step 3: Append to `plantforge/realbench.py`**

```python
def nmse_on_windows(model, u: torch.Tensor, y: torch.Tensor) -> float:
    """Same nMSE formula as evaluate.nmse: query-horizon MSE normalized by
    query-horizon variance, computed over a fixed batch of (u, y) windows."""
    u_n, y_n = _norm(u, y)
    with torch.no_grad():
        pred = model(u_n, y_n)
    return (((pred[:, T_CTX:] - y_n[:, T_CTX:]) ** 2).mean()
            / (y_n[:, T_CTX:] ** 2).mean()).item()


def load_model(ckpt_name: str):
    """Load a trained InContextSysID checkpoint in eval mode, or None if the
    checkpoint file doesn't exist (e.g. only one of corpus/wh_only has been
    trained so far)."""
    ck_path = CKPT_DIR / ckpt_name
    if not ck_path.exists():
        return None
    model = InContextSysID().to(DEV)
    ck = torch.load(ck_path, map_location=DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_realbench`
Expected: all 11 `PASS` lines printed, exit code 0. (Task 1 originally shipped 7 tests, but its review-driven fix added 2 more — `test_decimate_to_factor_prime_terminates` and `test_best_decimation_factor_no_crash_when_native_much_finer` — so the baseline going into this task is 9, not 7; this task's 2 new tests bring the total to 11.)

---

### Task 3: Dataset-specific loaders (network-dependent, manually verified)

**Files:**
- Modify: `plantforge/realbench.py` (append)

**Interfaces:**
- Consumes: `best_decimation_factor`, `decimate_to_factor`, `pooled_windows`, `make_windows`, `RATES` (from Tasks 1-2).
- Produces: `silverbox_windows() -> tuple[tuple[torch.Tensor, torch.Tensor] | None, float, int]`, `cascaded_tanks_windows() -> tuple[tuple[torch.Tensor, torch.Tensor] | None, float]`, `wienerhammer_status() -> tuple[float, float]`.

These functions call the network (via `nonlinear_benchmarks`) and are not covered by the automated offline test suite, per the spec. Verify manually in Step 2.

- [ ] **Step 1: Append to `plantforge/realbench.py`**

```python
def silverbox_windows():
    """Load Silverbox's test records, decimate to the nearest trained rate,
    and pool windows across all test records. Returns
    (windows_or_None, achieved_dt, decimation_factor)."""
    import nonlinear_benchmarks as nb
    _, test = nb.Silverbox()
    native_dt = test[0].sampling_time
    q, achieved_dt = best_decimation_factor(native_dt, RATES)
    records = []
    for rec in test:
        u = decimate_to_factor(np.asarray(rec.u, dtype=np.float64), q)
        y = decimate_to_factor(np.asarray(rec.y, dtype=np.float64), q)
        records.append((u, y))
    return pooled_windows(records), achieved_dt, q


def cascaded_tanks_windows():
    """Load Cascaded_Tanks' test record at its native rate (already coarser
    than every trained rate, so no decimation is applied). Returns
    (windows_or_None, native_dt)."""
    import nonlinear_benchmarks as nb
    _, test = nb.Cascaded_Tanks()
    native_dt = test.sampling_time
    u = np.asarray(test.u, dtype=np.float64)
    y = np.asarray(test.y, dtype=np.float64)
    return make_windows(u, y), native_dt


def wienerhammer_status():
    """WienerHammerBenchMark's total physical duration is shorter than one
    context window at any trained rate -- report why it's skipped rather
    than evaluate it. Returns (duration_seconds, min_seconds_needed)."""
    import nonlinear_benchmarks as nb
    _, test = nb.WienerHammerBenchMark()
    native_dt = test.sampling_time
    duration = len(test.u) * native_dt
    min_needed = D * min(RATES)
    return duration, min_needed
```

- [ ] **Step 2: Manually verify against the live network/package**

Run:
```bash
cd /data/nas07_new/PersonalData/phuocthien && python -c "
from plantforge.realbench import silverbox_windows, cascaded_tanks_windows, wienerhammer_status
w, dt, q = silverbox_windows()
print('silverbox', None if w is None else w[0].shape, dt, q)
w2, dt2 = cascaded_tanks_windows()
print('cascaded_tanks', None if w2 is None else w2[0].shape, dt2)
print('wh', wienerhammer_status())
"
```
Expected: `silverbox (8, 224) 0.0196... 12`, `cascaded_tanks (4, 224) 4.0`, `wh (<2.0, 4.48)` (some duration under 2 seconds, min_needed 4.48).

---

### Task 4: Report orchestration (`main`)

**Files:**
- Modify: `plantforge/realbench.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: `main()`, module runnable via `python -m plantforge.realbench`.

- [ ] **Step 1: Append to `plantforge/realbench.py`**

```python
def main():
    print("=== real-plant zero-shot transfer (in-context nMSE) ===")
    models = {name: load_model(f"eval_{name}.pt") for name in ("wh_only", "corpus")}
    missing = [name for name, m in models.items() if m is None]
    if missing:
        print(f"  (checkpoint(s) missing, skipped: {', '.join(missing)} -- "
              f"run `python -m plantforge.evaluate <mode>` first)")

    sb_windows, sb_dt, sb_q = silverbox_windows()
    print(f"  Silverbox (decimated {sb_q}x -> dt={sb_dt:.4f}s, ~50Hz-like):")
    if sb_windows is None:
        print("    SKIPPED -- no full window available after decimation")
    else:
        u, y = sb_windows
        for name, model in models.items():
            if model is None:
                continue
            v = nmse_on_windows(model, u, y)
            print(f"    {name} model: nMSE={v:.4f}  (n={u.shape[0]} windows)")

    ct_windows, ct_dt = cascaded_tanks_windows()
    print(f"  Cascaded_Tanks (native dt={ct_dt:.2f}s -- "
          f"{ct_dt / max(RATES):.0f}x coarser than trained range, EXTRAPOLATION):")
    if ct_windows is None:
        print("    SKIPPED -- record too short for one context window")
    else:
        u, y = ct_windows
        for name, model in models.items():
            if model is None:
                continue
            v = nmse_on_windows(model, u, y)
            print(f"    {name} model: nMSE={v:.4f}  (n={u.shape[0]} windows)")

    wh_duration, wh_min_needed = wienerhammer_status()
    print(f"  WienerHammerBenchMark: SKIPPED -- record duration "
          f"({wh_duration:.2f}s) shorter than one context window at any "
          f"trained rate (min {wh_min_needed:.2f}s) -- not evaluable zero-shot.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full offline test suite once more (regression check)**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_realbench && python -m plantforge.tests.run_all`
Expected: all realbench `PASS` lines, then all 6 existing invariant-test `PASS` lines (proves Task 4's additions didn't break imports for the existing suite).

- [ ] **Step 3: Run the end-to-end report**

Run: `cd /data/nas07_new/PersonalData/phuocthien && PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data python -m plantforge.realbench`
Expected: the full report prints with real nMSE numbers for both checkpoints on Silverbox and Cascaded_Tanks, and the WienerHammerBenchMark skip message. No tracebacks.

---

## Self-Review Notes

- **Spec coverage:** architecture (Task 1-4 imports match spec's reuse list) / per-dataset rate handling table (Task 3, one function per row) / windowing & pooling (Task 1) / report format (Task 4 matches the spec's example block) / testing (Task 1-2, offline-only, kept in a separate file from `test_plantforge.py`) / out-of-scope items (Bouc-Wen, fine-tuning, no changes to other files) — all respected, no gaps found.
- **Placeholder scan:** none — every step has complete, concrete code.
- **Type consistency:** `make_windows`/`pooled_windows` return `tuple[torch.Tensor, torch.Tensor] | None` consistently across Tasks 1, 3, 4; `load_model` returns `InContextSysID | None` consistently across Tasks 2 and 4; `best_decimation_factor` returns `tuple[int, float] | None` consistently across Tasks 1 and 3.
