# Paper-strengthening experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Multi-seed error bars for every headline number, classical ARX/poly-NARX baselines under the transformer's exact protocol, and an experiment showing the identifiability annotations predict in-context difficulty.

**Architecture:** Minimal seed-parameterization of `evaluate.py`; three new modules (`aggregate.py`, `baselines.py`, `ident_exp.py`) that reuse `evaluate`/`realbench` machinery; one driver script for the 8 background training runs; a final results document.

**Tech Stack:** Python, PyTorch, numpy, scipy (all already installed). Zero new dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-14-paper-experiments-design.md` — binding.
- Do not modify `families.py`, `excitation.py`, `identifiability.py`, `corpus.py`. `evaluate.py` gets ONLY the seed changes in Task 1. `realbench.py` gets ONLY the two checkpoint-name updates in Task 1.
- Reuse `evaluate`'s `_norm`, `make_batch`, `nmse`, `T_CTX`, `D`, `DEV`, `CKPT_DIR`, `FAMILIES`, `HOLD_FAMILY` and `realbench`'s `load_model`, `nmse_on_windows`, `silverbox_windows`, `cascaded_tanks_windows` — no reimplementing.
- New modules must not import `nonlinear_benchmarks` at top level (lazy only).
- Git repo root: `/data/nas07_new/PersonalData/phuocthien/plantforge`. Work on branch `paper-experiments` off `main`. Run `git` commands with cwd at the repo root; run `python -m plantforge...` with cwd `/data/nas07_new/PersonalData/phuocthien` (package parent).
- `PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data` for anything touching checkpoints or corpus shards. Corpus shards (60 cells, 4000 instances each) already exist under `$PLANTFORGE_DATA/corpus/`.
- Existing checkpoints `eval_wh_only.pt` / `eval_corpus.pt` were trained with `torch.manual_seed(0)` and pool salt `done*13` — they are exactly seed 0 under the new scheme.
- Aggregation convention everywhere: nMSE is the ratio of batch-mean query MSE to batch-mean query power (matching `evaluate.nmse` and `realbench.nmse_on_windows`), never a mean of per-window ratios.

---

### Task 1: Seed support + checkpoint migration

**Files:**
- Modify: `plantforge/evaluate.py` (4 small edits)
- Modify: `plantforge/realbench.py` (1 edit in `main()`)
- No new tests (covered by existing suites + a load check)

**Interfaces:**
- Consumes: current `evaluate.py` / `realbench.py` on `main`.
- Produces: `evaluate.SEED` (int, from env `PF_SEED`, default 0); checkpoints named `eval_{mode}_s{SEED}.pt`; training pool salt `done * 13 + SEED * 1_000_000`; `realbench.main()` loading `eval_{name}_s0.pt`.

- [ ] **Step 1: Edit `plantforge/evaluate.py`**

Edit 1 — add the SEED constant directly below the `CKPT_DIR` line:

```python
CKPT_DIR = pathlib.Path(os.environ.get("PLANTFORGE_DATA", "/home/coder/plantforge_data"))
SEED = int(os.environ.get("PF_SEED", "0"))   # training seed: init + data pool draws
```

Edit 2 — in `run()`, replace:

```python
    torch.manual_seed(0)
    ck_path = CKPT_DIR / f"eval_{mode}.pt"
```

with:

```python
    torch.manual_seed(SEED)
    ck_path = CKPT_DIR / f"eval_{mode}_s{SEED}.pt"
```

Edit 3 — in `run()`, replace:

```python
    pool = build_pool(mode, run_salt=done * 13)
```

with:

```python
    pool = build_pool(mode, run_salt=done * 13 + SEED * 1_000_000)
```

(Seed 0 reproduces the old salt exactly; evaluation seeds 900–905 in `nmse` are untouched.)

- [ ] **Step 2: Edit `plantforge/realbench.py`**

In `main()`, replace:

```python
    models = {name: load_model(f"eval_{name}.pt") for name in ("wh_only", "corpus")}
```

with:

```python
    models = {name: load_model(f"eval_{name}_s0.pt") for name in ("wh_only", "corpus")}
```

- [ ] **Step 3: Rename the existing checkpoints on disk (not git-tracked)**

```bash
mv /data/nas07_new/PersonalData/phuocthien/plantforge_data/eval_wh_only.pt \
   /data/nas07_new/PersonalData/phuocthien/plantforge_data/eval_wh_only_s0.pt
mv /data/nas07_new/PersonalData/phuocthien/plantforge_data/eval_corpus.pt \
   /data/nas07_new/PersonalData/phuocthien/plantforge_data/eval_corpus_s0.pt
```

- [ ] **Step 4: Verify**

```bash
cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_realbench && python -m plantforge.tests.run_all
cd /data/nas07_new/PersonalData/phuocthien && PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data python -c "
from plantforge.realbench import load_model
import torch, os, pathlib
for n in ('eval_wh_only_s0.pt', 'eval_corpus_s0.pt'):
    m = load_model(n)
    assert m is not None, n
    ck = torch.load(pathlib.Path(os.environ['PLANTFORGE_DATA']) / n, map_location='cpu')
    assert ck['step'] == 10000, (n, ck['step'])
print('both s0 checkpoints load, step 10000')
"
```

Expected: 12 + 6 PASS lines, then `both s0 checkpoints load, step 10000`.

- [ ] **Step 5: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add evaluate.py realbench.py
git commit -m "Parameterize training seed via PF_SEED; seed-suffixed checkpoint names"
```

---

### Task 2: Multi-seed training driver script

**Files:**
- Create: `plantforge/scripts/train_seeds.sh`

**Interfaces:**
- Consumes: Task 1's `PF_SEED` / checkpoint naming.
- Produces: a script the controller launches in the background to train all missing (seed, mode) checkpoints to step 10000.

- [ ] **Step 1: Create `plantforge/scripts/train_seeds.sh`**

```bash
#!/bin/bash
# Train eval_{wh_only,corpus}_s{SEED}.pt for every seed in $SEEDS to 10000
# steps, resuming per PF_BUDGET-bounded attempts. Skips finished checkpoints
# WITHOUT invoking evaluate.py (its pool build costs minutes even when
# there is nothing left to train).
set -uo pipefail
cd /data/nas07_new/PersonalData/phuocthien
export PLANTFORGE_DATA=${PLANTFORGE_DATA:-/data/nas07_new/PersonalData/phuocthien/plantforge_data}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export PYTHONUNBUFFERED=1
export PF_BUDGET=${PF_BUDGET:-500}
SEEDS=${SEEDS:-"0 1 2 3 4"}
TOTAL=10000

done_steps() {
    python - "$1" <<'EOF'
import sys, torch
try:
    print(torch.load(sys.argv[1], map_location="cpu")["step"])
except Exception:
    print(0)
EOF
}

for seed in $SEEDS; do
    for mode in headline corpus; do
        if [ "$mode" = headline ]; then name=wh_only; else name=corpus; fi
        ck="$PLANTFORGE_DATA/eval_${name}_s${seed}.pt"
        while true; do
            steps=$(done_steps "$ck")
            if [ "$steps" -ge "$TOTAL" ]; then
                echo "== seed $seed $mode: done ($steps/$TOTAL)"
                break
            fi
            echo "== seed $seed $mode: at $steps/$TOTAL, training..."
            PF_SEED=$seed python -m plantforge.evaluate "$mode" \
                || { echo "== FAILED seed $seed $mode"; exit 1; }
        done
    done
done
echo "== ALL SEEDS DONE"
```

- [ ] **Step 2: Verify the skip path (fast, no training)**

```bash
chmod +x /data/nas07_new/PersonalData/phuocthien/plantforge/scripts/train_seeds.sh
SEEDS=0 /data/nas07_new/PersonalData/phuocthien/plantforge/scripts/train_seeds.sh
```

Expected (seconds, no pool build): `== seed 0 headline: done (10000/10000)`, `== seed 0 corpus: done (10000/10000)`, `== ALL SEEDS DONE`.

- [ ] **Step 3: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add scripts/train_seeds.sh
git commit -m "Add multi-seed training driver script"
```

**Execution note for the controller:** after this task's review passes, launch `nohup .../train_seeds.sh > .../plantforge_data_train_seeds.log 2>&1 &` in the background (seeds 1–4 ≈ 5–7 h) and continue with Tasks 3–5 while it runs. Task 6 waits on it.

---

### Task 3: `aggregate.py` — multi-seed transfer matrix with error bars

**Files:**
- Create: `plantforge/aggregate.py`
- Test: `plantforge/tests/test_aggregate.py`

**Interfaces:**
- Consumes: `evaluate.nmse`, `evaluate.FAMILIES`, `evaluate.HOLD_FAMILY`, `realbench.load_model`, `realbench.nmse_on_windows`, `realbench.silverbox_windows`, `realbench.cascaded_tanks_windows`.
- Produces: `transfer_cells(mode) -> list[tuple[str, str, str, float]]` (label, family, exc, dt); `matrix(model, mode) -> dict[str, float]`; `aggregate_matrices(mats: list[dict]) -> dict[str, tuple[float, float, int]]` (label → mean, std, n); `mean_std_str(values: list[float]) -> str`; CLI `python -m plantforge.aggregate`.

- [ ] **Step 1: Write the failing tests**

Create `plantforge/tests/test_aggregate.py`:

```python
"""Offline tests for plantforge.aggregate's pure helpers -- no checkpoints,
no GPU, no network.

    python -m plantforge.tests.test_aggregate     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import math

from plantforge.aggregate import (
    transfer_cells, aggregate_matrices, mean_std_str,
)
from plantforge.evaluate import FAMILIES, HOLD_FAMILY


def test_transfer_cells_structure():
    for mode in ("wh_only", "corpus"):
        cells = transfer_cells(mode)
        labels = [c[0] for c in cells]
        assert labels[0] == "reference (train-like)"
        ref_family = cells[0][1]
        assert ref_family == ("wh" if mode == "wh_only" else "stribeck")
        # 1 ref + 3 held-out-family rates + 2 held-out excitations
        # + (len(FAMILIES)-1) held-out-rate families
        assert len(cells) == 1 + 3 + 2 + (len(FAMILIES) - 1)
        assert all(c[1] != HOLD_FAMILY or "family" in c[0] for c in cells)
    print("  PASS  test_transfer_cells_structure")


def test_aggregate_matrices_mean_std_n():
    mats = [{"a": 1.0, "b": 10.0}, {"a": 3.0, "b": 10.0}]
    agg = aggregate_matrices(mats)
    m, s, n = agg["a"]
    assert abs(m - 2.0) < 1e-12 and n == 2
    assert abs(s - 1.0) < 1e-12          # sample std (ddof=1) of [1, 3]
    m, s, n = agg["b"]
    assert abs(m - 10.0) < 1e-12 and abs(s - 0.0) < 1e-12
    print("  PASS  test_aggregate_matrices_mean_std_n")


def test_aggregate_matrices_single_seed_no_nan_std():
    agg = aggregate_matrices([{"a": 2.0}])
    m, s, n = agg["a"]
    assert m == 2.0 and n == 1
    assert not math.isnan(s) and s == 0.0   # n=1 reports std 0, not NaN
    print("  PASS  test_aggregate_matrices_single_seed_no_nan_std")


def test_mean_std_str_format():
    s = mean_std_str([0.1, 0.2, 0.3])
    assert "0.2000" in s and "±" in s and "n=3" in s
    print("  PASS  test_mean_std_str_format")


def _run_all():
    test_transfer_cells_structure()
    test_aggregate_matrices_mean_std_n()
    test_aggregate_matrices_single_seed_no_nan_std()
    test_mean_std_str_format()


if __name__ == "__main__":
    print("PLANTFORGE aggregate -- offline helper tests:")
    _run_all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_aggregate`
Expected: `ModuleNotFoundError: No module named 'plantforge.aggregate'`

- [ ] **Step 3: Write `plantforge/aggregate.py`**

```python
"""Multi-seed aggregation: recompute the transfer matrix for every finished
seed checkpoint and print each cell as mean +/- std, plus the real-plant
zero-shot numbers aggregated the same way.

    PLANTFORGE_DATA=... python -m plantforge.aggregate
Env: PF_SEEDS (default "0,1,2,3,4") -- seeds to look for.
"""
from __future__ import annotations

import os

import torch

from .evaluate import (FAMILIES, HOLD_FAMILY, CKPT_DIR, nmse)
from .realbench import load_model, nmse_on_windows

TOTAL_STEPS = 10000


def transfer_cells(mode: str):
    """(label, family, excitation, dt) rows, mirroring evaluate.report()."""
    ref_family = "wh" if mode == "wh_only" else "stribeck"
    cells = [("reference (train-like)", ref_family, "multisine", 0.05)]
    for dt in (0.10, 0.05, 0.02):
        cells.append((f"held-out family {HOLD_FAMILY} dt={dt:.2f}",
                      HOLD_FAMILY, "multisine", dt))
    for exc in ("chirp", "closedloop"):
        cells.append((f"held-out excitation {exc} (stribeck, dt=0.05)",
                      "stribeck", exc, 0.05))
    for fam in [f for f in FAMILIES if f != HOLD_FAMILY]:
        cells.append((f"held-out rate dt=0.05 {fam}", fam, "multisine", 0.05))
    return cells


def matrix(model, mode: str) -> dict:
    return {label: nmse(model, fam, exc, dt)
            for label, fam, exc, dt in transfer_cells(mode)}


def aggregate_matrices(mats: list) -> dict:
    """label -> (mean, sample std [0 when n=1], n) across per-seed matrices."""
    out = {}
    for label in mats[0]:
        vals = [m[label] for m in mats]
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
        out[label] = (mean, std, n)
    return out


def mean_std_str(values: list) -> str:
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f} (n={n})"


def _finished_models(mode: str, seeds):
    models = []
    for s in seeds:
        name = f"eval_{mode}_s{s}.pt"
        ck_path = CKPT_DIR / name
        if not ck_path.exists():
            print(f"  (seed {s}: {name} missing -- skipped)")
            continue
        step = torch.load(ck_path, map_location="cpu")["step"]
        if step < TOTAL_STEPS:
            print(f"  (seed {s}: {name} unfinished at step {step} -- skipped)")
            continue
        models.append(load_model(name))
    return models


def main():
    seeds = [int(s) for s in os.environ.get("PF_SEEDS", "0,1,2,3,4").split(",")]
    realplant = {}
    try:
        from .realbench import silverbox_windows, cascaded_tanks_windows
        sb, sb_dt, sb_q = silverbox_windows()
        ct, ct_dt = cascaded_tanks_windows()
        realplant = {f"Silverbox (decimated {sb_q}x -> dt={sb_dt:.4f}s)": sb,
                     f"Cascaded_Tanks (native dt={ct_dt:.2f}s, extrapolation)": ct}
    except Exception as e:
        print(f"(real-plant section skipped: {type(e).__name__}: {e})")

    for mode in ("wh_only", "corpus"):
        print(f"\n=== {mode}: transfer matrix, mean ± std over seeds ===")
        models = _finished_models(mode, seeds)
        if not models:
            print("  no finished checkpoints -- nothing to aggregate")
            continue
        mats = [matrix(m, mode) for m in models]
        agg = aggregate_matrices(mats)
        ref_mean = agg["reference (train-like)"][0]
        for label, (mean, std, n) in agg.items():
            print(f"  {label}: {mean:.4f} ± {std:.4f} (n={n})"
                  f"  [{mean / ref_mean:.1f}x ref]")
        for ds_label, windows in realplant.items():
            if windows is None:
                continue
            u, y = windows
            vals = [nmse_on_windows(m, u, y) for m in models]
            print(f"  real-plant {ds_label}: {mean_std_str(vals)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_aggregate`
Expected: 4 PASS lines, exit 0.

- [ ] **Step 5: Smoke the CLI against seed 0 only (checkpoints exist already)**

```bash
cd /data/nas07_new/PersonalData/phuocthien && \
PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data \
PF_SEEDS=0 python -m plantforge.aggregate
```

Expected: both modes print full matrices with `± 0.0000 (n=1)` and `[..x ref]` ratios; real-plant lines print for both datasets; missing seeds are not requested so no skip notes. Values should match the known single-seed numbers (e.g. corpus reference ≈ 0.0210, Silverbox corpus ≈ 0.3018).

- [ ] **Step 6: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add aggregate.py tests/test_aggregate.py
git commit -m "Add multi-seed aggregation with mean±std transfer matrix"
```

---

### Task 4: `baselines.py` — ARX and poly-NARX under the in-context protocol

**Files:**
- Create: `plantforge/baselines.py`
- Test: `plantforge/tests/test_baselines.py`

**Interfaces:**
- Consumes: `evaluate.make_batch` (returns `_norm`-alized `(B, D)` tensors), `evaluate._norm`, `evaluate.T_CTX`, `evaluate.D`, `aggregate.transfer_cells`.
- Produces: `predict_batch(u_b, y_b, poly: bool) -> np.ndarray (B, D-T_CTX)` (free-run query predictions from normalized windows); `baseline_nmse_batch(u_b, y_b, poly) -> float` (ratio-of-means, matching `nmse_on_windows` aggregation); `select_order(u, y, poly) -> int`; constants `LAGS = (2, 4, 8)`, `VAL = 32`, `RIDGE = 1e-6`; CLI `python -m plantforge.baselines [real]`.

- [ ] **Step 1: Write the failing tests**

Create `plantforge/tests/test_baselines.py`:

```python
"""Offline tests for plantforge.baselines -- no GPU training, no network.

    python -m plantforge.tests.test_baselines     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from plantforge.baselines import (
    predict_batch, baseline_nmse_batch, select_order, LAGS,
)
from plantforge.evaluate import T_CTX, D


def _arx_windows(B=4, seed=0):
    """Noise-free windows from a known ARX process (within model class for
    k>=2): y_t = 0.6 y_{t-1} - 0.2 y_{t-2} + 0.5 u_t + 0.1 u_{t-1}."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((B, D))
    y = np.zeros((B, D))
    for t in range(2, D):
        y[:, t] = (0.6 * y[:, t - 1] - 0.2 * y[:, t - 2]
                   + 0.5 * u[:, t] + 0.1 * u[:, t - 1])
    return (torch.tensor(u, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))


def test_arx_recovers_arx_process():
    u, y = _arx_windows()
    v = baseline_nmse_batch(u, y, poly=False)
    assert v < 1e-3, v
    print("  PASS  test_arx_recovers_arx_process")


def test_narx_handles_linear_process():
    u, y = _arx_windows()
    v = baseline_nmse_batch(u, y, poly=True)
    assert v < 1e-2, v
    print("  PASS  test_narx_handles_linear_process")


def test_select_order_returns_candidate():
    u, y = _arx_windows(B=1)
    k = select_order(u[0].numpy().astype(np.float64),
                     y[0].numpy().astype(np.float64), poly=False)
    assert k in LAGS, k
    print("  PASS  test_select_order_returns_candidate")


def test_freerun_never_peeks_at_query_y():
    u, y = _arx_windows()
    y2 = y.clone()
    y2[:, T_CTX:] = torch.randn_like(y2[:, T_CTX:])
    for poly in (False, True):
        p1 = predict_batch(u, y, poly)
        p2 = predict_batch(u, y2, poly)
        assert np.array_equal(p1, p2), "query-horizon y leaked into prediction"
    print("  PASS  test_freerun_never_peeks_at_query_y")


def _run_all():
    test_arx_recovers_arx_process()
    test_narx_handles_linear_process()
    test_select_order_returns_candidate()
    test_freerun_never_peeks_at_query_y()


if __name__ == "__main__":
    print("PLANTFORGE baselines -- offline tests:")
    _run_all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_baselines`
Expected: `ModuleNotFoundError: No module named 'plantforge.baselines'`

- [ ] **Step 3: Write `plantforge/baselines.py`**

```python
"""Classical SysID baselines (ARX, degree-2 polynomial NARX) under the SAME
in-context protocol as the transformer: per window, fit on the 192-sample
context only, then FREE-RUN the 32-sample query horizon given only u (the
transformer's yprev input is zeroed there too -- neither method sees
ground-truth y in the query horizon). Zero new dependencies.

    python -m plantforge.baselines          # synthetic transfer cells
    python -m plantforge.baselines real     # also real-plant windows (network)
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from .evaluate import make_batch, _norm, T_CTX, D
from .aggregate import transfer_cells

LAGS = (2, 4, 8)      # order candidates (na = nb = k)
VAL = 32              # last VAL context samples -> order selection
RIDGE = 1e-6          # Tikhonov regularization for conditioning
EVAL_SEEDS = range(900, 906)   # same eval seeds as evaluate.nmse


def _lag_vector(u, y, k, t):
    """Regressor for y[t]: k lags of y (t-1..t-k), k values of u (t..t-k+1)."""
    return np.concatenate([y[t - k:t][::-1], u[t - k + 1:t + 1][::-1]])


def _phi(z, poly):
    """[1, z] for ARX; [1, z, upper-triangle degree-2 products] for NARX."""
    if not poly:
        return np.concatenate([[1.0], z])
    quad = np.outer(z, z)[np.triu_indices(len(z))]
    return np.concatenate([[1.0], z, quad])


def _fit(u, y, k, poly, t_lo, t_hi):
    X = np.stack([_phi(_lag_vector(u, y, k, t), poly)
                  for t in range(max(k, t_lo), t_hi)])
    target = y[max(k, t_lo):t_hi]
    A = X.T @ X + RIDGE * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ target)


def _one_step_mse(u, y, w, k, poly, t_lo, t_hi):
    err = 0.0
    for t in range(t_lo, t_hi):
        pred = _phi(_lag_vector(u, y, k, t), poly) @ w
        err += (pred - y[t]) ** 2
    return err / (t_hi - t_lo)


def select_order(u, y, poly):
    """Pick k from LAGS by one-step-ahead error on the last VAL context
    samples; fit uses only the context before them. Never touches t>=T_CTX."""
    best_k, best_err = LAGS[0], np.inf
    for k in LAGS:
        w = _fit(u, y, k, poly, k, T_CTX - VAL)
        err = _one_step_mse(u, y, w, k, poly, T_CTX - VAL, T_CTX)
        if err < best_err:
            best_k, best_err = k, err
    return best_k


def _freerun(u, y, w, k, poly):
    """Simulate t = T_CTX..D-1 with predicted y fed back as lags; true y is
    used only for t < T_CTX."""
    yhat = y.copy()
    for t in range(T_CTX, D):
        yhat[t] = _phi(_lag_vector(u, yhat, k, t), poly) @ w
    return yhat[T_CTX:]


def predict_batch(u_b: torch.Tensor, y_b: torch.Tensor, poly: bool) -> np.ndarray:
    """Free-run query predictions for a batch of NORMALIZED (B, D) windows."""
    u_np = u_b.detach().cpu().numpy().astype(np.float64)
    y_np = y_b.detach().cpu().numpy().astype(np.float64)
    preds = []
    for i in range(u_np.shape[0]):
        u, y = u_np[i], y_np[i]
        k = select_order(u, y, poly)
        w = _fit(u, y, k, poly, k, T_CTX)
        preds.append(_freerun(u, y, w, k, poly))
    return np.stack(preds)


def baseline_nmse_batch(u_b, y_b, poly) -> float:
    """Ratio of batch-mean query MSE to batch-mean query power -- the same
    aggregation as evaluate.nmse / realbench.nmse_on_windows."""
    pred = predict_batch(u_b, y_b, poly)
    y_q = y_b.detach().cpu().numpy().astype(np.float64)[:, T_CTX:]
    return float(((pred - y_q) ** 2).mean() / (y_q ** 2).mean())


def synthetic_report():
    print("=== classical baselines on the corpus transfer cells "
          "(context-fit + free-run, nMSE) ===")
    for label, fam, exc, dt in transfer_cells("corpus"):
        vals = {"ARX": [], "NARX2": []}
        for sd in EVAL_SEEDS:
            u, y = make_batch(fam, exc, dt, 96, sd)   # already _norm-alized
            if not (torch.isfinite(u).all() and torch.isfinite(y).all()):
                continue                              # skip diverged draws
            vals["ARX"].append(baseline_nmse_batch(u, y, poly=False))
            vals["NARX2"].append(baseline_nmse_batch(u, y, poly=True))
        arx = sum(vals["ARX"]) / max(len(vals["ARX"]), 1)
        narx = sum(vals["NARX2"]) / max(len(vals["NARX2"]), 1)
        print(f"  {label}: ARX {arx:.4f} | NARX2 {narx:.4f}")


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


if __name__ == "__main__":
    synthetic_report()
    if len(sys.argv) > 1 and sys.argv[1] == "real":
        real_report()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_baselines`
Expected: 4 PASS lines, exit 0.

- [ ] **Step 5: Smoke the CLI (synthetic only — a few minutes of CPU lstsq)**

```bash
cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.baselines 2>&1 | tail -15
```

Expected: one `ARX ... | NARX2 ...` line per transfer cell, all values finite; no traceback.

- [ ] **Step 6: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add baselines.py tests/test_baselines.py
git commit -m "Add ARX/poly-NARX baselines under the in-context free-run protocol"
```

---

### Task 5: `ident_exp.py` — identifiability predicts difficulty

**Files:**
- Create: `plantforge/ident_exp.py`
- Test: `plantforge/tests/test_ident_exp.py`

**Interfaces:**
- Consumes: `evaluate._norm`, `evaluate.T_CTX`, `evaluate.D`, `evaluate.DEV`, `evaluate.CKPT_DIR`, `realbench.load_model`, `realbench.nmse_on_windows`, `corpus.OUT` (shard directory), `families.FAMILIES`, `excitation.EXCITATIONS`, `scipy.stats.spearmanr`.
- Produces: `qry_stats(model, u, y) -> (num (B,), den (B,))` per-instance query MSE and power; `nmse_per_instance(model, u, y) -> torch.Tensor (B,)`; `quartile_table(metric: np.ndarray, nmse: np.ndarray) -> list[tuple[float, float, int]]` (bin mean-metric, bin mean-nMSE, bin count); CLI `python -m plantforge.ident_exp`.
- **Spec correction (documented here deliberately):** the spec says "`nmse_per_instance`'s mean equals `nmse_on_windows`" — that is loose: `nmse_on_windows` is a ratio of batch means, which equals `num.mean()/den.mean()`, NOT `(num/den).mean()`. The test below asserts the correct identity via `qry_stats`.

- [ ] **Step 1: Write the failing tests**

Create `plantforge/tests/test_ident_exp.py`:

```python
"""Offline tests for plantforge.ident_exp -- untrained model, synthetic data.

    python -m plantforge.tests.test_ident_exp     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from plantforge.ident_exp import qry_stats, nmse_per_instance, quartile_table
from plantforge.evaluate import InContextSysID, D, DEV
from plantforge.realbench import nmse_on_windows


def test_qry_stats_reconstruct_batch_nmse():
    torch.manual_seed(0)
    model = InContextSysID().to(DEV)
    model.eval()
    u = torch.randn(8, D, device=DEV)
    y = torch.randn(8, D, device=DEV)
    num, den = qry_stats(model, u, y)
    assert num.shape == (8,) and den.shape == (8,)
    batch = nmse_on_windows(model, u, y)
    assert abs((num.mean() / den.mean()).item() - batch) < 1e-5
    per_inst = nmse_per_instance(model, u, y)
    assert per_inst.shape == (8,) and torch.isfinite(per_inst).all()
    print("  PASS  test_qry_stats_reconstruct_batch_nmse")


def test_quartile_table_monotone_construction():
    metric = np.arange(1.0, 9.0)          # 1..8
    nmse = metric * 2.0                   # perfectly increasing with metric
    rows = quartile_table(metric, nmse)
    assert len(rows) == 4
    assert all(rows[i][1] < rows[i + 1][1] for i in range(3))
    assert sum(r[2] for r in rows) == 8
    print("  PASS  test_quartile_table_monotone_construction")


def _run_all():
    test_qry_stats_reconstruct_batch_nmse()
    test_quartile_table_monotone_construction()


if __name__ == "__main__":
    print("PLANTFORGE ident_exp -- offline tests:")
    _run_all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_ident_exp`
Expected: `ModuleNotFoundError: No module named 'plantforge.ident_exp'`

- [ ] **Step 3: Write `plantforge/ident_exp.py`**

```python
"""Do the corpus's identifiability annotations predict in-context difficulty?
Scores corpus-model checkpoints per-instance on corpus shards (which carry
per-instance rel-CRLB and FIM condition annotations) and reports Spearman
correlations plus a quartile table.

Only dt in {0.05, 0.02} cells are usable: shards hold T_PHYS=12.8 s, so
10 Hz cells have 128 < D=224 samples.

    PLANTFORGE_DATA=... python -m plantforge.ident_exp
Env: PF_SEEDS (default "0,1,2,3,4"), PF_IDENT_N (instances/cell, default 1000).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from scipy.stats import spearmanr

from .corpus import OUT
from .evaluate import _norm, T_CTX, D, DEV, CKPT_DIR
from .families import FAMILIES
from .excitation import EXCITATIONS
from .realbench import load_model

USABLE_DTS = (0.05, 0.02)
CHUNK = 256
TOTAL_STEPS = 10000


def qry_stats(model, u, y):
    """Per-instance query-horizon (mse, power): num/den are (B,) tensors and
    num.mean()/den.mean() reproduces nmse_on_windows' batch ratio exactly."""
    u_n, y_n = _norm(u, y)
    with torch.no_grad():
        pred = model(u_n, y_n)
    num = ((pred[:, T_CTX:] - y_n[:, T_CTX:]) ** 2).mean(dim=1)
    den = (y_n[:, T_CTX:] ** 2).mean(dim=1)
    return num, den


def nmse_per_instance(model, u, y):
    num, den = qry_stats(model, u, y)
    return num / (den + 1e-12)


def quartile_table(metric: np.ndarray, nmse: np.ndarray):
    """Bin instances into metric quartiles; return
    [(bin mean metric, bin mean nMSE, bin count)] x 4."""
    qs = np.quantile(metric, [0.25, 0.5, 0.75])
    bins = np.digitize(metric, qs)
    rows = []
    for b in range(4):
        mask = bins == b
        rows.append((float(metric[mask].mean()), float(nmse[mask].mean()),
                     int(mask.sum())))
    return rows


def _corpus_models(seeds):
    models = []
    for s in seeds:
        name = f"eval_corpus_s{s}.pt"
        ck_path = CKPT_DIR / name
        if not ck_path.exists():
            continue
        if torch.load(ck_path, map_location="cpu")["step"] < TOTAL_STEPS:
            continue
        models.append(load_model(name))
    return models


def _score_cell(models, u, y):
    """Mean-over-seeds per-instance nMSE, chunked to bound GPU memory."""
    per_model = []
    for m in models:
        chunks = [nmse_per_instance(m, u[i:i + CHUNK], y[i:i + CHUNK])
                  for i in range(0, u.shape[0], CHUNK)]
        per_model.append(torch.cat(chunks))
    return torch.stack(per_model).mean(dim=0).cpu().numpy()


def main():
    seeds = [int(s) for s in os.environ.get("PF_SEEDS", "0,1,2,3,4").split(",")]
    n_per_cell = int(os.environ.get("PF_IDENT_N", "1000"))
    models = _corpus_models(seeds)
    if not models:
        print("no finished corpus checkpoints found -- train first")
        return
    print(f"=== identifiability vs in-context difficulty "
          f"(corpus model, {len(models)} seed(s), {n_per_cell}/cell) ===")

    fams, crlbs, conds, nmses = [], [], [], []
    for fam in FAMILIES:
        for exc in EXCITATIONS:
            for dt in USABLE_DTS:
                path = OUT / f"{fam}_{exc}_dt{int(1 / dt)}hz.pt"
                if not path.exists():
                    print(f"  (shard missing, skipped: {path.name})")
                    continue
                shard = torch.load(path, map_location="cpu")
                u = shard["u"].t()[:n_per_cell, :D].to(DEV)   # (B, D)
                y = shard["y"].t()[:n_per_cell, :D].to(DEV)
                v = _score_cell(models, u, y)
                crlb = shard["rel_crlb"][:n_per_cell].max(dim=1).values.numpy()
                cond = shard["log10_cond"][:n_per_cell].numpy()
                fams += [fam] * len(v)
                crlbs.append(crlb); conds.append(cond); nmses.append(v)

    crlb = np.concatenate(crlbs); cond = np.concatenate(conds)
    nmse = np.concatenate(nmses); fams = np.array(fams)
    ok = np.isfinite(crlb) & np.isfinite(cond) & np.isfinite(nmse)
    crlb, cond, nmse, fams = crlb[ok], cond[ok], nmse[ok], fams[ok]
    print(f"  instances scored: {len(nmse)} (dropped {int((~ok).sum())} non-finite)")

    r, p = spearmanr(crlb, nmse)
    print(f"  Spearman(max rel-CRLB, nMSE): r={r:.3f} (p={p:.1e})")
    r2, p2 = spearmanr(cond, nmse)
    print(f"  Spearman(log10 FIM cond, nMSE): r={r2:.3f} (p={p2:.1e})")
    for fam in FAMILIES:
        m = fams == fam
        if m.sum() > 10:
            rf, pf = spearmanr(crlb[m], nmse[m])
            print(f"    {fam}: r={rf:.3f} (p={pf:.1e}, n={int(m.sum())})")

    print("  quartiles of max rel-CRLB -> mean nMSE:")
    for i, (mm, mn, n) in enumerate(quartile_table(crlb, nmse), 1):
        print(f"    Q{i}: metric~{mm:.3g}  nMSE={mn:.4f}  (n={n})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_ident_exp`
Expected: 2 PASS lines, exit 0.

- [ ] **Step 5: Smoke the CLI with seed 0 and a small N**

```bash
cd /data/nas07_new/PersonalData/phuocthien && \
PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data \
PF_SEEDS=0 PF_IDENT_N=200 python -m plantforge.ident_exp
```

Expected: instance count line, two Spearman lines with finite r/p, per-family lines, 4 quartile rows; no traceback. (Direction is a research result, not asserted — but r for max rel-CRLB is expected positive.)

- [ ] **Step 6: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add ident_exp.py tests/test_ident_exp.py
git commit -m "Add identifiability-predicts-difficulty experiment"
```

---

### Task 6: Run everything, record results

**Precondition:** the Task 2 driver has finished (`== ALL SEEDS DONE` in its log; all 10 checkpoints at step 10000). The controller enforces this — do not start this task before then.

**Files:**
- Create: `docs/superpowers/results/2026-07-14-experiment-results.md`

**Interfaces:**
- Consumes: all previous tasks' CLIs.
- Produces: a committed results document with the three experiments' full outputs.

- [ ] **Step 1: Verify all checkpoints are finished**

```bash
cd /data/nas07_new/PersonalData/phuocthien && \
PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data python -c "
import torch, pathlib, os
d = pathlib.Path(os.environ['PLANTFORGE_DATA'])
for mode in ('wh_only', 'corpus'):
    for s in range(5):
        ck = d / f'eval_{mode}_s{s}.pt'
        step = torch.load(ck, map_location='cpu')['step'] if ck.exists() else None
        print(f'{ck.name}: {step}')
        assert step == 10000, ck.name
print('all 10 finished')
"
```

Expected: 10 lines all showing 10000, then `all 10 finished`.

- [ ] **Step 2: Run the three CLIs, capturing output**

```bash
cd /data/nas07_new/PersonalData/phuocthien
export PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data
python -m plantforge.aggregate        2>&1 | tee /tmp/agg.txt
python -m plantforge.baselines real   2>&1 | tee /tmp/base.txt
python -m plantforge.ident_exp        2>&1 | tee /tmp/ident.txt
```

Expected: no tracebacks; aggregate shows `(n=5)` on every cell.

- [ ] **Step 3: Write `docs/superpowers/results/2026-07-14-experiment-results.md`**

Structure (fill the code blocks with the verbatim captured outputs):

```markdown
# Experiment results — multi-seed, baselines, identifiability (2026-07-14)

Configuration: 5 seeds (PF_SEED 0–4) x 2 modes, 10k steps each; eval seeds
900–905 fixed across all models; corpus shards 4000 instances/cell;
identifiability run with PF_IDENT_N=1000 over dt in {0.05, 0.02}.

## A. Multi-seed transfer matrices (mean ± std, n=5)
<verbatim aggregate output>

## B. Classical baselines (ARX / degree-2 NARX, context-fit + free-run)
<verbatim baselines output>

## C. Identifiability vs difficulty (Spearman + quartiles)
<verbatim ident_exp output>

## Reading notes
<3-6 bullet points a co-author would need: does the corpus-vs-wh_only gap
survive error bars? where do baselines land relative to both models? is the
rel-CRLB correlation positive and significant? any surprises worth flagging.>
```

- [ ] **Step 4: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add docs/superpowers/results/2026-07-14-experiment-results.md
git commit -m "Record multi-seed, baseline, and identifiability experiment results"
```

---

## Self-Review Notes

- **Spec coverage:** A → Tasks 1-3 + 6; B → Task 4 (+6); C → Task 5 (+6). Spec's execution-order note → Task 2's controller note + Task 6's precondition. Spec's "10 Hz shards too short" constraint → `USABLE_DTS` in Task 5. All global constraints copied into the header.
- **Placeholder scan:** Task 6 Step 3's angle-bracket blocks are deliberate capture slots for run outputs that cannot exist until execution, with explicit instructions for filling them; everything else is complete code.
- **Type consistency:** `transfer_cells` (Task 3) consumed by `baselines.synthetic_report` (Task 4) with the same 4-tuple shape; `qry_stats`/`nmse_per_instance` shapes match their tests; `load_model` reused from `realbench` unchanged; checkpoint naming `eval_{mode}_s{seed}.pt` consistent across Tasks 1, 2, 3, 5, 6.
- **Correction to spec noted inline** (Task 5 interfaces): ratio-of-means vs mean-of-ratios identity, tested via `qry_stats`.
