# Architecture ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer whether the paper's headline transfer-gap claim (rate/excitation
closes, family gap halves) is sensitive to model capacity, via 4 architecture
variants (narrow/wide/shallow/deep) trained single-seed on the `corpus` recipe
and compared against the already-trained default on 2 target cells.

**Architecture:** Minimal size-parameterization of `evaluate.py` (env vars,
analogous to the existing `PF_SEED`); a new comparison script `ablation.py`;
a new resumable training driver `scripts/train_ablation.sh`.

**Tech Stack:** Python, PyTorch (all already in use, no new dependencies).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-14-architecture-ablation-design.md` — binding.
- Reuse `evaluate.InContextSysID`, `evaluate.run`, `evaluate.nmse`,
  `evaluate.HOLD_FAMILY`, `evaluate.FAMILIES` — no reimplementing.
- Do not modify `families.py`, `excitation.py`, `identifiability.py`,
  `corpus.py`, `realbench.py`, `aggregate.py`, `baselines.py`, `ident_exp.py`.
- Default-architecture checkpoint names (`eval_{mode}_s{seed}.pt`) MUST stay
  unchanged — `aggregate.py`/`realbench.py`/`ident_exp.py` all assume this
  exact name for the default architecture; the already-trained
  `eval_corpus_s0.pt` (d=160/layers=5/heads=8) must remain loadable by this
  plan's code with zero retraining.
- Non-default sizes get a checkpoint suffix `_d{WIDTH}L{LAYERS}` (e.g.
  `eval_corpus_s0_d80L5.pt`), only appended when `PF_WIDTH`/`PF_LAYERS` differ
  from the defaults (160/5) — so setting neither env var reproduces today's
  exact filename.
- Git repo root `/data/nas07_new/PersonalData/phuocthien/plantforge`, branch
  `architecture-ablation` off `main`. `git` commands run with cwd at the repo
  root; `python -m plantforge...` commands run with cwd
  `/data/nas07_new/PersonalData/phuocthien`. `PLANTFORGE_DATA` env var
  convention unchanged.

---

### Task 1: Size-parameterize `evaluate.py`

**Files:**
- Modify: `plantforge/evaluate.py`

**Interfaces:**
- Produces: `evaluate.WIDTH` (int, env `PF_WIDTH`, default 160),
  `evaluate.LAYERS` (int, env `PF_LAYERS`, default 5); `run()` constructs
  `InContextSysID(d=WIDTH, layers=LAYERS)` and derives the checkpoint path
  from `(mode, SEED, WIDTH, LAYERS)`.

- [ ] **Step 1: Add WIDTH/LAYERS constants**

Directly below the existing `SEED = int(os.environ.get("PF_SEED", "0"))` line
in `plantforge/evaluate.py`, add:

```python
WIDTH = int(os.environ.get("PF_WIDTH", "160"))    # InContextSysID d
LAYERS = int(os.environ.get("PF_LAYERS", "5"))    # InContextSysID layers
```

- [ ] **Step 2: Add a checkpoint-name helper and use it in `run()`**

Add this function directly above `def run(`:

```python
def _ckpt_name(mode: str) -> str:
    """eval_{mode}_s{SEED}.pt for the default architecture (160/5) --
    unchanged from before this plan, so the already-trained default
    checkpoints stay loadable with zero retraining. Non-default
    width/layers get an explicit suffix so they never collide with the
    default-architecture name."""
    suffix = "" if (WIDTH, LAYERS) == (160, 5) else f"_d{WIDTH}L{LAYERS}"
    return f"eval_{mode}_s{SEED}{suffix}.pt"
```

In `run()`, replace:

```python
    ck_path = CKPT_DIR / f"eval_{mode}_s{SEED}.pt"
    model = InContextSysID().to(DEV)
```

with:

```python
    ck_path = CKPT_DIR / _ckpt_name(mode)
    model = InContextSysID(d=WIDTH, layers=LAYERS).to(DEV)
```

- [ ] **Step 3: Verify no behavior change at defaults**

```bash
cd /data/nas07_new/PersonalData/phuocthien && python -c "
from plantforge.evaluate import _ckpt_name, WIDTH, LAYERS
assert (WIDTH, LAYERS) == (160, 5)
assert _ckpt_name('corpus') == 'eval_corpus_s0.pt'
assert _ckpt_name('wh_only') == 'eval_wh_only_s0.pt'
print('default checkpoint names unchanged:', _ckpt_name('corpus'), _ckpt_name('wh_only'))
"
cd /data/nas07_new/PersonalData/phuocthien && python -c "
import os
os.environ['PF_WIDTH'] = '80'
from plantforge.evaluate import _ckpt_name
assert _ckpt_name('corpus') == 'eval_corpus_s0_d80L5.pt'
print('non-default suffix works:', _ckpt_name('corpus'))
"
cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.run_all 2>&1 | tail -5
```

Expected: both assert blocks print without error, and the full offline suite
(30 tests) still passes — this task changes no test-covered behavior at
default settings.

- [ ] **Step 4: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add evaluate.py
git commit -m "Parameterize model width/layers via PF_WIDTH/PF_LAYERS for architecture ablation"
```

---

### Task 2: `ablation.py` — comparison across architecture variants

**Files:**
- Create: `plantforge/ablation.py`
- Test: `plantforge/tests/test_ablation.py`

**Interfaces:**
- Consumes: `evaluate.InContextSysID`, `evaluate.nmse`, `evaluate.HOLD_FAMILY`,
  `evaluate.CKPT_DIR`, `evaluate.DEV`, `evaluate._ckpt_name` (via constructing
  the same env-driven naming logic for each variant).
- Produces: `VARIANTS` (list of dicts: name, width, layers, params);
  `load_variant(name, width, layers) -> InContextSysID | None`;
  `param_count(width, layers) -> int`; CLI `python -m plantforge.ablation`.

- [ ] **Step 1: Write the failing tests**

Create `plantforge/tests/test_ablation.py`:

```python
"""Offline tests for plantforge.ablation's pure helpers -- no checkpoints,
no GPU, no network.

    python -m plantforge.tests.test_ablation     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from plantforge.ablation import VARIANTS, param_count, load_variant


def test_variants_include_default_and_four_others():
    names = {v["name"] for v in VARIANTS}
    assert names == {"default", "narrow", "wide", "shallow", "deep"}
    default = next(v for v in VARIANTS if v["name"] == "default")
    assert (default["width"], default["layers"]) == (160, 5)
    print("  PASS  test_variants_include_default_and_four_others")


def test_param_count_monotone_in_width_and_layers():
    base = param_count(160, 5)
    assert param_count(80, 5) < base < param_count(320, 5)
    assert param_count(160, 2) < base < param_count(160, 8)
    print("  PASS  test_param_count_monotone_in_width_and_layers")


def test_load_variant_missing_checkpoint_returns_none():
    assert load_variant("eval_this_checkpoint_does_not_exist.pt") is None
    print("  PASS  test_load_variant_missing_checkpoint_returns_none")


def _run_all():
    test_variants_include_default_and_four_others()
    test_param_count_monotone_in_width_and_layers()
    test_load_variant_missing_checkpoint_returns_none()


if __name__ == "__main__":
    print("PLANTFORGE ablation -- offline tests:")
    _run_all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_ablation`
Expected: `ModuleNotFoundError: No module named 'plantforge.ablation'`

- [ ] **Step 3: Write `plantforge/ablation.py`**

```python
"""Architecture ablation: does the paper's headline transfer-gap claim
(rate/excitation closes, family gap halves) depend on model capacity?
Compares 4 size variants (single seed 0, corpus recipe) against the
already-trained default on the two cells that carry the paper's claim.

    PLANTFORGE_DATA=... python -m plantforge.ablation
"""
from __future__ import annotations

import torch

from .evaluate import InContextSysID, nmse, HOLD_FAMILY, CKPT_DIR, DEV

VARIANTS = [
    {"name": "default", "width": 160, "layers": 5},
    {"name": "narrow", "width": 80, "layers": 5},
    {"name": "wide", "width": 320, "layers": 5},
    {"name": "shallow", "width": 160, "layers": 2},
    {"name": "deep", "width": 160, "layers": 8},
]
TOTAL_STEPS = 10000


def param_count(width: int, layers: int) -> int:
    return sum(p.numel() for p in InContextSysID(d=width, layers=layers).parameters())


def _ckpt_name_for(width: int, layers: int) -> str:
    suffix = "" if (width, layers) == (160, 5) else f"_d{width}L{layers}"
    return f"eval_corpus_s0{suffix}.pt"


def load_variant(ckpt_name: str):
    """Load a trained InContextSysID checkpoint (any width/layers, inferred
    from its own state dict shapes) in eval mode, or None if missing."""
    ck_path = CKPT_DIR / ckpt_name
    if not ck_path.exists():
        return None
    ck = torch.load(ck_path, map_location=DEV)
    sd = ck["model"]
    width = sd["inp.weight"].shape[0]
    layers = len({k.split(".")[2] for k in sd if k.startswith("tr.layers.")})
    model = InContextSysID(d=width, layers=layers).to(DEV)
    model.load_state_dict(sd)
    model.eval()
    return model


def main():
    print("=== architecture ablation: corpus recipe, seed 0, 2 target cells ===")
    for v in VARIANTS:
        name, width, layers = v["name"], v["width"], v["layers"]
        ckpt_name = _ckpt_name_for(width, layers)
        model = load_variant(ckpt_name)
        params = param_count(width, layers)
        if model is None:
            print(f"  {name:8s} (d={width:3d} L={layers} params={params:,}): "
                  f"MISSING -- run scripts/train_ablation.sh")
            continue
        ref = nmse(model, "stribeck", "multisine", 0.05)
        fam = nmse(model, HOLD_FAMILY, "multisine", 0.05)
        print(f"  {name:8s} (d={width:3d} L={layers} params={params:,}): "
              f"reference={ref:.4f}  family_backlash_dt.05={fam:.4f}  ({fam/ref:.1f}x ref)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_ablation`
Expected: 3 PASS lines, exit 0.

- [ ] **Step 5: Smoke the CLI against the already-trained default checkpoint only**

```bash
cd /data/nas07_new/PersonalData/phuocthien && \
PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data python -m plantforge.ablation
```

Expected: `default` row prints real `reference`/`family_backlash_dt.05` numbers
matching the known seed-0 corpus values (reference ≈0.0210-0.0215, family
dt=0.05 ≈0.29); the other 4 rows print `MISSING` (not yet trained). No traceback.

- [ ] **Step 6: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add ablation.py tests/test_ablation.py
git commit -m "Add architecture ablation comparison script"
```

---

### Task 3: Training driver for the 4 non-default variants

**Files:**
- Create: `plantforge/scripts/train_ablation.sh`

**Interfaces:**
- Consumes: Task 1's `PF_WIDTH`/`PF_LAYERS` env vars and `_ckpt_name` naming.
- Produces: a script the controller launches in the background to train
  `narrow`/`wide`/`shallow`/`deep` (corpus recipe, seed 0) to step 10000.

- [ ] **Step 1: Create `plantforge/scripts/train_ablation.sh`**

```bash
#!/bin/bash
# Train the 4 non-default architecture variants (corpus recipe, seed 0) to
# 10000 steps, resuming per PF_BUDGET-bounded attempts. Skips finished
# variants WITHOUT invoking evaluate.py, same pattern as train_seeds.sh.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_PARENT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PKG_PARENT"
export PLANTFORGE_DATA=${PLANTFORGE_DATA:-"$PKG_PARENT/plantforge_data"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export PF_BUDGET=${PF_BUDGET:-500}
export PF_SEED=0
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

# name width layers
VARIANTS="narrow:80:5 wide:320:5 shallow:160:2 deep:160:8"

for spec in $VARIANTS; do
    name=${spec%%:*}
    rest=${spec#*:}
    width=${rest%%:*}
    layers=${rest#*:}
    suffix=""
    if [ "$width" != "160" ] || [ "$layers" != "5" ]; then
        suffix="_d${width}L${layers}"
    fi
    ck="$PLANTFORGE_DATA/eval_corpus_s0${suffix}.pt"
    export PF_WIDTH=$width
    export PF_LAYERS=$layers
    prev_steps=-1
    while true; do
        steps=$(done_steps "$ck")
        if [ "$steps" -ge "$TOTAL" ]; then
            echo "== $name (d=$width L=$layers): done ($steps/$TOTAL)"
            break
        fi
        if [ "$steps" -le "$prev_steps" ]; then
            echo "== STALLED $name: step stuck at $steps after a successful run"
            exit 1
        fi
        prev_steps=$steps
        echo "== $name (d=$width L=$layers): at $steps/$TOTAL, training..."
        python -m plantforge.evaluate corpus \
            || { echo "== FAILED $name"; exit 1; }
    done
done
echo "== ALL VARIANTS DONE"
```

- [ ] **Step 2: Verify the skip path is inert when nothing is trained yet (fast, no training)**

This step intentionally does NOT run the script to completion (that would
start ~4 x 25-40min of training). Instead, verify the variant-parsing logic
in isolation:

```bash
chmod +x /data/nas07_new/PersonalData/phuocthien/plantforge/scripts/train_ablation.sh
bash -n /data/nas07_new/PersonalData/phuocthien/plantforge/scripts/train_ablation.sh && echo "syntax OK"
cd /data/nas07_new/PersonalData/phuocthien && python -c "
import os
os.environ['PF_WIDTH'] = '80'; os.environ['PF_LAYERS'] = '5'
from plantforge.evaluate import _ckpt_name
assert _ckpt_name('corpus') == 'eval_corpus_s0_d80L5.pt', _ckpt_name('corpus')
print('width=80,layers=5 ->', _ckpt_name('corpus'), '(matches script suffix logic)')
"
```

Expected: `syntax OK`, then the printed checkpoint name matching the shell
script's own suffix-construction logic (`_d80L5`) for the same
(width, layers) pair — this is the property that matters (script's naming
and Python's `_ckpt_name` must agree), verified without spending GPU time.

- [ ] **Step 3: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add scripts/train_ablation.sh
git commit -m "Add training driver for the 4 architecture-ablation variants"
```

**Execution note for the controller:** after this task's review passes,
launch `nohup .../train_ablation.sh > .../plantforge_data_train_ablation.log 2>&1 &`
in the background (4 variants x up to ~40min each, sequential on one GPU,
so up to ~2.5h worst case) and monitor to completion, then run
`python -m plantforge.ablation` for the final comparison table and record it
in a short results note (not editing `paper/main.tex` in this pass, per the
spec's explicit out-of-scope).

---

## Self-Review Notes

- **Spec coverage:** Task 1 = size-parameterization; Task 2 = comparison
  script + tests; Task 3 = training driver. Spec's "default checkpoint name
  unchanged" constraint verified explicitly in Task 1 Step 3 and Task 3 Step 2.
  Spec's "not updating paper/main.tex" constraint reflected in the controller
  execution note.
- **Placeholder scan:** none — every step has complete, concrete code.
- **Type consistency:** `_ckpt_name` (Task 1, inside `evaluate.py`) and
  `_ckpt_name_for` (Task 2, inside `ablation.py`) implement the same suffix
  rule independently (evaluate.py's depends on module-global SEED/WIDTH/LAYERS
  for training; ablation.py's takes explicit width/layers for reporting on
  arbitrary variants without mutating global state) — both produce identical
  strings for the same inputs, exercised by Task 3 Step 2's cross-check.
