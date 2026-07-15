# Multi-seed architecture ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train seeds 1-4 for the 4 non-default architecture variants and
extend `ablation.py` to report mean±std per variant, removing the
single-seed caveat from the architecture-ablation finding.

**Architecture:** One new training driver script; a small, backward-compatible
extension of `ablation.py`'s checkpoint naming and `main()`.

**Tech Stack:** Python, PyTorch, bash (all already in use, no new dependencies).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-15-multiseed-ablation-design.md` — binding.
- Do not modify `evaluate.py`, `families.py`, `excitation.py`,
  `identifiability.py`, `corpus.py`, `realbench.py`, `aggregate.py`,
  `baselines.py`, `ident_exp.py`, `paper/main.tex`.
- Reuse `evaluate.InContextSysID/nmse/HOLD_FAMILY/CKPT_DIR/DEV` and
  `aggregate.mean_std_str` — no reimplementing training or the mean/std formula.
- `load_variant(ckpt_name)` and `param_count(width, layers)` signatures in
  `ablation.py` stay unchanged.
- Git repo root `/data/nas07_new/PersonalData/phuocthien/plantforge`, branch
  `multiseed-ablation` off `main`. `git` commands run with cwd at the repo
  root; `python -m plantforge...` commands run with cwd
  `/data/nas07_new/PersonalData/phuocthien`. `PLANTFORGE_DATA` env var
  convention unchanged.

---

### Task 1: Multi-seed training driver

**Files:**
- Create: `plantforge/scripts/train_ablation_seeds.sh`

**Interfaces:**
- Consumes: `evaluate.py`'s existing `PF_SEED`/`PF_WIDTH`/`PF_LAYERS` env vars
  (no changes needed there).
- Produces: a script the controller launches in the background to train
  seeds 1-4 for narrow/wide/shallow/deep to step 10000 each.

- [ ] **Step 1: Create `plantforge/scripts/train_ablation_seeds.sh`**

```bash
#!/bin/bash
# Train seeds 1-4 (default: $SEEDS) for the 4 non-default architecture
# variants (narrow/wide/shallow/deep, corpus recipe) to 10000 steps each.
# The default architecture already has 5 seeds trained (0-4) from an earlier
# plan and is not retrained here. Resumable, stall-guarded, skips finished
# (seed, variant) pairs without invoking evaluate.py.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_PARENT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PKG_PARENT"
export PLANTFORGE_DATA=${PLANTFORGE_DATA:-"$PKG_PARENT/plantforge_data"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export PF_BUDGET=${PF_BUDGET:-500}
TOTAL=10000
SEEDS=${SEEDS:-"1 2 3 4"}

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

for seed in $SEEDS; do
    for spec in $VARIANTS; do
        name=${spec%%:*}
        rest=${spec#*:}
        width=${rest%%:*}
        layers=${rest#*:}
        suffix=""
        if [ "$width" != "160" ] || [ "$layers" != "5" ]; then
            suffix="_d${width}L${layers}"
        fi
        ck="$PLANTFORGE_DATA/eval_corpus_s${seed}${suffix}.pt"
        export PF_SEED=$seed
        export PF_WIDTH=$width
        export PF_LAYERS=$layers
        prev_steps=-1
        while true; do
            steps=$(done_steps "$ck")
            if [ "$steps" -ge "$TOTAL" ]; then
                echo "== seed $seed $name (d=$width L=$layers): done ($steps/$TOTAL)"
                break
            fi
            if [ "$steps" -le "$prev_steps" ]; then
                echo "== STALLED seed $seed $name: step stuck at $steps after a successful run"
                exit 1
            fi
            prev_steps=$steps
            echo "== seed $seed $name (d=$width L=$layers): at $steps/$TOTAL, training..."
            python -m plantforge.evaluate corpus \
                || { echo "== FAILED seed $seed $name"; exit 1; }
        done
    done
done
echo "== ALL SEEDS x VARIANTS DONE"
```

- [ ] **Step 2: Verify naming agreement and syntax (fast, no training)**

```bash
chmod +x /data/nas07_new/PersonalData/phuocthien/plantforge/scripts/train_ablation_seeds.sh
bash -n /data/nas07_new/PersonalData/phuocthien/plantforge/scripts/train_ablation_seeds.sh && echo "syntax OK"
cd /data/nas07_new/PersonalData/phuocthien && python -c "
import os
os.environ['PF_SEED'] = '3'; os.environ['PF_WIDTH'] = '320'; os.environ['PF_LAYERS'] = '5'
from plantforge.evaluate import _ckpt_name
assert _ckpt_name('corpus') == 'eval_corpus_s3_d320L5.pt', _ckpt_name('corpus')
print('seed=3,width=320,layers=5 ->', _ckpt_name('corpus'), '(matches script suffix logic)')
"
```

Do NOT run the script to completion in this step — that starts real GPU
training (~7h). The controller launches it separately after this task's
review passes.

Expected: `syntax OK`, then the printed checkpoint name matching the
script's own suffix-construction logic.

- [ ] **Step 3: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add scripts/train_ablation_seeds.sh
git commit -m "Add multi-seed training driver for the 4 architecture-ablation variants"
```

---

### Task 2: Multi-seed aggregation in `ablation.py`

**Files:**
- Modify: `plantforge/ablation.py`
- Modify: `plantforge/tests/test_ablation.py`

**Interfaces:**
- Consumes: `aggregate.mean_std_str`.
- Produces: `_ckpt_name_for(width, layers, seed=0)` (seed now a parameter,
  default preserves old 2-arg call sites); `_finished_variant_models(name,
  width, layers, seeds)`; `main()` prints mean±std per variant.
  `load_variant(ckpt_name)` and `param_count(width, layers)` unchanged.

- [ ] **Step 1: Write the failing test**

Update the import line and add one new test in
`plantforge/tests/test_ablation.py`:

```python
from plantforge.ablation import VARIANTS, param_count, load_variant, _ckpt_name_for
```

```python
def test_ckpt_name_for_seed_parameter():
    assert _ckpt_name_for(160, 5, seed=0) == "eval_corpus_s0.pt"
    assert _ckpt_name_for(160, 5, seed=3) == "eval_corpus_s3.pt"
    assert _ckpt_name_for(80, 5, seed=2) == "eval_corpus_s2_d80L5.pt"
    print("  PASS  test_ckpt_name_for_seed_parameter")
```

Add the call inside `_run_all()`:

```python
    test_ckpt_name_for_seed_parameter()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_ablation`
Expected: `TypeError: _ckpt_name_for() got an unexpected keyword argument 'seed'`

- [ ] **Step 3: Edit `plantforge/ablation.py`**

Replace the whole file with:

```python
"""Architecture ablation: does the paper's headline transfer-gap claim
(rate/excitation closes, family gap halves) depend on model capacity?
Compares 4 size variants (corpus recipe) against the default, aggregating
mean+-std over every trained seed found per variant (seed 0 alone is used
when that's all that exists, reported as n=1).

    PLANTFORGE_DATA=... python -m plantforge.ablation
Env: PF_SEEDS (default "0,1,2,3,4") -- seeds to look for per variant.
"""
from __future__ import annotations

import os

import torch

from .evaluate import InContextSysID, nmse, HOLD_FAMILY, CKPT_DIR, DEV
from .aggregate import mean_std_str

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


def _ckpt_name_for(width: int, layers: int, seed: int = 0) -> str:
    suffix = "" if (width, layers) == (160, 5) else f"_d{width}L{layers}"
    return f"eval_corpus_s{seed}{suffix}.pt"


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


def _finished_variant_models(name: str, width: int, layers: int, seeds):
    """Load every finished (step >= TOTAL_STEPS) checkpoint for this variant
    across the given seeds; print a skip note for missing/unfinished ones."""
    models = []
    for seed in seeds:
        ckpt_name = _ckpt_name_for(width, layers, seed)
        ck_path = CKPT_DIR / ckpt_name
        if not ck_path.exists():
            print(f"    (seed {seed}: {ckpt_name} missing -- skipped)")
            continue
        step = torch.load(ck_path, map_location="cpu")["step"]
        if step < TOTAL_STEPS:
            print(f"    (seed {seed}: {ckpt_name} unfinished at step {step} -- skipped)")
            continue
        models.append(load_variant(ckpt_name))
    return models


def main():
    seeds = [int(s) for s in os.environ.get("PF_SEEDS", "0,1,2,3,4").split(",")]
    print(f"=== architecture ablation: corpus recipe, seeds {seeds}, 2 target cells ===")
    for v in VARIANTS:
        name, width, layers = v["name"], v["width"], v["layers"]
        params = param_count(width, layers)
        print(f"  {name} (d={width} L={layers} params={params:,}):")
        models = _finished_variant_models(name, width, layers, seeds)
        if not models:
            print(f"    MISSING -- no finished seeds for this variant")
            continue
        refs = [nmse(m, "stribeck", "multisine", 0.05) for m in models]
        fams = [nmse(m, HOLD_FAMILY, "multisine", 0.05) for m in models]
        ref_mean = sum(refs) / len(refs)
        print(f"    reference: {mean_std_str(refs)}")
        print(f"    family_backlash_dt.05: {mean_std_str(fams)}"
              f"  ({(sum(fams) / len(fams)) / ref_mean:.1f}x ref)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_ablation`
Expected: all 4 tests PASS (the 3 pre-existing + the new one), exit 0.

- [ ] **Step 5: Smoke the CLI against currently-available checkpoints**

```bash
cd /data/nas07_new/PersonalData/phuocthien && \
PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data python -m plantforge.ablation
```

Expected: `default` shows `n=5` for both cells with numbers matching
`paper/main.tex` Table 1's corpus row (reference 0.0215±0.0017, family dt=0.05
0.2899±0.0095); `narrow`/`wide`/`shallow`/`deep` each show `n=1` (only seed 0
exists yet) with numbers matching `docs/superpowers/results/2026-07-15-architecture-ablation-results.md`.
No traceback.

- [ ] **Step 6: Run the full offline suite for import safety**

```bash
cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.run_all
```

Expected: all 33 tests still PASS (this task doesn't add new test files, so
the count doesn't change; `test_ablation`'s 4th test runs as part of the
existing `ablation` section).

- [ ] **Step 7: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add ablation.py tests/test_ablation.py
git commit -m "Aggregate architecture ablation over multiple seeds (mean+-std per variant)"
```

**Execution note for the controller:** after both tasks' reviews pass and
the final whole-branch review approves, launch
`nohup .../train_ablation_seeds.sh > .../plantforge_data_train_ablation_seeds.log 2>&1 &`
in the background (16 runs, ~7h estimated from single-seed timing), monitor
to completion, then run `python -m plantforge.ablation` for the final
multi-seed table and record it in a results note (not editing
`paper/main.tex` in this pass, per the spec's explicit out-of-scope).

---

## Self-Review Notes

- **Spec coverage:** Task 1 = training driver; Task 2 = aggregation. Spec's
  "no evaluate.py changes needed" constraint reflected (Task 1 only adds a
  new script). Spec's "not updating paper/main.tex" constraint reflected in
  the controller execution note.
- **Placeholder scan:** none — every step has complete, concrete code.
- **Type consistency:** `_ckpt_name_for(width, layers, seed=0)`'s new
  signature is backward-compatible (old 2-arg calls still work); `main()`'s
  per-variant loop structure mirrors `aggregate.main()`'s per-mode loop
  structure, reusing `mean_std_str` rather than duplicating its formula a
  third time (the whole-branch review of the prior plan flagged this
  duplication as an acceptable-but-noted trade-off; this plan removes it
  where it's now easy to).
