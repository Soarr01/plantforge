# Leave-one-family-out sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer the paper's own flagged Limitation ("we have not verified
whether `backlash` is representative of the corpus's hardest held-out-family
case or an outlier") by training 4 new single-seed models, one per
alternative held-out family, and comparing to the already-trained 5-seed
`backlash` baseline.

**Architecture:** Minimal `HOLD_FAMILY` parameterization of `evaluate.py`
(env var, same pattern as `PF_SEED`/`PF_WIDTH`/`PF_LAYERS`); a new
comparison script `leave_one_out.py` that defines its own reference metric
(deliberately NOT reusing `evaluate.report()`, which hardcodes `stribeck` as
reference family — invalid when `stribeck` is the held-out choice); a new
resumable training driver.

**Tech Stack:** Python, PyTorch, bash (all already in use, no new dependencies).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-15-leave-one-family-out-design.md` — binding.
- Reuse `evaluate.InContextSysID`, `evaluate.nmse`, `evaluate.FAMILIES`,
  `evaluate.CKPT_DIR`, `evaluate.DEV` — no reimplementing training or eval.
- **Do not modify `evaluate.report()`** — it hardcodes `"stribeck"` as
  reference family and stays that way; the new module defines its own
  reference metric instead.
- Do not modify `families.py`, `excitation.py`, `identifiability.py`,
  `corpus.py`, `realbench.py`, `aggregate.py`, `baselines.py`,
  `ablation.py`, `ident_exp.py`, `paper/main.tex`.
- Default (`HOLD_FAMILY="backlash"`, the paper's existing setting) checkpoint
  names MUST stay unchanged — the already-trained 5-seed `backlash`
  checkpoints (`eval_corpus_s{0..4}.pt`) must remain loadable with zero
  retraining.
- Non-default held family gets a checkpoint suffix `_ho{FAMILY_UPPER}`,
  composing correctly with the existing width/layers suffix (e.g. a
  non-default-architecture, non-default-held-family run would be
  `eval_corpus_s0_d80L5_hoSTRIBECK.pt`) — only appended when
  `PF_HOLD_FAMILY` differs from the default `"backlash"`.
- **Known non-blocking cosmetic issue, not to be "fixed":** when the training
  driver runs `python -m plantforge.evaluate corpus` with
  `PF_HOLD_FAMILY=stribeck`, the script's own `__main__` block calls
  `report()`, which hardcodes `nmse(model, "stribeck", ...)` as its
  "reference (train-like)" line. This still executes without error (`nmse`
  works on any family regardless of training), but prints a semantically
  misleading number in the training log for that one run (evaluating
  out-of-distribution `stribeck` performance under a label that says
  "train-like"). This is expected, harmless to this plan's actual results
  (which come from `leave_one_out.py`'s own correctly-defined reference
  metric, never from parsing `report()`'s printed text), and explicitly
  NOT something any task in this plan should attempt to fix by touching
  `report()`.
- Git repo root `/data/nas07_new/PersonalData/phuocthien/plantforge`, branch
  `leave-one-family-out` off `main`. `git` commands run with cwd at the
  repo root; `python -m plantforge...` commands run with cwd
  `/data/nas07_new/PersonalData/phuocthien`. `PLANTFORGE_DATA` env var
  convention unchanged.

---

### Task 1: `PF_HOLD_FAMILY` parameterization of `evaluate.py`

**Files:**
- Modify: `plantforge/evaluate.py`

**Interfaces:**
- Produces: `evaluate.HOLD_FAMILY` (str, env `PF_HOLD_FAMILY`, default
  `"backlash"`); `_ckpt_name(mode)` gains the `_ho{FAMILY}` suffix component.

- [ ] **Step 1: Change the `HOLD_FAMILY` constant**

Replace:

```python
HOLD_FAMILY = "backlash"            # held-out family (the hardest per the gate)
```

with:

```python
HOLD_FAMILY = os.environ.get("PF_HOLD_FAMILY", "backlash")   # held-out family
```

- [ ] **Step 2: Extend `_ckpt_name`'s suffix logic**

Replace:

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

with:

```python
def _ckpt_name(mode: str) -> str:
    """eval_{mode}_s{SEED}.pt for the default architecture (160/5) and
    default held-out family (backlash) -- unchanged from before this plan,
    so the already-trained default checkpoints stay loadable with zero
    retraining. Non-default width/layers and/or non-default held family
    each add an explicit suffix component so they never collide with the
    default name or each other."""
    suffix = ""
    if (WIDTH, LAYERS) != (160, 5):
        suffix += f"_d{WIDTH}L{LAYERS}"
    if HOLD_FAMILY != "backlash":
        suffix += f"_ho{HOLD_FAMILY.upper()}"
    return f"eval_{mode}_s{SEED}{suffix}.pt"
```

- [ ] **Step 3: Verify no behavior change at defaults, and correct suffix composition**

```bash
cd /data/nas07_new/PersonalData/phuocthien && python -c "
from plantforge.evaluate import _ckpt_name, HOLD_FAMILY
assert HOLD_FAMILY == 'backlash'
assert _ckpt_name('corpus') == 'eval_corpus_s0.pt'
print('default unchanged:', _ckpt_name('corpus'))
"
cd /data/nas07_new/PersonalData/phuocthien && python -c "
import os
os.environ['PF_HOLD_FAMILY'] = 'stribeck'
from plantforge.evaluate import _ckpt_name
assert _ckpt_name('corpus') == 'eval_corpus_s0_hoSTRIBECK.pt', _ckpt_name('corpus')
print('non-default held family:', _ckpt_name('corpus'))
"
cd /data/nas07_new/PersonalData/phuocthien && python -c "
import os
os.environ['PF_WIDTH'] = '80'; os.environ['PF_HOLD_FAMILY'] = 'drivetrain'
from plantforge.evaluate import _ckpt_name
assert _ckpt_name('corpus') == 'eval_corpus_s0_d80L5_hoDRIVETRAIN.pt', _ckpt_name('corpus')
print('composed suffix:', _ckpt_name('corpus'))
"
cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.run_all 2>&1 | tail -5
```

Expected: all three assert blocks print without error, and the full offline
suite (34 tests) still passes — this task changes no test-covered behavior
at default settings.

- [ ] **Step 4: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add evaluate.py
git commit -m "Parameterize held-out family via PF_HOLD_FAMILY for leave-one-family-out sweep"
```

---

### Task 2: `leave_one_out.py` — comparison across held-out-family choices

**Files:**
- Create: `plantforge/leave_one_out.py`
- Test: `plantforge/tests/test_leave_one_out.py`

**Interfaces:**
- Consumes: `evaluate.InContextSysID`, `evaluate.nmse`, `evaluate.FAMILIES`,
  `evaluate.CKPT_DIR`, `evaluate.DEV`.
- Produces: `HOLD_CHOICES` (list of the 4 non-default families, plus
  `"backlash"` handled specially as the existing 5-seed baseline);
  `_ckpt_name_for(held_family, seed=0) -> str`; `load_variant(ckpt_name) ->
  InContextSysID | None`; `reference_and_heldout(model, held_family) ->
  tuple[float, float]` (mean-over-other-4-families nMSE, held-out-family
  nMSE); CLI `python -m plantforge.leave_one_out`.

- [ ] **Step 1: Write the failing tests**

Create `plantforge/tests/test_leave_one_out.py`:

```python
"""Offline tests for plantforge.leave_one_out's pure helpers -- no
checkpoints, no GPU, no network.

    python -m plantforge.tests.test_leave_one_out     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from plantforge.leave_one_out import HOLD_CHOICES, _ckpt_name_for, load_variant
from plantforge.evaluate import FAMILIES


def test_hold_choices_are_the_four_non_default_families():
    assert set(HOLD_CHOICES) == set(FAMILIES) - {"backlash"}
    assert len(HOLD_CHOICES) == 4
    print("  PASS  test_hold_choices_are_the_four_non_default_families")


def test_ckpt_name_for_matches_evaluate_suffix_rule():
    assert _ckpt_name_for("backlash", seed=0) == "eval_corpus_s0.pt"
    assert _ckpt_name_for("backlash", seed=3) == "eval_corpus_s3.pt"
    assert _ckpt_name_for("stribeck", seed=0) == "eval_corpus_s0_hoSTRIBECK.pt"
    assert _ckpt_name_for("drivetrain", seed=0) == "eval_corpus_s0_hoDRIVETRAIN.pt"
    print("  PASS  test_ckpt_name_for_matches_evaluate_suffix_rule")


def test_load_variant_missing_checkpoint_returns_none():
    assert load_variant("eval_this_checkpoint_does_not_exist.pt") is None
    print("  PASS  test_load_variant_missing_checkpoint_returns_none")


def _run_all():
    test_hold_choices_are_the_four_non_default_families()
    test_ckpt_name_for_matches_evaluate_suffix_rule()
    test_load_variant_missing_checkpoint_returns_none()


if __name__ == "__main__":
    print("PLANTFORGE leave_one_out -- offline tests:")
    _run_all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_leave_one_out`
Expected: `ModuleNotFoundError: No module named 'plantforge.leave_one_out'`

- [ ] **Step 3: Write `plantforge/leave_one_out.py`**

```python
"""Leave-one-family-out sweep: is the paper's held-out family choice
(backlash) representative of the corpus's hardest case, or an outlier?
Trains one model per alternative held-out family (single seed 0, corpus
recipe) and compares each to the already-trained 5-seed backlash baseline.

Deliberately does NOT use evaluate.report() (which hardcodes "stribeck" as
the reference family -- invalid when stribeck itself is held out). Instead
defines its own reference metric: mean nMSE over whichever 4 families were
actually trained, which is well-defined for any held-out choice.

    PLANTFORGE_DATA=... python -m plantforge.leave_one_out
"""
from __future__ import annotations

import torch

from .evaluate import InContextSysID, nmse, FAMILIES, CKPT_DIR, DEV

HOLD_CHOICES = [f for f in FAMILIES if f != "backlash"]
BACKLASH_SEEDS = range(5)   # the already-trained 5-seed baseline
TOTAL_STEPS = 10000


def _ckpt_name_for(held_family: str, seed: int = 0) -> str:
    suffix = "" if held_family == "backlash" else f"_ho{held_family.upper()}"
    return f"eval_corpus_s{seed}{suffix}.pt"


def load_variant(ckpt_name: str):
    """Load a trained InContextSysID checkpoint in eval mode, or None if
    missing or not finished (step < TOTAL_STEPS)."""
    ck_path = CKPT_DIR / ckpt_name
    if not ck_path.exists():
        return None
    ck = torch.load(ck_path, map_location=DEV)
    if ck["step"] < TOTAL_STEPS:
        return None
    model = InContextSysID().to(DEV)
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def reference_and_heldout(model, held_family: str):
    """mean nMSE over the 4 trained (non-held) families, and nMSE on the
    held-out family itself, both at dt=0.05, multisine."""
    others = [f for f in FAMILIES if f != held_family]
    ref_vals = [nmse(model, fam, "multisine", 0.05) for fam in others]
    reference = sum(ref_vals) / len(ref_vals)
    held_out = nmse(model, held_family, "multisine", 0.05)
    return reference, held_out


def main():
    print("=== leave-one-family-out sweep: corpus recipe, "
          "reference = mean nMSE over 4 trained families ===")

    print("  backlash (existing 5-seed baseline):")
    refs, outs = [], []
    for seed in BACKLASH_SEEDS:
        model = load_variant(_ckpt_name_for("backlash", seed))
        if model is None:
            print(f"    (seed {seed}: missing or unfinished -- skipped)")
            continue
        r, o = reference_and_heldout(model, "backlash")
        refs.append(r); outs.append(o)
    if refs:
        ref_mean = sum(refs) / len(refs)
        out_mean = sum(outs) / len(outs)
        print(f"    reference={ref_mean:.4f}  held_out={out_mean:.4f}  "
              f"({out_mean/ref_mean:.1f}x ref)  n={len(refs)}")
    else:
        print("    MISSING -- no finished backlash checkpoints found")

    for held_family in HOLD_CHOICES:
        ckpt_name = _ckpt_name_for(held_family, seed=0)
        model = load_variant(ckpt_name)
        print(f"  {held_family} (seed 0):")
        if model is None:
            print(f"    MISSING -- run scripts/train_leave_one_out.sh")
            continue
        r, o = reference_and_heldout(model, held_family)
        print(f"    reference={r:.4f}  held_out={o:.4f}  ({o/r:.1f}x ref)  n=1")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_leave_one_out`
Expected: 3 PASS lines, exit 0.

- [ ] **Step 5: Smoke the CLI against the already-trained backlash checkpoints**

```bash
cd /data/nas07_new/PersonalData/phuocthien && \
PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data python -m plantforge.leave_one_out
```

Expected: `backlash` row shows real numbers with `n=5` (reference and
held_out values in the same ballpark as the paper's Table 1 corpus row,
though NOT expected to match exactly since this task's `reference` metric
--- mean over 4 other families --- differs from the paper's `stribeck`-only
reference); the 4 other rows show `MISSING` (not yet trained). No traceback.

- [ ] **Step 6: Run the full offline suite for import safety**

```bash
cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.run_all
```

Expected: 37 PASS lines total (34 prior + 3 new).

- [ ] **Step 7: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add leave_one_out.py tests/test_leave_one_out.py
git commit -m "Add leave-one-family-out comparison script"
```

---

### Task 3: Training driver for the 4 alternative held-out families

**Files:**
- Create: `plantforge/scripts/train_leave_one_out.sh`

**Interfaces:**
- Consumes: Task 1's `PF_HOLD_FAMILY` env var and `_ckpt_name` naming.
- Produces: a script the controller launches in the background to train
  seed 0 for each of the 4 alternative held-out families to step 10000.

- [ ] **Step 1: Create `plantforge/scripts/train_leave_one_out.sh`**

```bash
#!/bin/bash
# Train seed 0 for each of the 4 alternative held-out families
# (stribeck, saturate, boucwen, drivetrain -- corpus recipe) to 10000
# steps. The existing default (backlash, already 5-seed trained) is not
# retrained. Resumable, stall-guarded, skips finished families without
# invoking evaluate.py, same pattern as scripts/train_ablation.sh.
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
FAMILIES=${FAMILIES:-"stribeck saturate boucwen drivetrain"}

done_steps() {
    python - "$1" <<'EOF'
import sys, torch
try:
    print(torch.load(sys.argv[1], map_location="cpu")["step"])
except Exception:
    print(0)
EOF
}

for fam in $FAMILIES; do
    fam_upper=$(echo "$fam" | tr '[:lower:]' '[:upper:]')
    ck="$PLANTFORGE_DATA/eval_corpus_s0_ho${fam_upper}.pt"
    export PF_HOLD_FAMILY=$fam
    prev_steps=-1
    while true; do
        steps=$(done_steps "$ck")
        if [ "$steps" -ge "$TOTAL" ]; then
            echo "== hold=$fam: done ($steps/$TOTAL)"
            break
        fi
        if [ "$steps" -le "$prev_steps" ]; then
            echo "== STALLED hold=$fam: step stuck at $steps after a successful run"
            exit 1
        fi
        prev_steps=$steps
        echo "== hold=$fam: at $steps/$TOTAL, training..."
        python -m plantforge.evaluate corpus \
            || { echo "== FAILED hold=$fam"; exit 1; }
    done
done
echo "== ALL HELD-OUT FAMILIES DONE"
```

- [ ] **Step 2: Verify naming agreement and syntax (fast, no training)**

```bash
chmod +x /data/nas07_new/PersonalData/phuocthien/plantforge/scripts/train_leave_one_out.sh
bash -n /data/nas07_new/PersonalData/phuocthien/plantforge/scripts/train_leave_one_out.sh && echo "syntax OK"
cd /data/nas07_new/PersonalData/phuocthien && python -c "
import os
os.environ['PF_HOLD_FAMILY'] = 'boucwen'
from plantforge.evaluate import _ckpt_name
assert _ckpt_name('corpus') == 'eval_corpus_s0_hoBOUCWEN.pt', _ckpt_name('corpus')
print('hold=boucwen ->', _ckpt_name('corpus'), '(matches script suffix logic)')
"
```

Do NOT run the script to completion in this step — that starts real GPU
training (4 runs, up to ~40min each, up to ~2.7h worst case). The
controller launches it separately after this task's review passes.

Expected: `syntax OK`, then the printed checkpoint name matching the
script's own `fam_upper` suffix-construction logic.

- [ ] **Step 3: Commit**

```bash
cd /data/nas07_new/PersonalData/phuocthien/plantforge
git add scripts/train_leave_one_out.sh
git commit -m "Add training driver for the leave-one-family-out sweep"
```

**Execution note for the controller:** after all 3 tasks' reviews pass and
the final whole-branch review approves, launch
`nohup .../train_leave_one_out.sh > .../plantforge_data_train_loo.log 2>&1 &`
in the background (4 runs, ~2-3h estimated), monitor to completion, then
run `python -m plantforge.leave_one_out` for the final comparison table and
record it in a results note (not editing `paper/main.tex` in this pass, per
the spec's explicit out-of-scope). Remember the training log's `report()`
output for the `hold=stribeck` run will print a misleadingly-labeled
"reference (train-like)" line (see Global Constraints) — this is expected
and does not affect this plan's actual results.

---

## Self-Review Notes

- **Spec coverage:** Task 1 = `HOLD_FAMILY` parameterization; Task 2 =
  comparison script + tests; Task 3 = training driver. Spec's "do not touch
  `report()`" constraint reflected explicitly in Global Constraints and in
  `leave_one_out.py`'s own docstring/design (defines its own reference
  metric). Spec's "not updating paper/main.tex" constraint reflected in the
  controller execution note.
- **Placeholder scan:** none — every step has complete, concrete code.
- **Type consistency:** `_ckpt_name_for` (Task 2, `leave_one_out.py`) and
  `evaluate._ckpt_name`'s new suffix logic (Task 1) implement the same
  `_ho{FAMILY_UPPER}` rule independently for the same inputs — both produce
  identical strings, exercised by Task 2's own test
  (`test_ckpt_name_for_matches_evaluate_suffix_rule`) and Task 3's manual
  cross-check in Step 2. `load_variant` (Task 2) returns
  `InContextSysID | None` consistently with the pattern established in
  `ablation.py`'s `load_variant`.
