# Leave-one-family-out multi-seed follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confirm the single-seed leave-one-family-out ranking (backlash and
drivetrain tied hardest) at 5-seed scale, by training seeds 1-4 for the 4
non-default held-out families and aggregating mean+-std in `leave_one_out.py`.

**Architecture:** One new resumable/stall-guarded shell script trains the 16
missing (seed, family) checkpoints. `leave_one_out.py` is refactored so both
the `backlash` baseline and the 4 other families go through one shared
`_report_family` helper that loads however many finished seeds exist per
family and prints `mean_std_str`-formatted reference/held-out numbers,
replacing the current hardcoded "seed 0 only" path for the 4 non-default
families.

**Tech Stack:** Python 3, PyTorch, bash. Reuses `evaluate.py`'s
`InContextSysID`/`nmse`/`FAMILIES`/`CKPT_DIR`/`DEV` and `aggregate.py`'s
`mean_std_str` — no new dependencies.

## Global Constraints

- Checkpoint naming stays byte-identical to the existing suffix rule:
  `eval_corpus_s{seed}_ho{FAMILY_UPPER}.pt` for non-default held families,
  `eval_corpus_s{seed}.pt` for `backlash`. Do not invent a second naming
  convention.
- Do not modify `evaluate.py`, `families.py`, `ablation.py`, `aggregate.py`,
  `realbench.py`, `baselines.py`, `ident_exp.py`, `paper/main.tex`.
- `backlash`'s existing 5-seed checkpoints and the numbers computed from
  them must not change — same seeds (0-4), same metric
  (`reference_and_heldout`), same `_ckpt_name_for` output
  (`eval_corpus_s{seed}.pt`).
- Reuse `evaluate.nmse`, `evaluate.FAMILIES`, `evaluate.CKPT_DIR`,
  `evaluate.DEV`, `evaluate.InContextSysID`, `aggregate.mean_std_str`. Do
  not reimplement loading, nMSE, or mean/std logic.
- New test modules are wired into `tests/run_all.py` ONLY at the final
  whole-branch review stage, never within an individual task (established
  convention in this repo — see `.superpowers/sdd/progress.md`).

---

### Task 1: Multi-seed training driver

**Files:**
- Create: `scripts/train_leave_one_out_seeds.sh`

**Interfaces:**
- Consumes: `plantforge.evaluate` CLI (`python -m plantforge.evaluate corpus`)
  driven by env vars `PF_SEED`, `PF_HOLD_FAMILY`, `PLANTFORGE_DATA`,
  `PF_BUDGET` — same contract `scripts/train_leave_one_out.sh` and
  `scripts/train_ablation_seeds.sh` already use.
- Produces: checkpoints at
  `$PLANTFORGE_DATA/eval_corpus_s{seed}_ho{FAMILY_UPPER}.pt` for
  `seed in {1,2,3,4}`, `family in {stribeck,saturate,boucwen,drivetrain}` —
  the exact filenames `leave_one_out._ckpt_name_for` (Task 2) expects.

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# Train seeds 1-4 (default: $SEEDS) for the 4 non-default held-out families
# (stribeck, saturate, boucwen, drivetrain -- corpus recipe) to 10000 steps
# each. backlash already has 5 seeds trained and is not retrained here.
# Resumable, stall-guarded, skips finished (seed, family) pairs without
# invoking evaluate.py.
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

for seed in $SEEDS; do
    for fam in $FAMILIES; do
        fam_upper=$(echo "$fam" | tr '[:lower:]' '[:upper:]')
        ck="$PLANTFORGE_DATA/eval_corpus_s${seed}_ho${fam_upper}.pt"
        export PF_SEED=$seed
        export PF_HOLD_FAMILY=$fam
        prev_steps=-1
        while true; do
            steps=$(done_steps "$ck")
            if [ "$steps" -ge "$TOTAL" ]; then
                echo "== seed $seed hold=$fam: done ($steps/$TOTAL)"
                break
            fi
            if [ "$steps" -le "$prev_steps" ]; then
                echo "== STALLED seed $seed hold=$fam: step stuck at $steps after a successful run"
                exit 1
            fi
            prev_steps=$steps
            echo "== seed $seed hold=$fam: at $steps/$TOTAL, training..."
            python -m plantforge.evaluate corpus \
                || { echo "== FAILED seed $seed hold=$fam"; exit 1; }
        done
    done
done
echo "== ALL SEEDS x FAMILIES DONE"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/train_leave_one_out_seeds.sh`

- [ ] **Step 3: Verify the checkpoint path matches Task 2's `_ckpt_name_for`**

Run this from the repo root (no GPU, no training — just checks the string
the script would construct for `seed=1 fam=stribeck` against what
`leave_one_out._ckpt_name_for(held_family="stribeck", seed=1)` returns
after Task 2 lands; if Task 2 hasn't landed yet, compare by hand against
the existing rule in `leave_one_out.py`'s current `_ckpt_name_for` —
`"" if held_family == "backlash" else f"_ho{held_family.upper()}"`
composed as `f"eval_corpus_s{seed}{suffix}.pt"`):

```bash
python3 -c "
fam = 'stribeck'; seed = 1
fam_upper = fam.upper()
script_name = f'eval_corpus_s{seed}_ho{fam_upper}.pt'
expected = f'eval_corpus_s{seed}_ho{fam.upper()}.pt'
assert script_name == expected, (script_name, expected)
print('OK', script_name)
"
```
Expected: `OK eval_corpus_s1_hoSTRIBECK.pt`

- [ ] **Step 4: Commit**

```bash
git add scripts/train_leave_one_out_seeds.sh
git commit -m "Add multi-seed training driver for leave-one-family-out sweep"
```

---

### Task 2: Multi-seed aggregation in `leave_one_out.py`

**Files:**
- Modify: `leave_one_out.py` (whole file — see below)
- Test: `tests/test_leave_one_out.py` (add tests; do not remove existing ones)

**Interfaces:**
- Consumes: `evaluate.InContextSysID`, `evaluate.nmse`, `evaluate.FAMILIES`,
  `evaluate.CKPT_DIR`, `evaluate.DEV`, `aggregate.mean_std_str(values: list) -> str`
  (formats `"{mean:.4f} ± {std:.4f} (n={n})"`, `std=0.0` when `n==1`).
- Produces: `HOLD_CHOICES`, `_ckpt_name_for(held_family, seed=0)`,
  `load_variant(ckpt_name)`, `reference_and_heldout(model, held_family)` —
  all UNCHANGED signatures (existing tests in
  `tests/test_leave_one_out.py` import these three by name: `HOLD_CHOICES`,
  `_ckpt_name_for`, `load_variant`). New: `ALL_SEEDS = range(5)`,
  `_finished_models_for(held_family: str, seeds) -> list`,
  `_report_family(held_family: str, seeds, label: str) -> None`.

- [ ] **Step 1: Replace the file contents**

```python
"""Leave-one-family-out sweep: is the paper's held-out family choice
(backlash) representative of the corpus's hardest case, or an outlier?
Trains one model per alternative held-out family (corpus recipe) and
compares each to the already-trained 5-seed backlash baseline, aggregating
mean+-std over however many finished seeds exist per family.

Deliberately does NOT use evaluate.report() (which hardcodes "stribeck" as
the reference family -- invalid when stribeck itself is held out). Instead
defines its own reference metric: mean nMSE over whichever 4 families were
actually trained, which is well-defined for any held-out choice.

    PLANTFORGE_DATA=... python -m plantforge.leave_one_out
"""
from __future__ import annotations

import torch

from .evaluate import InContextSysID, nmse, FAMILIES, CKPT_DIR, DEV
from .aggregate import mean_std_str

HOLD_CHOICES = [f for f in FAMILIES if f != "backlash"]
ALL_SEEDS = range(5)
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


def _finished_models_for(held_family: str, seeds):
    """Load every finished (step >= TOTAL_STEPS) checkpoint for this
    held-out family across the given seeds; print a skip note for
    missing/unfinished ones."""
    models = []
    for seed in seeds:
        ckpt_name = _ckpt_name_for(held_family, seed)
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


def _report_family(held_family: str, seeds, label: str):
    print(f"  {label}:")
    models = _finished_models_for(held_family, seeds)
    if not models:
        print("    MISSING -- run scripts/train_leave_one_out.sh "
              "and scripts/train_leave_one_out_seeds.sh")
        return
    refs, outs = [], []
    for m in models:
        r, o = reference_and_heldout(m, held_family)
        refs.append(r)
        outs.append(o)
    ref_mean = sum(refs) / len(refs)
    out_mean = sum(outs) / len(outs)
    print(f"    reference: {mean_std_str(refs)}")
    print(f"    held_out:  {mean_std_str(outs)}  ({out_mean / ref_mean:.1f}x ref)")


def main():
    print("=== leave-one-family-out sweep: corpus recipe, "
          "reference = mean nMSE over 4 trained families ===")
    _report_family("backlash", ALL_SEEDS, "backlash (existing baseline)")
    for held_family in HOLD_CHOICES:
        _report_family(held_family, ALL_SEEDS, held_family)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add tests for the new helpers**

This module runs standalone via `python -m plantforge.tests.test_leave_one_out`
(no pytest fixture injection), so use stdlib `io.StringIO` +
`contextlib.redirect_stdout` to capture output rather than a `capsys` fixture.

Add these imports at the top of `tests/test_leave_one_out.py`, alongside the
existing ones:

```python
import io
import contextlib

from plantforge.leave_one_out import _finished_models_for, _report_family
```

Append these two new test functions (keep the three existing test functions
untouched):

```python
def test_finished_models_for_missing_family_returns_empty_list():
    models = _finished_models_for("this_family_has_no_checkpoints", [0, 1, 2])
    assert models == []
    print("  PASS  test_finished_models_for_missing_family_returns_empty_list")


def test_report_family_missing_prints_message_and_does_not_raise():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _report_family("this_family_has_no_checkpoints", [0, 1, 2], "nonexistent")
    assert "MISSING" in buf.getvalue()
    print("  PASS  test_report_family_missing_prints_message_and_does_not_raise")
```

Update `_run_all` in the same file to also call these two new functions:

```python
def _run_all():
    test_hold_choices_are_the_four_non_default_families()
    test_ckpt_name_for_matches_evaluate_suffix_rule()
    test_load_variant_missing_checkpoint_returns_none()
    test_finished_models_for_missing_family_returns_empty_list()
    test_report_family_missing_prints_message_and_does_not_raise()
```

- [ ] **Step 3: Run the test module directly**

Run: `cd /data/nas07_new/PersonalData/phuocthien/plantforge && PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data python -m plantforge.tests.test_leave_one_out`

Expected: all 5 tests print `PASS`, no traceback.

- [ ] **Step 4: Run the existing offline CLI smoke check**

Run: `cd /data/nas07_new/PersonalData/phuocthien/plantforge && PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data python -m plantforge.leave_one_out`

Expected: prints the `backlash (existing baseline)` block with the same
5-seed numbers as before (reference/held_out via `mean_std_str`, matching
the previously reported `reference=0.0600 held_out=0.2899` within
formatting), then one block per `HOLD_CHOICES` family — each still `n=1`
at this point since Task 1's training hasn't run yet, printing `(seed 1:
... missing -- skipped)` etc. for seeds 1-4. No traceback.

- [ ] **Step 5: Commit**

```bash
git add leave_one_out.py tests/test_leave_one_out.py
git commit -m "Aggregate multi-seed leave-one-out results via shared _report_family helper"
```

---

## After both tasks: training + final report (controller steps, not a task)

These are not implementer-subagent tasks — the controller runs them
directly after both tasks pass review, per this repo's established
pattern (training runs and `tests/run_all.py` wiring happen at the
controller/final-review stage, not inside a task):

1. Wire `test_leave_one_out` into `tests/run_all.py` if not already
   present (it already is, from the prior plan — verify, don't re-add).
2. Launch `scripts/train_leave_one_out_seeds.sh` on a free GPU, monitor to
   completion (16 runs).
3. Run `python -m plantforge.leave_one_out` once training finishes and
   record the 5-seed numbers in a new results doc.
