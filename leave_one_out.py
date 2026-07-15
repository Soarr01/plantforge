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
