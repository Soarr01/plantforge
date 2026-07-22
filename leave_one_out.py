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

from .evaluate import InContextSysID, nmse, FAMILIES, TRAIN_RATES, CKPT_DIR, DEV
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


def _has_finite_weights(model) -> bool:
    """True iff every learned parameter (not buffers -- the model's causal
    mask buffer is legitimately -inf by design) is finite. A fully-NaN
    checkpoint (training diverged and never recovered) still satisfies
    step >= TOTAL_STEPS, so this is a separate check from "finished"."""
    return all(torch.isfinite(p).all() for p in model.parameters())


def _finished_models_for(held_family: str, seeds):
    """Load every finished (step >= TOTAL_STEPS) checkpoint with finite
    weights for this held-out family across the given seeds; print a skip
    note for missing/unfinished/diverged ones."""
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
        model = load_variant(ckpt_name)
        if not _has_finite_weights(model):
            print(f"    (seed {seed}: {ckpt_name} diverged to non-finite weights -- skipped)")
            continue
        models.append(model)
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
