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


def _has_finite_weights(model) -> bool:
    """True iff every learned parameter (not buffers -- the model's causal
    mask buffer is legitimately -inf by design) is finite. A fully-NaN
    checkpoint (training diverged and never recovered) still satisfies
    step >= TOTAL_STEPS, so this is a separate check from "finished"."""
    return all(torch.isfinite(p).all() for p in model.parameters())


def _finished_variant_models(name: str, width: int, layers: int, seeds):
    """Load every finished (step >= TOTAL_STEPS) checkpoint with finite
    weights for this variant across the given seeds; print a skip note for
    missing/unfinished/diverged ones."""
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
        model = load_variant(ckpt_name)
        if not _has_finite_weights(model):
            print(f"    (seed {seed}: {ckpt_name} diverged to non-finite weights -- skipped)")
            continue
        models.append(model)
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
