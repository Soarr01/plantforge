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
