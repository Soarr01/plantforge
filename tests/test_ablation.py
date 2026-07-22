"""Offline tests for plantforge.ablation's pure helpers -- no checkpoints,
no GPU, no network.

    python -m plantforge.tests.test_ablation     (from project root)
"""
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import io
import contextlib
import os

import torch

from plantforge.ablation import (
    VARIANTS, param_count, load_variant, _ckpt_name_for,
    _has_finite_weights, _finished_variant_models,
)
from plantforge.evaluate import InContextSysID, TRAIN_RATES


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


def test_ckpt_name_for_seed_parameter():
    assert _ckpt_name_for(160, 5, seed=0) == "eval_corpus_s0.pt"
    assert _ckpt_name_for(160, 5, seed=3) == "eval_corpus_s3.pt"
    assert _ckpt_name_for(80, 5, seed=2) == "eval_corpus_s2_d80L5.pt"
    print("  PASS  test_ckpt_name_for_seed_parameter")


def test_has_finite_weights_true_for_fresh_model():
    model = InContextSysID()
    assert _has_finite_weights(model) is True
    print("  PASS  test_has_finite_weights_true_for_fresh_model")


def test_has_finite_weights_false_when_a_parameter_is_nan():
    model = InContextSysID()
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    assert _has_finite_weights(model) is False
    print("  PASS  test_has_finite_weights_false_when_a_parameter_is_nan")


def test_finished_variant_models_skips_diverged_checkpoint():
    """A checkpoint that reached TOTAL_STEPS but has NaN weights (a real
    failure mode observed during the 2026-07-21 retraining campaign: the
    'wide' variant's seed 2 diverged mid-training) must be skipped, not
    silently included in the aggregated report."""
    import plantforge.ablation as abl
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = pathlib.Path(tmp)
        model = InContextSysID()
        with torch.no_grad():
            next(model.parameters()).fill_(float("nan"))
        ckpt_name = _ckpt_name_for(160, 5, seed=0)
        torch.save({"model": model.state_dict(), "step": abl.TOTAL_STEPS}, tmp_dir / ckpt_name)

        original_ckpt_dir = abl.CKPT_DIR
        abl.CKPT_DIR = tmp_dir
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                models = _finished_variant_models("default", 160, 5, [0])
            assert models == []
            assert "diverged to non-finite weights" in buf.getvalue()
        finally:
            abl.CKPT_DIR = original_ckpt_dir
    print("  PASS  test_finished_variant_models_skips_diverged_checkpoint")


def test_main_reference_averages_over_train_rates():
    """The bug this task fixes: ablation.py's 'reference' used to be
    nmse(m, 'stribeck', 'multisine', 0.05) -- dt=0.05 is a held-out rate.
    main() must now average over TRAIN_RATES (0.10, 0.02) for the reference,
    which changes the printed 'reference:' value and the 'x ref' ratio."""
    import plantforge.ablation as abl
    from plantforge.evaluate import TRAIN_RATES

    calls = []

    def fake_nmse(model, fam, exc, dt):
        calls.append((fam, exc, dt))
        return dt  # deterministic: returned value IS dt

    def fake_finished_variant_models(name, width, layers, seeds):
        return [object()] if name == "default" else []  # one fake "model" for default only

    original_nmse = abl.nmse
    original_finished = abl._finished_variant_models
    abl.nmse = fake_nmse
    abl._finished_variant_models = fake_finished_variant_models
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            abl.main()
    finally:
        abl.nmse = original_nmse
        abl._finished_variant_models = original_finished

    ref_calls = [c for c in calls if c[0] == "stribeck" and c[1] == "multisine"
                 and c[2] in TRAIN_RATES]
    assert len(ref_calls) >= 2, f"expected reference calls at TRAIN_RATES, got {calls}"
    assert not any(c[0] == "stribeck" and c[2] == 0.05 for c in calls), \
        "reference must not be computed at the held-out rate 0.05"
    print("  PASS  test_main_reference_averages_over_train_rates")


def _run_all():
    test_variants_include_default_and_four_others()
    test_param_count_monotone_in_width_and_layers()
    test_load_variant_missing_checkpoint_returns_none()
    test_ckpt_name_for_seed_parameter()
    test_has_finite_weights_true_for_fresh_model()
    test_has_finite_weights_false_when_a_parameter_is_nan()
    test_finished_variant_models_skips_diverged_checkpoint()
    test_main_reference_averages_over_train_rates()


if __name__ == "__main__":
    print("PLANTFORGE ablation -- offline tests:")
    _run_all()
