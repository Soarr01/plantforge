"""Offline tests for plantforge.ablation's pure helpers -- no checkpoints,
no GPU, no network.

    python -m plantforge.tests.test_ablation     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from plantforge.ablation import VARIANTS, param_count, load_variant, _ckpt_name_for


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


def _run_all():
    test_variants_include_default_and_four_others()
    test_param_count_monotone_in_width_and_layers()
    test_load_variant_missing_checkpoint_returns_none()
    test_ckpt_name_for_seed_parameter()


if __name__ == "__main__":
    print("PLANTFORGE ablation -- offline tests:")
    _run_all()
