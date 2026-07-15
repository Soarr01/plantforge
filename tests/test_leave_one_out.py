"""Offline tests for plantforge.leave_one_out's pure helpers -- no
checkpoints, no GPU, no network.

    python -m plantforge.tests.test_leave_one_out     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import io
import contextlib

from plantforge.leave_one_out import HOLD_CHOICES, _ckpt_name_for, load_variant, _finished_models_for, _report_family
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


def _run_all():
    test_hold_choices_are_the_four_non_default_families()
    test_ckpt_name_for_matches_evaluate_suffix_rule()
    test_load_variant_missing_checkpoint_returns_none()
    test_finished_models_for_missing_family_returns_empty_list()
    test_report_family_missing_prints_message_and_does_not_raise()


if __name__ == "__main__":
    print("PLANTFORGE leave_one_out -- offline tests:")
    _run_all()
