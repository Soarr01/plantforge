"""Discriminating tests for PLANTFORGE.

    python -m plantforge.tests.test_plantforge     (from /home/coder)
"""
import sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from plantforge.families import ALL, FAMILIES, sample, simulate
from plantforge.excitation import EXCITATIONS, generate, open_loop_input
from plantforge.identifiability import identifiability
from plantforge.corpus import gen_cell


# ── Test 1: every family x excitation is finite and parameter-dependent ─────────────
def test_families_finite_and_param_dependent():
    g = torch.Generator().manual_seed(0)
    for fam in ALL:
        p = sample(fam, 8, g)
        for exc in EXCITATIONS:
            u, y = generate(fam, p, exc, 96, 8, 0.05, torch.Generator().manual_seed(3))
            assert torch.isfinite(y).all(), (fam, exc)
        # different instances must produce different outputs (params matter)
        u = open_loop_input("prbs", 96, 8, 0.05, torch.Generator().manual_seed(4))
        y = simulate(fam, p, u, 0.05)
        spread = y.std(dim=1).mean()
        assert spread > 1e-3, f"{fam}: outputs identical across instances"
    print("  PASS  test_families_finite_and_param_dependent")


# ── Test 2: THE multi-rate invariant — same plant + same physical input at dt and
#            dt/4 agree at common instants (exact ZOH / substepped state-NL) ─────────
def test_multirate_ground_truth_consistency():
    g = torch.Generator().manual_seed(1)
    for fam in ALL:
        p = sample(fam, 4, g)
        u_c = open_loop_input("prbs", 64, 4, 0.08, torch.Generator().manual_seed(7))
        y_c = simulate(fam, p, u_c, 0.08)
        y_f = simulate(fam, p, u_c.repeat_interleave(4, dim=0), 0.02)
        err = (y_c - y_f[3::4]).abs().max().item()
        scale = y_c.abs().max().item() + 1e-6
        assert err / scale < 1e-3, f"{fam}: multirate mismatch {err:.2e} (scale {scale:.2f})"
    print("  PASS  test_multirate_ground_truth_consistency")


# ── Test 3: physical consistency — (u, y) obey the stated parameters (re-simulation
#            with the ground-truth params reproduces y exactly; no hidden normalizer) ─
def test_physical_consistency_of_pairs():
    g = torch.Generator().manual_seed(2)
    for fam in FAMILIES:
        p = sample(fam, 4, g)
        u, y = generate(fam, p, "multisine", 96, 4, 0.05, torch.Generator().manual_seed(5))
        y2 = simulate(fam, p, u, 0.05)
        assert torch.allclose(y, y2, atol=1e-6), f"{fam}: (u,y) not reproducible from params"
    print("  PASS  test_physical_consistency_of_pairs")


# ── Test 4: closed-loop excitation is genuinely closed-loop — the defining property
#            is that u DEPENDS ON THE PLANT (same seed, different plant -> different u),
#            while open-loop u is plant-independent ─────────────────────────────────
def test_closedloop_is_closed():
    p1 = sample("stribeck", 8, torch.Generator().manual_seed(3))
    p2 = sample("stribeck", 8, torch.Generator().manual_seed(30))
    u_ol1, _ = generate("stribeck", p1, "multisine", 128, 8, 0.05,
                        torch.Generator().manual_seed(6))
    u_ol2, _ = generate("stribeck", p2, "multisine", 128, 8, 0.05,
                        torch.Generator().manual_seed(6))
    u_cl1, _ = generate("stribeck", p1, "closedloop", 128, 8, 0.05,
                        torch.Generator().manual_seed(6))
    u_cl2, _ = generate("stribeck", p2, "closedloop", 128, 8, 0.05,
                        torch.Generator().manual_seed(6))
    assert torch.equal(u_ol1, u_ol2), "open-loop u must be plant-independent"
    dev = (u_cl1 - u_cl2).abs().mean() / (u_cl1.abs().mean() + 1e-9)
    assert dev > 0.05, f"closed-loop u must react to the plant: rel-dev {dev:.3f}"
    print(f"  PASS  test_closedloop_is_closed (u plant-dependence rel-dev {dev:.2f})")


# ── Test 5: identifiability annotation is directionally correct — an excitation that
#            never leaves the deadzone makes the backlash width unidentifiable ───────
def test_identifiability_flags_unexcited_parameter():
    B = 6
    g = torch.Generator().manual_seed(4)
    p = sample("backlash", B, g)
    p["db"] = torch.full((B,), 0.6)
    T = 96
    strong = 2.0 * open_loop_input("prbs", T, B, 0.05, torch.Generator().manual_seed(8))
    weak = 0.12 * strong                 # amp 0.24-0.43 < db=0.6: deadzone never exits
    i_strong = identifiability("backlash", p, strong, 0.05)
    i_weak = identifiability("backlash", p, weak, 0.05)
    k = i_strong["keys"].index("db")
    r_strong = i_strong["rel_crlb"][:, k].median()
    r_weak = i_weak["rel_crlb"][:, k].median()
    assert r_weak > 5 * r_strong, \
        f"deadzone width should be far less identifiable under weak excitation: " \
        f"{r_weak:.2f} vs {r_strong:.2f}"
    print(f"  PASS  test_identifiability_flags_unexcited_parameter "
          f"(rel-CRLB weak {r_weak:.2f} vs strong {r_strong:.2f})")


# ── Test 6: corpus shard roundtrip ───────────────────────────────────────────────────
def test_corpus_cell_roundtrip():
    from plantforge.corpus import gen_cell
    shard = gen_cell("saturate", "multisine", 0.05, n_inst=16, ident=False)
    with tempfile.TemporaryDirectory() as d:
        tmp = str(pathlib.Path(d) / "pf_cell.pt")
        torch.save(shard, tmp)
        back = torch.load(tmp)
    assert torch.equal(shard["u"], back["u"]) and torch.equal(shard["theta"], back["theta"])
    y2 = simulate("saturate",
                  {k: back["theta"][:, i] for i, k in enumerate(back["keys"])},
                  back["u"], back["dt"])
    assert torch.allclose(back["y"], y2, atol=1e-6), "shard not reproducible from theta"
    print("  PASS  test_corpus_cell_roundtrip")


def test_gen_cell_seeds_are_deterministic_across_hash_randomization():
    """gen_cell's per-instance draws must not depend on Python's
    per-process-randomized hash() -- two calls with the same explicit seed
    must produce identical output regardless of hash seed randomization
    (simulated here by calling gen_cell twice in the same process, which
    would already differ under PYTHONHASHSEED=random if hash() were still
    used, since hash('stribeck') is stable WITHIN one process but not
    ACROSS processes -- the real regression this guards is a fixed,
    order-stable index lookup, not process-level hash volatility, which a
    single-process test can only partially exercise. The stronger
    guarantee -- byte-identical shards from two separate `python -m
    plantforge.corpus` invocations -- is verified operationally, not by
    this unit test)."""
    shard1 = gen_cell("stribeck", "multisine", 0.05, n_inst=16, seed=42, ident=False)
    shard2 = gen_cell("stribeck", "multisine", 0.05, n_inst=16, seed=42, ident=False)
    assert torch.equal(shard1["u"], shard2["u"]), \
        "same explicit seed must give identical output within the same process"
    assert torch.equal(shard1["theta"], shard2["theta"])
    print("  PASS  test_gen_cell_seeds_are_deterministic_across_hash_randomization")


def test_gen_cell_different_families_get_different_seeds():
    """Two different families with the same seed/lo must NOT draw the same
    parameter values (would happen if the family-index lookup collided).

    Checked via `y`, not `u`: for "prbs" (open-loop) excitation, `u` is
    plant-independent by construction (same property test_closedloop_is_closed
    relies on), so it is identical across families regardless of whether the
    family-index seed lookup collides -- comparing `u` here would never catch
    a real collision. `y` depends on the family's dynamics/parameters even
    when `u` is shared, so it is the field that actually exercises the
    family-index lookup."""
    shard_a = gen_cell("stribeck", "prbs", 0.05, n_inst=16, seed=0, ident=False)
    shard_b = gen_cell("saturate", "prbs", 0.05, n_inst=16, seed=0, ident=False)
    assert not torch.equal(shard_a["y"], shard_b["y"]), \
        "different families must not produce identical trajectories"
    print("  PASS  test_gen_cell_different_families_get_different_seeds")


def test_gen_cell_returns_exactly_n_inst_even_with_divergence():
    """gen_cell must always return exactly n_inst instances, even if some
    draws diverge to non-finite values and must be retried."""
    for fam in FAMILIES:
        shard = gen_cell(fam, "closedloop", 0.05, n_inst=32, seed=0, ident=False)
        assert shard["u"].shape[1] == 32, (fam, shard["u"].shape)
        assert torch.isfinite(shard["u"]).all(), f"{fam}: non-finite u in returned shard"
        assert torch.isfinite(shard["y"]).all(), f"{fam}: non-finite y in returned shard"
    print("  PASS  test_gen_cell_returns_exactly_n_inst_even_with_divergence")


def _run_all():
    test_families_finite_and_param_dependent()
    test_multirate_ground_truth_consistency()
    test_physical_consistency_of_pairs()
    test_closedloop_is_closed()
    test_identifiability_flags_unexcited_parameter()
    test_corpus_cell_roundtrip()
    test_gen_cell_seeds_are_deterministic_across_hash_randomization()
    test_gen_cell_different_families_get_different_seeds()
    test_gen_cell_returns_exactly_n_inst_even_with_divergence()


if __name__ == "__main__":
    _run_all()
