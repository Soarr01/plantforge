"""Offline tests for plantforge.evaluate's pure helpers -- no GPU training,
no checkpoints, no network.

    python -m plantforge.tests.test_evaluate     (from project root)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from plantforge.evaluate import _norm, T_CTX


def test_norm_uses_context_only_std():
    """_norm's scale factor must come from the first T_CTX samples only --
    changing the query-horizon (last T_QRY) samples must not change the
    normalization applied to the context portion."""
    torch.manual_seed(0)
    u = torch.randn(4, 224)
    y = torch.randn(4, 224)
    u1, y1 = _norm(u.clone(), y.clone())

    u_altered_query = u.clone()
    y_altered_query = y.clone()
    u_altered_query[:, T_CTX:] *= 1000.0
    y_altered_query[:, T_CTX:] *= 1000.0
    u2, y2 = _norm(u_altered_query, y_altered_query)

    # Context portion (same raw values in both calls) must normalize
    # identically -- only reachable if the scale factor ignores the query.
    assert torch.allclose(u1[:, :T_CTX], u2[:, :T_CTX], atol=1e-5), \
        "context normalization changed when only query values changed -- leak"
    assert torch.allclose(y1[:, :T_CTX], y2[:, :T_CTX], atol=1e-5), \
        "context normalization changed when only query values changed -- leak"
    print("  PASS  test_norm_uses_context_only_std")


def test_norm_context_std_is_approximately_one():
    """Sanity check: normalizing by the context's own std should leave the
    context portion with std close to 1 (not the full-sequence std, which
    would differ if the query portion has different scale)."""
    torch.manual_seed(1)
    u = torch.randn(8, 192) * 3.0
    u_query = torch.randn(8, 32) * 0.01   # very different scale from context
    u_full = torch.cat([u, u_query], dim=1)
    y_full = torch.randn(8, 224)

    u_n, _ = _norm(u_full, y_full)
    ctx_std = u_n[:, :T_CTX].std(dim=1)
    assert torch.allclose(ctx_std, torch.ones(8), atol=0.15), \
        f"context std after norm should be ~1, got {ctx_std}"
    print("  PASS  test_norm_context_std_is_approximately_one")


def _run_all():
    test_norm_uses_context_only_std()
    test_norm_context_std_is_approximately_one()


if __name__ == "__main__":
    print("PLANTFORGE evaluate -- offline tests:")
    _run_all()
