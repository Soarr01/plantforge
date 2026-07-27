# Fix identifiability.py's pinv bug: zero-sensitivity parameters must show large, not zero, rel-CRLB

Status: approved design, not yet implemented.

## Purpose

A full-repo Codex adversarial review (2026-07-27), while investigating
beyond its assigned checklist, found a real correctness bug in
`identifiability.py`: `torch.linalg.pinv(F)` on the Tikhonov-regularized
Fisher information matrix (`F = F + 1e-9 * eye(K)`) treats any
near-singular eigenvalue direction as rank-deficient (null space) and
returns **0** for that direction in the pseudo-inverse, rather than a
large value. This is the exact opposite of correct CRLB semantics: a
parameter with zero (or near-zero) sensitivity to the excitation is
**maximally unidentifiable** and should show a **large** `rel_crlb`, not a
near-zero one.

Independently reproduced (`docs/superpowers/results/` — see the
2026-07-27 investigation trail) and quantified:

```python
F = torch.diag(torch.tensor([1.0, 1e-9]))
torch.linalg.pinv(F).diag()   # [1.0, 0.0]      <- wrong: should be huge, not 0
torch.linalg.inv(F).diag()    # [1.0, 1e9]      <- correct semantics
```

**Scope of impact** (measured directly against the released corpus
shards under `$PLANTFORGE_DATA/corpus/`):
- 22,398 of 1,488,000 total `(instance, parameter)` entries (9.3% of all
  240,000 corpus rows) have a spuriously-zero `rel_crlb` for at least one
  parameter.
- Overwhelmingly concentrated in the `saturate` family's `sat` parameter:
  22,260 of 48,000 `saturate`-family rows (46.4%) are affected — every
  case where the excitation amplitude never reaches the sampled
  saturation limit.
- Smaller counts in `backlash` (`c0`, `c1`, `p1`, `p2` — not `db`, the
  parameter the paper's own Section 3 design-invariant example uses) and
  one `drivetrain` row.

**Effect on the paper's Table 3** (identifiability vs. prediction
difficulty, `ident_exp.py`): `ident_exp.py` does NOT read a shard's
stored `rel_crlb` field — it recomputes `identifiability()` live on the
224-sample prediction window (an earlier, already-merged fix). So fixing
`identifiability.py` alone automatically corrects Table 3, without
needing to touch the released corpus shards at all, for the paper's own
numbers to be correct. A full independent re-analysis (all 40 cells, all
5 seeds, corrected float64 regularized inverse) was run and confirms the
paper's qualitative headline finding survives (weak, negative, not the
hypothesized direction) but the exact reported numbers shift:

| | median r (base) | median r (power-controlled) |
|---|---|---|
| current (buggy pinv) | −0.112 | −0.113 |
| corrected | −0.093 | −0.075 |

**Effect on the released corpus** (separate concern, independent of the
paper): `corpus.py`'s `gen_cell` calls the SAME `identifiability()`
function at generation time to compute the `rel_crlb`/`log10_cond`
fields baked into the shards already published on Hugging Face
(`stark4062/plantforge`). Those stored fields carry the same bug for
downstream users who read them directly rather than recomputing (as
`ident_exp.py` does). User has approved fixing this too (Direction:
regenerate + re-publish), not just leaving it as a paper-only fix.

**Not affected:** the paper's Section 3 "design invariants" example
(`backlash`'s `db` parameter, ~5×10⁴ under a weak-but-nonzero
excitation, ~0.3 under a strong one) — verified directly, unchanged by
the fix, because that example's "weak" excitation still produces
nonzero (if small) sensitivity, keeping the relevant eigenvalue above
the point where `pinv`'s internal rank-detection would zero it.

## Design

### Change 1 — fix `identifiability()`'s inverse (`identifiability.py`)

Replace `torch.linalg.pinv` (float32, SVD-based rank truncation) with an
explicit regularized inverse computed via eigendecomposition in float64
(every eigenvalue direction is inverted — none are treated as null
space, since the added `1e-9` floor already guarantees a well-defined,
if tiny, minimum eigenvalue everywhere):

Current (`identifiability.py`, full function):

```python
@torch.no_grad()
def identifiability(family: str, p: dict, u: torch.Tensor, dt: float):
    """Returns dict: rel_crlb (B,K), log10_cond (B,), keys (list of K names)."""
    theta, keys = param_vector(family, p)              # (B,K)
    B, K = theta.shape
    T = u.shape[0]
    sens = torch.zeros(B, K, T)
    for i, key in enumerate(keys):
        h = (p[key].abs() * REL_STEP).clamp(min=1e-3)
        pp = {k: v.clone() for k, v in p.items()}
        pp[key] = p[key] + h
        y_hi = simulate(family, pp, u, dt)
        pp[key] = p[key] - h
        y_lo = simulate(family, pp, u, dt)
        sens[:, i] = ((y_hi - y_lo) / (2 * h).unsqueeze(0)).t()
    F = torch.einsum("bkt,blt->bkl", sens, sens) / SIGMA_REF ** 2
    F = F + 1e-9 * torch.eye(K)                        # numerical floor
    eigs = torch.linalg.eigvalsh(F)
    log_cond = torch.log10(eigs[:, -1].clamp(min=1e-30) / eigs[:, 0].clamp(min=1e-30))
    Finv = torch.linalg.pinv(F)
    rel_crlb = torch.sqrt(torch.diagonal(Finv, dim1=1, dim2=2).clamp(min=0)) \
        / theta.abs().clamp(min=1e-3)
    return {"rel_crlb": rel_crlb, "log10_cond": log_cond, "keys": keys}
```

New:

```python
@torch.no_grad()
def identifiability(family: str, p: dict, u: torch.Tensor, dt: float):
    """Returns dict: rel_crlb (B,K), log10_cond (B,), keys (list of K names)."""
    theta, keys = param_vector(family, p)              # (B,K)
    B, K = theta.shape
    T = u.shape[0]
    sens = torch.zeros(B, K, T)
    for i, key in enumerate(keys):
        h = (p[key].abs() * REL_STEP).clamp(min=1e-3)
        pp = {k: v.clone() for k, v in p.items()}
        pp[key] = p[key] + h
        y_hi = simulate(family, pp, u, dt)
        pp[key] = p[key] - h
        y_lo = simulate(family, pp, u, dt)
        sens[:, i] = ((y_hi - y_lo) / (2 * h).unsqueeze(0)).t()
    F = torch.einsum("bkt,blt->bkl", sens, sens) / SIGMA_REF ** 2
    # Regularized true inverse via eigendecomposition, in float64. NOT
    # torch.linalg.pinv: pinv's SVD-based rank truncation treats any
    # near-singular eigenvalue direction (e.g. a parameter with ~zero
    # sensitivity, like `sat` when the excitation never reaches the
    # saturation limit) as null space and returns 0 for that entry of the
    # inverse -- the OPPOSITE of correct CRLB semantics: zero sensitivity
    # means maximally UNidentifiable, which must show a LARGE rel_crlb,
    # not a near-zero one. The 1e-9 floor below guarantees every direction
    # has a well-defined (if tiny) eigenvalue, so a true regularized
    # inverse (invert every direction, never truncate) is both correct
    # and numerically safe. float64 avoids float32 precision loss when
    # dividing by eigenvalues as small as 1e-9.
    F64 = F.double() + 1e-9 * torch.eye(K, dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(F64)
    evals = evals.clamp(min=1e-9)
    log_cond = torch.log10(evals[:, -1] / evals[:, 0])
    Finv = (evecs * (1.0 / evals).unsqueeze(1)) @ evecs.transpose(1, 2)
    rel_crlb = torch.sqrt(torch.diagonal(Finv, dim1=1, dim2=2).clamp(min=0)) \
        / theta.double().abs().clamp(min=1e-3)
    return {"rel_crlb": rel_crlb.float(), "log10_cond": log_cond.float(), "keys": keys}
```

Notes:
- Return dtype stays `float32` (`.float()` at the end) so every caller
  (`corpus.py`, `ident_exp.py`) sees an unchanged dtype contract.
- `torch.linalg.eigh` on `F64` (symmetric by construction, `sens @
  sens.T`-derived) is well-defined and replaces both the old
  `eigvalsh` call (for `log_cond`) and the `pinv` call (for
  `rel_crlb`) in one pass — no redundant decomposition.
- Function signature, return dict keys/shapes, and every caller's usage
  are unchanged.

### Change 2 — add a test that catches the exact bug

Add to `tests/test_plantforge.py`, immediately after the existing
`test_identifiability_flags_unexcited_parameter` (which uses a
weak-but-nonzero excitation and therefore does not exercise the
exact-zero-sensitivity path this bug lives in):

```python
# ── Test 5b: a parameter with EXACTLY zero sensitivity (the excitation never
#            reaches it at all, e.g. saturation never engaged) must show a
#            LARGE rel_crlb, not a near-zero one -- the pinv-based inverse
#            used to get this backwards (see 2026-07-27 fix) ───────────────
def test_identifiability_zero_sensitivity_gives_large_not_zero_crlb():
    B = 6
    g = torch.Generator().manual_seed(2)
    p = sample("saturate", B, g)
    p["sat"] = torch.full((B,), 5.0)   # push the saturation limit far out of reach
    T = 96
    u = 0.1 * open_loop_input("prbs", T, B, 0.05, torch.Generator().manual_seed(3))
    idn = identifiability("saturate", p, u, 0.05)
    k = idn["keys"].index("sat")
    assert idn["rel_crlb"][:, k].min() > 100, \
        f"a parameter with zero sensitivity must show a LARGE rel_crlb, not near-zero: " \
        f"{idn['rel_crlb'][:, k].tolist()}"
    print(f"  PASS  test_identifiability_zero_sensitivity_gives_large_not_zero_crlb "
          f"(sat rel-CRLB min {idn['rel_crlb'][:, k].min():.1f})")
```

Register the new test in `test_plantforge.py`'s `_run_all()` (or
equivalent test-runner list), immediately after
`test_identifiability_flags_unexcited_parameter()`.

Verified directly (both by hand-running the exact code above): on the
current buggy `identifiability()`, `rel_crlb[:, k]` is exactly `0.0` for
all 6 instances (test fails); after Change 1, it is `~6324.6` (test
passes, comfortably above the `>100` threshold). The pre-existing
`test_identifiability_flags_unexcited_parameter` still passes after
Change 1 (verified: weak/strong ratio 233×, comfortably above the
required 5×).

### Change 3 — recompute the released corpus's own annotations in place

New file `fix_identifiability_annotations.py` at the repo root (mirrors
every other top-level module's `python -m plantforge.X` invocation
convention):

```python
"""One-time corpus-maintenance pass: recompute rel_crlb/log10_cond in
place on every existing shard under $PLANTFORGE_DATA/corpus/, using the
fixed identifiability() (2026-07-27 pinv-bug fix) -- WITHOUT
re-simulating. Reuses each shard's own u and theta exactly (identical to
what corpus.gen_cell used when the shard was first generated), so u, y,
theta, keys, dt, family, and excitation are byte-identical before and
after; only rel_crlb and log10_cond change.

    PLANTFORGE_DATA=... python -m plantforge.fix_identifiability_annotations
"""
from __future__ import annotations

import pathlib

import torch

from .corpus import OUT
from .identifiability import identifiability


def main():
    paths = sorted(OUT.glob("*.pt"))
    print(f"found {len(paths)} shards under {OUT}")
    total_rows = total_changed = total_zero_before = total_zero_after = 0
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        keys = shard["keys"]
        theta = shard["theta"]
        p = {k: theta[:, i] for i, k in enumerate(keys)}
        old_rel_crlb = shard["rel_crlb"]
        idn = identifiability(shard["family"], p, shard["u"], shard["dt"])
        new_rel_crlb, new_log10_cond = idn["rel_crlb"], idn["log10_cond"]

        changed = int((~torch.isclose(old_rel_crlb, new_rel_crlb, rtol=1e-4, atol=1e-6)).sum())
        zero_before = int((old_rel_crlb == 0).sum())
        zero_after = int((new_rel_crlb == 0).sum())
        total_rows += old_rel_crlb.numel()
        total_changed += changed
        total_zero_before += zero_before
        total_zero_after += zero_after
        print(f"  {path.name}: changed={changed} zero_entries {zero_before}->{zero_after}")

        shard["rel_crlb"] = new_rel_crlb
        shard["log10_cond"] = new_log10_cond
        torch.save(shard, path)

    print(f"SUMMARY: {len(paths)} shards, {total_rows} total (instance,param) entries, "
          f"{total_changed} changed, zero_entries {total_zero_before} -> {total_zero_after}")


if __name__ == "__main__":
    main()
```

Notes:
- `u`, `y`, `theta`, `keys`, `dt`, `family`, `excitation` keys in each
  shard dict are never reassigned — only `rel_crlb`/`log10_cond` are
  overwritten. `torch.save(shard, path)` re-serializes the WHOLE dict
  (including the untouched fields) back to the same file.
- No re-simulation, no re-drawing of instances (no calls to `sample`,
  `generate`, or `simulate` outside of `identifiability()`'s own
  finite-difference sensitivity probes, which only perturb parameters
  around their EXISTING sampled values already stored in `theta` —
  this is the same probing `identifiability()` always did, just with a
  correct inverse at the end).
- Measured cost: ~2.0s for the largest shard (`saturate_prbs_dt50hz.pt`,
  B=4000, T=640); ~2-5 minutes total across all 60 shards. CPU-only, no
  GPU needed.
- Idempotent: safe to re-run (recomputing already-correct annotations a
  second time changes nothing beyond floating-point noise at the
  `rtol=1e-4` comparison tolerance).

### Change 4 — verify the regeneration

A verification pass (controller-run, not part of the module itself)
after `fix_identifiability_annotations.py` completes:

```python
# u/y/theta must be byte-identical to a hash taken before the run
# (compare against a pre-run hash snapshot of every shard's u/y/theta,
# taken before Change 3 runs, e.g. via hashlib.sha256 on each tensor's
# .numpy().tobytes()).
```

Confirms: (a) `u`/`y`/`theta` unchanged on every shard (proves no
re-simulation accidentally happened), (b) the specific `saturate`
family's zero-entry count drops from the measured 22,260 to
approximately 0 (a few genuinely-exact-zero cases may remain
legitimate, e.g. a truly flat/constant-output family variant, so "drops
to near-zero" is the right bar, not "exactly 0" as a hard assert), (c)
full offline test suite still passes.

### Change 5 — rerun `ident_exp.py`, update every downstream number

Controller-run steps, after Changes 1-4 are merged and verified (NOT
individually SDD-task-reviewed line items — mirrors how prior
retraining/regeneration campaigns in this repo were handled as
post-merge controller steps):

1. Re-run `python -m plantforge.ident_exp` fresh (do not reuse the
   exploratory verification numbers already computed during the
   investigation — those used a hand-written reimplementation for speed,
   not the actual `ident_exp.py` module; the official numbers must come
   from running the real, merged module).
2. Save output to a new dated file,
   `docs/superpowers/results/2026-07-27-ident-exp-post-identifiability-fix.txt`
   (the existing `2026-07-23-ident-exp-post-fix.txt` stays as historical
   record, per this repo's established convention of never deleting
   superseded dated result files).
3. Update `paper/main.tex`:
   - Abstract's `Spearman $r=-0.112$` mention.
   - Table 3 (`tab:ident`): base/power-controlled/range-filtered
     median/mean r, positive-cell counts.
   - The "confound we caught ourselves" paragraph's pooled
     Spearman r values and per-family breakdown.
   - The "positive cells concentrated in..." paragraph (rewritten once
     already this session against the pre-identifiability-fix numbers;
     must be re-derived from the new run, not hand-patched from the old
     breakdown, since some of the affected cells had spuriously-zero
     entries that could shift which rows count as "positive").
   - The quartile table (`tab:ident`'s second table / quartile
     discussion) and its prose (mean/median contrast).
   - Re-verify (not just assume) the Section 3 design-invariant example
     (`backlash`/`db`, ~5×10⁴) is still accurate with the new code path.
4. Update `README.md`'s identifiability section to match.
5. Regenerate `figures/fig3_within_cell_spearman.{png,pdf}` and
   `figures/fig4_quartile_artifact.{png,pdf}` via
   `figures/make_figures.py`, with `CELL_R` and the quartile arrays
   updated from the new `ident_exp.py` run's actual per-cell output (not
   hand-derived).
6. Rebuild `paper/main.pdf` and `paper/arxiv_submission/` (mirrors the
   exact process already used earlier this session: pdflatex + bibtex +
   pdflatex×2, flatten figure paths for the submission bundle, verify a
   clean standalone compile).
7. Add one line to `docs/DATASHEET.md`'s "Will the dataset be updated?"
   answer, noting the 2026-07-27 annotation-correctness revision and its
   cause (the pinv bug), pointing at the specific HF revision (Change 6).

### Change 6 — publish the corrected corpus as a new Hugging Face revision

A separate, explicitly-confirmed final step, NOT bundled into the same
approval as Changes 1-5. After every local verification above passes:

- Push the corrected shard files as a **new revision** on the existing
  `stark4062/plantforge` HF dataset repo (git-based versioning — the
  prior revision remains addressable, existing downloaders are not
  silently changed underneath them), with a commit message describing
  the fix and its cause.
- Requires the controller to explicitly ask for and receive confirmation
  before running any upload command — this is a public, hard-to-fully-
  undo action, handled with the same care as a `git push --force` or
  similar in this project's established norms. The environment is
  already authenticated to Hugging Face as `stark4062` (verified via
  `hf auth whoami`), but authentication existing is not itself
  authorization to act.

## Global constraints

- `identifiability()`'s signature, return dict keys (`rel_crlb`,
  `log10_cond`, `keys`), and dtypes are unchanged — every caller
  (`corpus.py:gen_cell`, `ident_exp.py`) needs zero changes to keep
  working.
- Do not modify `families.py`'s simulation logic, `excitation.py`, or
  any other module's numerics — this is scoped entirely to
  `identifiability.py`'s own inverse computation.
- `fix_identifiability_annotations.py` must never re-simulate or
  re-draw instances — `u`, `y`, `theta` must be byte-identical
  before/after on every shard, verified explicitly (Change 4), not
  assumed.
- Every code change gets an offline test (no GPU, no network); the new
  test must be shown to fail on the current buggy code and pass after
  the fix (both already hand-verified during this design's
  investigation).
- The corpus regeneration (Change 3) and the paper/figures rebuild
  (Change 5) are controller-run steps after Changes 1-2 are merged,
  matching this repo's established pattern for every prior
  training/regeneration campaign in this session (brainstorm → spec →
  plan → SDD for the code fix; controller-run for the data/paper
  regeneration that follows).
- The Hugging Face re-upload (Change 6) requires a separate, explicit,
  final confirmation from the user before executing — never bundled
  into a broader "go ahead" for the code/data work.
- Branch off `main` for Changes 1-2 (the only tasks that go through
  full subagent-driven-development); same process as every prior plan
  in this repo.

## Out of scope

- Any change to the physical simulation (`families.py`), excitation
  generation (`excitation.py`), or the training/evaluation pipeline
  (`evaluate.py`, `aggregate.py`, `ablation.py`, `leave_one_out.py`,
  `baselines.py`) — none of these call `identifiability()` and are
  entirely unaffected.
- Re-running the 50-checkpoint training campaign — unaffected by this
  fix (checkpoints don't depend on identifiability annotations at all).
- Any change to how `corpus.gen_cell` draws new instances (seeding,
  chunking, finite-filtering) — Change 3 explicitly reuses existing
  instances rather than generating new ones.
