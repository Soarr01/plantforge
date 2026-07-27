# Identifiability pinv Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `identifiability.py`'s `identifiability()` function, which
currently uses `torch.linalg.pinv()` on the regularized Fisher information
matrix and silently returns `rel_crlb = 0` for any parameter with
near-zero sensitivity — the opposite of correct CRLB semantics (zero
sensitivity means maximally UNidentifiable, which must show a LARGE
rel_crlb, not a near-zero one).

**Architecture:** Single-function change in `identifiability.py`: replace
the float32 `torch.linalg.pinv(F)` call with an explicit regularized
inverse computed via `torch.linalg.eigh` in float64 (every eigenvalue
direction is inverted, none truncated to zero, since the existing `1e-9`
floor already guarantees every direction has a well-defined minimum
eigenvalue). Return dtype stays `float32`; the function's signature and
return dict keys are unchanged, so every caller (`corpus.py`,
`ident_exp.py`) needs zero changes.

**Tech Stack:** Python 3, PyTorch. No new dependencies.

## Global Constraints

- Modify ONLY `identifiability.py` and `tests/test_plantforge.py`. Do not
  touch `families.py`'s simulation logic, `excitation.py`, `corpus.py`, or
  `ident_exp.py` — none of their own logic needs to change for this fix
  (they call `identifiability()` and get corrected behavior automatically).
- `identifiability()`'s signature (`identifiability(family: str, p: dict,
  u: torch.Tensor, dt: float)`), return dict keys (`rel_crlb`,
  `log10_cond`, `keys`), and return dtypes (`float32`) are unchanged.
- The existing test `test_identifiability_flags_unexcited_parameter` must
  continue to pass unmodified after the fix (verify this, don't just
  assume it).
- New behavior gets an offline test (no GPU, no network) in
  `tests/test_plantforge.py`, following the repo's established pattern —
  the new test must be shown to FAIL on the current buggy code and PASS
  after the fix.
- Branch off `main`; same subagent-driven-development process as every
  prior plan in this repo.

---

### Task 1: Fix `identifiability()`'s inverse and add a regression test

**Files:**
- Modify: `identifiability.py` (the `identifiability()` function body)
- Test: `tests/test_plantforge.py` (extend)

**Interfaces:**
- Consumes: `torch` (already imported in both files); `sample` and
  `open_loop_input` (already imported in `tests/test_plantforge.py` from
  `plantforge.families` and `plantforge.excitation` respectively).
- Produces: `identifiability()`'s external contract (signature, return
  dict shape/dtype) is unchanged — only the numeric values returned for
  `rel_crlb` change (correctly, for near-singular directions).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_plantforge.py`, immediately after the existing
`test_identifiability_flags_unexcited_parameter` function (which ends
around line 97 with its `print(...)` call) and before
`test_corpus_cell_roundtrip`:

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

Update `_run_all()` (near the bottom of the file) to call the new test
immediately after `test_identifiability_flags_unexcited_parameter()`:

```python
def _run_all():
    test_families_finite_and_param_dependent()
    test_multirate_ground_truth_consistency()
    test_physical_consistency_of_pairs()
    test_closedloop_is_closed()
    test_identifiability_flags_unexcited_parameter()
    test_identifiability_zero_sensitivity_gives_large_not_zero_crlb()
    test_corpus_cell_roundtrip()
    test_gen_cell_seeds_are_deterministic_across_hash_randomization()
    test_gen_cell_different_families_get_different_seeds()
    test_gen_cell_returns_exactly_n_inst_even_with_divergence()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_plantforge`

Expected: FAIL on `test_identifiability_zero_sensitivity_gives_large_not_zero_crlb`
— the current buggy `identifiability()` returns `rel_crlb[:, k] == 0.0`
for all 6 instances (verified during design), so
`idn["rel_crlb"][:, k].min() > 100` is `False`.

- [ ] **Step 3: Implement the fix**

In `identifiability.py`, replace the current `identifiability()` function
body:

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

with:

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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.test_plantforge`
(same invocation as Step 2). Expected: all tests PASS, including both
`test_identifiability_flags_unexcited_parameter` (pre-existing — confirm
it still passes, not just that it wasn't touched) and the new
`test_identifiability_zero_sensitivity_gives_large_not_zero_crlb`.

- [ ] **Step 5: Run the full offline suite to confirm no regressions**

Run: `cd /data/nas07_new/PersonalData/phuocthien && python -m plantforge.tests.run_all`
Expected: all tests PASS, no failures/tracebacks, exit 0.

- [ ] **Step 6: Commit**

```bash
git add identifiability.py tests/test_plantforge.py
git commit -m "Fix identifiability()'s pinv bug: zero-sensitivity params must show large, not zero, rel-CRLB"
```

---

## After the task: controller-run corpus regeneration and paper update (not a task step)

Per this repo's established convention (mirrors the training-divergence-
guard and reference-rate-fix plans), the following happen as controller-
run steps after Task 1 is reviewed and merged to `main` — NOT part of the
task's TDD steps, and NOT re-entering subagent-driven-development:

1. **Recompute the released corpus's own annotations in place.** Create
   `fix_identifiability_annotations.py` at the repo root (module
   invocation: `PLANTFORGE_DATA=... python -m
   plantforge.fix_identifiability_annotations`). For every shard under
   `$PLANTFORGE_DATA/corpus/*.pt`: load it, reconstruct `p` from
   `theta`+`keys`, call the now-fixed `identifiability(shard["family"], p,
   shard["u"], shard["dt"])`, overwrite ONLY `shard["rel_crlb"]` and
   `shard["log10_cond"]`, save back to the same path. Do not re-simulate
   or re-draw instances — `u`, `y`, `theta`, `keys`, `dt`, `family`,
   `excitation` must stay byte-identical. See
   `docs/superpowers/specs/2026-07-27-identifiability-pinv-fix-design.md`
   (Change 3) for the exact reference implementation. Measured cost:
   ~2-5 minutes across all 60 shards, CPU-only.
2. **Verify the regeneration.** Before running step 1, snapshot a hash
   (e.g. `hashlib.sha256` over each shard's `u`/`y`/`theta` tensor bytes)
   for every shard. After step 1 completes, re-hash and confirm every
   shard's `u`/`y`/`theta` hash is unchanged. Confirm the `saturate`
   family's spurious-zero-`rel_crlb` count drops from the previously
   measured 22,260/48,000 to near-zero. Re-run
   `python -m plantforge.tests.run_all` once more.
3. **Rerun `ident_exp.py` and record fresh results.** Run
   `PLANTFORGE_DATA=... python -m plantforge.ident_exp` fresh (do not
   reuse any exploratory/investigation numbers from the design phase —
   those used a hand-written reimplementation for speed, not the actual
   merged module). Save output to
   `docs/superpowers/results/2026-07-27-ident-exp-post-identifiability-fix.txt`
   (keep the existing `2026-07-23-ident-exp-post-fix.txt` as historical
   record — do not delete or overwrite it).
4. **Update every downstream number.** In `paper/main.tex`: the
   abstract's `Spearman $r=-0.112$` mention, Table 3 (`tab:ident`,
   base/power-controlled/range-filtered median/mean r and positive-cell
   counts), the "confound we caught ourselves" paragraph's pooled r
   values and per-family breakdown, the "positive cells concentrated
   in..." paragraph (re-derive from the new run's actual per-cell output
   — do not hand-patch the existing prose), the quartile table and its
   discussion. Re-verify (don't assume) the Section 3 design-invariant
   example (`backlash`/`db`, ~5×10⁴) is still accurate under the new code
   path. Update `README.md`'s identifiability section to match. Add one
   line to `docs/DATASHEET.md`'s "Will the dataset be updated?" answer
   noting the 2026-07-27 annotation-correctness revision.
5. **Regenerate figures.** Update `figures/make_figures.py`'s `CELL_R`
   list and the quartile `q_mean`/`q_median` arrays from the new
   `ident_exp.py` run's actual per-cell output, then run
   `python figures/make_figures.py` to regenerate
   `figures/fig3_within_cell_spearman.{png,pdf}` and
   `figures/fig4_quartile_artifact.{png,pdf}`.
6. **Rebuild the paper and arXiv bundle.** `cd paper && pdflatex ... &&
   bibtex main && pdflatex ... && pdflatex ...` (same process already
   used earlier this session), verify a clean compile (zero undefined
   references), then rebuild `paper/arxiv_submission/` and
   `paper/plantforge_arxiv_submission.tar.gz` the same way, and verify
   the rebuilt bundle compiles standalone (extract to a clean directory,
   pdflatex-only, no bibtex needed since `main.bbl` is bundled).
7. **Publish the corrected corpus as a new Hugging Face revision** — a
   separate, explicitly-confirmed final step. Do not execute any upload
   command without first asking the user for and receiving explicit
   confirmation, even though the environment is already authenticated as
   `stark4062` (`hf auth whoami`). Push as a new revision on the existing
   `stark4062/plantforge` dataset repo (git-based versioning — the prior
   revision must remain addressable), with a commit message describing
   the fix and its cause.
