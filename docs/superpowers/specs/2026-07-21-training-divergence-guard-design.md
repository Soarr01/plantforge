# Training-divergence guard: skip non-finite-gradient steps, refuse all-NaN checkpoints

Status: approved design, not yet implemented.

## Purpose

After the normalization-leak fix (context-only `_norm`), 20 of 50 retrained
checkpoints diverged to all-NaN weights while still "finishing" 10,000 steps
and being saved silently. The investigation
(`docs/superpowers/results/2026-07-21-normalization-divergence-investigation.md`)
established, by direct reproduction, that:

- Rare training batches produce a **finite forward loss but non-finite
  backward gradients** under context-only normalization.
- `nn.utils.clip_grad_norm_` then computes `total_norm = nan` and multiplies
  every gradient by a nan clip coefficient, so `opt.step()` writes nan into
  every weight.
- The existing `if not torch.isfinite(loss)` skip never fires (loss was
  finite), so the corruption is silent and the run persists an all-NaN
  checkpoint.

The user approved **Direction 1**: a gradient-finiteness guard (skip the
optimizer step when the gradient is non-finite) plus a hard refusal to save an
all-NaN checkpoint. The context-only normalization is kept unchanged (it is a
genuine correctness fix; the divergence is a separate training-loop robustness
gap). The user also approved retraining **all 50** checkpoints under the fixed
loop.

## Scope

A single-function change in `evaluate.py`'s `run()` (the training loop), plus
offline tests, plus the controller-run retraining campaign afterward. No
change to `_norm`, the model, the data pipeline, or any other module.

## Design

### Change 1 — skip the optimizer step on non-finite gradients

Current loop body (evaluate.py, inside `run()`):

```python
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if i % 1000 == 0:
            print(f"  [{mode}] step {i:5d} mse {loss.item():.4f}")
        i += 1
```

`nn.utils.clip_grad_norm_` RETURNS the total gradient norm computed **before**
clipping; if any gradient is non-finite this return value is `nan`/`inf`.
Capture it and skip the step when it is non-finite:

```python
        opt.zero_grad(); loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            # A rare batch can produce a finite forward loss but non-finite
            # backward gradients under context-only normalization; clip_grad_norm_
            # then returns a non-finite total norm and has already multiplied every
            # gradient by a non-finite clip coefficient. Skipping the step here (the
            # grads are discarded by the next opt.zero_grad()) prevents opt.step()
            # from writing nan into every weight -- the confirmed root cause of the
            # 2026-07-21 all-NaN-checkpoint divergences. Bad batches are rare
            # (<~1%), so skipping them has negligible training impact.
            print(f"  [{mode}] step {i}: non-finite gradient (norm={gnorm}), skipping step")
            i += 1
            continue
        opt.step()
        if i % 1000 == 0:
            print(f"  [{mode}] step {i:5d} mse {loss.item():.4f}")
        i += 1
```

Notes:
- The existing `if not torch.isfinite(loss)` skip (a few lines above, which
  handles a non-finite *loss* by replacing the poisoned pooled batch) stays
  exactly as-is. This new guard handles the distinct case of a *finite loss
  with non-finite gradients*, which that guard misses.
- We do NOT replace the pooled batch on the grad-skip path (unlike the
  loss-skip path). `i` still increments, so the loop advances to the next
  batch immediately; re-encountering the same rare batch ~once per POOL_N-step
  cycle and re-skipping it is negligible, and NOT mutating the pool keeps the
  change minimal and side-effect-free. (A reviewer may argue for symmetry with
  the loss-skip's pool replacement; this is a deliberate, documented choice for
  simplicity, not an oversight.)
- `torch.isfinite(gnorm)` where `gnorm` is the tensor returned by
  `clip_grad_norm_` is a single clean scalar check that catches exactly the
  nan/inf-gradient case.

### Change 2 — refuse to save an all-NaN (diverged) checkpoint

Current save (end of `run()`):

```python
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": i}, ck_path)
    print(f"[{mode}] checkpoint at step {i}")
```

Guard against silently persisting a diverged model:

```python
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError(
            f"[{mode}] refusing to save checkpoint at step {i}: model weights are "
            f"non-finite (training diverged). This should be unreachable given the "
            f"non-finite-gradient skip guard above -- if it fires, a new divergence "
            f"path exists and must be investigated, not silently persisted.")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": i}, ck_path)
    print(f"[{mode}] checkpoint at step {i}")
```

Rationale: with Change 1, the model should never reach all-NaN weights, so this
assertion should never fire in practice. It is defense-in-depth: it converts
any *future* silent divergence into a loud, un-missable failure, complementing
the `_has_finite_weights` report-time guards already present in `ablation.py`
and `leave_one_out.py` (those catch a diverged checkpoint at *read* time; this
prevents writing one at *train* time). A loud `RuntimeError` is the correct
behavior — an all-NaN model has no salvageable progress to preserve, and the
resumable shell drivers' stall-guard already handles a run that fails to
advance.

## Global constraints

- Do NOT modify `_norm`, `InContextSysID`, `make_batch`, `build_pool`, the
  data pipeline, or any file other than `evaluate.py` (and its test).
- `run()`'s signature, return value (`(model, finished_bool)`), checkpoint
  format/keys (`model`, `opt`, `step`), resumability contract, and printed
  output format all stay unchanged except for the two additions above. The
  resumable shell training drivers (`scripts/train_*.sh`) must need zero
  changes.
- The existing `if not torch.isfinite(loss)` loss-skip path is unchanged.
- Every new behavior gets an offline test (no GPU, no network) in
  `tests/test_evaluate.py`, following the repo's established pattern.
- Branch off `main`; same subagent-driven-development process as every prior
  plan.

## Testing approach

The tricky part is testing a training-loop guard offline (no GPU, no real
training). Approach: unit-test the guard *conditions* in isolation rather than
driving a full `run()`:

1. **Non-finite-gradient detection**: construct a tiny scenario where
   `clip_grad_norm_` returns a non-finite value (e.g. a model whose `.grad`
   tensors are manually set to contain `nan`/`inf`), and assert
   `torch.isfinite(clip_grad_norm_(...))` is `False` — documenting the exact
   signal the guard keys on. This locks in the assumption that
   `clip_grad_norm_`'s return value is the right thing to check.
2. **All-NaN checkpoint refusal**: factor the weight-finiteness check into a
   tiny testable helper (or test the `all(torch.isfinite(p).all() for p in
   model.parameters())` predicate directly on a model with one parameter
   `fill_`-ed with nan → predicate is `False`; on a fresh model → `True`).
   Mirror the `_has_finite_weights` test pattern already in
   `tests/test_ablation.py` / `tests/test_leave_one_out.py`.

A full end-to-end "does training no longer diverge" confirmation is NOT an
offline test — it is verified by the controller-run retraining campaign
afterward (re-scanning all 50 checkpoints for NaN, expecting 0), exactly as
prior training-affecting plans in this repo verified their effects post-merge.

## Out of scope

- Normalization stabilization (context-std flooring, etc.) — Direction 2,
  explicitly not chosen. Bad batches are rare enough that the skip guard
  suffices; changing the normalization again would require re-validating the
  leak fix.
- Any change to how the diverged/healthy checkpoints already on disk are
  handled — they will all be deleted and regenerated by the retraining
  campaign (a controller step after this code change merges).
- The paper/README/datasheet text updates that depend on the final retrained
  numbers — deferred to the same post-retraining text pass already pending
  from the adversarial-review-fixes work (findings 2, 4, 5, 8 there, plus the
  new corrected numbers).
