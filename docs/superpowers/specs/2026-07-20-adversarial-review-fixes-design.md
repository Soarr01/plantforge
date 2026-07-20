# Adversarial review fixes: normalization leak, reference mislabeling, corpus determinism, identifiability window mismatch

Status: approved design, not yet implemented.

## Purpose

A Codex adversarial review (2026-07-20, full-repo read, xhigh reasoning
effort) surfaced 11 findings against `paper/main.tex` and its supporting
code. Each finding was independently re-verified against live code
execution and real measurements (not taken on Codex's word) before being
accepted into this plan. Verified disposition:

| # | Finding | Verified? | Action |
|---|---|---|---|
| 1 | "Reference (in-distribution)" cell is actually `stribeck/multisine/dt=0.05`, a held-out rate, not a trained-rate cell | **Confirmed, serious** | Fix code (Nhóm B) |
| 2 | Paper says "white-noise"; code always uses `multisine` excitation, no white-noise class exists | Confirmed | Text-only fix (paper) |
| 3 | `_norm` computes std over the full 224-sample sequence (192 context + 32 query), leaking query-horizon scale into model input | **Confirmed, most serious** — measured 9%-145% nMSE shift across cells when recomputed with context-only std | Fix code (Nhóm A), **forces full retrain** |
| 4 | Training never loads the released corpus shards (always generates on-the-fly); the 10 Hz shard (128 samples) is shorter than the 224-sample eval window, so it could not supply the eval protocol if it were used | Confirmed | Text-only fix (README/paper/datasheet), no pipeline change |
| 5 | "Halves the cross-family gap" is arithmetically wrong: 92.6x -> 13.5x is a 6.9x ratio reduction; 0.3678 -> 0.2899 is a 21% absolute reduction, neither is a halving | Confirmed | Text-only fix (paper) |
| 6 | Leave-one-out reference has no true with/without-family counterfactual (no model was ever trained on all 5 families) | Confirmed real design limitation, but already hedged in the paper's own text | No code change; out of scope for this plan |
| 7 | Identifiability annotations are computed over the full corpus-generation trajectory (128-640 samples depending on rate), but `ident_exp.py` scores prediction difficulty only on the first 224 samples | **Confirmed** | Fix code (Nhóm C), does not require retraining the main models |
| 8 | Architecture ablation "capacity-invariant" framing | Reviewer overclaimed the paper's actual claim (paper only claims the *corpus-model* family gap is stable across capacity, not a WH-vs-corpus comparison at every capacity) | Minor text hedge only, not a code/rerun item |
| 9 | Corpus generation seeds use Python's `hash(family)`/`hash(exc)`, which is randomized per-process (`PYTHONHASHSEED`), breaking the determinism the datasheet promises | **Confirmed by direct measurement**: `hash('stribeck')` differs across two invocations | Fix code (Nhóm B) |
| 10 | Datasheet claims `gen_cell` filters out divergent (non-finite) instances during generation; the actual code has no such filter | **Confirmed** by reading `gen_cell` — unconditional `.append()` | Fix code (Nhóm B) |
| 11 | ZOH substep timing / `backlash` naming concerns | Verified not to actually apply: `round(dt/DT_INT)` gives exact 2ms substeps for all 3 corpus rates in use; paper already glosses `backlash` as "input deadzone" in-text | No action |

This plan covers findings **1, 3, 4, 5, 7, 9, 10** — the ones that survived
independent verification and require either a code fix or a text fix.
Findings 6, 8, 11 are explicitly out of scope (no code change warranted).

## Scope

Three independent groups of code changes, plus a text-only pass. Because
Group A forces every existing checkpoint to be invalid (the model's own
input distribution changes), Groups A and B are combined into a **single**
retraining campaign rather than two, to avoid paying the GPU cost twice.
Group C does not touch model checkpoints and can proceed independently.

### Group A — normalization leak (finding 3), forces full retrain

`evaluate.py`'s `_norm(u, y)` currently computes `std` over the full
224-sample sequence (`u.std(1, keepdim=True)`), which includes the 32
query-horizon samples the model is never supposed to see. Fix: compute
`std` over the 192-sample context only, and apply that same scale factor
to the full sequence (context stays correctly normalized; query values are
scaled by a context-derived factor, never their own future statistics).

```python
def _norm(u, y):
    """Per-series normalization using CONTEXT-ONLY statistics -- avoids
    leaking query-horizon scale into the model's input (the prior version
    computed std over the full 224-sample sequence, including the 32
    query samples the model must not see)."""
    u_ctx_std = u[:, :T_CTX].std(1, keepdim=True) + 1e-6
    y_ctx_std = y[:, :T_CTX].std(1, keepdim=True) + 1e-6
    return u / u_ctx_std, y / y_ctx_std
```

`baselines.py` imports `_norm` directly (`from .evaluate import ... _norm`)
so ARX/NARX baselines automatically inherit the fix -- no separate change
needed there.

**Consequence:** every existing checkpoint (`eval_{headline,corpus}_s*.pt`,
all architecture-ablation variants, all leave-one-out held-family variants
-- on the order of 50 checkpoints) was trained under the old (leaking)
input distribution. They must be deleted and retrained from scratch. This
plan's execution section covers only the code fix; the retraining campaign
itself is a controller-run post-merge step (same pattern as every prior
GPU-training plan in this repo), not part of the task-by-task
implementation.

### Group B — reference-rate mislabeling (1), corpus determinism (9), datasheet accuracy (10)

**Finding 1 — `evaluate.py`'s `report()`:** the "reference (train-like)"
cell for `corpus` mode is `nmse(model, "stribeck", "multisine", 0.05)` --
`dt=0.05` is a held-out rate (`TRAIN_RATES = [0.10, 0.02]`), so this cell
is not actually in-distribution. Fix: the corpus-mode reference becomes the
mean of `stribeck` nMSE at the two actually-trained rates (0.10 and 0.02).
The `wh_only`-mode reference is unaffected (`wh`/`multisine`/`0.05` *is*
`wh_only`'s only trained rate/family/excitation combination, so it was
already a valid in-distribution reference -- do not touch it).

```python
def report(model, mode):
    print(f"\n=== {mode}: transfer matrix (in-context nMSE) ===")
    if mode == "wh_only":
        ref = nmse(model, "wh", "multisine", 0.05)
    else:
        ref = sum(nmse(model, "stribeck", "multisine", dt) for dt in TRAIN_RATES) / len(TRAIN_RATES)
    print(f"  reference (train-like): {ref:.4f}")
    ...
```

This changes the printed reference number for `corpus` mode (previously
identical to the held-out-rate `stribeck` row; now a genuine
trained-rate average), which changes every `x ref` ratio derived from it
in `report()`'s own printed output. It does **not** change any
`nmse(...)` value itself -- only which cell is used as the denominator.

**Finding 9 — `corpus.py`'s `gen_cell`:** replace `hash(family)` /
`hash(exc)` (randomized per Python process unless `PYTHONHASHSEED` is
pinned) with a fixed, order-stable index lookup.

```python
_FAMILY_IDX = {f: i for i, f in enumerate(FAMILIES)}
_EXC_IDX = {e: i for i, e in enumerate(EXCITATIONS)}

def gen_cell(family, exc, dt, n_inst, seed=0, ident=True, chunk=64):
    ...
    for lo in range(0, n_inst, chunk):
        B = min(chunk, n_inst - lo)
        gen_p = torch.Generator().manual_seed(seed * 7919 + _FAMILY_IDX[family] * 10007 + lo)
        p = sample(family, B, gen_p)
        gen_u = torch.Generator().manual_seed(seed * 104729 + _EXC_IDX[exc] * 3571 + lo)
        ...
```

**Finding 10 — `corpus.py`'s `gen_cell`:** add a real finite-value filter
so the datasheet's existing claim ("divergent draws are dropped during
generation, not retained with missing fields") becomes true. Filter
per-chunk before appending: drop any instance (column) where `u` or `y`
is non-finite, re-draw to top up the chunk back to its target size (bounded
retry, matching the existing `build_pool` divergence-handling pattern in
`evaluate.py` for consistency of style).

```python
def gen_cell(family, exc, dt, n_inst, seed=0, ident=True, chunk=64):
    T = round(T_PHYS / dt)
    us, ys, thetas, crlbs, conds = [], [], [], [], []
    keys = None
    collected = 0
    attempt = 0
    while collected < n_inst:
        lo = collected
        B = min(chunk, n_inst - lo)
        attempt += 1
        gen_p = torch.Generator().manual_seed(seed * 7919 + _FAMILY_IDX[family] * 10007 + lo * 1000 + attempt)
        p = sample(family, B, gen_p)
        gen_u = torch.Generator().manual_seed(seed * 104729 + _EXC_IDX[exc] * 3571 + lo * 1000 + attempt)
        u, y = generate(family, p, exc, T, B, dt, gen_u)
        finite = torch.isfinite(u).all(dim=0) & torch.isfinite(y).all(dim=0)
        if not finite.all():
            u, y = u[:, finite], y[:, finite]
            p = {k: v[finite] for k, v in p.items()}
        if u.shape[1] == 0:
            continue   # whole chunk diverged, retry with a fresh seed
        us.append(u); ys.append(y)
        from .families import param_vector
        th, keys = param_vector(family, p)
        thetas.append(th)
        if ident:
            idn = identifiability(family, p, u, dt)
            crlbs.append(idn["rel_crlb"]); conds.append(idn["log10_cond"])
        collected += u.shape[1]
    ...
```

Note the seed formula changes shape slightly (`lo * 1000 + attempt` instead
of bare `lo`) to guarantee a fresh, non-repeating draw on retry after a
divergent chunk -- this is a deliberate, documented part of the fix, not
an incidental side effect.

### Group C — identifiability FIM/prediction window mismatch (finding 7)

`ident_exp.py`'s `main()` currently reads `shard["rel_crlb"]` /
`shard["log10_cond"]` directly from the released corpus shard, which
`identifiability.py` computed over the *full* trajectory (`T = round(12.8
/ dt)` samples: 128/256/640 depending on rate). The prediction task those
annotations are compared against only ever sees the first 224 samples
(`D = T_CTX + T_QRY`). Fix: recompute rel-CRLB/log10-cond specifically over
the first 224 samples of `u` for this experiment, rather than reusing the
full-trajectory annotation already stored in the shard. This does not
modify the released corpus's own annotations (those remain full-trajectory
by design, documented as such) -- it only changes what `ident_exp.py`
uses internally for its own correlation analysis.

`param_vector(family, p) -> (theta, keys)` (`families.py:164-166`) builds
`theta` by sorting `p`'s keys and stacking the corresponding values -- the
inverse is therefore a direct, lossless one-liner (`keys` is already sorted,
so zipping it back against `theta`'s columns exactly reconstructs `p`), no
new helper function needed in `families.py`:

```python
from .identifiability import identifiability

...
for fam in FAMILIES:
    for exc in EXCITATIONS:
        for dt in USABLE_DTS:
            path = OUT / f"{fam}_{exc}_dt{int(1 / dt)}hz.pt"
            if not path.exists():
                continue
            shard = torch.load(path, map_location="cpu")
            u = shard["u"].t()[:n_per_cell, :D].to(DEV)
            y = shard["y"].t()[:n_per_cell, :D].to(DEV)
            # Recompute identifiability on the SAME 224-sample window the
            # prediction task uses, rather than the shard's full-trajectory
            # annotation -- the two must cover the same record length to be
            # a meaningful comparison. theta/keys in the shard already fully
            # capture the sampled p dict (param_vector's forward direction),
            # so reconstruction here is exact and lossless.
            theta_window = shard["theta"][:n_per_cell]           # (n_per_cell, K)
            keys = shard["keys"]
            p_window = {k: theta_window[:, i] for i, k in enumerate(keys)}
            # identifiability() expects u as (T, B) -- shard["u"] is already
            # stored (T_full, B_total) uncut (see corpus.py's
            # `torch.cat(us, dim=1)`), so slice directly without transposing;
            # this is a DIFFERENT slice of u than the (B, D)-shaped `u`
            # variable above used for the model's forward pass.
            u_for_ident = shard["u"][:D, :n_per_cell]            # (D, n_per_cell) = (T, B)
            idn = identifiability(fam, p_window, u_for_ident, dt)
            crlb = idn["rel_crlb"].max(dim=1).values.numpy()
            cond = idn["log10_cond"].numpy()
            ...
```

### Text-only fixes (findings 2, 4, 5, 8) -- separate, later step

Not part of this code plan. After Group A/B/C land, merge, and (for A/B)
the retraining campaign completes with new numbers, a **separate** pass
updates `paper/main.tex`, `README.md`, and `docs/DATASHEET.md`:
- Finding 2: replace "white-noise" with "multisine" (or a hedge explaining
  the incumbent recipe's excitation, if literature precedent uses that term
  loosely) everywhere it appears (4 locations already found by grep).
- Finding 4: clarify the corpus-trained models learn from the *generator
  distribution*, not the static released shards; the released 240k-instance
  corpus is a separate, versioned snapshot used for reproducibility,
  identifiability-annotation reference, and this project's own
  `ident_exp.py` -- not the transformer's literal training set.
- Finding 5: replace "halves" with the accurate reduction figures (6.9x /
  6.4x ratio reduction, ~21% absolute reduction) -- exact new numbers
  depend on the Group A/B rerun, so this must wait until those land.
- Finding 8: soften "capacity-invariant" to scope it explicitly to "the
  corpus-trained model's family gap," not a WH-vs-corpus comparison at
  every capacity (no experiment establishes the latter).

This text pass is deliberately deferred to its own plan/step, following
this project's established convention of never editing the paper with
numbers that haven't been produced and reviewed yet (see prior plans'
Global Constraints).

## Global constraints

- Do not modify `families.py`'s physical simulation logic
  (`_core2`, `_zoh`, `Stepper.step`), `excitation.py`'s signal-generation
  logic, or any test file's assertions beyond what's needed to reflect the
  new `_norm`/`gen_cell`/`report`/`ident_exp` behavior -- this plan fixes
  measurement/protocol bugs, not physics.
- `_norm`'s new signature stays `(u, y) -> (u, y)`, same as before --
  callers (`make_batch`, `baselines.py`, `ident_exp.py`) are unaffected by
  the signature, only by the changed numeric output.
- `gen_cell`'s new signature and return shape (the `shard` dict's keys)
  must stay identical to the current version -- only the seeding scheme
  and the addition of a finite-filter change; nothing downstream
  (`corpus.py main()`, `ident_exp.py`, the HF dataset loader) should need
  to change to accommodate this.
- `report()`'s printed output format (line structure, `x ref` label) stays
  the same -- only the reference value computation changes.
- Every new/changed function gets an offline test (no GPU, no network) in
  the existing `tests/` module structure, following this repo's established
  pattern (`tests/test_plantforge.py` for corpus/families, `tests/test_evaluate.py`
  if one needs to be created for `_norm`/`report`, etc.).
- Do not touch `identifiability.py`'s FIM computation itself (`SIGMA_REF`,
  `REL_STEP`, the finite-difference sensitivity method) -- Group C changes
  *what window* is passed to it, not how it computes.
- Branch off `main`, same subagent-driven-development process (task briefs,
  task-scoped review, final whole-branch review) as every prior plan in
  this repo.
- Do not launch any GPU training as part of this plan's task execution --
  the retraining campaign is a controller-run step after this plan's code
  changes are reviewed and merged, exactly like every prior plan that
  changed a training-affecting invariant in this repo.

## Out of scope

- Findings 6, 8 (partially), 11 -- no code change, see table above.
- The text-only paper/README/datasheet updates for findings 2, 4, 5, 8 --
  deferred to a follow-up plan once new numbers exist.
- Re-running `figures/make_figures.py` -- deferred to the same follow-up
  text/number plan, once new results exist to plot.
- The leave-one-out true with/without-family counterfactual (finding 6) --
  would require training a model on all 5 families, a new experiment this
  repo has never run; not requested, not designed here.
- Croissant metadata / arXiv submission bundle updates -- unaffected by
  this plan's code changes until the retraining campaign produces new
  numbers to put in the paper.
