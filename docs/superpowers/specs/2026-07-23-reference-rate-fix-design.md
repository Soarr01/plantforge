# Fix the reference-rate bug in aggregate.py, ablation.py, leave_one_out.py

Status: approved design, not yet implemented.

## Purpose

The 2026-07-20 adversarial-review-fixes plan (Finding 1) fixed a bug in
`evaluate.py`'s `report()`: the corpus-mode "reference (train-like)" cell
used `nmse(model, "stribeck", "multisine", 0.05)` — but `dt=0.05` is a
**held-out** rate (`TRAIN_RATES = [0.10, 0.02]`), so the cell was not
actually in-distribution despite its label. The fix averaged over
`TRAIN_RATES` instead.

That fix touched only `evaluate.py`. While preparing to regenerate paper
numbers on the freshly retrained (2026-07-23, all-50-finite) checkpoints, a
review of every script that computes a "reference" baseline found the same
underlying pattern — reference measured at the held-out rate — duplicated in
three more places:

- **`aggregate.py`**: `transfer_cells()`'s reference row is literally labeled
  `"reference (train-like)"` — the exact same label, computed the exact same
  wrong way (`dt=0.05`). This is the script that produces the paper's
  multi-seed mean±std tables, so this is the most consequential instance.
- **`ablation.py`**: `refs = [nmse(m, "stribeck", "multisine", 0.05) ...]` —
  used as the denominator for the "Nx ref" family-gap ratio in the headline
  architecture-capacity-invariance claim. Not labeled "train-like", but
  measuring the baseline at an out-of-distribution rate makes it artificially
  worse, which understates the true family-gap ratio.
- **`leave_one_out.py`**: `reference_and_heldout()`'s `ref_vals` (mean nMSE
  over the 4 non-held families) is also computed at `dt=0.05`. Same
  understatement risk for the family-gap ratio.

User-approved decision: fix all three, for consistency — every "reference"
computed anywhere in this codebase should be a genuine in-distribution
(trained-rate) baseline.

## Design

The shared principle, applied three times: wherever a "reference" baseline
value is computed, replace a bare `nmse(model, fam, exc, 0.05)` call with an
average over `TRAIN_RATES` (`[0.10, 0.02]`) — mirroring the `evaluate.report()`
fix exactly. Values that are deliberately probing a *specific held-out*
condition (a held-out family, excitation, or rate row) are untouched — only
the "what does this model do in-distribution" baseline changes.

### Change 1 — `aggregate.py`

`transfer_cells()` stays structurally the same (still returns the
`"reference (train-like)"` label with a `dt=0.05` placeholder tuple, since
other code — e.g. `test_transfer_cells_structure` — inspects its label/family
structure). The averaging happens in `matrix()`, which is the function that
actually calls `nmse()`:

```python
def matrix(model, mode: str) -> dict:
    out = {}
    for label, fam, exc, dt in transfer_cells(mode):
        if label == "reference (train-like)" and mode != "wh_only":
            # dt=0.05 is a held-out rate (TRAIN_RATES = [0.10, 0.02]) --
            # average over the actually-trained rates instead, mirroring the
            # fix in evaluate.report() (2026-07-20 adversarial-review-fixes,
            # Finding 1). wh_only mode is unaffected: dt=0.05 IS wh_only's
            # only trained rate, so it was already a valid reference.
            out[label] = sum(nmse(model, fam, exc, r) for r in TRAIN_RATES) / len(TRAIN_RATES)
        else:
            out[label] = nmse(model, fam, exc, dt)
    return out
```

Add `TRAIN_RATES` to the existing `from .evaluate import (...)` line.

### Change 2 — `ablation.py`

```python
refs = [sum(nmse(m, "stribeck", "multisine", dt) for dt in TRAIN_RATES) / len(TRAIN_RATES)
        for m in models]
```
replaces
```python
refs = [nmse(m, "stribeck", "multisine", 0.05) for m in models]
```
`fams` (the held-out-family row) is untouched. Add `TRAIN_RATES` to the
existing `from .evaluate import (...)` line.

### Change 3 — `leave_one_out.py`

```python
def reference_and_heldout(model, held_family: str):
    """mean nMSE over the 4 trained (non-held) families, averaged over the
    actually-trained rates (TRAIN_RATES) so the reference is genuinely
    in-distribution; and nMSE on the held-out family itself at dt=0.05 (the
    held-out rate), matching how held-out family generalization is reported
    elsewhere (evaluate.report())."""
    others = [f for f in FAMILIES if f != held_family]
    ref_vals = [nmse(model, fam, "multisine", dt) for fam in others for dt in TRAIN_RATES]
    reference = sum(ref_vals) / len(ref_vals)
    held_out = nmse(model, held_family, "multisine", 0.05)
    return reference, held_out
```
`held_out` is untouched — it deliberately probes the held-out family at the
held-out rate, matching how held-out-family generalization is reported
elsewhere. Add `TRAIN_RATES` to the existing `from .evaluate import (...)`
line.

## Global constraints

- Modify ONLY `aggregate.py`, `ablation.py`, `leave_one_out.py`, and their
  three test files (`tests/test_aggregate.py`, `tests/test_ablation.py`,
  `tests/test_leave_one_out.py`). Do NOT touch `evaluate.py` (already fixed),
  `baselines.py` or `ident_exp.py` (confirmed to have no instance of this
  pattern — `baselines.py` has no reference-rate logic at all;
  `ident_exp.py`'s only `0.05` is a p-value significance threshold, unrelated).
- No `nmse(...)` VALUE changes — only which cell(s) are averaged together to
  form each "reference" denominator. Every other row/cell is unchanged.
- Held-out-condition probes (`fams` in `ablation.py`, `held_out` in
  `leave_one_out.py`, every non-reference row in `aggregate.py`) stay
  evaluated exactly as before — do not average these.
- `aggregate.py`'s `transfer_cells()` return structure (list of
  `(label, family, excitation, dt)` tuples) is unchanged, since
  `test_transfer_cells_structure` and any other caller inspects it directly;
  the reference averaging is added in `matrix()`, not `transfer_cells()`.
- New/changed behavior gets offline tests (no GPU, no checkpoints, no
  network) via monkeypatching each module's imported `nmse` name to a
  deterministic recording stub — this repo's established pattern for testing
  functions that call `nmse()` without real model inference (see
  `tests/test_ablation.py`'s `abl.CKPT_DIR = tmp_dir` monkeypatch style for
  the precedent of patching module-level names for isolated testing).
- Branch off `main`; same subagent-driven-development process as every prior
  plan in this repo.

## Testing approach

`nmse()` requires a real (even if untrained/random-init) `InContextSysID`
forward pass; existing offline tests in this repo do not call it directly
(too slow/heavyweight for a unit test and not this repo's established
pattern). Instead, monkeypatch each module's own imported `nmse` binding
(e.g., `plantforge.aggregate.nmse`, not `plantforge.evaluate.nmse` — each
module imported its own reference to the name) to a deterministic stub that
records its call arguments and returns a value derived from `dt` (e.g.
`return dt`), so the averaging arithmetic is exactly checkable:

- **`aggregate.py`**: stub `matrix(None, "corpus")`'s reference cell equals
  `(0.10 + 0.02) / 2 = 0.06` (exact, since `TRAIN_RATES = [0.10, 0.02]`), and
  confirm the stub was called with `dt` in `{0.10, 0.02}` for that cell (never
  `0.05`). A second test confirms `matrix(None, "wh_only")`'s reference is
  still computed with a single `dt=0.05` call (unchanged).
- **`ablation.py`**: same averaging-value assertion for `refs`, via
  monkeypatching `plantforge.ablation.nmse`.
- **`leave_one_out.py`**: `reference_and_heldout()`'s `reference` return value
  equals the mean of the stub's `dt`-derived values over 4 families × 2 rates
  (8 values), while `held_out` still equals the stub's return for the single
  `dt=0.05` call — confirming the two return values diverge in behavior
  (reference now multi-rate, held_out still single-rate).

## Out of scope

- Re-running `aggregate.py` / `ablation.py` / `leave_one_out.py` to produce
  final paper numbers, and folding those numbers (plus the still-pending
  text-only fixes from the original adversarial-review-fixes work) into
  `paper/main.tex` / `README.md` / `docs/DATASHEET.md` — a controller step
  after this fix merges, using the already-retrained (2026-07-23, 50/50
  finite) checkpoints.
- Any change to `evaluate.py` (already correct), `baselines.py`,
  `ident_exp.py`, or `corpus.py`.
- Any change to which rates/families/excitations are trained or held out.
