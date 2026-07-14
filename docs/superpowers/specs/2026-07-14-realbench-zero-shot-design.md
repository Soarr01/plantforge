# Zero-shot real-plant evaluation (`plantforge/realbench.py`)

Status: approved, not yet implemented.

## Purpose

README's roadmap item: "zero-shot corpus-trained models onto these real plants
(turns the incumbent into the validator)." Evaluate the already-trained
`eval_corpus.pt` and `eval_wh_only.pt` checkpoints (no fine-tuning) on real
measured data from nonlinearbenchmark.org, using the same in-context nMSE
methodology as `evaluate.py`'s transfer-matrix report.

## Scope

Three datasets, loaded via the official `nonlinear_benchmarks` pip package
(already installed and verified working):

- `Silverbox`
- `Cascaded_Tanks`
- `WienerHammerBenchMark`

Bouc-Wen is explicitly out of scope: it is not exposed by
`nonlinear_benchmarks`'s public API (`dir(nonlinear_benchmarks)` has no
Bouc-Wen loader), and hand-rolling a separate loader/format for it was
rejected as unnecessary risk for this iteration.

## Architecture

New file `plantforge/realbench.py`, standalone from `evaluate.py` /
`corpus.py` — the `nonlinear_benchmarks` dependency is imported only here, so
it never becomes a hard dependency of the core corpus/eval pipeline.

Reuses from `evaluate.py` (no changes needed there, all already
module-level):  `InContextSysID`, `_norm`, `T_CTX`, `T_QRY`, `D`, `DEV`,
`CKPT_DIR`.

Loads both checkpoints (`CKPT_DIR / "eval_corpus.pt"`,
`CKPT_DIR / "eval_wh_only.pt"`) in eval mode. No training occurs in this
module.

Entry point: `python -m plantforge.realbench`.

## Per-dataset rate handling

This is the crux of the design — real benchmark sample rates do not line up
with the corpus's trained rates (dt ∈ {0.02, 0.05, 0.10}s), and the mismatch
differs in kind per dataset:

| dataset | native dt | native duration (test split) | handling |
|---|---|---|---|
| Silverbox | ≈0.001638s (~610 Hz) | tens of seconds, ample | **decimate** (`scipy.signal.decimate`, chained if factor > 13, with anti-aliasing) by ~12× → effective dt ≈ 0.0197s, matches the 50 Hz (dt=0.02) trained rate |
| Cascaded_Tanks | 4.0s | 1024 samples × 4s = 4096s, ample in time but coarse in samples | **no decimation possible** — already 40× coarser than the coarsest trained dt (0.10s). Use native dt as-is. Report explicitly flags this as *extrapolation beyond the corpus's rate range*, not interpolation. |
| WienerHammerBenchMark | ≈1.953e-5s (~51 kHz) | test ≈1.54s, train ≈1.95s total physical duration | **not evaluable** — even the coarsest usable window (`D=224` samples at dt=0.02s) needs 4.48s of physical signal, more than the entire record. Report explicitly states "insufficient physical duration for a single context window" rather than forcing degenerate decimation. |

Decimation implementation note: `scipy.signal.decimate` recommends factors
≤13 per call; for any factor above that, chain multiple calls (only relevant
if a dataset needed a much larger factor — Silverbox's ~12× fits in one
call).

## Windowing & evaluation

For each evaluable dataset (Silverbox, Cascaded_Tanks):

1. Take the benchmark's test split(s) (standard community convention for
   reporting — the model never trains on any of this data either way, so
   train/test only matters for comparability with published numbers).
   Silverbox's loader returns three separate test records (two multisine
   variants + one "arrow no extrapolation"); concatenate windows drawn from
   all three into a single pool rather than reporting them separately, since
   the design goal is one headline number per dataset per checkpoint.
2. Decimate (Silverbox only).
3. Slice into non-overlapping windows of length `D=224` per record, pool
   across records (see above), cap the total pool at 8 windows (fewer if the
   available records are shorter — e.g. Cascaded_Tanks yields ~4 from its
   single 1024-sample test record).
4. Normalize each window with `_norm` (per-series std normalization),
   identical to how `evaluate.py` does it — a method choice applied
   consistently, not a new convention.
5. Run both checkpoints (`corpus`, `wh_only`) zero-shot over all windows,
   average nMSE per dataset per checkpoint (same nMSE formula as
   `evaluate.nmse`: MSE over the query horizon `T_CTX:` normalized by
   query-horizon variance).

## Report format

Printed table analogous to `evaluate.report()`, e.g.:

```
=== real-plant zero-shot transfer (in-context nMSE) ===
  Silverbox (decimated ~12x -> dt=0.0197s, ~50Hz-like):
    wh_only model:  nMSE=...
    corpus model:   nMSE=...
  Cascaded_Tanks (native dt=4.0s -- 40x coarser than trained range, EXTRAPOLATION):
    wh_only model:  nMSE=...
    corpus model:   nMSE=...
  WienerHammerBenchMark: SKIPPED -- record duration (~1.5s) shorter than one
    context window at any trained rate (min 4.48s) -- not evaluable zero-shot.
```

## Testing

One lightweight test added under `tests/` covering only the decimation and
windowing helpers (pure functions, synthetic input arrays — no network
access, no checkpoint required), so it runs alongside the existing 6
invariant tests without requiring internet or trained checkpoints. It is
kept separate from `test_plantforge.py`'s `_run_all()` since it exercises
different (offline-safe but currently network-optional) code paths — actual
concern is just not silently breaking `run_all` in network-less CI-like
environments; the new test itself must not touch the network either.

## Out of scope for this iteration

- Bouc-Wen (see Scope above).
- Fine-tuning / adapting the model to real data — this is strictly zero-shot.
- Any change to `evaluate.py`, `corpus.py`, `families.py`, `excitation.py`,
  `identifiability.py`.
