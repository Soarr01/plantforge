# PLANTFORGE — a procedural control-plant corpus for in-context system identification

Released, static, reproducible corpus of **control/mechatronics plants** organized along
three axes no existing dynamics corpus has together: **nonlinearity family × excitation
class × sampling rate (exact ZOH)**, with per-instance **Fisher-information
identifiability annotations**. Gap adversarially verified 2026-07-12 (two independent
verifications, both NARROWED to exactly this design); gate PASSED same day.

## The two headline experiments (both run, both reproducible)

**1. The failure** (gate, `/home/coder/plantforge_gate`): an in-context SysID transformer
trained on Wiener-Hammerstein-only white-noise data — the private-generator recipe the
forgi86 lineage regenerates every paper — collapses off its training distribution
(in-family nMSE 0.0042): cross-family → backlash **32.6×**, cross-rate → dt/4 **10.6×**.

**2. The fix — and its honest limit** (`evaluate corpus`, 10k steps, trained on 4-of-5
families × {multisine, prbs} × {10, 50 Hz}):

| held-out axis | corpus-model nMSE (vs train-like 0.020) |
|---|---|
| rate 20 Hz (between trained rates) | stribeck **1.0×** · saturate 1.3× · boucwen 0.8× · drivetrain 7.3× |
| excitation (chirp / closed-loop) | **1.6× / 0.9×** |
| family (backlash, never seen) | **14–19×** (vs WH-model's 32.6×) |

Multi-axis training data **solves the rate and excitation axes** (the WH-model's 10.6×
rate collapse becomes ~1×; note: interpolated rate — the corpus's multi-rate axis is
exactly what makes that training data generatable) and **halves but does not close** the
cross-family gap. Held-out-family is therefore shipped as the corpus's open challenge
track, not claimed solved. Resonant `drivetrain` also resists rate interpolation (7.3×) —
a second open finding invisible to single-family generators.

## Design invariants (tested, `tests/`)

1. **(u, y) pairs are physically consistent with the stated parameters** — no hidden
   output normalization in ground truth (normalization is a method choice, done in eval
   preprocessing). Test: re-simulation from θ reproduces y to 1e-6.
2. **Multi-rate = the SAME continuous-time plant**: state-nonlinear families substep at
   ≤2 ms internally with ZOH-held input, so dt and dt/4 agree at common instants (≤1e-3
   relative; exact for LTI cores). This is the axis nobody else generates exactly.
3. **Closed-loop excitation is a true sequential loop** (PI against the actual plant
   stepper — same seed + different plant ⇒ different u), not a two-pass imitation.
4. **Identifiability annotations are directionally physical**: an excitation that never
   exits the deadzone makes the backlash width unidentifiable (rel-CRLB 5e4 vs 0.3).

## Families (5 corpus + WH for baselines)

`stribeck` (velocity friction, state NL) · `backlash` (input deadzone) · `saturate`
(input clipping) · `boucwen` (output hysteresis with memory) · `drivetrain` (two-inertia
motor/gear/compliant-load + load Coulomb — the mechatronics staple) · [`wh` — the
incumbent generators' only family, excluded from the corpus, kept for experiments]

## Axes

- **Excitations**: `prbs` (0.25 s physical hold) · `multisine` (8 tones ≤1.85 Hz) ·
  `chirp` (0.05→3 Hz) · `closedloop` (PI tracking, correlated u — the hard ID case)
- **Rates**: 10 / 20 / 50 Hz from one continuous-time truth (exact ZOH)
- **Ground truth**: named physical parameters per instance + per-parameter relative
  CRLB and FIM condition number per (instance, excitation, rate)

## Use

```
cd /home/coder
python -m plantforge.tests.run_all                     # 6 invariant tests
python -m plantforge.corpus --instances 200            # generate corpus shards (CPU)
CUDA_VISIBLE_DEVICES=1 python -m plantforge.evaluate headline   # WH-only model: the failure
CUDA_VISIBLE_DEVICES=1 python -m plantforge.evaluate corpus     # corpus model: the fix
```
`evaluate` trains in-context transformers (checkpoint-resumable; rerun until
"step 14000") and prints the transfer matrix: held-out family (backlash), held-out rate
(20 Hz), held-out excitations (chirp, closedloop).

## Honest positioning / cite head-on
- **arXiv 2412.00395** (foundation model for dynamics from purely-synthetic RKHS data) —
  generic dynamics, none of the four axes; the most dangerous citation.
- **DynaDojo** (NeurIPS'23 D&B, 18★) — generic ODE/chaos/PDE scaling platform, no
  control nonlinearities, no excitation/rate/identifiability axes.
- **nonlinearbenchmark.org** (Silverbox/WH/Bouc-Wen/Cascaded-Tanks) — the community's
  real-measured default: 5 fixed plants, no parameter truth, single records. Roadmap:
  zero-shot corpus-trained models onto these real plants (turns the incumbent into the
  validator).
- **forgi86 lineage** (LCSS'23/IFAC'24/RAL'25) — regenerates private WH-only data per
  paper; the corpus's demand proof.

## Layout
| file | role |
|---|---|
| `families.py` | 6 plant families, Stepper API, exact-ZOH + substepped simulation |
| `excitation.py` | 4 excitation classes in physical time |
| `identifiability.py` | FIM / relative-CRLB annotations |
| `corpus.py` | resumable shard generation + registry |
| `evaluate.py` | in-context transfer experiments (headline & corpus modes) |
