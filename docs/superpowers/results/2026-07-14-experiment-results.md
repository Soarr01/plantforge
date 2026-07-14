# Experiment results — multi-seed, baselines, identifiability (2026-07-14)

Configuration: 5 seeds (PF_SEED 0–4) x 2 modes, 10k steps each; eval seeds
900–905 fixed across all models; corpus shards 4000 instances/cell;
identifiability run with PF_IDENT_N=1000 over dt in {0.05, 0.02}.

Runs: `CUDA_VISIBLE_DEVICES=6`, `OMP_NUM_THREADS=1`,
`PLANTFORGE_DATA=/data/nas07_new/PersonalData/phuocthien/plantforge_data`.
`ident_exp` was run with default `PF_SEEDS`/`PF_IDENT_N` (5 seeds x 40 cells x
1000 instances/cell — the full-scale run, not the seed-0 smoke test).

## A. Multi-seed transfer matrices (mean ± std, n=5)

```
=== wh_only: transfer matrix, mean ± std over seeds ===
  reference (train-like): 0.0040 ± 0.0004 (n=5)  [1.0x ref]
  held-out family backlash dt=0.10: 0.3880 ± 0.0322 (n=5)  [97.7x ref]
  held-out family backlash dt=0.05: 0.3678 ± 0.0408 (n=5)  [92.6x ref]
  held-out family backlash dt=0.02: 0.4665 ± 0.0195 (n=5)  [117.5x ref]
  held-out excitation chirp (stribeck, dt=0.05): 0.1533 ± 0.0645 (n=5)  [38.6x ref]
  held-out excitation closedloop (stribeck, dt=0.05): 0.0545 ± 0.0056 (n=5)  [13.7x ref]
  held-out rate dt=0.05 stribeck: 0.0341 ± 0.0039 (n=5)  [8.6x ref]
  held-out rate dt=0.05 saturate: 0.0141 ± 0.0016 (n=5)  [3.5x ref]
  held-out rate dt=0.05 boucwen: 0.0916 ± 0.0177 (n=5)  [23.1x ref]
  held-out rate dt=0.05 drivetrain: 0.2180 ± 0.0156 (n=5)  [54.9x ref]
  real-plant Silverbox (decimated 12x -> dt=0.0197s): 0.9583 ± 0.2170 (n=5)
  real-plant Cascaded_Tanks (native dt=4.00s, extrapolation): 0.3937 ± 0.0814 (n=5)

=== corpus: transfer matrix, mean ± std over seeds ===
  reference (train-like): 0.0215 ± 0.0017 (n=5)  [1.0x ref]
  held-out family backlash dt=0.10: 0.3107 ± 0.0080 (n=5)  [14.5x ref]
  held-out family backlash dt=0.05: 0.2899 ± 0.0095 (n=5)  [13.5x ref]
  held-out family backlash dt=0.02: 0.3927 ± 0.0102 (n=5)  [18.3x ref]
  held-out excitation chirp (stribeck, dt=0.05): 0.0383 ± 0.0143 (n=5)  [1.8x ref]
  held-out excitation closedloop (stribeck, dt=0.05): 0.0195 ± 0.0031 (n=5)  [0.9x ref]
  held-out rate dt=0.05 stribeck: 0.0215 ± 0.0017 (n=5)  [1.0x ref]
  held-out rate dt=0.05 saturate: 0.0285 ± 0.0021 (n=5)  [1.3x ref]
  held-out rate dt=0.05 boucwen: 0.0178 ± 0.0014 (n=5)  [0.8x ref]
  held-out rate dt=0.05 drivetrain: 0.1722 ± 0.0111 (n=5)  [8.0x ref]
  real-plant Silverbox (decimated 12x -> dt=0.0197s): 0.3306 ± 0.0397 (n=5)
  real-plant Cascaded_Tanks (native dt=4.00s, extrapolation): 0.0843 ± 0.0430 (n=5)
```

Timing: 4m 51s wall (`time python -m plantforge.aggregate`). No tracebacks;
every cell shows `(n=5)`.

## B. Classical baselines (ARX / degree-2 NARX, context-fit + free-run)

```
=== classical baselines on the corpus transfer cells (context-fit + free-run, nMSE) ===
  reference (train-like): ARX 0.0077 | NARX2 80562087383.8275
  held-out family backlash dt=0.10: ARX 0.3116 | NARX2 33437981158.7110
  held-out family backlash dt=0.05: ARX 0.3271 | NARX2 120692312753.1690
  held-out family backlash dt=0.02: ARX 0.4630 | NARX2 143884443588.5283
  held-out excitation chirp (stribeck, dt=0.05): ARX 0.0039 | NARX2 450212432381.8998
  held-out excitation closedloop (stribeck, dt=0.05): ARX 0.0036 | NARX2 30608253972.9392
  held-out rate dt=0.05 stribeck: ARX 0.0077 | NARX2 80562087383.8275
  held-out rate dt=0.05 saturate: ARX 1346357896.3545 | NARX2 69342700687.7888
  held-out rate dt=0.05 boucwen: ARX 0.0311 | NARX2 72943611806.3270
  held-out rate dt=0.05 drivetrain: ARX 0.0253 | NARX2 50321803139.9900
=== classical baselines on real-plant windows (nMSE) ===
  Silverbox: ARX 0.0028 | NARX2 0.0028
  Cascaded_Tanks: ARX 0.0075 | NARX2 3665027259.1796
```

Timing: 6m 17s wall (`time python -m plantforge.baselines real`). This is the
first exercise of the `real` code path (flagged as untested in the Task 4
review) — it ran to completion with no traceback and produced sane ARX
numbers on both real-plant benchmarks; not a silent-crash finding.

## C. Identifiability vs difficulty (Spearman + quartiles)

```
=== identifiability vs in-context difficulty (corpus model, 5 seed(s), 1000/cell) ===
  instances scored: 39994 (dropped 6 non-finite)
  PRIMARY -- within-cell Spearman(max rel-CRLB, nMSE), aggregated over cells:
    median r = -0.122  mean r = -0.092  (cells: 40, positive r: 10/40, p<0.05 & r>0: 5/40 (descriptive; no multiplicity correction -- ~2/40 expected by chance))
    power-controlled: median r = -0.117  mean r = -0.103  (excl. bottom-decile den per cell)
    range-filtered: median r = -0.122  (cells with rel-CRLB IQR > 1e-6: 40/40)
      stribeck     prbs         dt=0.05  r=-0.140 (p=8.5e-06, n=1000)  r_pow=-0.115 (p=5.4e-04, n=900)  IQR=17.5
      stribeck     prbs         dt=0.02  r=-0.014 (p=6.6e-01, n=1000)  r_pow=-0.016 (p=6.3e-01, n=900)  IQR=10.9
      stribeck     multisine    dt=0.05  r=-0.241 (p=9.8e-15, n=1000)  r_pow=-0.249 (p=3.6e-14, n=900)  IQR=44.1
      stribeck     multisine    dt=0.02  r=-0.165 (p=1.6e-07, n=1000)  r_pow=-0.178 (p=8.2e-08, n=900)  IQR=26.6
      stribeck     chirp        dt=0.05  r=-0.010 (p=7.5e-01, n=1000)  r_pow=-0.003 (p=9.2e-01, n=900)  IQR=29.9
      stribeck     chirp        dt=0.02  r=-0.133 (p=2.4e-05, n=1000)  r_pow=-0.117 (p=4.6e-04, n=900)  IQR=18.5
      stribeck     closedloop   dt=0.05  r=-0.158 (p=4.7e-07, n=1000)  r_pow=-0.210 (p=2.1e-10, n=900)  IQR=74.2
      stribeck     closedloop   dt=0.02  r=-0.082 (p=9.1e-03, n=1000)  r_pow=-0.086 (p=1.0e-02, n=900)  IQR=45.9
      backlash     prbs         dt=0.05  r=-0.325 (p=5.7e-26, n=1000)  r_pow=-0.348 (p=5.5e-27, n=900)  IQR=1.74
      backlash     prbs         dt=0.02  r=-0.295 (p=1.6e-21, n=1000)  r_pow=-0.276 (p=3.1e-17, n=900)  IQR=1.11
      backlash     multisine    dt=0.05  r=+0.142 (p=6.1e-06, n=1000)  r_pow=+0.093 (p=5.4e-03, n=900)  IQR=3.3
      backlash     multisine    dt=0.02  r=+0.158 (p=5.2e-07, n=1000)  r_pow=+0.101 (p=2.5e-03, n=900)  IQR=2.08
      backlash     chirp        dt=0.05  r=-0.193 (p=7.7e-10, n=1000)  r_pow=-0.192 (p=6.0e-09, n=900)  IQR=2.04
      backlash     chirp        dt=0.02  r=-0.059 (p=6.4e-02, n=1000)  r_pow=-0.018 (p=6.0e-01, n=900)  IQR=1.27
      backlash     closedloop   dt=0.05  r=+0.327 (p=2.4e-26, n=1000)  r_pow=+0.246 (p=6.6e-14, n=900)  IQR=4.68
      backlash     closedloop   dt=0.02  r=+0.315 (p=1.6e-24, n=1000)  r_pow=+0.228 (p=4.2e-12, n=900)  IQR=2.99
      saturate     prbs         dt=0.05  r=-0.239 (p=2.0e-14, n=1000)  r_pow=-0.228 (p=4.3e-12, n=900)  IQR=1.31
      saturate     prbs         dt=0.02  r=-0.228 (p=2.7e-13, n=1000)  r_pow=-0.235 (p=9.2e-13, n=900)  IQR=0.827
      saturate     multisine    dt=0.05  r=-0.236 (p=4.2e-14, n=1000)  r_pow=-0.257 (p=4.9e-15, n=900)  IQR=3.13
      saturate     multisine    dt=0.02  r=-0.169 (p=7.7e-08, n=1000)  r_pow=-0.184 (p=2.8e-08, n=900)  IQR=1.94
      saturate     chirp        dt=0.05  r=-0.270 (p=4.0e-18, n=1000)  r_pow=-0.289 (p=9.8e-19, n=900)  IQR=1.86
      saturate     chirp        dt=0.02  r=-0.194 (p=6.4e-10, n=1000)  r_pow=-0.182 (p=3.9e-08, n=900)  IQR=1.18
      saturate     closedloop   dt=0.05  r=-0.254 (p=3.9e-16, n=1000)  r_pow=-0.270 (p=1.6e-16, n=900)  IQR=3.96
      saturate     closedloop   dt=0.02  r=-0.130 (p=3.6e-05, n=1000)  r_pow=-0.155 (p=2.8e-06, n=900)  IQR=2.47
      boucwen      prbs         dt=0.05  r=-0.213 (p=1.0e-11, n=1000)  r_pow=-0.206 (p=4.6e-10, n=900)  IQR=0.512
      boucwen      prbs         dt=0.02  r=-0.154 (p=1.1e-06, n=1000)  r_pow=-0.208 (p=3.3e-10, n=900)  IQR=0.277
      boucwen      multisine    dt=0.05  r=-0.101 (p=1.4e-03, n=1000)  r_pow=-0.109 (p=1.1e-03, n=900)  IQR=0.879
      boucwen      multisine    dt=0.02  r=-0.103 (p=1.2e-03, n=1000)  r_pow=-0.106 (p=1.4e-03, n=900)  IQR=0.505
      boucwen      chirp        dt=0.05  r=-0.114 (p=3.1e-04, n=1000)  r_pow=-0.117 (p=4.2e-04, n=900)  IQR=0.703
      boucwen      chirp        dt=0.02  r=-0.055 (p=8.3e-02, n=1000)  r_pow=-0.018 (p=5.9e-01, n=900)  IQR=0.392
      boucwen      closedloop   dt=0.05  r=-0.260 (p=7.2e-17, n=1000)  r_pow=-0.284 (p=3.5e-18, n=900)  IQR=1.45
      boucwen      closedloop   dt=0.02  r=-0.136 (p=1.5e-05, n=1000)  r_pow=-0.168 (p=4.0e-07, n=900)  IQR=0.91
      drivetrain   prbs         dt=0.05  r=+0.109 (p=5.5e-04, n=1000)  r_pow=+0.136 (p=4.4e-05, n=900)  IQR=0.41
      drivetrain   prbs         dt=0.02  r=-0.018 (p=5.7e-01, n=1000)  r_pow=-0.003 (p=9.3e-01, n=900)  IQR=0.271
      drivetrain   multisine    dt=0.05  r=+0.012 (p=7.1e-01, n=1000)  r_pow=+0.033 (p=3.3e-01, n=900)  IQR=1.64
      drivetrain   multisine    dt=0.02  r=+0.018 (p=5.8e-01, n=1000)  r_pow=-0.010 (p=7.8e-01, n=900)  IQR=1.03
      drivetrain   chirp        dt=0.05  r=-0.109 (p=5.3e-04, n=1000)  r_pow=-0.147 (p=9.7e-06, n=900)  IQR=0.869
      drivetrain   chirp        dt=0.02  r=+0.035 (p=2.7e-01, n=1000)  r_pow=+0.035 (p=3.0e-01, n=900)  IQR=0.551
      drivetrain   closedloop   dt=0.05  r=+0.010 (p=7.5e-01, n=1000)  r_pow=+0.014 (p=6.8e-01, n=900)  IQR=2
      drivetrain   closedloop   dt=0.02  r=+0.002 (p=9.6e-01, n=1000)  r_pow=-0.014 (p=6.7e-01, n=900)  IQR=1.27
  SECONDARY (pooled across cells -- confounded by between-cell structure; see within-cell above):
    Spearman(max rel-CRLB, nMSE): r=-0.088 (p=3.1e-69)
    Spearman(log10 FIM cond, nMSE): r=-0.316 (p=0.0e+00)
      stribeck: r=-0.081 (p=4.4e-13, n=8000)
      backlash: r=0.294 (p=2.6e-159, n=8000)
      saturate: r=-0.148 (p=1.9e-40, n=8000)
      boucwen: r=-0.050 (p=8.2e-06, n=8000)
      drivetrain: r=0.091 (p=3.8e-16, n=7994)
  quartiles of max rel-CRLB -> nMSE (mean is heavy-tail-dominated; median is the defensible summary):
    Q1: metric~0.262  mean nMSE=1517.2062  median nMSE=0.0367  (n=9999)
    Q2: metric~0.752  mean nMSE=93761480.0000  median nMSE=0.0375  (n=9998)
    Q3: metric~2.12  mean nMSE=490145888.0000  median nMSE=0.0404  (n=9998)
    Q4: metric~91  mean nMSE=1589712128.0000  median nMSE=0.0211  (n=9999)
```

Timing: 58s wall (`time python -m plantforge.ident_exp`, default 5 seeds x 40
cells x 1000 instances/cell). No tracebacks; 39994/40000 instances scored (6
dropped as non-finite).

## Reading notes

- **The corpus-vs-wh_only gap mostly survives error bars, with one clean
  exception.** Using mean±1std as an interval, corpus beats wh_only with
  non-overlapping bands on 10 of 11 held-out cells (all three backlash dts,
  both held-out excitations, stribeck/boucwen/drivetrain rate variants, and
  both real-plant benchmarks — most dramatically Silverbox: wh_only
  0.958±0.217 vs corpus 0.331±0.040, and Cascaded_Tanks: wh_only 0.394±0.081
  vs corpus 0.084±0.043). The exception is held-out rate saturate, where
  wh_only is *better* (0.0141±0.0016 vs corpus 0.0285±0.0021, non-overlapping
  the other way) — worth flagging rather than smoothing over, since it
  contradicts the "corpus training always helps" framing on at least one
  cell. **Statistical caveat:** non-overlapping ±1std is a descriptive
  heuristic across 11 simultaneous cell comparisons, not a
  multiplicity-corrected test (the same caveat already applied to the
  identifiability experiment's within-cell aggregate below). At n=5 seeds,
  std is a conservative bar relative to the standard error of the mean, and
  the load-bearing cells (Silverbox 0.958 vs 0.331, Cascaded_Tanks 0.394 vs
  0.084) clear it by a wide margin, so the headline conclusion is not
  multiplicity-sensitive — but the count "10 of 11" should be read as
  descriptive, the same way the ident_exp positive-cell counts are.
- **ARX is a strong baseline that both neural models lose to on several
  cells, and it is not uniformly weaker than the "hard" cases.** ARX beats
  both corpus and wh_only outright on held-out excitation chirp (0.0039 vs
  0.0383/0.1533), held-out excitation closedloop (0.0036 vs 0.0195/0.0545),
  held-out rate drivetrain (0.0253 vs 0.1722/0.2180), and crushes both models
  on real-plant Silverbox (0.0028 vs 0.331/0.958) and Cascaded_Tanks (0.0075
  vs 0.084/0.394). It is roughly tied with corpus on held-out family backlash
  dt=0.10 (0.3116 vs 0.3107). This says the family-transfer degradation isn't
  just "hard for everyone" — a simple linear context-fit is *better* than
  both trained transformers on several distribution shifts, which should
  temper any claim that in-context transformers are uniformly stronger
  general-purpose identifiers. **Protocol note (verified by review):** ARX
  and the transformer are evaluated under an identical protocol — same
  windows, same normalization, same 192/32 context/query split, neither sees
  query-horizon ground truth. ARX fits a linear model per window (order
  k∈{2,4,8} selected on held-out context samples); the transformer adapts
  in-context via a frozen forward pass — both methods "look at the context
  and adapt," so this is not an artifact of unfair per-window fitting. On
  rate-shifted and real-plant cells specifically, ARX's per-window re-fit at
  the test rate makes it structurally immune to the sample-rate mismatch the
  frozen transformer must extrapolate through (most acute on Cascaded_Tanks,
  a 40x rate extrapolation for the transformer) — this is a genuine
  limitation of the frozen in-context approach, not a comparison bug.
  **Sample-size note:** real-plant nMSE is computed over ≤8 non-overlapping
  windows (`WINDOW_CAP=8`); ARX/NARX2 there are single deterministic
  evaluations, whereas the transformer ±std reflects 5-seed model variance on
  the same fixed windows. The 30x–300x ARX margins far exceed this variance,
  but the real-plant numbers are not seed-averaged and should be reported as
  such.
- **NARX2 diverges broadly in free-run and should not be treated as a
  comparable number** — only on real-plant Silverbox does it produce a sane
  nMSE (0.0028, matching ARX); every other cell is in the 1e10–1e11 range,
  including even the "reference" (train-like) cell. The one new
  divergence spotted this run: **ARX itself explodes on held-out rate
  saturate** (1.35e9), unlike its behavior everywhere else — a linear
  context-fit apparently goes unstable in free-run specifically on the
  saturate family, while both neural models stay well-behaved there. That's
  a second, independent anomaly worth a footnote, distinct from NARX2's
  known general divergence.
- **The `real` baseline path ran clean on first real exercise.** It was
  flagged in the Task 4 review as untested against actual network-fetched
  real-plant loaders; this run completed in 6m17s with no traceback and
  produced physically sensible ARX numbers on both Silverbox and
  Cascaded_Tanks. No crash to report — a green result, not a null one.
- **Multi-seed within-cell Spearman confirms the seed-0 finding: weak,
  mostly negative, and non-uniform in sign across families.** Median r
  across cells is -0.122 (base) / -0.117 (power-controlled) / -0.122
  (range-filtered, all 40/40 cells pass the IQR filter) — close to the
  seed-0 values of -0.137/-0.115 previously reported, confirming those
  weren't a seed-0 artifact. Only 10/40 cells have positive r, concentrated
  in backlash-multisine and backlash-closedloop (r up to +0.327); everything
  else is negative or near zero. The negative sign means *higher*
  identifiability difficulty (rel-CRLB) tends to associate with *lower*
  prediction error, not higher — the opposite of the naive intuition that
  hard-to-identify systems are hard to predict. The quartile table makes the
  same point structurally: median nMSE is flat-to-declining across rel-CRLB
  quartiles (0.037 -> 0.037 -> 0.040 -> 0.021), while the mean is wrecked by
  heavy tails (1.5e3 -> 9.4e7 -> 4.9e8 -> 1.6e9). This is consistent with the
  paper's framing — prediction difficulty decouples from parameter-recovery
  difficulty — rather than any claim that identifiability annotations
  predict prediction difficulty.
- **Pooled (secondary) correlations are sign-inconsistent across families,
  which is exactly why the within-cell analysis is primary.** The pooled
  log10-FIM-condition correlation is r=-0.316 (p≈0) with per-family pooled
  breakdowns of opposite sign: backlash +0.294 and drivetrain +0.091 vs
  stribeck -0.081, saturate -0.148, boucwen -0.050. A single pooled number
  here would be an artifact of between-family structure (different families
  occupy different nMSE and rel-CRLB ranges), not a within-family
  identifiability-difficulty effect — the within-cell Spearman table is the
  number to cite.
