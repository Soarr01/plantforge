# Training-divergence investigation after the normalization-leak fix (2026-07-21)

## Context

The adversarial-review fix campaign (branch `adversarial-review-fixes`, merged
`6a5dd9e`) changed `evaluate._norm` from full-sequence-std normalization to
context-only-std normalization, to remove a query-horizon information leak.
After deleting all 50 old checkpoints and retraining under the fixed code
(across GPU 0 + GPU 1), a checkpoint scan found **20 of 50 checkpoints
diverged to all-NaN weights** despite all reaching `step=10000` and looking
"finished". Most alarmingly, **all 5 default-corpus checkpoints
(`eval_corpus_s0..4.pt`) — the paper's central config — were all-NaN**, while
all 5 `wh_only` checkpoints were fine.

Divergence rate by config:
- default corpus / architecture-ablation variants (all `backlash` held out): 17/25 diverged
- leave-one-out (non-backlash held out): 3/20 diverged (boucwen s1/s4, stribeck s1)
- wh_only: 0/5

This document records the read-only investigation that established the root
cause before any fix was designed. Two Sonnet-tier investigation agents plus
several direct controlled experiments were used; **every claim below is backed
by a reproduced measurement, not a hypothesis.**

## What was ruled OUT

1. **Not a simple normalization scale blowup.** A per-cell scan (60 batches ×
   20 cells) showed the only catastrophic scale blowup is `backlash|multisine`
   (context std hits 0.0 via the deadzone locking the output flat → normalized
   query magnitude up to **36,670** under context-only norm vs 12.4 under
   full-sequence). BUT `backlash` is *held out* of the default corpus recipe,
   so those batches are never trained on. Across the default recipe's 4
   actually-trained families (stribeck, saturate, boucwen, drivetrain),
   **0 of ~92,000 sampled rows** exceeded a normalized-query magnitude of 12.6
   — nothing catastrophic. So the config that diverges most never even reaches
   the pathological scale regime. (This paradox is why the naive "clamp the
   scale" fix would not work.)

2. **Not the `-inf` causal attention mask** (a classic softmax-backward nan
   footgun). Swapping the model's `-inf` mask for a finite `-1e9` mask did
   **not** prevent the nan gradients (see the A/B table below).

3. **Not reproducible with a single fixed data pool.** A clean 2500-step
   single-pool run under the new norm did NOT diverge — because the real
   training driver *rebuilds the data pool on every resume*
   (`run_salt = done*13 + SEED*1e6`, chunked by `PF_BUDGET=500s`), so over a
   full run it draws far more distinct batches and eventually hits a rare bad
   one. The clean single-pool test is not representative.

## Root cause (confirmed by direct reproduction)

The real `eval_corpus_s0.pt` run diverged at step ~1519, in the data pool
built at its resume point `done=1441` (`run_salt = 1441*13 = 18733`).
Rebuilding that exact pool and probing it:

- The pool's max normalized-query magnitude is only **~10.6** (boucwen/prbs) —
  **not** catastrophic, consistent with the scan above.
- Probing the worst batches through a fresh model, **batch 78 (boucwen, prbs,
  dt=0.02) has a perfectly normal finite loss of 3.54 but produces non-finite
  gradients in the backward pass**, which makes
  `nn.utils.clip_grad_norm_(model.parameters(), 1.0)` return
  `total_norm = nan`.

The mechanism, end to end:
1. A rare batch produces a **finite forward loss but non-finite backward
   gradients** under context-only normalization.
2. `clip_grad_norm_` computes `total_norm = nan`, so
   `clip_coef = max_norm / (total_norm + 1e-6) = nan`, and multiplies **every**
   gradient by nan → all gradients become nan. (Gradient clipping, meant to be
   the safety net, is exactly where the nan spreads.)
3. `opt.step()` writes nan into **every** model weight.
4. The existing `if not torch.isfinite(loss): skip` guard never fires, because
   the loss was finite (3.54).
5. From then on every forward pass gives nan loss → the "step N: non-finite
   loss, skipping batch" spam seen in the logs → the run burns through to
   step 10000 and **silently saves an all-NaN checkpoint** that passes the
   step-count "finished" check.

### The decisive A/B (same raw boucwen/prbs/dt=0.02 draw, seed 78812)

| condition | loss | grad non-finite? |
|---|---|---|
| **NEW norm** (context-only), `-inf` mask | 3.536 | **YES** |
| OLD norm (full-sequence), `-inf` mask | 3.208 | no |
| NEW norm, `-1e9` finite mask | 3.543 | **YES** |
| OLD norm, `-1e9` finite mask | 3.208 | no |

→ The context-only normalization is the **confirmed causal trigger** (same
batch, only the normalization differs). The mask is not involved. The forward
loss is finite and small in every case — this is a **backward-pass** numerical
fragility that context-only normalization exposes (most likely a LayerNorm /
attention backward hitting a near-zero-variance or extreme intermediate on the
context-only-scaled inputs), not a forward overflow.

## Frequency

The nan-gradient batches are **rare**: a 960-batch scan (60 draws × 16 trained
cells, fresh model) found **0** — yet a deterministic one exists in the actual
divergence pool (found within a few hundred tries when scanning the divergence
pool's seed range, and reproduced above). So the rate is well under ~1%, but
across a full multi-resume run (thousands of distinct batches) the probability
of hitting at least one is high — matching the observed per-config divergence
rates (higher for configs that draw more boucwen/prbs batches).

Rarity is the key design fact: **skipping the rare bad batch costs training
essentially nothing.**

## Implication for the fix

- The correctness of the context-only normalization (the leak fix) is NOT in
  question and should be kept.
- The training loop has a real robustness gap: it steps on non-finite
  gradients (via `clip_grad_norm_` propagating nan) and silently persists
  all-NaN checkpoints. A gradient-finiteness guard that skips the step when the
  gradient norm is non-finite would have prevented **all** of these
  divergences, and — because bad batches are rare — with negligible training
  impact. A separate guard should refuse to save an all-NaN checkpoint (fail
  loudly) so any future divergence can never again be silent.

Design and implementation follow in
`docs/superpowers/specs/2026-07-21-training-divergence-guard-design.md` and its
plan.
