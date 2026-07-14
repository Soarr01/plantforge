# Figures

Regenerate all four from source: `python figures/make_figures.py` (from the
repo root; needs `matplotlib`, listed in `requirements.txt`).
Every number plotted is transcribed verbatim from
[`docs/superpowers/results/2026-07-14-experiment-results.md`](../docs/superpowers/results/2026-07-14-experiment-results.md)
— the script computes nothing new, it only visualizes already-reviewed results.

| file | shows |
|---|---|
| `fig1_transfer_matrix` | Synthetic held-out transfer matrix, `wh_only` vs `corpus`, 5-seed mean±std, log-y. The headline "does multi-axis training help" result. |
| `fig2_real_plant` | Zero-shot nMSE on Silverbox / Cascaded_Tanks: `wh_only`, `corpus`, and the ARX classical baseline. The "ARX beats both transformers on real data" finding, with its sample-size/protocol caveats in the figure footnote. |
| `fig3_within_cell_spearman` | Within-cell Spearman r (max rel-CRLB vs. per-instance nMSE), all 40 (family × excitation × rate) cells, corpus model, 5 seeds. The identifiability-does-not-predict-difficulty null result — mostly negative, heterogeneous by family. |
| `fig4_quartile_artifact` | Same data, binned into rel-CRLB quartiles: bin **mean** nMSE (heavy-tail-dominated, looks like a strong trend) vs. bin **median** nMSE (flat — the defensible summary). Illustrates why the pooled/mean framing in an earlier design-iteration of this experiment was misleading (see the plan/spec docs for that history).

Palette and mark choices (categorical order, hatching for print/CVD safety,
diverging blue/red for signed correlations, direct value labels) follow the
`dataviz` skill's color-formula and anti-patterns guidance; the categorical
palette was validated with `scripts/validate_palette.js` before use.
