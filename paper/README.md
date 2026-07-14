# Paper draft

`main.tex` — NeurIPS 2024 D&B-track-formatted draft (default/anonymous
submission mode; the style file forces the "Anonymous Author(s)" placeholder
regardless of the `\author{}` content — this is expected NeurIPS behavior, not
a bug). `main.pdf` is the compiled, committed output so the draft is readable
without a LaTeX install.

Every number, table, and figure in the draft is transcribed from
[`../docs/superpowers/results/2026-07-14-experiment-results.md`](../docs/superpowers/results/2026-07-14-experiment-results.md)
and [`../figures/`](../figures/) — no new computation happens in the paper
source. All citations were independently web-verified before inclusion (see
commit history); one citation present in an earlier README draft
("forgi86 lineage ... RAL'25") could not be verified and was dropped, not
included on trust.

**Before this becomes a real submission**, at minimum:
- Fill in real author names/affiliation and swap to `\usepackage[final]{neurips_2024}` for the camera-ready version (never for the anonymous review version).
- Restore the dataset citation URL in `references.bib` (currently omitted for anonymity) in the camera-ready `\bibitem`.
- Expand Related Work if a reviewer asks for broader coverage — this draft cites only the four sources already load-bearing in this repo's README, not a full literature sweep.
- Address the ablation/family-sweep limitations named in Section 5 if reviewers require them, or defend their absence.

## Rebuild

```
sudo apt-get install texlive-latex-base texlive-latex-recommended \
  texlive-fonts-recommended texlive-latex-extra   # one-time
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
