"""Regenerate all paper figures from the verified experiment results.

Every number below is transcribed verbatim from the 2026-07-23 post-fix
result files (docs/superpowers/results/2026-07-23-{aggregate,ident-exp}-post-fix.txt;
5 seeds, PF_SEED=0..4, 10k steps, all 50 checkpoints retrained under the
training-divergence guard; identifiability run at full scale, 5 seeds x 40
cells x 1000 instances/cell) -- this script does not compute anything new,
it only visualizes already-reported, already-reviewed numbers.

    python figures/make_figures.py
writes fig1_transfer_matrix.{png,pdf} .. fig4_quartile_artifact.{png,pdf}
into figures/.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent

# Reference categorical palette (dataviz skill, references/palette.md),
# fixed order: wh_only=blue, corpus=aqua, ARX=yellow, NARX2=red.
C_WH = "#2a78d6"
C_CORPUS = "#1baf7a"
C_ARX = "#eda100"
C_NARX2 = "#e34948"
C_DIVERGE_POS = "#2a78d6"   # diverging pair poles (blue<->red), gray midpoint
C_DIVERGE_NEG = "#e34948"
C_NEUTRAL = "#8a8a86"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e2dc"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
})


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png / {name}.pdf")


# ---------------------------------------------------------------------------
# Figure 1: synthetic transfer matrix, wh_only vs corpus, mean+-std, log-y
# ---------------------------------------------------------------------------
def fig1_transfer_matrix():
    cells = [
        "reference\n(train-like)",
        "backlash\ndt=.10",
        "backlash\ndt=.05",
        "backlash\ndt=.02",
        "chirp",
        "closedloop",
        "stribeck",
        "saturate",
        "boucwen",
        "drivetrain",
    ]
    wh_mean = [0.0034, 0.3800, 0.3883, 0.6112, 0.1417, 0.0636, 0.0300, 0.0140, 0.0869, 0.2036]
    wh_std = [0.0001, 0.0081, 0.0085, 0.0065, 0.0393, 0.0162, 0.0012, 0.0005, 0.0068, 0.0022]
    co_mean = [0.0192, 0.3205, 0.3448, 0.5988, 0.0355, 0.0273, 0.0210, 0.0287, 0.0186, 0.1691]
    co_std = [0.0007, 0.0147, 0.0161, 0.0155, 0.0129, 0.0112, 0.0017, 0.0029, 0.0015, 0.0108]

    x = np.arange(len(cells))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 3.0)
    ax.axvspan(-0.5, 3.5, color=GRID, alpha=0.5, zorder=0)
    ax.bar(x - w / 2, wh_mean, w, yerr=wh_std, capsize=2, color=C_WH,
           label="wh–only (WH-only training)", hatch="//", edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x + w / 2, co_mean, w, yerr=co_std, capsize=2, color=C_CORPUS,
           label="corpus (multi-axis training)", edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_ylabel("in-context nMSE (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(cells, fontsize=8)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Synthetic transfer matrix — 5-seed mean±std (n=5)", loc="left")

    # group super-labels below the x-axis tick labels
    group_spans = [(-0.5, 3.5, "held-out FAMILY (open challenge)"),
                   (3.5, 5.5, "held-out EXCITATION"),
                   (5.5, 9.5, "held-out RATE (dt=0.05)")]
    for lo, hi, label in group_spans:
        ax.annotate("", xy=(lo + 0.08, -0.17), xytext=(hi - 0.08, -0.17),
                    xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7))
        ax.text((lo + hi) / 2, -0.21, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.8, color=MUTED)

    _save(fig, "fig1_transfer_matrix")


# ---------------------------------------------------------------------------
# Figure 2: real-plant zero-shot, wh_only vs corpus vs ARX, log-y
# ---------------------------------------------------------------------------
def fig2_real_plant():
    datasets = ["Silverbox\n(decimated ≈50Hz)", "Cascaded_Tanks\n(native dt=4s, extrapolation)",
                "Bouc-Wen\n(decimated 15x, exact 50Hz)"]
    wh_mean = [0.7941, 0.3653, 1.9684]
    wh_std = [0.2098, 0.1564, 0.4163]
    co_mean = [0.4863, 0.0415, 1.3497]
    co_std = [0.0937, 0.0121, 0.1756]
    arx = [0.0028, 0.0073, 0.0616]   # single deterministic run, no std -- noted below

    x = np.arange(len(datasets))
    w = 0.22
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.set_yscale("log")
    ax.set_ylim(1.5e-3, 5.0)
    ax.bar(x - w, wh_mean, w, yerr=wh_std, capsize=3, color=C_WH, hatch="//",
           edgecolor="white", linewidth=0.5, label="wh–only")
    ax.bar(x, co_mean, w, yerr=co_std, capsize=3, color=C_CORPUS,
           edgecolor="white", linewidth=0.5, label="corpus")
    ax.bar(x + w, arx, w, color=C_ARX, hatch="xx", edgecolor="white",
           linewidth=0.5, label="ARX (classical, per-window fit)")

    for xi, (wv, cv, av) in enumerate(zip(wh_mean, co_mean, arx)):
        ax.annotate(f"{wv:.3f}", (xi - w, wv), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=6.5, color=MUTED)
        ax.annotate(f"{cv:.3f}", (xi, cv), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=6.5, color=MUTED)
        ax.annotate(f"{av:.4f}", (xi + w, av), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=6.5, color=MUTED)

    ax.set_ylabel("zero-shot nMSE (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=8)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=7.5, borderaxespad=0)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Zero-shot on real plants — a classical baseline\nbeats both trained transformers",
                 loc="left", fontsize=9)
    ax.text(0.0, -0.30, "ARX: single deterministic run, ≤8 windows (no std shown; NARX2 omitted --\n"
            "diverges except on Silverbox/Bouc-Wen, see paper Section 4.3).\n"
            "wh_only/corpus: mean±std, 5 training seeds, same fixed windows.",
            transform=ax.transAxes, fontsize=6, color=MUTED, va="top")

    _save(fig, "fig2_real_plant")


# ---------------------------------------------------------------------------
# Figure 3: within-cell Spearman r, 40 cells, sign-colored strip + median
# ---------------------------------------------------------------------------
CELL_R = [
    ("stribeck", "prbs", 0.05, -0.152), ("stribeck", "prbs", 0.02, -0.082),
    ("stribeck", "multisine", 0.05, -0.292), ("stribeck", "multisine", 0.02, -0.247),
    ("stribeck", "chirp", 0.05, -0.011), ("stribeck", "chirp", 0.02, -0.217),
    ("stribeck", "closedloop", 0.05, -0.179), ("stribeck", "closedloop", 0.02, -0.110),
    ("backlash", "prbs", 0.05, -0.335), ("backlash", "prbs", 0.02, -0.312),
    ("backlash", "multisine", 0.05, 0.170), ("backlash", "multisine", 0.02, 0.198),
    ("backlash", "chirp", 0.05, -0.152), ("backlash", "chirp", 0.02, -0.106),
    ("backlash", "closedloop", 0.05, 0.331), ("backlash", "closedloop", 0.02, 0.340),
    ("saturate", "prbs", 0.05, -0.252), ("saturate", "prbs", 0.02, -0.242),
    ("saturate", "multisine", 0.05, -0.216), ("saturate", "multisine", 0.02, -0.138),
    ("saturate", "chirp", 0.05, -0.246), ("saturate", "chirp", 0.02, -0.044),
    ("saturate", "closedloop", 0.05, -0.257), ("saturate", "closedloop", 0.02, -0.149),
    ("boucwen", "prbs", 0.05, -0.190), ("boucwen", "prbs", 0.02, -0.115),
    ("boucwen", "multisine", 0.05, -0.091), ("boucwen", "multisine", 0.02, -0.086),
    ("boucwen", "chirp", 0.05, -0.114), ("boucwen", "chirp", 0.02, -0.028),
    ("boucwen", "closedloop", 0.05, -0.248), ("boucwen", "closedloop", 0.02, -0.090),
    ("drivetrain", "prbs", 0.05, 0.095), ("drivetrain", "prbs", 0.02, 0.014),
    ("drivetrain", "multisine", 0.05, 0.027), ("drivetrain", "multisine", 0.02, 0.031),
    ("drivetrain", "chirp", 0.05, -0.222), ("drivetrain", "chirp", 0.02, 0.011),
    ("drivetrain", "closedloop", 0.05, 0.001), ("drivetrain", "closedloop", 0.02, -0.018),
]
FAMILIES = ["stribeck", "backlash", "saturate", "boucwen", "drivetrain"]


def fig3_within_cell_spearman():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rng = np.random.default_rng(0)
    for fi, fam in enumerate(FAMILIES):
        vals = [r for f, _, _, r in CELL_R if f == fam]
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        colors = [C_DIVERGE_POS if v >= 0 else C_DIVERGE_NEG for v in vals]
        ax.scatter(np.full(len(vals), fi) + jitter, vals, c=colors, s=26,
                   edgecolor="white", linewidth=0.4, zorder=3)
        med = np.median(vals)
        ax.plot([fi - 0.28, fi + 0.28], [med, med], color=TEXT, linewidth=1.6, zorder=4)

    all_r = [r for _, _, _, r in CELL_R]
    overall_med = np.median(all_r)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="-", zorder=1)
    ax.axhline(overall_med, color=TEXT, linewidth=1.0, linestyle="--", zorder=2)
    ax.text(4.65, overall_med, f"  overall median r = {overall_med:.3f}",
            fontsize=7, color=TEXT, va="center")

    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels(FAMILIES, fontsize=8.5)
    ax.set_ylabel("within-cell Spearman r\n(max rel-CRLB vs. per-instance nMSE)")
    ax.set_title("Identifiability does not positively predict prediction difficulty\n"
                 "40 cells (5 families × 4 excitations × 2 rates), corpus model, 5 seeds",
                 loc="left", fontsize=9)
    ax.text(0.99, 0.03,
            f"10/40 cells positive · black tick = per-family median · "
            f"blue = r≥0, red = r<0",
            transform=ax.transAxes, fontsize=6.5, color=MUTED, ha="right")
    ax.set_ylim(-0.4, 0.4)

    _save(fig, "fig3_within_cell_spearman")


# ---------------------------------------------------------------------------
# Figure 4: quartile mean vs median -- the heavy-tail artifact
# ---------------------------------------------------------------------------
def fig4_quartile_artifact():
    q_labels = ["Q1\n(metric≈.41)", "Q2\n(metric≈1.2)", "Q3\n(metric≈3.2)", "Q4\n(metric≈250)"]
    q_mean = [2.4397, 11425202.0, 317037280.0, 931860480.0]
    q_median = [0.0345, 0.0396, 0.0430, 0.0207]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    x = np.arange(4)

    ax1.plot(x, q_mean, marker="o", color=C_DIVERGE_NEG, linewidth=1.8, markersize=5)
    ax1.set_yscale("log")
    ax1.set_ylim(1, 3e9)
    ax1.set_title("bin MEAN nMSE\n(looks like a strong positive trend)", fontsize=8.5, loc="left")
    ax1.set_ylabel("nMSE (log scale)")
    for xi, v in zip(x, q_mean):
        dx = 10 if xi == 0 else 0
        ax1.annotate(f"{v:.1e}", (xi, v), textcoords="offset points", xytext=(dx, 6),
                     ha="left" if xi == 0 else "center", fontsize=6.5, color=MUTED)

    ax2.plot(x, q_median, marker="o", color=C_DIVERGE_POS, linewidth=1.8, markersize=5)
    ax2.set_title("bin MEDIAN nMSE\n(the defensible summary: flat)", fontsize=8.5, loc="left")
    ax2.set_ylabel("nMSE")
    ax2.set_ylim(0, 0.05)
    for xi, v in zip(x, q_median):
        ax2.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=6.5, color=MUTED)

    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(q_labels, fontsize=7)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.suptitle("Quartiles of max rel-CRLB → per-instance nMSE: mean is\n"
                 "heavy-tail-dominated, median is flat (n≈10,000/bin)",
                 fontsize=9, x=0.02, ha="left", y=0.99)

    _save(fig, "fig4_quartile_artifact")


if __name__ == "__main__":
    fig1_transfer_matrix()
    fig2_real_plant()
    fig3_within_cell_spearman()
    fig4_quartile_artifact()
