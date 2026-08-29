"""Figures for the SAE selection-rule ablation report. Reads results.json only."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from geodesic_style import COLORS, FIGURE_SIZES, apply_style, save_figure

HERE = Path(__file__).resolve().parent
OUT = HERE / "build" / "figs"

JUDGE = (COLORS["blue"], "o", "-", "judge pool")
ALLD = (COLORS["orange"], "s", "--", "full dictionary")


def load():
    return json.loads((HERE / "results.json").read_text())


def frontier(block, pool):
    return sorted((v for k, v in block.items() if k.startswith(pool + " ")),
                  key=lambda r: r["fp"])


def _panel_letters(axes):
    for ax, letter in zip(axes, "ABCDEF"):
        ax.set_title(letter, loc="left", fontweight="bold")


def _roc_line(ax, d, pop, color, ls, label):
    """Solid curve over the supported (positive-score) region; dashed
    tie-segment to (1,1) for the zero-score mass."""
    c = d[f"curve_{pop}"]
    f = np.asarray(c["fpr"]); t = np.asarray(c["tpr"])
    sup = d.get(f"support_{pop}")
    if sup and sup["fpr0"] < 0.999:
        f0, t0 = sup["fpr0"], sup["tpr0"]
        m = f <= f0
        fs = np.concatenate([f[m], [f0]]); ts = np.concatenate([t[m], [t0]])
        ax.plot(fs, ts, ls, color=color, label=label, lw=1.4)
        ax.plot([f0, 1.0], [t0, 1.0], ":", color=color, lw=0.9, alpha=0.55)
    else:
        ax.plot(f, t, ls, color=color, label=label, lw=1.4)


def fig_roc(R):
    """Page-1 figure: ROC curves of the continuous score, judge vs full dict."""
    roc = R["roc_65k"]
    series = [
        ("judge rho>=10", COLORS["blue"], "-", "judge · screen ρ≥10"),
        ("all rho>=35", COLORS["orange"], "--", "full dict. · screen ρ≥35"),
        ("judge pool (no screen)", COLORS["sky_blue"], "-.", "judge · no screen"),
        ("full dictionary (no screen)", COLORS["black"], ":", "full dict. · no screen"),
    ]
    pops = [("all_forget", "TPR, all held-out forget"),
            ("span", "TPR, human-labelled spans"),
            ("c2", "TPR, held-out category")]
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_SIZES["double_column"][0],
                                            FIGURE_SIZES["double_column"][1] * 0.85))
    for ax, (pop, ylab) in zip(axes, pops):
        for key, color, ls, label in series:
            _roc_line(ax, roc[key], pop, color, ls, label)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("control false-positive rate")
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, loc="upper left", fontsize=6.5)
    _panel_letters(axes)
    fig.tight_layout()
    save_figure(fig, OUT / "fig_roc")


def fig_judge(R):
    blk = R["judge_frontier_k4"]
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_SIZES["double_column"][0],
                                            FIGURE_SIZES["double_column"][1] * 0.85))
    panels = [("F2all", "held-out F2 (all domains)"),
              ("F2in", "held-out F2 (in-distribution)"),
              ("F2out", "held-out F2 (held-out domain)")]
    for ax, (metric, ylab) in zip(axes, panels):
        for pool, (color, marker, ls, label) in (("judge", JUDGE), ("all", ALLD)):
            rows = frontier(blk, pool)
            xs = np.array([r["fp"] for r in rows])
            ys = np.array([r[metric] for r in rows])
            es = np.array([r["se"][metric] for r in rows])
            ax.plot(xs, ys, ls, marker=marker, color=color, label=label, lw=1.4, ms=4)
            ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.18, lw=0)
        ax.set_xscale("log")
        ax.set_xlabel("control false-positive rate")
        ax.set_ylabel(ylab)
        ax.set_ylim(0.1, 1.0)
    axes[0].legend(frameon=False, loc="lower right", fontsize=7)
    _panel_letters(axes)
    fig.tight_layout()
    save_figure(fig, OUT / "fig_judge")


def fig_vote(R):
    blk = R["vote_k"]
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_SIZES["double_column"][0],
                                            FIGURE_SIZES["double_column"][1] * 0.85))
    styles = {5: (COLORS["bluish_green"], "o", "-", "screen ρ≥5"),
              10: (COLORS["reddish_purple"], "s", "--", "screen ρ≥10")}
    ks = [1, 2, 3, 4, 6, 8]
    for metric, ylab, ax in (("F2all", "held-out F2 (all domains)", axes[0]),
                             ("Rspan", "span recall", axes[1])):
        for rho, (color, marker, ls, label) in styles.items():
            rows = [blk[f"rho={rho} k={k}"] for k in ks]
            ax.errorbar(ks, [r[metric] for r in rows],
                        yerr=[r["se"][metric] for r in rows],
                        fmt=ls, marker=marker, color=color, label=label,
                        lw=1.4, ms=4, capsize=2, elinewidth=0.9)
        ax.set_xlabel("vote k")
        ax.set_ylabel(ylab)
        ax.set_xticks(ks)
    ax = axes[2]
    for rho, (color, marker, ls, label) in styles.items():
        rows = [blk[f"rho={rho} k={k}"] for k in ks]
        ax.errorbar(ks, [r["fp"] for r in rows], yerr=[r["se"]["fp"] for r in rows],
                    fmt=ls, marker=marker, color=color, label=label,
                    lw=1.4, ms=4, capsize=2, elinewidth=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("vote k")
    ax.set_ylabel("control false-positive rate")
    ax.set_xticks(ks)
    axes[0].legend(frameon=False, loc="lower center", fontsize=7)
    _panel_letters(axes)
    fig.tight_layout()
    save_figure(fig, OUT / "fig_vote")


def fig_boundary_k(R):
    """Boundary rules crossed with the vote k, as k-swept operating curves."""
    blk = R["boundary_k"]
    ks = [1, 2, 3, 4, 6, 8]
    bstyles = [("seed", COLORS["sky_blue"], "o", "-", "seeds only"),
               ("window_ab", COLORS["bluish_green"], "^", "--", "growth, above-threshold"),
               ("window", COLORS["vermillion"], "D", "-.", "growth, any active (paper)")]
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_SIZES["double_column"][0],
                                            FIGURE_SIZES["double_column"][1] * 0.85))
    for ax, rho in zip(axes[:2], (5, 10)):
        for b, color, marker, ls, label in bstyles:
            rows = [blk[f"rho={rho} k={k} {b}"] for k in ks]
            ax.errorbar(ks, [r["F2all"] for r in rows],
                        yerr=[r["se"]["F2all"] for r in rows],
                        fmt=ls, marker=marker, color=color, label=label,
                        lw=1.4, ms=4, capsize=2, elinewidth=0.9)
        ax.set_xlabel("vote k")
        ax.set_ylabel(f"held-out F2 (all domains), ρ≥{rho}")
        ax.set_xticks(ks)
        ax.set_ylim(0.2, 0.9)
    ax = axes[2]
    for b, color, marker, ls, label in bstyles:
        rows = [blk[f"rho=10 k={k} {b}"] for k in ks]
        ax.errorbar([r["fp"] for r in rows], [r["Rspan"] for r in rows],
                    xerr=[r["se"]["fp"] for r in rows],
                    yerr=[r["se"]["Rspan"] for r in rows],
                    fmt=ls, marker=marker, color=color, label=label,
                    lw=1.2, ms=4, capsize=1.5, elinewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("control false-positive rate")
    ax.set_ylabel("span recall (k swept 1–8, ρ≥10)")
    axes[0].legend(frameon=False, loc="lower center", fontsize=6.5)
    _panel_letters(axes)
    fig.tight_layout()
    save_figure(fig, OUT / "fig_boundary_k")


def fig_width(R):
    """16k against 65k: judged-pool frontier and ROC."""
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_SIZES["double_column"][0],
                                            FIGURE_SIZES["double_column"][1] * 0.85))
    w16 = sorted(R["width_16k_frontier_k4"].values(), key=lambda r: r["fp"])
    w65 = frontier(R["judge_frontier_k4"], "judge")
    styles = [("65k", w65, COLORS["blue"], "o", "-"),
              ("16k", w16, COLORS["yellow"], "v", "--")]
    ax = axes[0]
    for label, rows, color, marker, ls in styles:
        xs = np.array([r["fp"] for r in rows])
        ys = np.array([r["F2all"] for r in rows])
        es = np.array([r["se"]["F2all"] for r in rows])
        ax.plot(xs, ys, ls, marker=marker, color=color, label=label, lw=1.4, ms=4)
        ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("control false-positive rate")
    ax.set_ylabel("held-out F2 (all domains)")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    for ax, (pop, ylab) in zip(axes[1:], [("span", "TPR, human-labelled spans"),
                                          ("c2", "TPR, held-out category")]):
        for wkey, color, ls, label in (("roc_65k", COLORS["blue"], "-", "65k"),
                                       ("roc_16k", COLORS["yellow"], "--", "16k")):
            if wkey not in R:
                continue
            _roc_line(ax, R[wkey]["judge rho>=10"], pop, color, ls, label)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("control false-positive rate")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, loc="upper left", fontsize=7)
    _panel_letters(axes)
    fig.tight_layout()
    save_figure(fig, OUT / "fig_width")


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    R = load()
    fig_roc(R)
    fig_judge(R)
    fig_vote(R)
    fig_boundary_k(R)
    fig_width(R)


if __name__ == "__main__":
    main()
