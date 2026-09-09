"""Shared plotting style for the numerical figures in the rewritten paper."""

from __future__ import annotations

import matplotlib.pyplot as plt


MARKERS = ("o", "s", "^", "D", "v", "P", "*", "X")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)))


def apply_paper_style() -> None:
    """Apply the white-background style used by the paper's reference plots."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 17,
            "axes.titlesize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 1.45,
            "lines.markersize": 4.3,
            "axes.linewidth": 0.95,
            "figure.dpi": 180,
            "savefig.dpi": 350,
            "savefig.bbox": "tight",
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def setup_axis(ax, xlog: bool = False, ylog: bool = False) -> None:
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.15, linewidth=0.50, color="#b8b8b8")
    ax.grid(False, which="minor")
    ax.tick_params(
        direction="in", which="major", top=True, right=True, length=5
    )
    ax.tick_params(
        direction="in", which="minor", top=True, right=True, length=2.8
    )
    ax.minorticks_on()


def sparse_staggered_markevery(
    npts: int,
    curve_idx: int,
    ncurves: int,
    target_markers: int = 5,
    minimum_step: int = 18,
) -> tuple[int, int]:
    """Place a small number of markers and offset them between curves."""
    step = max(minimum_step, npts // max(1, target_markers))
    offset = int(round(curve_idx * step / max(1, ncurves)))
    return offset, step


def add_top_legend(fig, handles, labels, ncol: int, y: float = 0.975):
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=True,
        fancybox=True,
        edgecolor="#aaaaaa",
        columnspacing=1.0,
        handlelength=2.4,
        borderpad=0.35,
        labelspacing=0.7,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    return legend


def add_panel_label(
    ax, text: str, x: float = 0.0, y: float = 1.02, fontsize: float = 16
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
    )
