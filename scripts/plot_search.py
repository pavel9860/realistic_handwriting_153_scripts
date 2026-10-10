"""Plots the quality-diversity search log (docs/qd_log.txt) to docs/search.png: median quality and instability of
the archive elites and the number of filled cells per iteration.

python scripts/plot_search.py
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"iter (\d+): added (\d+), cells (\d+), quality median ([-\d.]+), instability median ([\d.]+)")


def main():
    rows = [
        tuple(float(x) for x in m.groups())
        for m in map(PAT.search, (ROOT / "docs" / "qd_log.txt").read_text().splitlines())
        if m
    ]
    it = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=150)
    for ax, k, title, color in (
        (axes[0], 3, "median quality of elites", "#2a78d6"),
        (axes[1], 4, "median baseline instability (x-heights)", "#eb6834"),
        (axes[2], 2, "filled archive cells (24 types)", "#1baf7a"),
    ):
        ax.plot(it, [r[k] for r in rows], color=color, lw=2, marker="o", ms=5, markeredgecolor="white")
        ax.set_title(title, fontsize=9.5, color="#1f1f1c", loc="left")
        ax.set_xlabel("iteration", fontsize=8.5, color="#3d3d39")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#c9c8c0")
        ax.tick_params(colors="#6b6b65", labelsize=8)
        ax.grid(axis="y", color="#ebeae4", lw=0.8)
    axes[1].axhspan(0.13, 0.35, color="#eb6834", alpha=0.08, lw=0)
    axes[1].text(0.2, 0.335, "band: clean real lines (0.13-0.35)", fontsize=7.5, color="#6b6b65")
    fig.suptitle(
        "Quality-diversity search (MAP-Elites): elites become more realistic and steadier",
        fontsize=11,
        x=0.01,
        ha="left",
        color="#1f1f1c",
    )
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "search.png")


if __name__ == "__main__":
    main()
