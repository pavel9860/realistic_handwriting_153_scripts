"""Plots docs/progression.json (realism per pipeline stage and style) to docs/progression.png.

python scripts/plot_progression.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"casual": "#2a78d6", "formal": "#eb6834", "hurried": "#1baf7a", "calligraphic": "#eda100"}
MARKERS = {"casual": "o", "formal": "s", "hurried": "^", "calligraphic": "D"}


def main():
    res = json.loads((ROOT / "docs" / "progression.json").read_text())
    stages = list(next(iter(res.values())))
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    ax.axhline(0, color="#8a8a85", lw=1, ls=(0, (4, 3)))
    ax.text(
        len(stages) - 0.95,
        0.02,
        "real handwriting (held-out documents)",
        color="#55554f",
        fontsize=8.5,
        ha="right",
        va="bottom",
    )
    ends = []
    for st, rows in res.items():
        m = [rows[s]["mean"] for s in stages]
        e = [rows[s]["sem"] for s in stages]
        ax.errorbar(
            range(len(stages)),
            m,
            yerr=e,
            color=COLORS[st],
            lw=2,
            marker=MARKERS[st],
            ms=6,
            capsize=2,
            elinewidth=1,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=st,
        )
        ends.append([m[-1], st])
    ends.sort()
    lo, hi = ax.get_ylim()
    for i in range(1, len(ends)):  # end labels at least 4 % of the axis apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.04 * (hi - lo))
    for y, st in ends:
        ax.text(len(stages) - 1 + 0.15, y, st, color="#2b2b28", fontsize=9, va="center")
    ax.set_xticks(range(len(stages)), [s.replace(" ", "\n") for s in stages], fontsize=9, color="#3d3d39")
    ax.set_ylabel("realism (z-score vs real lines)", fontsize=9.5, color="#3d3d39")
    ax.set_title(
        "From font rendering to natural handwriting: realism as each model is added",
        fontsize=11,
        color="#1f1f1c",
        loc="left",
    )
    ax.set_xlim(-0.3, len(stages) - 0.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c9c8c0")
    ax.tick_params(colors="#6b6b65", labelsize=8.5)
    ax.grid(axis="y", color="#ebeae4", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower right", bbox_to_anchor=(1, 0.02))
    fig.text(
        0.01,
        0.01,
        "Mean ± s.e. over 16 writers × 2 texts per point; higher = more typical of real scanned "
        "handwriting (IAM test forms, 0 = their mean).\nAfter the writer stage the measure changes little; the 2D map resolves the "
        "later stages better. Writers from type priors.",
        fontsize=7.5,
        color="#6b6b65",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(ROOT / "docs" / "progression.png")


if __name__ == "__main__":
    main()
