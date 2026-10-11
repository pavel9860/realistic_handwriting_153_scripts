"""Plots docs/map2d.json to docs/map2d.png. Left: Latin, real lines (IAM and web) and the paths of four styles as
the models are added. Right: other scripts, each real script as a 1-sigma ellipse and the path of generated text in
that script from font rendering to the searched bank.

python scripts/plot_map.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

ROOT = Path(__file__).resolve().parents[1]
STYLE_COLORS = {"casual": "#2a78d6", "formal": "#eb6834", "hurried": "#1baf7a", "calligraphic": "#eda100"}
SCRIPT_COLORS = {
    "Cyrillic": "#1c5cab",
    "Japanese": "#d95926",
    "Hebrew": "#199e70",
    "Greek": "#c98500",
    "Arabic": "#d55181",
}
INK, MUTED, GRID = "#2b2b28", "#6b6b65", "#ebeae4"


def ellipse(ax, p, color, lw=1.5, ls="-", fill=False):
    c, cov = p.mean(0), np.cov(p.T)
    w, v = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(
        Ellipse(
            c,
            2 * np.sqrt(w[1]),
            2 * np.sqrt(w[0]),
            angle=ang,
            fill=fill,
            ec=color,
            fc=color,
            alpha=0.12 if fill else 1,
            lw=lw,
            ls=ls,
            zorder=2,
        )
    )
    return c, np.sqrt(w.max())


def style_axes(ax, title):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c9c8c0")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10, color="#1f1f1c", loc="left")
    ax.set_xlabel("PC 1 of real lines", fontsize=8.5, color=INK)


def main():
    m = json.loads((ROOT / "docs" / "map2d.json").read_text())
    real = {k: np.array(v["points"]) for k, v in m["real"].items()}
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 6.2), dpi=150, gridspec_kw=dict(width_ratios=(1.15, 1)))
    iam = real["Latin (IAM)"]
    a.scatter(iam[:, 0], iam[:, 1], s=3, color="#b9b8b0", alpha=0.25, lw=0, zorder=1)
    c1, _ = ellipse(a, iam, "#6b6b65", lw=1.8)
    c2, _ = ellipse(a, real["Latin (web)"], "#6b6b65", lw=1.2, ls=(0, (4, 3)))
    a.annotate(
        "real IAM lines (1σ)",
        c1,
        xytext=(-60, -95),
        textcoords="offset points",
        fontsize=8,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )
    a.annotate(
        "real web lines, Latin (1σ)",
        c2,
        xytext=(-40, 60),
        textcoords="offset points",
        fontsize=8,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
    )
    pts = [c1, c2]
    for st, stages in m["styles"].items():
        names = list(stages)
        xy = np.array([stages[s]["mean"] for s in names])
        pts += list(xy)
        col = STYLE_COLORS[st]
        a.plot(xy[:, 0], xy[:, 1], color=col, lw=2, zorder=3, label=st)
        a.scatter(xy[:, 0], xy[:, 1], s=30, color=col, edgecolor="white", lw=1.2, zorder=4)
        a.scatter(xy[:1, 0], xy[:1, 1], s=60, marker="s", color=col, edgecolor="white", lw=1.2, zorder=4)
        a.annotate("", xy=xy[-1], xytext=xy[-2], arrowprops=dict(arrowstyle="-|>", color=col, lw=2), zorder=4)
        if st == "casual":
            for k in range(len(names)):
                a.annotate(str(k), xy[k], xytext=(5, 5), textcoords="offset points", fontsize=8, color=INK, zorder=6)
            key = "  ".join(f"{k} {s}" for k, s in enumerate(names))
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.25 * (hi - lo)
    a.set_xlim(min(lo[0], c1[0] - 0.35) - pad[0], hi[0] + pad[0])
    a.set_ylim(min(lo[1], c1[1] - 0.3) - pad[1], max(hi[1], c2[1] + 0.2) + pad[1])
    style_axes(a, "Latin: four styles from font rendering (square) to the bank and personas")
    a.set_ylabel("PC 2 of real lines", fontsize=8.5, color=INK)
    a.legend(title="style", frameon=False, fontsize=8.5, title_fontsize=8.5, loc="upper right")
    a.text(
        0.0,
        -0.16,
        "stages (numbered on casual): " + key.replace("  4", "\n4"),
        transform=a.transAxes,
        fontsize=7,
        color=MUTED,
    )
    pts = []
    for scr, stages in m["scripts"].items():
        if scr not in real or len(real[scr]) < 5:
            continue
        col = SCRIPT_COLORS[scr]
        c, _ = ellipse(b, real[scr], col, lw=1.2, fill=True)
        b.scatter(*c, s=70, marker="*", color=col, edgecolor="white", lw=0.8, zorder=5)
        names = list(stages)
        xy = np.array([stages[s]["mean"] for s in names])
        b.plot(xy[:, 0], xy[:, 1], color=col, lw=1.8, zorder=3)
        b.scatter(xy[1:, 0], xy[1:, 1], s=22, color=col, edgecolor="white", lw=1, zorder=4)
        b.scatter(xy[:1, 0], xy[:1, 1], s=55, marker="s", color=col, edgecolor="white", lw=1, zorder=4)
        b.annotate("", xy=xy[-1], xytext=xy[-2], arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8), zorder=4)
        off = {"Greek": (6, -12), "Cyrillic": (-70, -14)}.get(scr, (6, 6))
        b.annotate(
            f"{scr} ({len(real[scr])} real lines)",
            c,
            xytext=off,
            textcoords="offset points",
            fontsize=8,
            color=INK,
            zorder=6,
        )
        pts += list(xy) + [c]
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.2 * (hi - lo)
    b.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    b.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    style_axes(b, "Other scripts: generated text (square = font) and real script (★, 1σ)")
    b.text(
        0.0,
        -0.16,
        "path: font → writer → hand movement → scan → searched bank\nreal clouds are small and possibly contaminated",
        transform=b.transAxes,
        fontsize=7,
        color=MUTED,
    )
    fig.suptitle(
        "From mechanical to real handwriting: one plane fitted to real lines only (embeddings + stroke features)",
        fontsize=11,
        x=0.01,
        ha="left",
        color="#1f1f1c",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96), w_pad=3)
    fig.savefig(ROOT / "docs" / "map2d.png")


if __name__ == "__main__":
    main()
