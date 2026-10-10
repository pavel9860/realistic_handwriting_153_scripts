"""Hand-drawn figures: basic shapes, arrows, axes with a plotted curve, ticks and marks, drawn by a writer.

Shapes are planned as pen paths in mm with the imperfections of freehand drawing (circles that do not quite close,
corners that overshoot, lines that bow), then moved and inked like handwriting (Writer.finish, pen.render)."""

import numpy as np

from .page import Sample, compose, paper
from .pen import render
from .skeleton import resample
from .writer import Stroke


def _closed(rng, pts, overshoot=0.08):
    """Closed polygon drawn in one movement: the end overshoots or stops short of the start."""
    pts = np.vstack([pts, pts[:1]])
    d = pts[-1] - pts[-2]
    return np.vstack([pts[:-1], pts[-1] + d * rng.uniform(-overshoot, overshoot)])


def _bowed(rng, a, b, bow=0.02):
    """Straight line drawn by hand: a slight bow and uneven ends."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.linspace(0, 1, 30)[:, None]
    n = np.array([-(b - a)[1], (b - a)[0]])
    return a + t * (b - a) + np.sin(np.pi * t) * n * rng.normal(0, bow)


def shape(kind, size, rng, **kw):
    """Pen paths (list of (N, 2) arrays, mm, y down, origin top-left) of a figure about size mm wide."""
    s, r = size, size / 2
    c = np.array((r, r))
    if kind == "circle":
        a0 = rng.uniform(0, 2 * np.pi)
        t = np.linspace(a0, a0 + 2 * np.pi * rng.uniform(0.95, 1.06), 80)
        e = 1 + rng.normal(0, 0.04)
        return [c + np.c_[r * e * np.cos(t), r / e * np.sin(t)]]
    if kind == "ellipse":
        t = np.linspace(0, 2 * np.pi * rng.uniform(0.96, 1.05), 80)
        return [c + np.c_[r * np.cos(t), 0.55 * r * np.sin(t)]]
    if kind == "triangle":
        p = np.array([(r, 0), (s, 0.87 * s), (0, 0.87 * s)]) + rng.normal(0, 0.02 * s, (3, 2))
        return [_closed(rng, p)]
    if kind in ("square", "rectangle", "box"):
        h = s if kind == "square" else 0.6 * s
        p = np.array([(0, 0), (s, 0), (s, h), (0, h)]) + rng.normal(0, 0.015 * s, (4, 2))
        return [_closed(rng, p)]
    if kind == "arrow":
        head = 0.18 * s
        return [_bowed(rng, (0, r), (s, r)), np.array([(s - head, r - 0.6 * head), (s, r), (s - head, r + 0.6 * head)])]
    if kind == "star":
        a = np.radians(np.arange(5) * 144 - 90)
        return [_closed(rng, c + r * np.c_[np.cos(a), np.sin(a)], 0.05)]
    if kind == "check":
        return [np.array([(0, 0.55 * s), (0.35 * s, s), (s, 0)]) + rng.normal(0, 0.02 * s, (3, 2))]
    if kind == "cross":
        return [_bowed(rng, (0, 0), (s, s)), _bowed(rng, (s, 0), (0, s))]
    if kind == "axes":
        fn = kw.get("fn", np.sin)
        x = np.linspace(0, 1, 60)
        y = fn(x * kw.get("span", 2 * np.pi))
        y = (y - y.min()) / (np.ptp(y) + 1e-9)
        o = np.array((0.1 * s, 0.9 * s))
        paths = [_bowed(rng, o, (s, o[1])), _bowed(rng, o, (o[0], 0))]
        paths += [
            np.array([(s - 0.04 * s, o[1] - 0.025 * s), (s, o[1]), (s - 0.04 * s, o[1] + 0.025 * s)]),
            np.array([(o[0] - 0.025 * s, 0.04 * s), (o[0], 0), (o[0] + 0.025 * s, 0.04 * s)]),
        ]
        for k in range(1, 5):  # ticks
            xt = o[0] + k * 0.2 * s
            paths.append(np.array([(xt, o[1] - 0.015 * s), (xt, o[1] + 0.015 * s)]))
        curve = np.c_[o[0] + 0.05 * s + x * 0.8 * s, o[1] - 0.1 * s - y * 0.7 * s]
        return paths + [curve + rng.normal(0, 0.003 * s, curve.shape)]
    if kind == "underline":
        return [_bowed(rng, (0, 0), (s, rng.normal(0, 0.02 * s)), 0.01)]
    raise ValueError(f"unknown figure {kind!r}")


FIGURES = (
    "circle",
    "ellipse",
    "triangle",
    "square",
    "rectangle",
    "arrow",
    "star",
    "check",
    "cross",
    "axes",
    "underline",
)


def figure(kind, writer, size=30.0, dpi=150, margin=5.0, **kw):
    """Sample of one hand-drawn figure about size mm wide."""
    rng = writer.rng
    xh = writer.p["size"]
    strokes = [Stroke(resample(p, 0.02 * xh)) for p in shape(kind, size, rng, **kw)]
    writer.finish(strokes, xh)
    k = dpi / 25.4
    allp = np.vstack([s.p for s in strokes])
    origin = allp.min(0) - margin
    shp = tuple(np.ceil((allp.max(0) - origin + margin) * k).astype(int)[::-1])
    ink, _ = render(strokes, writer.pen, writer.p, dpi, rng, shp, origin)
    img = compose(ink, writer.pen.ink, paper(shp, rng, None, dpi))
    return Sample((np.clip(img, 0, 1) * 255).astype(np.uint8), ink, [], dict(figure=kind, writer=writer.name))
