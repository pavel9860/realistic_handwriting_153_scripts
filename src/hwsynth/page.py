"""Text block on paper: writer layout -> ink -> composed image with word boxes in pixels."""

from dataclasses import dataclass

import cv2
import numpy as np

from .pen import render

PAPERS = ((0.97, 0.965, 0.94), (0.99, 0.99, 0.98), (0.96, 0.95, 0.90), (0.98, 0.97, 0.93))


@dataclass
class Sample:
    image: np.ndarray  # (H, W, 3) uint8 RGB
    ink: np.ndarray  # (H, W) float32 ink density in [0, 1]
    words: list  # [(text, (x0, y0, x1, y1) px, struck)]
    meta: dict


def paper(shape, rng, rules=None, dpi=150, first=0.0):
    """Paper RGB (H, W, 3) float in [0, 1]: tint, fibre noise, slow shading; rules: ruling pitch (mm) through the
    baseline at first (mm from the top)."""
    h, w = shape
    base = np.array(PAPERS[int(rng.integers(len(PAPERS)))])
    tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2) * 0.012
    tex += (
        cv2.resize(
            rng.normal(0, 1, (h // 40 + 2, w // 40 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC
        )
        * 0.01
    )
    img = np.clip(base[None, None] + tex[..., None], 0, 1)
    if rules:
        k = dpi / 25.4
        col = np.array((0.62, 0.75, 0.9)) if rng.random() < 0.7 else np.array((0.7, 0.7, 0.7))
        y = first % rules
        while y * k < h:
            yy = int(round(y * k))
            img[max(yy - 1, 0) : yy + 1] = img[max(yy - 1, 0) : yy + 1] * 0.4 + col * 0.6
            y += rules
    return img


def smudge(ink, pen, dpi, rng):
    """Left hand dragging over wet ink: faint smears trailing right and slightly down along some lines (fluid inks
    and graphite; ballpoint paste barely smears)."""
    amount = {"fluid": 0.25, "solid": 0.2, "viscous": 0.05}[pen.spec.ink] * rng.uniform(0.3, 1)
    k = dpi / 25.4
    n = max(int(rng.uniform(2, 5) * k), 3)
    ker = np.zeros((3, 2 * n + 1), np.float32)
    ker[1, : n + 1] = np.exp(-np.arange(n, -1, -1) / (n / 2))  # correlation: output trails to the right
    ker /= ker.sum()
    s = cv2.filter2D(ink, -1, ker)
    s = cv2.warpAffine(s, np.float32([[1, 0, 0], [0, 1, 0.3 * k]]), s.shape[::-1])
    band = cv2.resize(rng.random((max(ink.shape[0] // int(4 * k), 1), 1)).astype(np.float32), ink.shape[::-1]) > 0.5
    return np.clip(np.maximum(ink, s * amount * band), 0, 1)


def compose(ink, rgb, bg):
    """Subtractive ink on paper: bg * (1 - density * (1 - ink colour))."""
    return bg * (1 - ink[..., None] * (1 - np.asarray(rgb)[None, None]))


def write(text, writer, width_mm=150.0, dpi=150, ruled=None, margin=6.0):
    """Sample of text written by writer on paper; ruled: True, False or None (by chance, 40 %)."""
    rng = writer.rng
    if ruled is None:
        ruled = {"blank": False, "ruled": True}.get(writer.paper, rng.random() < 0.4)
    xh = max(writer.p["size"], 4 * writer.pen.edge, 5 * writer.pen.width)
    rules = round(max(writer.p["pitch"], writer.min_pitch()) * xh / 0.5) * 0.5 if ruled else None
    lay = writer.lines(text, width_mm, rules)
    k = dpi / 25.4
    first = lay.pitch + margin
    shape = (int((lay.lines * lay.pitch + first + margin) * k), int((width_mm + 2 * margin) * k))
    origin = np.array((-margin, -first))
    ink, _ = render(lay.strokes, writer.pen, writer.p, dpi, rng, shape, origin)
    if writer.left:
        ink = smudge(ink, writer.pen, dpi, rng)
    img = compose(ink, writer.pen.ink, paper(shape, rng, rules, dpi, first))
    words = [
        (w.text, tuple(int(round(v)) for v in ((np.array(w.box) - np.r_[origin, origin]) * k)), w.struck)
        for w in lay.words
        if w.box
    ]
    return Sample((np.clip(img, 0, 1) * 255).astype(np.uint8), ink, words, dict(lay.meta, ruled=bool(ruled), dpi=dpi))


def line_crop(writer, text, dpi=120, layout=False):
    """Grayscale crop of one line of text on light paper, padded like a text-line detector box (None if empty);
    with layout=True also the layout (pen paths)."""
    lay = writer.lines(text, 400)
    ink, _ = render(lay.strokes, writer.pen, writer.p, dpi, writer.rng)
    g = (compose(ink, writer.pen.ink, np.full(ink.shape + (3,), 0.96)).mean(2) * 255).astype(np.uint8)
    ys, xs = np.nonzero(ink > 0.2)
    if not len(ys):
        return (None, lay) if layout else None
    pad = int(0.15 * (ys.max() - ys.min())) + 2
    crop = g[max(ys.min() - pad, 0) : ys.max() + pad, max(xs.min() - pad, 0) : xs.max() + pad]
    return (crop, lay) if layout else crop
