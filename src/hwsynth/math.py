"""Handwritten mathematics and chemistry in a small TeX-like syntax, laid out as boxes of pen strokes.

  x^2  x^{n+1}  a_i  a_{ij}  \\frac{a}{b}  \\sqrt{x}  \\int \\sum \\pi \\alpha ... (Unicode symbols)
  ->  <-  <=>  =>  (drawn arrows)   \\ce{H2SO4 + 2NaOH -> Na2SO4 + 2H2O}   \\benzene
A box holds strokes in mm with the origin at its left end on the baseline (y down), its width and its height above
(asc) and depth below (desc) the baseline. Everything is drawn by the writer: glyphs through Writer.word, rules,
radicals, arrows and rings as pen strokes; the writer then moves the pen over all of them (Writer.finish).
"""

from dataclasses import dataclass, field

import numpy as np

from .skeleton import resample
from .writer import Stroke

SYMBOLS = {
    "int": "∫",
    "sum": "∑",
    "prod": "∏",
    "pi": "π",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "Delta": "Δ",
    "epsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "sigma": "σ",
    "Sigma": "Σ",
    "phi": "φ",
    "omega": "ω",
    "Omega": "Ω",
    "infty": "∞",
    "partial": "∂",
    "nabla": "∇",
    "pm": "±",
    "times": "×",
    "cdot": "·",
    "div": "÷",
    "leq": "≤",
    "geq": "≥",
    "neq": "≠",
    "approx": "≈",
    "in": "∈",
    "forall": "∀",
    "exists": "∃",
    "degree": "°",
    "angle": "∠",
    "perp": "⊥",
}
ARROWS = {"<=>": "eq", "->": "right", "<-": "left", "=>": "double"}
SUP, SUB, FRAC, ROOT = 0.62, 0.62, 0.8, 0.9  # scales of scripts, fraction parts and radicands


@dataclass
class Box:
    strokes: list = field(default_factory=list)
    w: float = 0.0
    asc: float = 0.0
    desc: float = 0.0

    def shifted(self, dx, dy):
        return [np.asarray(s) + (dx, dy) for s in self.strokes]


def _group(s, i):
    """Content of a {...} group starting at s[i] == '{', or a single character; returns (content, next index)."""
    if i < len(s) and s[i] == "{":
        depth, j = 0, i
        while j < len(s):
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            if depth == 0:
                return s[i + 1 : j], j + 1
            j += 1
        return s[i + 1 :], len(s)
    return (s[i], i + 1) if i < len(s) else ("", i)


def parse(s):
    """Nodes: ('text', str) ('sup', nodes) ('sub', nodes) ('frac', nodes, nodes) ('sqrt', nodes) ('arrow', kind)
    ('space',) ('benzene',)."""
    out, i, run = [], 0, ""

    def flush():
        nonlocal run
        if run:
            out.append(("text", run))
            run = ""

    while i < len(s):
        arrow = next((a for a in ARROWS if s.startswith(a, i)), None)
        if arrow:
            flush()
            out.append(("arrow", ARROWS[arrow]))
            i += len(arrow)
        elif s[i] == " ":
            flush()
            out.append(("space",))
            i += 1
        elif s[i] in "^_":
            flush()
            mark = s[i]
            g, i = _group(s, i + 1)
            out.append(("sup" if mark == "^" else "sub", parse(g)))
        elif s[i] == "\\":
            flush()
            j = i + 1
            while j < len(s) and s[j].isalpha():
                j += 1
            name = s[i + 1 : j]
            i = j
            if name == "frac":
                a, i = _group(s, i)
                b, i = _group(s, i)
                out.append(("frac", parse(a), parse(b)))
            elif name == "sqrt":
                a, i = _group(s, i)
                out.append(("sqrt", parse(a)))
            elif name == "ce":
                a, i = _group(s, i)
                out += chem(a)
            elif name == "benzene":
                out.append(("benzene",))
            else:
                run += SYMBOLS.get(name, name)
        else:
            run += s[i]
            i += 1
    flush()
    return out


def chem(s):
    """Chemical equation: digits after an element or a closing bracket become subscripts, + and - after an
    element, digit or ^ become charges, arrows are drawn; leading coefficients stay on the line."""
    out = []
    for k, part in enumerate(s.split(" ")):
        if k:
            out.append(("space",))
        if part in ARROWS:
            out.append(("arrow", ARROWS[part]))
            continue
        i, run = 0, ""
        coef = True
        while i < len(part):
            ch = part[i]
            if ch.isdigit() and not coef:
                j = i
                while j < len(part) and part[j].isdigit():
                    j += 1
                if run:
                    out.append(("text", run))
                    run = ""
                if j < len(part) and part[j] in "+-" and (j + 1 == len(part)):
                    out.append(("sup", [("text", part[i : j + 1])]))
                    i = j + 1
                    continue
                out.append(("sub", [("text", part[i:j])]))
                i = j
                continue
            if ch == "^":
                g, i = _group(part, i + 1)
                if run:
                    out.append(("text", run))
                    run = ""
                out.append(("sup", [("text", g)]))
                continue
            if ch in "+-" and i == len(part) - 1 and len(part) > 1:
                if run:
                    out.append(("text", run))
                    run = ""
                out.append(("sup", [("text", ch)]))
                i += 1
                continue
            if ch.isalpha() or ch in "()[]":
                coef = False
            run += ch
            i += 1
        if run:
            out.append(("text", run))
    return out


def _arc(c, r, a0, a1, n=24):
    t = np.linspace(a0, a1, n)
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]


def _arrow(kind, xh):
    L, y, h = 2.2 * xh, -0.5 * xh, 0.3 * xh
    line = lambda x0, x1, yy: np.array([[x0, yy], [x1, yy]])
    head = lambda x, yy, d: np.array([[x - d * h, yy - 0.6 * h], [x, yy], [x - d * h, yy + 0.6 * h]])
    if kind == "right":
        return [line(0, L, y), head(L, y, 1)], L
    if kind == "left":
        return [line(0, L, y), head(0, y, -1)], L
    if kind == "double":
        return [line(0, L - 0.2 * xh, y - 0.15 * xh), line(0, L - 0.2 * xh, y + 0.15 * xh), head(L, y, 1)], L
    up, dn = y - 0.15 * xh, y + 0.15 * xh  # equilibrium: half-headed pair
    return [np.array([[0, up], [L, up], [L - h, up - 0.6 * h]]), np.array([[L, dn], [0, dn], [h, dn + 0.6 * h]])], L


def _benzene(xh):
    R = 1.3 * xh
    c = np.array((R, -0.5 * xh))
    hexa = np.array([c + R * np.array((np.cos(a), np.sin(a))) for a in np.radians(np.arange(30, 391, 60))])
    return [hexa, _arc(c, 0.6 * R, 0, 2 * np.pi * 0.97)], 2 * R


def layout_nodes(writer, nodes, xh):
    """Box of a node list at x-height xh."""
    box = Box()

    def put(b, dy=0.0):
        box.strokes += b.shifted(box.w, dy)
        box.asc = max(box.asc, b.asc - dy)
        box.desc = max(box.desc, b.desc + dy)
        box.w += b.w

    prev = Box(asc=xh, desc=0)
    for node in nodes:
        kind = node[0]
        if kind == "text":
            strokes, adv = writer.word(node[1], xh, 0.5)
            pts = [s.p for s in strokes if s.p is not None]
            b = Box([s.p for s in strokes if s.p is not None], adv + 0.08 * xh)
            if pts:
                a = np.vstack(pts)
                b.asc, b.desc = max(-a[:, 1].min(), 0), max(a[:, 1].max(), 0)
            put(b)
            prev = b
        elif kind == "space":
            box.w += 0.6 * xh
        elif kind in ("sup", "sub"):
            b = layout_nodes(writer, node[1], xh * (SUP if kind == "sup" else SUB))
            dy = -(prev.asc * 0.75) if kind == "sup" else 0.45 * xh
            put(b, dy)
        elif kind == "frac":
            n, d = layout_nodes(writer, node[1], xh * FRAC), layout_nodes(writer, node[2], xh * FRAC)
            w = max(n.w, d.w) + 0.4 * xh
            axis = -0.55 * xh
            b = Box(w=w)
            b.strokes = n.shifted((w - n.w) / 2, axis - 0.25 * xh - n.desc) + d.shifted(
                (w - d.w) / 2, axis + 0.3 * xh + d.asc
            )
            b.strokes.append(np.array([[0.05 * xh, axis], [w - 0.05 * xh, axis]]))
            b.asc, b.desc = -axis + 0.25 * xh + n.desc + n.asc, axis + 0.3 * xh + d.asc + d.desc
            put(b)
            prev = b
        elif kind == "sqrt":
            r = layout_nodes(writer, node[1], xh * ROOT)
            top = -(r.asc + 0.25 * xh)
            lead = 0.7 * xh
            rad = np.array(
                [
                    [0, -0.35 * xh],
                    [0.2 * xh, -0.45 * xh],
                    [0.4 * xh, r.desc + 0.1 * xh],
                    [lead, top],
                    [lead + r.w + 0.15 * xh, top],
                ]
            )
            b = Box([rad] + r.shifted(lead + 0.05 * xh, 0), lead + r.w + 0.3 * xh, -top, r.desc + 0.1 * xh)
            put(b)
            prev = b
        elif kind == "arrow":
            strokes, w = _arrow(node[1], xh)
            box.w += 0.3 * xh
            put(Box(strokes, w + 0.3 * xh, 0.7 * xh, 0))
        elif kind == "benzene":
            strokes, w = _benzene(xh)
            box.w += 0.2 * xh
            put(Box(strokes, w + 0.3 * xh, 1.9 * xh, 0.9 * xh))
    return box


def layout(writer, expr, xh):
    """(strokes, advance) of a math or chemistry expression, like Writer.word."""
    b = layout_nodes(writer, parse(expr), xh)
    return [Stroke(resample(np.asarray(p, float), 0.02 * xh)) for p in b.strokes if len(p) > 1], b.w
