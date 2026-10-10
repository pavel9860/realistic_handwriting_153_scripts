"""Glyph centrelines: a font glyph is rasterised, skeletonised and traced into ordered pen strokes (em units, y down,
baseline at y = 0)."""

from collections import Counter
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize

from .config import GlyphCfg
from .fonts import pil_font

NB8 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def resample(p, step):
    """Polyline p (N, 2) resampled at equal arc-length steps; a polyline shorter than step keeps its ends."""
    p = np.asarray(p, float)
    if len(p) < 2:
        return np.vstack([p, p])
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    if d[-1] < step:
        return p[[0, -1]]
    s = np.linspace(0, d[-1], int(d[-1] / step) + 1)
    return np.c_[np.interp(s, d, p[:, 0]), np.interp(s, d, p[:, 1])]


def smooth(p, sigma):
    """Gaussian smoothing along a polyline (sigma in samples); open ends stay fixed, closed loops wrap."""
    p = np.asarray(p, float)
    if sigma < 0.3 or len(p) < 4:
        return p
    closed = len(p) > 8 and np.allclose(p[0], p[-1])
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    q = np.pad(p[:-1] if closed else p, ((r, r), (0, 0)), mode="wrap" if closed else "edge")
    o = np.c_[np.convolve(q[:, 0], k, "valid"), np.convolve(q[:, 1], k, "valid")]
    if closed:
        return np.vstack([o, o[:1]])
    o[0], o[-1] = p[0], p[-1]
    return o


def _graph(sk):
    pts = set(zip(*np.nonzero(sk)))
    nbr = {p: [(p[0] + a, p[1] + b) for a, b in NB8 if (p[0] + a, p[1] + b) in pts] for p in pts}
    return pts, nbr


def trace(b, cfg=GlyphCfg(), px=None):
    """Strokes (list of (N, 2) xy pixel arrays) along the skeleton of the boolean image b: edges between junction
    clusters, short free spurs pruned, edges continued straight through junctions (smallest turn)."""
    px = px or cfg.px
    pts, nbr = _graph(skeletonize(b))
    node = {p for p, n in nbr.items() if len(n) != 2}
    cl, n_cl = {}, 0
    for p in node:
        if p in cl:
            continue
        stack, cl[p] = [p], n_cl
        while stack:
            for r in nbr[stack.pop()]:
                if r in node and r not in cl:
                    cl[r] = n_cl
                    stack.append(r)
        n_cl += 1
    members = {}
    for p, c in cl.items():
        members.setdefault(c, []).append(p)
    centre = {c: np.mean(v, 0) for c, v in members.items()}
    edges, used = [], set()
    for p in node:
        for m in nbr[p]:
            if m in node or (p, m) in used:
                continue
            path, prev, cur = [p, m], p, m
            used.add((p, m))
            while cur not in node:
                nxt = [r for r in nbr[cur] if r != prev and r not in path[-3:]]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            used.add((cur, prev))
            if cur in node:
                edges.append([cl[p], cl[cur], path])
    seen = node | {q for _, _, e in edges for q in e}
    for p in sorted(pts - seen):
        if p in seen:
            continue
        path, prev, cur = [p], None, p
        while True:
            seen.add(cur)
            nxt = [r for r in nbr[cur] if r != prev and r not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
        edges.append([None, None, path + [p]])
    touched = {c for a, b_, _ in edges for c in (a, b_)}
    edges += [[None, None, [tuple(centre[c]), tuple(centre[c] + 0.5)]] for c in centre if c not in touched]
    spur = cfg.spur * px
    for _ in range(2):
        deg = Counter(c for a, b_, _ in edges for c in (a, b_) if c is not None)
        edges = [
            e
            for e in edges
            if not (
                len(e[2]) < spur
                and e[0] is not None
                and e[1] is not None
                and ((deg[e[0]] == 1 and deg[e[1]] >= 3) or (deg[e[1]] == 1 and deg[e[0]] >= 3))
            )
        ]
    deg = Counter(c for a, b_, _ in edges for c in (a, b_) if c is not None)
    out = []
    for a, b_, path in edges:
        p = np.array(path, float)[:, ::-1]
        if a is not None and deg[a] >= 2:
            p = np.vstack([centre[a][::-1], p])
        if b_ is not None and deg[b_] >= 2:
            p = np.vstack([p, centre[b_][::-1]])
        out.append((a, b_, p))
    return _chain(out, cfg.join_turn)


def _chain(edges, turn):
    def out_dir(i, end):
        p = edges[i][2] if end else edges[i][2][::-1]
        k = min(len(p) - 1, 8)
        d = p[-1 - k] - p[-1]
        return d / (np.hypot(*d) + 1e-9)

    ends = {}
    for i, (a, b, _) in enumerate(edges):
        for end, c in ((0, a), (1, b)):
            if c is not None:
                ends.setdefault(c, []).append((i, end))
    pair = {}
    for es in ends.values():
        cand = sorted(
            (float(out_dir(*u) @ out_dir(*v)), u, v) for x, u in enumerate(es) for v in es[x + 1 :] if u[0] != v[0]
        )
        for d, u, v in cand:
            if d < turn and u not in pair and v not in pair:
                pair[u], pair[v] = v, u
    done, strokes = set(), []
    starts = [(i, e) for i in range(len(edges)) for e in (0, 1) if (i, e) not in pair] + [
        (i, 0) for i in range(len(edges))
    ]
    for i, e in starts:
        if i in done:
            continue
        seq = []
        while i not in done:
            done.add(i)
            p = edges[i][2]
            seq.append(p if e == 0 else p[::-1])
            nxt = pair.get((i, 1 - e))
            if nxt is None:
                break
            i, e = nxt
        strokes.append(np.vstack(seq))
    return strokes


@lru_cache(200000)
def glyph(path, unit, cfg=GlyphCfg()):
    """(strokes, advance) of a text unit in a font: strokes in em (baseline y = 0, y down), ordered left to right,
    each starting at its upper-left end; advance in em."""
    f = pil_font(path, cfg.px)
    asc, desc = f.getmetrics()
    w = f.getlength(unit)
    pad = cfg.px // 2
    im = Image.new("L", (int(w) + 2 * pad, asc + desc + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad, pad + asc), unit, fill=255, font=f, anchor="ls")
    b = np.asarray(im) > cfg.threshold
    if b.sum() < 3:
        return (), w / cfg.px
    strokes = []
    for s in trace(b, cfg):
        s = smooth(resample(s, cfg.pre_step), cfg.pre_smooth)
        strokes.append(resample((s - (pad, pad + asc)) / cfg.px, cfg.step))
    strokes.sort(key=lambda s: s[:, 0].min())
    lead = np.array((1.0, 0.6))
    return tuple(s if s[0] @ lead <= s[-1] @ lead else s[::-1] for s in strokes), w / cfg.px
