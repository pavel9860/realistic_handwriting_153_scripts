"""Writing tools and ink: along each stroke the speed (curvature, sigma-lognormal-like ramps at the ends) and the
pressure (writer push, variation, downstroke emphasis, lift-off) set width and ink density by nib shape and ink type
(after Franke & Rose's ink-deposition classes: solid, viscous, fluid). Strokes are drawn as dense discs in a
supersampled patch and combined with tool-specific overlap, bleed and graphite grain."""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import TOOL_PRIOR, TOOLS, InkCfg, RenderCfg
from .noise import noise1d


@dataclass
class Pen:
    tool: str
    width: float  # tip width (mm)
    edge: float  # broad nib edge (mm)
    angle: float  # broad nib angle to the baseline (rad)
    density: float
    ink: tuple  # RGB transmittance
    ink_name: str

    @property
    def kind(self):
        return self.tool

    @property
    def spec(self):
        return TOOLS[self.tool]

    @classmethod
    def sample(cls, rng, prior=None, inks=InkCfg()):
        prior = prior or TOOL_PRIOR
        names = list(prior)
        w = np.array([prior[n] for n in names], float)
        t = TOOLS[names[rng.choice(len(names), p=w / w.sum())]]
        if t.ink == "solid":
            ink, name = inks.graphite, "graphite"
        else:
            cols = list(inks.inks)
            pw = np.array([inks.inks[c][1] for c in cols])
            name = cols[rng.choice(len(cols), p=pw / pw.sum())]
            ink = inks.inks[name][0]
        return cls(
            t.name,
            rng.uniform(*t.width),
            rng.uniform(*t.edge) if t.nib == "broad" else 0.0,
            np.radians(rng.uniform(*t.angle)),
            rng.uniform(*t.density),
            ink,
            name,
        )

    def reach(self):
        """Largest half-width of a mark (mm)."""
        t = self.spec
        return 0.5 * (self.width * (1 + t.flex) + self.edge) + 0.2


def kinematics(q, k, xh, p, rng, speed=None):
    """Normalised speed v, pressure and direction angle along a stroke (q px, k px/mm); speed: the hand movement
    model's speed along the stroke (resampled to q), else estimated from curvature."""
    n = len(q)
    seg = np.diff(q, axis=0, prepend=q[:1])
    sl = np.hypot(*seg.T)
    arc = np.cumsum(sl) / k
    L = arc[-1] + 1e-6
    ang = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    if n > 2:
        ang[0] = ang[1]
        curv = np.abs(np.gradient(ang)) / (sl / k + 1e-3)
        if n > 8:
            curv = np.convolve(np.pad(curv, 4, mode="edge"), np.ones(9) / 9, "valid")
    else:
        curv = np.zeros(n)
    if speed is None:
        ramp = np.clip(np.minimum(arc / (0.15 * xh), (L - arc) / (0.2 * xh)), 0, 1)
        v = (0.3 + 0.7 * np.sqrt(ramp)) / (1 + 0.6 * curv * xh)
    else:
        v = speed
    down = np.clip(np.sin(ang), 0, 1)
    pr = (
        0.45
        + 0.6 * p["press"]
        + p["press_sd"] * 0.3 * noise1d(rng, n, max(n / 8, 2), 1)
        + 0.35 * p["down"] * (down - 0.5)
        - 0.2 * (v - 0.5)
    )
    pr *= (0.55 + 0.45 * np.minimum(1, (L - arc) / (0.08 * xh))) * (0.85 + 0.15 * np.minimum(1, arc / (0.04 * xh)))
    return v, np.clip(pr, 0.05, 1.4), ang, arc, L


def marks(pen, v, pr, ang, arc, L, p, kss, xh, rng):
    """Width (mm) and ink density along a stroke."""
    t, n = pen.spec, len(v)
    if t.nib == "broad":
        wd = pen.width + pen.edge * np.abs(np.sin(ang + pen.angle)) * (1 + t.flex * (pr - 0.75))
    elif t.nib == "brush":
        taper = np.clip(np.minimum(arc, L - arc) / (0.12 * xh), 0.15, 1)
        wd = pen.width * (0.4 + t.flex * pr**1.5) * taper
    else:
        wd = pen.width * (1 + t.flex * (pr - 0.75))
    if t.ink == "solid":
        d = pen.density * (0.5 + 0.65 * pr) - 0.08 * v
    elif t.ink == "viscous":
        d = pen.density * (0.85 + 0.2 * pr) - 0.3 * v * p["speed"] + noise1d(rng, n, 0.2 * kss, 0.06)
        if t.skip:
            gap = (v * p["speed"] > 0.55) & (noise1d(rng, n, 0.35 * kss, 1) > 1.1 / t.skip)
            d = np.where(gap, d * 0.25, d)
        d[: max(int(0.02 * kss), 1)] += 0.12
    else:
        d = pen.density * (0.95 + 0.1 * pr) + 0.15 * (1 - v)
    return np.maximum(wd, 0.02), np.clip(d, 0, 1)


def render(strokes, pen, p, dpi, rng, shape=None, origin=None, cfg=RenderCfg()):
    """Ink density (H, W) float32 in [0, 1] and the origin (mm at pixel (0, 0)); p: decoded writer genes."""
    k = dpi / 25.4
    ss = cfg.supersample
    xh = p["size"]
    t = pen.spec
    cap = t.max_xh * xh
    if shape is None:
        allp = np.vstack([s.p for s in strokes])
        m = 3 * pen.reach() + 1
        origin = allp.min(0) - m
        shape = tuple(np.ceil((allp.max(0) - origin + m) * k).astype(int)[::-1])
    origin = np.asarray(origin, float)
    C = np.zeros(shape, np.float32)
    flow = 1.0
    for s in strokes:
        flow = float(np.clip(flow + rng.normal(0, 0.03) + 0.05 * (1 - flow), 0.75, 1.1))  # ink flow drifts

        q = (s.p - origin) * k
        if len(q) < 2:
            continue
        rmin = max(min(pen.width, cap) * k * ss / 2, cfg.min_radius)
        n0 = len(q)
        q = _dense(q, max(cfg.min_step, cfg.step * rmin) / ss)
        sp = None if s.v is None else np.interp(np.linspace(0, 1, len(q)), np.linspace(0, 1, n0), s.v)
        v, pr, ang, arc, L = kinematics(q, k, xh, p, rng, sp)
        wd, d = marks(pen, v, pr, ang, arc, L, p, k * ss, xh, rng)
        d = np.clip(d * flow, 0, 1)
        if s.dot:
            wd = wd * 1.15
        r = np.maximum(np.minimum(wd, cap) * k * ss / 2, cfg.min_radius)
        x0, y0 = np.floor(q.min(0) - r.max() / ss - 2).astype(int)
        x1, y1 = np.ceil(q.max(0) + r.max() / ss + 2).astype(int)
        ya, yb, xa, xb = max(y0, 0), min(y1, shape[0]), max(x0, 0), min(x1, shape[1])
        if ya >= yb or xa >= xb:
            continue
        P = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
        Q = ((q - (x0, y0)) * ss * 16).astype(np.int32)
        R = (r * 16).astype(np.int32)
        D = (d * 255).astype(np.int32)
        for i in range(len(q)):
            cv2.circle(P, (int(Q[i, 0]), int(Q[i, 1])), int(R[i]), int(D[i]), -1, cv2.LINE_AA, 4)
        P = cv2.resize(P, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        P = P[ya - y0 : yb - y0, xa - x0 : xb - x0]
        c = C[ya:yb, xa:xb]
        hi = np.maximum(c, P)
        C[ya:yb, xa:xb] = hi + t.overlap * np.minimum(c, P) * (1 - hi)
    return np.clip(_texture(C, t, k, rng), 0, 1), origin


def _dense(q, step):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
    if d[-1] < step:
        return q
    s = np.linspace(0, d[-1], int(d[-1] / step) + 1)
    return np.c_[np.interp(s, d, q[:, 0]), np.interp(s, d, q[:, 1])]


def _texture(C, t, k, rng):
    mm = lambda x: max(x * k, 0.01)
    if t.grain:
        g = cv2.GaussianBlur(rng.normal(1, 0.5, C.shape).astype(np.float32), (0, 0), mm(0.04))
        fib = cv2.GaussianBlur(rng.normal(0, 1, C.shape).astype(np.float32), (0, 0), sigmaX=mm(0.25), sigmaY=mm(0.03))
        tex = np.clip(g * 0.8 + 0.25 * fib / (fib.std() + 1e-6) + 0.2, 0, 1.2)
        C = C * (1 - t.grain + t.grain * tex)
    if t.bleed:
        C = np.maximum(C, cv2.GaussianBlur(C, (0, 0), mm(0.08)) * t.bleed * 2)
    return cv2.GaussianBlur(C, (0, 0), mm(t.blur))
