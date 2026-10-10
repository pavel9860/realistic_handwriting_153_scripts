"""Writer: gene vector -> pen strokes of text in mm (y down, first baseline at y = 0), with word boxes."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import scripts as S
from .config import GENE_INDEX, GENES, LEFT_HANDED, GlyphCfg, MotionCfg
from .fonts import FontIndex, em_per_xh
from .noise import Field, noise1d
from .pen import Pen
from .motion import effort_lambda, min_jerk, pivot, synthesise
from .skeleton import glyph, resample, smooth

AR = 0.6  # correlation of size, slant and baseline jitter between neighbouring letters


def decode(genes):
    """Gene vector in [0, 1] -> {name: natural value}."""
    return {g.name: g.lo + (g.hi - g.lo) * float(v) for g, v in zip(GENES, genes)}


def encode(values):
    """{name: natural value} -> gene vector (missing genes 0.5)."""
    g = np.full(len(GENES), 0.5)
    for n, v in values.items():
        i = GENE_INDEX[n]
        g[i] = (v - GENES[i].lo) / (GENES[i].hi - GENES[i].lo)
    return np.clip(g, 0, 1)


@dataclass
class Stroke:
    p: np.ndarray  # (N, 2) mm
    dot: bool = False
    word: int = -1
    v: np.ndarray = None  # normalised pen speed along p (hand movement model), None: estimated by the pen
    script: str = "Latin"
    plan: np.ndarray = None  # planned movement (after hand movement, before tremor and page warp), mm


@dataclass
class Word:
    text: str
    line: int
    struck: bool = False
    box: tuple = ()  # x0, y0, x1, y1 (mm), set by Writer.lines


@dataclass
class Layout:
    strokes: list
    words: list
    lines: int
    pitch: float  # mm
    width: float  # mm
    rtl: bool = False
    meta: dict = field(default_factory=dict)


def _sharpen(s, tol):
    """Polyline simplified to straight runs (Douglas-Peucker, tolerance in em) and resampled: angular corners."""
    if tol <= 0 or len(s) < 5:
        return s
    q = cv2.approxPolyDP(s.astype(np.float32)[:, None], tol, False)[:, 0].astype(float)
    return resample(q, GlyphCfg().step) if len(q) >= 2 else s


class Writer:
    """genes: vector in [0, 1] (config.GENES); font_class: preferred allograph class; pens: pen prior."""

    def __init__(self, genes, seed=0, font_class="any", pens=None, fonts=None, name="", paper="any"):
        self.genes = np.asarray(genes, float)
        self.p = decode(self.genes)
        self.rng = np.random.default_rng(seed)
        self.font_class = font_class
        self.fonts = fonts or _index()
        self.pen = Pen.sample(self.rng, pens)
        self.name = name
        self.paper = paper
        self.identity = int(np.round(self.genes * 50).astype(int) @ (31 ** np.arange(len(self.genes)) % 1000003))
        self.slant_drift = 0.0
        self.mcfg = MotionCfg()
        self.pivot_r = self.rng.uniform(*self.mcfg.pivot_radius)
        self.lam = effort_lambda(self.p["effort"], self.mcfg)
        self.left = self.p["hand"] > LEFT_HANDED
        if self.left:
            self.p["slant"] -= 8 * self.rng.random()
            self.pen.angle += np.radians(15)

    # fonts ------------------------------------------------------------------------------------------------------
    def font(self, scr):
        """The writer's font for a script: its Latin font when that serves the script too (one hand across
        scripts), else its own choice among the script's candidates."""
        main = self.fonts.choose("Latin", self.p["font"], self.font_class)
        if scr != "Latin" and main in self.fonts.candidates(scr, self.font_class):
            return main
        return self.fonts.choose(scr, self.p["font"], self.font_class)

    # one word ---------------------------------------------------------------------------------------------------
    def word(self, text, xh, t):
        """Strokes of one word in mm, origin at its start on the baseline; returns (strokes, advance)."""
        p, rng = self.p, self.rng
        scr = S.text_script(text)
        size = xh * (1 + p["drift"] * (2 * t - 1)) * (1 + rng.normal(0, p["size_sd"] * 0.4))
        units = S.visual_units(text, scr)
        base_font = self.font(scr)
        alt_font = self.alt_font(scr)
        sl0 = np.radians(p["slant"] + self.slant_drift + rng.normal(0, p["slant_sd"] * 0.5))
        uphill = np.radians(rng.normal(0, 0.3 + p["slant_sd"] * 0.12))
        ar = np.zeros(4)
        n = len(units)
        rtl = S.text_rtl(text)
        x, out, prev = 0.0, [], None
        for k, u in enumerate(units):
            f = self.fonts.for_unit(alt_font if alt_font and self._uses_alt(u) else base_font, u)
            if f is None:
                continue
            strokes, adv = glyph(f, u)
            em = size * em_per_xh(f, scr if scr != "Common" else "Latin", GlyphCfg().px)
            kk = (n - 1 - k) if rtl else k
            decay = 1 - p["trail"] * 0.3 * max(0.0, kk / max(n - 1, 1) - 0.4) / 0.6 if n > 3 else 1.0
            big = p["cap"] if u.isupper() else 1.0
            ar = AR * ar + np.sqrt(1 - AR**2) * rng.normal(0, 1, 4)  # neighbouring letters vary together
            sx = p["xs"] * big * (1 + ar[0] * p["size_sd"]) * (1 + 0.3 * (1 - decay))
            sy = big * (1 + ar[1] * p["size_sd"]) * decay
            sl = sl0 + np.radians(ar[2] * p["slant_sd"])
            rot = np.radians(rng.normal(0, p["rot_sd"]))
            dy = ar[3] * p["base_sd"] * size - np.tan(uphill) * x
            c, s_ = np.cos(rot), np.sin(rot)
            A = np.array([[c, -s_], [s_, c]]) @ np.array([[1, -np.tan(sl)], [0, 1]]) @ np.diag([sx * em, sy * em])
            zone = 1 / em_per_xh(f, scr if scr != "Common" else "Latin", GlyphCfg().px)
            gs = []
            for st in strokes:
                ext = np.ptp(st, axis=0)
                dot = len(strokes) > 1 and np.hypot(*ext) < 0.1
                q = self._shape(self._ends(self._hand(self._zones(st, zone)), dot), dot) @ A.T + (x, dy)
                if dot and scr in S.JOINABLE:
                    q = self._dot(q, size, sl)
                gs.append(Stroke(q, dot, script=scr))
            if (
                prev is not None
                and scr in S.JOINABLE
                and u.isalpha()
                and prev[1].isalpha()
                and rng.random() < p["connect"]
            ):
                self._join(prev[0], gs, size)
            out += gs
            prev = (gs, u)
            x += adv * em * sx + (p["ls"] + rng.normal(0, p["ls_sd"])) * size
        return [s for s in out if s.p is not None], x

    def _hand(self, s):
        """Left hand: horizontal strokes pulled right to left, loops often clockwise (stroke direction sets where
        hooks, flicks and pressure ramps fall)."""
        if not self.left or len(s) < 3:
            return s
        d = s[-1] - s[0]
        closed = np.hypot(*d) < 0.02
        if (not closed and abs(d[0]) > 2 * abs(d[1]) and d[0] > 0) or (closed and self.rng.random() < 0.5):
            return s[::-1]
        return s

    def _ends(self, s, dot):
        """Stroke ends overshoot or stop short of where they meet (em): no two strokes join exactly."""
        if dot or len(s) < 6:
            return s
        rng, sd = self.rng, 0.012 + 0.02 * self.p["speed"]
        out = s
        for end in (0, -1):
            e = rng.normal(0, sd)
            nb = s[1] if end == 0 else s[-2]
            d = (s[end] - nb) / (np.hypot(*(s[end] - nb)) + 1e-9)
            if e > 0:
                ext = s[end] + d * e
                out = np.vstack([ext, out]) if end == 0 else np.vstack([out, ext])
            else:
                cut = min(int(-e / GlyphCfg().step), len(out) // 4)
                if cut:
                    out = out[cut:] if end == 0 else out[:-cut]
        return out

    def _uses_alt(self, unit):
        """Letters written in the writer's second allograph: a fixed share of the letters (identity), each of them
        also switching now and then (state)."""
        import zlib

        h = zlib.crc32(f"{self.identity}/{unit}".encode()) / 2**32
        return (h < self.p["allo"]) != (self.rng.random() < 0.04)

    def alt_font(self, scr):
        """Second allograph source: another font of the same class serving the script."""
        cands = [f for f in self.fonts.candidates(scr, self.font_class) if f != self.font(scr)]
        if not cands:
            return None
        return cands[int(((self.p["font"] * 7.31) % 1) * len(cands))]

    def _zones(self, s, zone):
        """Ascender, x-height and descender zones scaled separately (asc, desc, flat genes); em units, y down."""
        p = self.p
        fz = 1 - 0.35 * p["flat"]
        y = s[:, 1]
        y2 = np.where(y < -zone, -zone * fz + (y + zone) * p["asc"], np.where(y > 0, y * p["desc"], y * fz))
        return np.c_[s[:, 0], y2]

    def _shape(self, s, dot):
        p, rng = self.p, self.rng
        if dot:
            return s
        s = s + noise1d(rng, len(s), 10, p["wobble"] * 0.007)[:, None] * rng.normal(0, 1, 2)
        s = _sharpen(s, p["angular"] * 0.03)
        s = smooth(s, p["round"] * 2.0 * (1 - 0.8 * p["angular"]))
        if len(s) > 6 and rng.random() < p["hook"] * 0.5:
            d = s[0] - s[3]
            d = d / (np.hypot(*d) + 1e-9)
            h = 0.03 + 0.04 * rng.random()
            s = np.vstack([s[0] + h * np.array([d[1], -d[0]]) * rng.choice((-1, 1)) + d * h * 0.5, s])
        if len(s) > 6 and rng.random() < p["flick"] * 0.6:
            d = s[-1] - s[-4]
            d = d / (np.hypot(*d) + 1e-9)
            s = np.vstack([s, s[-1] + d * (0.04 + 0.12 * rng.random()) + (0, -0.02)])
        return s

    def _dot(self, q, size, sl):
        p, rng = self.p, self.rng
        c = q.mean(0) + (p["dot_dx"] * size * rng.uniform(0.3, 1.5), rng.normal(0, 0.08) * size)
        ln = size * p["dot_len"] * rng.uniform(0, 0.35)
        a = np.radians(rng.uniform(-60, 20)) - sl * 0.5
        d = np.array([np.cos(a), np.sin(a)]) * ln / 2
        return np.vstack([c - d, c + d]) if ln > 0.02 else np.vstack([c, c + (0.01, 0.01)])

    @staticmethod
    def _join(a, b, size):
        """Joins the exit of the previous letter (rightmost low end) to the entry of this one (leftmost end) with a
        cubic Bezier along both end tangents; the joined path replaces the entry stroke."""
        ea = [s for s in a if not s.dot and s.p is not None]
        eb = [s for s in b if not s.dot and s.p is not None]
        if not ea or not eb:
            return
        lead = np.array((1.0, 0.4))
        sa = max(ea, key=lambda s: max(s.p[0] @ lead, s.p[-1] @ lead))
        if sa.p[0] @ lead > sa.p[-1] @ lead:
            sa.p = sa.p[::-1]
        sb = min(eb, key=lambda s: min(s.p[0, 0], s.p[-1, 0]))
        if sb.p[-1, 0] < sb.p[0, 0]:
            sb.p = sb.p[::-1]
        p0, p1 = sa.p[-1], sb.p[0]
        L = np.hypot(*(p1 - p0))
        if L > 1.1 * size or p0[1] < -1.5 * size:
            return
        d0 = p0 - sa.p[max(len(sa.p) - 4, 0)]
        d1 = sb.p[min(3, len(sb.p) - 1)] - p1
        d0, d1 = d0 / (np.hypot(*d0) + 1e-9), d1 / (np.hypot(*d1) + 1e-9)
        t = np.linspace(0, 1, max(int(L / 0.05), 3))[:, None]
        c0, c1 = p0 + d0 * L * 0.4, p1 - d1 * L * 0.4
        bez = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c0 + 3 * (1 - t) * t**2 * c1 + t**3 * p1
        sb.p = np.vstack([sa.p, bez[1:-1], sb.p])
        sa.p = None

    def _scribble(self, x0, x1, size):
        rng, kind = self.rng, self.rng.integers(3)
        y = -size * rng.uniform(0.3, 0.6)
        line = lambda dy: np.c_[np.linspace(x0, x1, 20), y + dy + np.linspace(0, rng.normal(0, 0.12) * size, 20)]
        if kind == 0:
            return [line(0)]
        if kind == 1:
            return [line(-0.15 * size), line(0.15 * size)]
        n = int(rng.integers(5, 12))
        zz = np.c_[np.linspace(x0, x1, 2 * n), y + np.tile([-0.5, 0.5], n) * size * rng.uniform(0.6, 1.2)]
        return [smooth(resample(zz, size * 0.05), 2)]

    def _typo(self, w):
        rng = self.rng
        ch = list(w)
        j = int(rng.integers(len(ch)))
        ch[j] = ch[int(rng.integers(len(ch)))] if rng.random() < 0.5 else ch[j] * 2
        return "".join(ch[: int(rng.integers(2, len(ch) + 1))])

    def _case(self, w):
        p, rng = self.p, self.rng
        if p["upper"] > 0.5:
            return w.upper()
        if p["caps"] > 0 and len(w) > 1:
            return w[0] + "".join(c.upper() if rng.random() < p["caps"] else c for c in w[1:])
        return w

    # text -------------------------------------------------------------------------------------------------------
    def tokens(self, text):
        """(text, script, space after) tokens: words; scripts without spaces split into short chunks (single
        characters for Han and kana) with closing punctuation kept on the chunk before it."""
        out = []
        for w in text.split():
            scr = S.text_script(w)
            if scr in S.NOSPACE and len(w) > 6:
                cs = S.UNIT.findall(w)
                step = 1 if scr in S.CHARWISE else int(self.rng.integers(3, 7))
                chunks = []
                for j in range(0, len(cs), step):
                    c = "".join(cs[j : j + step])
                    if chunks and all(S.script(x) == "Common" and not x.isalnum() for x in c):
                        chunks[-1] += c
                    else:
                        chunks.append(c)
                out += [(c, scr, False) for c in chunks]
                out[-1] = (out[-1][0], scr, True)
            else:
                out.append((self._case(w), scr, True))
        return out

    def lines(self, text, width_mm, rules=None):
        """Layout of text wrapped to width_mm; rules: baseline pitch (mm) of ruled paper to follow."""
        p, rng = self.p, self.rng
        p["size"] = max(p["size"], 4 * self.pen.edge, 5 * self.pen.width)  # x-height of at least 4 nib widths
        xh = p["size"]
        pitch = rules or max(p["pitch"], self.min_pitch()) * xh
        rtl = S.text_rtl(text)
        toks = self.tokens(text)
        placed, words = [], []
        line, x = 0, rng.normal(0, p["margin_sd"] * xh * 0.3)
        for i, (w, scr, space) in enumerate(toks):
            todo = [(self._typo(w), True)] if rng.random() < p["cross"] and len(w) > 2 and w.isalpha() else []
            self.slant_drift = 0.9 * self.slant_drift + rng.normal(0, 0.6 * p["slant_sd"] + 0.3)
            for ww, struck in todo + [(w, False)]:
                strokes, adv = self.word(ww, xh, i / max(len(toks) - 1, 1))
                if rng.random() < p["retrace"] and len(strokes) > 1:
                    s = strokes[int(rng.integers(len(strokes)))]
                    strokes.append(Stroke(s.p + rng.normal(0, 0.06, 2) * xh))
                if x + adv > width_mm and placed and placed[-1][0] == line:
                    line += 1
                    x = rng.normal(0, p["margin_sd"] * xh * 0.3)
                if struck:
                    strokes += [Stroke(q) for q in self._scribble(-0.05 * xh, adv + 0.05 * xh, xh)]
                for s in strokes:
                    s.word = len(words)
                words.append(Word(ww, line, struck))
                placed.append((line, x, strokes, adv))
                gap = p["ws"] * xh * (1 + rng.normal(0, 0.25)) if space else rng.normal(0.05, 0.05) * xh
                x += adv + (max(gap, 0.3 * xh) if space else gap)
        n_lines = line + 1
        on_rules = rules is not None
        slope = np.tan(np.radians(p["slope"])) * (0.15 if on_rules else 1)
        warp = Field(rng, p["elastic"] * xh * 0.04, xh * 6)
        bends = [self._baseline(xh, width_mm, on_rules) for _ in range(n_lines)]
        strokes = []
        for ln, x, ss, adv in placed:
            X = width_mm - x - adv if rtl else x
            base = ln * pitch + (0 if on_rules else rng.normal(0, 0.06) * xh)
            for s in ss:
                q = s.p + (X, base)
                q[:, 1] += slope * q[:, 0] + bends[ln](q[:, 0])
                s.p = q
                strokes.append(s)
        self._move(strokes, xh)
        for s in strokes:
            s.p = warp(s.p)
            if len(s.p) > 3 and p["tremor"] > 0.05:
                n = len(s.p)
                s.p = s.p + np.c_[noise1d(rng, n, 1.2, 1), noise1d(rng, n, 1.2, 1)] * p["tremor"] * xh * 0.012
        for i, w in enumerate(words):
            pts = [s.p for s in strokes if s.word == i]
            if pts:
                a = np.vstack(pts)
                w.box = (*a.min(0), *a.max(0))
        return Layout(
            strokes,
            words,
            n_lines,
            pitch,
            width_mm,
            rtl,
            dict(
                writer=self.name, pen=self.pen.tool, ink=self.pen.ink_name, xh=xh, hand="left" if self.left else "right"
            ),
        )

    def motion(self, scr):
        """(overlap, mu, sigma) of the writer's sub-movements in a script: speed sets fluency and tempo, tremor and
        slowness widen the response, discrete-stroke scripts overlap less."""
        p, m = self.p, self.mcfg
        ov = m.overlap[1] - (m.overlap[1] - m.overlap[0]) * (0.75 * p["speed"] + 0.25 * p["connect"])
        ov = float(np.clip(ov + m.script_overlap.get(scr, 0.0) + 0.1 * p["angular"], 0.15, 0.95))
        mu = m.mu[1] - (m.mu[1] - m.mu[0]) * p["speed"]
        sigma = m.sigma[0] + (m.sigma[1] - m.sigma[0]) * max(p["tremor"], 1 - p["speed"]) * 0.8
        return ov, mu, sigma

    def _move(self, strokes, xh):
        """Strokes re-synthesised as lognormal hand movements; dots and two-point marks stay as they are."""
        for s in strokes:
            if s.dot or len(s.p) < 4:
                continue
            ov, mu, sigma = self.motion(s.script)
            s.p, s.v = synthesise(s.p, self.rng, ov, mu, sigma, self.mcfg.tol * xh, 0.02 * xh)
            s.p = min_jerk(s.p, self.lam)
            s.plan = s.p.copy()

    def min_pitch(self):
        """Line pitch (x-heights) below which ascenders of a line meet descenders of the line above."""
        return 1.0 + 0.8 * self.p["asc"] + 0.7 * self.p["desc"]

    def _baseline(self, xh, width, on_rules):
        """Vertical offset of a line's baseline along x: bow plus smooth random wander (correlation length 15-60 mm),
        both damped on ruled paper."""
        p, rng = self.p, self.rng
        k = 0.3 if on_rules else 1.0
        bow = p["bow"] * xh * rng.uniform(0.3, 1) * k
        n = int(width) + 40
        wander = noise1d(rng, n, rng.uniform(15, 60), p["wander"] * xh * 0.18 * k)
        w = max(width, 1)
        shifts = np.cumsum(rng.uniform(*self.mcfg.hand_shift, 40)) - 30
        side = (1 if self.left else -1) * k
        return lambda x: (
            bow * 4 * (x / w) * (1 - x / w)
            + np.interp(x + 20, np.arange(n), wander)
            + pivot(x, np.r_[-25.0, shifts], self.pivot_r, side)
        )


_INDEX = []


def _index():
    if not _INDEX:
        _INDEX.append(FontIndex())
    return _INDEX[0]
