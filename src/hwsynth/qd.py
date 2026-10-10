"""Quality-diversity search (MAP-Elites) over the individuals of all handwriting types.

Each type has an archive over two measured behaviour descriptors of its writing: slant (from the pen paths) and
pen lifts per letter (connectedness). A cell keeps its best individual by quality:
  realism      typicality of its lines among the real fit lines (realism.Realism)
  - stability  inst_w * max(0, instability - inst_free): measured residual of the word centres around the line
               (in x-heights) beyond natural variation, so realism cannot be bought with shaky writing
  - 1          if the individual is nearer another type than its own (separation: contour hinge and path
               signature descriptors against the type centroids)
Lines are rendered in the casual style and one random other style. New individuals are iso-line variations of two
elites of the same type (Vassiliades & Mouret 2018), kept within the type's range. At the end every elite is
re-evaluated with fresh seeds and kept when it still reaches min_q and stays nearer its own type.
"""

import json
import time
from multiprocessing import Pool

import numpy as np

from .config import PERSON_SPREAD, PERSONS, STYLES, Paths, QDCfg
from .features import feats
from .page import line_crop
from .styles import person_genes, styled, type_centre
from .writer import encode
from .writerid import chi2, descriptor, trajectory_descriptor


def bounds(person):
    lo, hi = np.zeros(len(encode({}))), np.ones(len(encode({})))
    for i, c in type_centre(person).items():
        lo[i], hi[i] = max(c - PERSON_SPREAD, 0), min(c + PERSON_SPREAD, 1)
    return lo, hi


def measures(lay, text, xh):
    """(slant deg, pen lifts per letter, instability) of a written line, from its planned pen paths."""
    paths = [s.plan if s.plan is not None else s.p for s in lay.strokes if not s.dot and len(s.p) > 1]
    seg = np.vstack([np.diff(p, axis=0) for p in paths])
    seg = seg * np.where(seg[:, 1:2] < 0, -1, 1)  # all pointing down (y down)
    L = np.hypot(*seg.T)
    v = np.abs(seg[:, 1]) > 2 * np.abs(seg[:, 0])
    slant = float(np.degrees(np.average(np.arctan2(-seg[v, 0], seg[v, 1]), weights=L[v]))) if v.any() else 0.0
    lifts = len(paths) / max(sum(c.isalpha() for c in text), 1)
    cx, cy = [], []
    for i in range(len(lay.words)):
        pts = [s.p for s in lay.strokes if s.word == i and not s.dot]
        if pts:
            a = np.vstack(pts)
            cx.append(a[:, 0].mean())
            cy.append(np.median(a[:, 1]))
    if len(cx) < 3:
        return slant, lifts, 0.0
    res = np.array(cy) - np.polyval(np.polyfit(cx, cy, 1), cx)
    inst = res.std() / xh
    return slant, lifts, float(inst)


def render_lines(args):
    """[(crop, stroke features, hinge descriptor, trajectory descriptor, measures)] of an individual over the
    texts, or None when any line fails."""
    person, genes, seed, texts = args
    rng = np.random.default_rng(seed)
    styles = ["casual", list(STYLES)[int(rng.integers(len(STYLES)))]]
    out = []
    for j, t in enumerate(texts):
        w = styled(person, genes, styles[j % 2], int(rng.integers(1 << 31)))
        c, lay = line_crop(w, t, layout=True, scan=None)
        if c is None or c.shape[0] <= 8:
            return None
        f, h, tr = feats(c), descriptor(c), trajectory_descriptor(lay.strokes, w.p["size"])
        if f is None or h is None or tr is None:
            return None
        m = measures(lay, t, w.p["size"])
        v = image_instability(c)
        out.append((c, f, h, tr, (m[0], m[1], m[2] if v is None else v)))
    return out


class Judge:
    """Renders and scores individuals: realism, image embeddings, hinge and trajectory descriptors, measures."""

    def __init__(self, realism, pool, cfg, rng, log=print):
        self.realism, self.pool, self.cfg, self.rng, self.log = realism, pool, cfg, rng, log
        self.tr_mu = self.tr_sd = None

    def __call__(self, items, seeds):
        from .realism import embed

        cfg = self.cfg
        jobs = [(p, g, int(self.rng.integers(1 << 31)), cfg.texts) for p, g in items for _ in range(seeds)]
        t0 = time.time()
        res = self.pool.map(render_lines, jobs, chunksize=4)
        n = len(items)
        out = dict(q=np.full(n, -9.0), E=np.full((n, 512), np.nan), H=None, T=None, M=np.full((n, 3), np.nan))
        trs = []
        for i in range(n):
            rs = res[i * seeds : (i + 1) * seeds]
            if any(r is None for r in rs):
                trs.append(None)
                continue
            lines = [x for r in rs for x in r]
            e = embed([x[0] for x in lines])
            out["q"][i] = self.realism(e, [x[1] for x in lines])
            out["E"][i] = e.mean(0)
            h = np.mean([x[2] for x in lines], 0)
            if out["H"] is None:
                out["H"] = np.full((n, len(h)), np.nan)
            out["H"][i] = h
            trs.append(np.mean([x[3] for x in lines], 0))
            out["M"][i] = np.mean([x[4] for x in lines], 0)
        dim = next((len(t) for t in trs if t is not None), 1)
        T = np.array([t if t is not None else np.full(dim, np.nan) for t in trs])
        if self.tr_mu is None:
            ok = ~np.isnan(T).any(1)
            self.tr_mu, self.tr_sd = T[ok].mean(0), T[ok].std(0) + 1e-9
        out["T"] = (T - self.tr_mu) / self.tr_sd
        if out["H"] is None:
            out["H"] = np.full((n, 1), np.nan)
        self.log(f"  {len(jobs)} renders and scores in {time.time() - t0:.0f} s")
        return out


class Centroids:
    """Type centroids of hinge and trajectory descriptors (running means over everything evaluated)."""

    def __init__(self):
        self.sum, self.n = {}, {}

    def add(self, person, h, t):
        if np.isnan(h).any() or np.isnan(t).any():
            return
        s = self.sum.setdefault(person, [np.zeros_like(h), np.zeros_like(t)])
        s[0] += h
        s[1] += t
        self.n[person] = self.n.get(person, 0) + 1

    def separation(self, person, h, t):
        """(nearest other - own) distance, hinge and trajectory each divided by the own distance, averaged."""
        if np.isnan(h).any() or person not in self.n or len(self.n) < 2:
            return -9.0
        c = {p: (s[0] / self.n[p], s[1] / self.n[p]) for p, s in self.sum.items()}
        oh = chi2(h, c[person][0])[0, 0] + 1e-9
        ot = np.linalg.norm(t - c[person][1]) + 1e-9
        nh = min(chi2(h, v[0])[0, 0] for p, v in c.items() if p != person)
        nt = min(np.linalg.norm(t - v[1]) for p, v in c.items() if p != person)
        return float(0.5 * ((nh - oh) / oh + (nt - ot) / ot))


def cell(m, cfg):
    return int(np.digitize(m[0], cfg.slant_edges)), int(np.digitize(m[1], cfg.lift_edges))


def quality(q, inst, sep, cfg):
    return q - cfg.inst_w * max(0.0, inst - cfg.inst_free) - (1.0 if sep <= 0 else 0.0)


def search(realism, persons=None, cfg=QDCfg(), paths=Paths(), workers=8, seed=0, log=print):
    """MAP-Elites over all types; writes one bank per type and returns {type: [elite]}."""
    persons = persons or list(PERSONS)
    rng = np.random.default_rng(seed)
    B = {p: bounds(p) for p in persons}
    archive = {p: {} for p in persons}  # cell -> dict(genes, q, inst, sep, fit, m)
    cents = Centroids()

    def insert(items, res):
        for (p, g), q, h, t, m in zip(items, res["q"], res["H"], res["T"], res["M"]):
            cents.add(p, h, t)
        added = 0
        for (p, g), q, h, t, m in zip(items, res["q"], res["H"], res["T"], res["M"]):
            if q <= -9 or np.isnan(m).any():
                continue
            sep = cents.separation(p, h, t)
            fit = quality(q, m[2], sep, cfg)
            k = cell(m, cfg)
            if k not in archive[p] or fit > archive[p][k]["fit"]:
                archive[p][k] = dict(genes=g, q=float(q), inst=float(m[2]), sep=sep, fit=float(fit), m=m.tolist())
                added += 1
        return added

    with Pool(workers) as pool:
        judge = Judge(realism, pool, cfg, rng, log)
        items = [(p, person_genes(p, rng)) for p in persons for _ in range(cfg.init)]
        insert(items, judge(items, 1))
        for it in range(cfg.iters):
            items = []
            for p in persons:
                el = [e["genes"] for e in archive[p].values()] or [person_genes(p, rng)]
                lo, hi = B[p]
                for _ in range(cfg.batch):
                    a, b = el[rng.integers(len(el))], el[rng.integers(len(el))]
                    c = a + rng.normal(0, cfg.iso_sd, len(a)) + rng.normal(0, cfg.line_sd) * (b - a)
                    items.append((p, np.clip(c, lo, hi)))
            added = insert(items, judge(items, 1))
            fits = [e["fit"] for a in archive.values() for e in a.values()]
            insts = [e["inst"] for a in archive.values() for e in a.values()]
            log(
                f"iter {it}: added {added}, cells {len(fits)}, quality median {np.median(fits):.3f}, "
                f"instability median {np.median(insts):.3f}"
            )
        log("re-evaluating elites with fresh seeds")
        items = [(p, e["genes"]) for p in persons for e in archive[p].values()]
        res = judge(items, cfg.final_seeds)
        banks = {p: [] for p in persons}
        for (p, g), q, h, t, m in zip(items, res["q"], res["H"], res["T"], res["M"]):
            sep = cents.separation(p, h, t)
            if q > -9 and q >= cfg.min_q and sep > 0 and m[2] <= cfg.inst_max:
                banks[p].append(
                    dict(
                        genes=np.round(g, 4).tolist(),
                        q=round(float(q), 4),
                        sep=round(sep, 3),
                        inst=round(float(m[2]), 4),
                        slant=round(float(m[0]), 1),
                        lifts=round(float(m[1]), 2),
                    )
                )
    for p in persons:
        paths.bank(p).parent.mkdir(parents=True, exist_ok=True)
        paths.bank(p).write_text(json.dumps(banks[p]))
    log("bank sizes: " + ", ".join(f"{p} {len(b)}" for p, b in banks.items()))
    return banks


def image_instability(crop):
    """Baseline instability of a line image: residual of the word ink centres around their fitted line, in
    x-heights (core band of the row profile); None when the line has fewer than three words."""
    import cv2

    b = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    prof = (b > 0).sum(1).astype(float)
    if prof.max() == 0:
        return None
    xh = max(int((prof > 0.5 * prof.max()).sum()), 3)
    words = cv2.dilate(b, np.ones((1, max(int(0.6 * xh), 3)), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(words, 8)
    cx, cy = [], []
    for i in range(1, n):
        if st[i, 2] < xh or st[i, 4] < xh * xh * 0.5:
            continue
        ys, xs = np.nonzero((lab == i) & (b > 0))
        cx.append(xs.mean())
        cy.append(np.median(ys))
    if len(cx) < 3:
        return None
    res = np.array(cy) - np.polyval(np.polyfit(cx, cy, 1), cx)
    return float(res.std() / xh)
