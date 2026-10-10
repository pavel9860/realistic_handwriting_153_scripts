"""Joint evolution of the individuals of all handwriting types.

Each type keeps a population of individuals within its range (type centre +- spread). Fitness of an individual:
  realism    typicality of its lines among the real fit lines (realism.Realism)
  separation how much nearer its writer descriptors (image: contour hinge, chi-square; pen paths: path signatures,
             z-scored Euclidean) are to its own type's centroid than to the nearest other type's centroid
  novelty    distance of its image embedding to the nearest individuals of its own type (population and bank)
Lines are rendered in the casual style and one random other style, so an individual must hold up in every
situation. An individual enters its type's bank when a re-evaluation with fresh seeds also reaches min_q, it is
nearer its own type than any other, and it differs from every banked individual of its type by min_nov.
"""

import json
import time
from multiprocessing import Pool

import numpy as np

from .config import PERSON_SPREAD, PERSONS, STYLES, EvolveCfg, Paths
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


def render_lines(args):
    """[(crop, stroke features, hinge descriptor, trajectory descriptor)] of an individual over the texts, or None
    when any line fails."""
    person, genes, seed, texts = args
    rng = np.random.default_rng(seed)
    styles = ["casual", list(STYLES)[int(rng.integers(len(STYLES)))]]
    out = []
    for j, t in enumerate(texts):
        w = styled(person, genes, styles[j % 2], int(rng.integers(1 << 31)))
        c, lay = line_crop(w, t, layout=True)
        if c is None or c.shape[0] <= 8:
            return None
        f, h, tr = feats(c), descriptor(c), trajectory_descriptor(lay.strokes, w.p["size"])
        if f is None or h is None or tr is None:
            return None
        out.append((c, f, h, tr))
    return out


class Judge:
    """Scores individuals of all types: realism, separation and image embeddings."""

    def __init__(self, realism, pool, cfg, rng, log=print):
        self.realism, self.pool, self.cfg, self.rng, self.log = realism, pool, cfg, rng, log
        self.tr_mu = self.tr_sd = None

    def __call__(self, items):
        """items: [(person, genes)] -> q, embedding means, hinge means, trajectory means (NaN rows on failure)."""
        from .realism import embed

        cfg = self.cfg
        jobs = [(p, g, int(self.rng.integers(1 << 31)), cfg.texts) for p, g in items for _ in range(cfg.seeds)]
        t0 = time.time()
        res = self.pool.map(render_lines, jobs, chunksize=4)
        t1 = time.time()
        n = len(items)
        q = np.full(n, -9.0)
        E, Hh = np.full((n, 512), np.nan), None
        trs = []
        for i in range(n):
            rs = res[i * cfg.seeds : (i + 1) * cfg.seeds]
            if any(r is None for r in rs):
                trs.append(None)
                continue
            lines = [x for r in rs for x in r]
            e = embed([x[0] for x in lines])
            q[i] = self.realism(e, [x[1] for x in lines])
            E[i] = e.mean(0)
            h = np.mean([x[2] for x in lines], 0)
            if Hh is None:
                Hh = np.full((n, len(h)), np.nan)
            Hh[i] = h
            trs.append(np.mean([x[3] for x in lines], 0))
        if Hh is None:
            Hh = np.full((n, 1), np.nan)
        self.log(f"  {len(jobs)} renders {t1 - t0:.0f} s, scoring {time.time() - t1:.0f} s")
        dim = next((len(t) for t in trs if t is not None), 1)
        Tr = np.array([t if t is not None else np.full(dim, np.nan) for t in trs])
        if self.tr_mu is None:
            ok = ~np.isnan(Tr).any(1)
            self.tr_mu, self.tr_sd = Tr[ok].mean(0), Tr[ok].std(0) + 1e-9
        return q, E, Hh, (Tr - self.tr_mu) / self.tr_sd


def separation(Hh, Tr, labels):
    """Per row: (nearest other centroid - own centroid) distance, for hinge (chi-square) and trajectory (Euclidean),
    each divided by its median own-centroid distance and averaged."""
    ok = ~np.isnan(Hh).any(1)
    types = sorted(set(labels))
    cH = {t: np.nanmean(Hh[(labels == t) & ok], 0) for t in types}
    cT = {t: np.nanmean(Tr[(labels == t) & ok], 0) for t in types}
    out = np.full(len(labels), -9.0)
    dh_own, dt_own = [], []
    rows = []
    for i in np.nonzero(ok)[0]:
        t = labels[i]
        others = [u for u in types if u != t]
        dh = chi2(Hh[i], np.array([cH[u] for u in others]))[0].min(), chi2(Hh[i], cH[t][None])[0, 0]
        dtt = [np.linalg.norm(Tr[i] - cT[u]) for u in others]
        dt = min(dtt), np.linalg.norm(Tr[i] - cT[t])
        rows.append((i, dh, dt))
        dh_own.append(dh[1])
        dt_own.append(dt[1])
    sh, st = np.median(dh_own) + 1e-9, np.median(dt_own) + 1e-9
    for i, dh, dt in rows:
        out[i] = 0.5 * ((dh[0] - dh[1]) / sh + (dt[0] - dt[1]) / st)
    return out


def evolve_all(realism, persons=None, cfg=EvolveCfg(), paths=Paths(), workers=8, seed=0, cap=150, log=print):
    persons = persons or list(PERSONS)
    rng = np.random.default_rng(seed)
    per = max(cfg.pop // 4, 8)
    B = {p: bounds(p) for p in persons}
    pops = {p: np.array([person_genes(p, rng) for _ in range(per)]) for p in persons}
    banks = {p: [] for p in persons}
    bankE = {p: np.zeros((0, 512)) for p in persons}
    with Pool(workers) as pool:
        judge = Judge(realism, pool, cfg, rng, log)
        for gen in range(cfg.gens):
            items = [(p, g) for p in persons for g in pops[p]]
            labels = np.array([p for p, _ in items])
            q, E, Hh, Tr = judge(items)
            sep = separation(Hh, Tr, labels)
            nov = np.zeros(len(items))
            for p in persons:
                m = np.nonzero(labels == p)[0]
                ref = np.vstack([bankE[p], np.nan_to_num(E[m], nan=9.0)])
                for i in m:
                    d = np.sort(np.linalg.norm(ref - np.nan_to_num(E[i], nan=9.0), axis=1))
                    nov[i] = d[1 : cfg.nov_k + 1].mean() if len(d) > 1 else 0
            fit = q + cfg.sep_w * np.clip(sep, -3, 3) + cfg.nov_w * nov
            cand = [i for i in np.argsort(-fit) if q[i] >= cfg.min_q and sep[i] > 0 and len(banks[labels[i]]) < cap][
                : 3 * len(persons)
            ]
            if cand:
                q2, E2, H2, T2 = judge([items[i] for i in cand])
                for i, qq, ee in zip(cand, q2, E2):
                    p = labels[i]
                    if (
                        qq >= cfg.min_q
                        and not np.isnan(ee).any()
                        and len(banks[p]) < cap
                        and (not len(bankE[p]) or np.linalg.norm(bankE[p] - ee, axis=1).min() > cfg.min_nov)
                    ):
                        banks[p].append(
                            dict(
                                genes=items[i][1].round(4).tolist(),
                                q=round(float((q[i] + qq) / 2), 4),
                                sep=round(float(sep[i]), 3),
                            )
                        )
                        bankE[p] = np.vstack([bankE[p], ee])
            for p in persons:
                m = np.nonzero(labels == p)[0]
                el = pops[p][np.argsort(-fit[m])[: max(int(per * cfg.elite), 2)]]
                lo, hi = B[p]
                kids = []
                while len(kids) < per - len(el):
                    a, b = el[rng.choice(len(el), 2, replace=False)]
                    c = np.where(rng.random(len(lo)) < 0.5, a, b)
                    c = c + (rng.random(len(lo)) < cfg.mut_p) * rng.normal(0, cfg.mut_sd, len(lo))
                    c = np.where(rng.random(len(lo)) < cfg.reset_p, rng.uniform(lo, hi), c)
                    kids.append(np.clip(c, lo, hi))
                pops[p] = np.vstack([el, kids])
            ok = q > -9
            log(
                f"gen {gen}: realism {np.median(q[ok]):.3f} (max {q.max():.3f}) separation {np.median(sep[ok]):.3f} "
                f"(>0: {(sep[ok] > 0).mean():.2f}) novelty {nov.mean():.3f} bank {sum(map(len, banks.values()))}"
            )
            for p in persons:
                paths.bank(p).parent.mkdir(parents=True, exist_ok=True)
                paths.bank(p).write_text(json.dumps(banks[p]))
    return banks
