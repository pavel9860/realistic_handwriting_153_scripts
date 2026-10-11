"""Personas fitted to real writers (analysis-by-synthesis).

For every page of the fit split of the primary references (one writer per page, IAM handwriting forms) a small
evolution strategy searches the genes whose writing matches the writer: the distance is the chi-square distance of
contour-direction/hinge distributions (Bulacu & Schomaker) plus the standardised distance of stroke features, both
averaged over the writer's lines and over two generated lines. Each fitted persona keeps its page id, distance and
nearest handwriting type (whose font class and symbol habits it uses). Only fit-split pages are used, so the test
split stays unseen."""

import json
from multiprocessing import Pool

import numpy as np

from .config import GENE_INDEX, PERSONS, FitCfg, Paths
from .features import feats
from .page import line_crop
from .styles import person_genes, styled, type_centre
from .writerid import chi2, descriptor

_REF = {}


def _init(path):
    r = np.load(path, allow_pickle=False)
    m = ~r["test"]
    F = r["F"][m]
    _REF.update(mu=F.mean(0), sd=F.std(0) + 1e-9, H=r["H"][m], F=F, doc=r["doc"][m])


def _distance(person, genes, seed, target, cfg):
    hs, fs = [], []
    rng = np.random.default_rng(seed)
    for t in cfg.texts:
        c = line_crop(styled(person, genes, "casual", int(rng.integers(1 << 31))), t)
        if c is None or c.shape[0] <= 8:
            return 9.0
        h, f = descriptor(c), feats(c)
        if h is None or f is None:
            return 9.0
        hs.append(h)
        fs.append((np.asarray(f) - _REF["mu"]) / _REF["sd"])
    th, tf = target
    return float(chi2(np.mean(hs, 0), th)[0, 0] + cfg.feat_w * np.mean((np.mean(fs, 0) - tf) ** 2))


def nearest_type(genes):
    best, bd = None, np.inf
    for p in PERSONS:
        c = type_centre(p)
        d = np.mean([(genes[i] - v) ** 2 for i, v in c.items()])
        if d < bd:
            best, bd = p, d
    return best


def fit_page(args):
    """(page, genes, distance, type) of one real writer; (page, None, error, None) if it fails."""
    doc, seed, cfg = args
    try:
        return _fit_page(doc, seed, cfg)
    except Exception as e:  # one bad page never stops the run
        return doc, None, repr(e), None


def _fit_page(doc, seed, cfg):
    m = _REF["doc"] == doc
    target = (_REF["H"][m].mean(0), ((_REF["F"][m] - _REF["mu"]) / _REF["sd"]).mean(0))
    rng = np.random.default_rng(seed)
    types = list(PERSONS)
    pop = [(types[int(rng.integers(len(types)))], None) for _ in range(cfg.init)]
    pop = [(p, person_genes(p, rng)) for p, _ in pop]
    scored = sorted(
        ((_distance(p, g, int(rng.integers(1 << 31)), target, cfg), p, g) for p, g in pop), key=lambda x: x[0]
    )
    sigma = cfg.sigma
    free = [GENE_INDEX[n] for n in cfg.genes]
    for _ in range(cfg.gens):
        elite = scored[: cfg.elite]
        kids = []
        for _ in range(cfg.batch):
            d, p, g = elite[int(rng.integers(len(elite)))]
            c = g.copy()
            c[free] = np.clip(c[free] + rng.normal(0, sigma, len(free)), 0, 1)
            kids.append((p, c))
        scored = sorted(
            elite + [(_distance(p, g, int(rng.integers(1 << 31)), target, cfg), p, g) for p, g in kids],
            key=lambda x: x[0],
        )
        sigma *= cfg.sigma_decay
    d, p, g = scored[0]
    return doc, g.round(4).tolist(), round(d, 4), nearest_type(g)


def fit_personas(paths=Paths(), cfg=FitCfg(), workers=8, log=print):
    """Fits personas to fit-split pages; resumable: results are saved after every page and pages already fitted (or
    failed) in paths.personas are skipped."""
    r = np.load(paths.refs, allow_pickle=False)
    docs = sorted(set(r["doc"][~r["test"]]))
    docs = list(np.random.default_rng(cfg.seed).permutation(docs)[: cfg.pages])
    out = json.loads(paths.personas.read_text()) if paths.personas.exists() else []
    failed_path = paths.personas.with_suffix(".failed.json")
    failed = json.loads(failed_path.read_text()) if failed_path.exists() else {}
    done = {o["page"] for o in out} | set(failed)
    todo = [(d, i, cfg) for i, d in enumerate(docs) if d not in done]
    log(f"{len(done)} pages done, {len(todo)} to fit")
    with Pool(workers, initializer=_init, initargs=(str(paths.refs),)) as pool:
        for doc, g, d, t in pool.imap_unordered(fit_page, todo):
            if g is None:
                failed[doc] = d
                failed_path.write_text(json.dumps(failed))
                continue
            out.append(dict(page=doc, genes=g, distance=d, type=t))
            paths.personas.write_text(json.dumps(out))
            if len(out) % 25 == 0:
                log(f"{len(out)} personas, median distance {np.median([o['distance'] for o in out]):.3f}")
    log(f"{len(out)} personas, {len(failed)} failed")
    return out
