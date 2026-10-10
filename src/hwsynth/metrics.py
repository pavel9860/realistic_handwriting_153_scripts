"""Evaluation on held-out real lines (test documents) with texts disjoint from the fitness texts.

Per system: realism (test split), Frechet distance of embeddings and of stroke features to the real test lines,
k-NN precision and recall (Kynkaanniemi et al. 2019), writer diversity, writer identifiability (1-NN writer
accuracy over lines), legibility (Tesseract character error rate on English lines) and speed.
"""

import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
from scipy import linalg

from .config import GENE_INDEX, PERSONS, STYLES
from .features import feats
from .page import line_crop
from .styles import individual, styled
from .writer import Writer

TEXTS_EN = (
    "Please send the invoice before Friday afternoon",
    "We met at the station near the old bridge",
    "Remember to buy milk, bread and seven apples",
    "The meeting moved to room twelve on Monday",
    "Thank you for the lovely flowers and the card",
)
TEXTS_OTHER = (
    "Bitte schicken Sie mir die Unterlagen bis Mittwoch",
    "Je vous écris pour confirmer notre rendez-vous",
    "Завтра мы поедем на дачу к бабушке",
    "Llegaremos tarde porque perdimos el autobús",
)


def frechet(a, b):
    """Frechet distance between Gaussians fitted to the rows of a and b."""
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    s = linalg.sqrtm(ca @ cb)
    return float(((ma - mb) ** 2).sum() + np.trace(ca + cb - 2 * s.real))


def precision_recall(real, fake, k=3):
    """Share of fake samples inside the real k-NN manifold (precision) and of real inside the fake one (recall)."""

    def radii(x):
        d = np.linalg.norm(x[:, None] - x[None], axis=2)
        return np.sort(d, axis=1)[:, k]

    def inside(x, ref, r):
        d = np.linalg.norm(x[:, None] - ref[None], axis=2)
        return float((d <= r[None]).any(1).mean())

    return inside(fake, real, radii(real)), inside(real, fake, radii(fake))


def cer(ref, hyp):
    """Character error rate (Levenshtein distance / reference length)."""
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def ocr(crop):
    """Tesseract (English, single line) transcription of a crop, None when Tesseract is missing."""
    if not shutil.which("tesseract"):
        return None
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        cv2.imwrite(f.name, cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_REPLICATE))
        r = subprocess.run(["tesseract", f.name, "-", "-l", "eng", "--psm", "7"], capture_output=True, text=True)
    return " ".join(r.stdout.split())


def font_baseline(text, seed):
    """Plain font rendering: a handwriting font, no writer model, no pen model."""
    from PIL import Image, ImageDraw
    from .writer import _index

    fi = _index()
    f = fi.choose("Latin", np.random.default_rng(seed).random())
    from .fonts import pil_font

    pf = pil_font(f, 48)
    w = int(pf.getlength(text)) + 40
    im = Image.new("L", (w, 90), 245)
    ImageDraw.Draw(im).text((20, 60), text, fill=25, font=pf, anchor="ls")
    a = np.asarray(im)
    ys, xs = np.nonzero(a < 128)
    pad = int(0.15 * (ys.max() - ys.min())) + 2
    return a[max(ys.min() - pad, 0) : ys.max() + pad, max(xs.min() - pad, 0) : xs.max() + pad]


def _ablate(w, what):
    p = w.p
    if what == "no_noise":
        for n in ("elastic", "wobble", "tremor", "slant_sd", "size_sd", "rot_sd", "base_sd", "ls_sd", "wander"):
            p[n] = 0.0
    elif what == "no_joins":
        p["connect"] = 0.0
    elif what == "no_pen":
        w.pen.tool, w.pen.edge = "fineliner", 0.0
        p["press_sd"], p["down"] = 0.0, 0.0
    return w


SYSTEMS = ("font", "random", "prior", "bank", "bank_no_pen", "bank_no_noise", "bank_no_joins")


def writer_for(system, i, style="casual"):
    """i-th writer of a system (deterministic)."""
    if system == "random":
        return Writer(np.random.default_rng(10_000 + i).random(len(GENE_INDEX)), 20_000 + i)
    person = list(PERSONS)[i % len(PERSONS)]
    per, g = individual(person, 30_000 + i, bank=system != "prior")
    w = styled(per, g, style, 40_000 + i)
    return _ablate(w, system[5:]) if system.startswith("bank_") else w


def generate(system, n_writers, texts, style="casual", motion=None):
    """Line crops with writer ids; motion: list collecting (energy, entropy) of the planned pen paths."""
    from .motion import energy, turning_entropy

    crops, wid = [], []
    for i in range(n_writers):
        w = None if system == "font" else writer_for(system, i, style)
        for t in texts:
            if w is None:
                c = font_baseline(t, i)
            else:
                c, lay = line_crop(w, t, layout=True)
                if motion is not None:
                    P = [s.plan for s in lay.strokes if s.plan is not None and len(s.plan) > 8]
                    motion += [(energy(q, w.p["size"]), turning_entropy(q)) for q in P]
            if c is not None and c.shape[0] > 8:
                crops.append(c)
                wid.append(i)
    return crops, np.array(wid)


def evaluate(system, realism, real, n_writers=60, style="casual", ocr_lines=40):
    """Metrics of one system against the real test references (realism: Realism('test'), real: its arrays)."""
    from .realism import embed

    t0 = time.time()
    mot = []
    crops, wid = generate(system, n_writers, TEXTS_EN[:2] + TEXTS_OTHER[:2], style, mot)
    dt = (time.time() - t0) / max(len(crops), 1)
    F = [feats(c) for c in crops]
    ok = [i for i, f in enumerate(F) if f is not None]
    crops, wid, F = [crops[i] for i in ok], wid[ok], np.array([F[i] for i in ok])
    E = embed(crops)
    Z = (F - realism.mu) / realism.sd
    prec, rec = precision_recall(real.E, E)
    means = np.array([E[wid == w].mean(0) for w in np.unique(wid)])
    div = float(np.mean(1 - (means @ means.T)[np.triu_indices(len(means), 1)]))
    sim = E @ E.T
    np.fill_diagonal(sim, -9)
    ident = float((wid[sim.argmax(1)] == wid).mean())
    errs = []
    if shutil.which("tesseract"):
        en, _ = generate(system, ocr_lines, TEXTS_EN[2:3], style)
        errs = [cer(TEXTS_EN[2], ocr(c) or "") for c in en]
    return dict(
        system=system,
        style=style,
        lines=len(crops),
        realism=round(realism(E, F), 3),
        fd_embed=round(frechet(real.E, E), 3),
        fd_feats=round(frechet(real.F, Z), 2),
        precision=round(prec, 3),
        recall=round(rec, 3),
        diversity=round(div, 4),
        writer_id=round(ident, 3),
        energy=float(f"{np.median([e for e, _ in mot]):.3g}") if mot else None,
        entropy=round(float(np.median([h for _, h in mot])), 3) if mot else None,
        cer=round(float(np.mean(errs)), 3) if errs else None,
        ms_per_line=round(1000 * dt, 1),
    )


def real_baseline(realism_test, n=240, seed=0):
    """The metrics of real lines (half of the test lines against the other half, split by document)."""
    rng = np.random.default_rng(seed)
    docs = np.unique(realism_test.doc)
    half = set(rng.permutation(docs)[: len(docs) // 2])
    a = np.array([d in half for d in realism_test.doc])
    Ea, Eb = realism_test.E[a][:n], realism_test.E[~a]
    Fa, Fb = realism_test.F[a][:n], realism_test.F[~a]
    prec, rec = precision_recall(Eb, Ea)
    return dict(
        system="real (held-out half)",
        style="-",
        lines=len(Ea),
        fd_embed=round(frechet(Eb, Ea), 3),
        fd_feats=round(frechet(Fb, Fa), 2),
        precision=round(prec, 3),
        recall=round(rec, 3),
    )


__all__ = ["SYSTEMS", "STYLES", "cer", "evaluate", "frechet", "precision_recall", "real_baseline"]


def distinguishability(per_type=4, styles=("casual", "formal", "hurried", "medical", "calligraphic"), bank=True):
    """Top-1 identification of type (matched to other individuals), individual (other texts) and style (other
    individuals) on held-out texts, with image (contour hinge, chi-square) and pen-path (signatures) descriptors."""
    from .writerid import descriptor, euclid, identification, trajectory_descriptor

    D, T, lab = [], [], []
    for ti, p in enumerate(PERSONS):
        for k in range(per_type):
            per, g = individual(p, 50_000 + 100 * ti + k, bank=bank)
            for si, st in enumerate(styles):
                for j, text in enumerate(TEXTS_EN[3:5]):
                    w = styled(per, g, st, 60_000 + 1000 * ti + 100 * k + 10 * si + j)
                    c, lay = line_crop(w, text, layout=True)
                    d = descriptor(c) if c is not None else None
                    tr = trajectory_descriptor(lay.strokes, w.p["size"])
                    if d is None or tr is None:
                        continue
                    D.append(d)
                    T.append(tr)
                    lab.append((ti, ti * 100 + k, si, j))
    D, T, lab = np.array(D), np.array(T), np.array(lab)
    T = (T - T.mean(0)) / (T.std(0) + 1e-9)
    out = dict(types=len(PERSONS), styles=len(styles), lines=len(lab))
    for name, X, kw in (("image", D, {}), ("pen_path", T, {"dist": euclid})):
        out[f"{name}_type"] = round(identification(X, lab[:, 0], lab[:, 1], **kw), 3)
        out[f"{name}_individual"] = round(identification(X, lab[:, 1], lab[:, 1] * 10 + lab[:, 3], **kw), 3)
        out[f"{name}_style"] = round(identification(X, lab[:, 2], lab[:, 1], **kw), 3)
    out.update(
        chance_type=round(1 / len(PERSONS), 3),
        chance_individual=round(1 / (len(PERSONS) * per_type), 4),
        chance_style=round(1 / len(styles), 3),
    )
    return out
