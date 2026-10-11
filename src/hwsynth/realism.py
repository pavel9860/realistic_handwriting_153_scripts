"""Real handwritten line references and the realism score.

References: text lines cut from scanned handwritten pages, cleaned (no rules, staves or bars; contrast; ink share),
kept when most of their nearest labelled lines are handwriting, embedded (ResNet18, ImageNet weights) and described
by stroke features. Documents are split into fit (evolution) and test (metrics) sets; documents that occur in the
labels are never test documents. Realism of a writer: typicality of its lines among the fit (or test) references,
leave-one-document-out normalised.
"""

import hashlib
import pickle
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .config import Paths, RefsCfg
from .features import feats


@lru_cache(1)
def embedder():
    import torch
    import torchvision

    torch.set_num_threads(4)
    return torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()


def embed(crops, batch=64):
    """Unit embeddings (N, 512) of grayscale line crops: 64 px high, 639 px folded into three 213 px rows."""
    import torch

    m = embedder()
    if not hasattr(m, "_cut"):
        m.fc = torch.nn.Identity()
        m._cut = True
    xs = []
    for c in crops:
        c = cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX)
        w = min(max(int(c.shape[1] * 64 / c.shape[0]), 1), 639)
        t = np.full((64, 639), 255, np.uint8)
        t[:, :w] = cv2.resize(c, (w, 64))[:, :639]
        xs.append(np.vstack([t[:, i * 213 : (i + 1) * 213] for i in range(3)]))
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            x = torch.tensor(np.array(xs[i : i + batch]) / 255.0, dtype=torch.float32)[:, None].repeat(1, 3, 1, 1)
            out.append(m((x - 0.45) / 0.25).numpy())
    E = np.vstack(out) if out else np.zeros((0, 512))
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def page_lines(g, cfg=RefsCfg()):
    """Text-line crops of a grayscale page: ink bands of the row profile after removing long rules."""
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    for k in ((g.shape[1] // 8, 1), (1, g.shape[0] // 8)):
        b = cv2.subtract(b, cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k)))
    prof = cv2.GaussianBlur((b > 0).sum(1).astype(np.float32)[:, None], (1, 0), 3)[:, 0]
    on = prof > max(prof.max() * 0.08, 4)
    out, y = [], 0
    while y < len(on):
        if not on[y]:
            y += 1
            continue
        y1 = y
        while y1 < len(on) and on[y1]:
            y1 += 1
        if cfg.min_h <= y1 - y <= cfg.max_h:
            cols = np.nonzero((b[y:y1] > 0).sum(0) > 0)[0]
            if len(cols) and cols[-1] - cols[0] >= cfg.min_aspect * (y1 - y):
                pad = (y1 - y) // 6
                out.append(g[max(y - pad, 0) : y1 + pad, max(cols[0] - pad, 0) : cols[-1] + pad])
        y = y1
    return out


def clean(c):
    """No rules, staves or bars (no row half ink), enough contrast, some but not too much ink."""
    lo, hi = np.percentile(c, (2, 60))
    ink = c < (lo + hi) / 2
    return hi - lo > 70 and 0.03 < ink.mean() < 0.3 and ink.mean(1).max() < 0.5


def is_test(doc, share):
    return int(hashlib.md5(doc.encode()).hexdigest(), 16) % 1000 < share * 1000


def build_refs(pdf_dir, labels=None, out=Paths().refs_web, pages=None, cfg=RefsCfg(), log=print):
    """Reference set from the first page of each PDF in pdf_dir. labels: pickle of dicts (crop, hw, file) of
    labelled lines used by the handwriting filter. Writes out (npz) and a contact sheet next to it."""
    import pymupdf

    files = sorted(Path(pdf_dir).glob("*.pdf"))[:pages]
    lab = pickle.loads(Path(labels).read_bytes()) if labels else []
    lab_docs = {Path(x["file"]).stem for x in lab}
    crops, docs = [], []
    for f in files:
        try:
            pix = pymupdf.open(f)[0].get_pixmap(dpi=cfg.dpi, colorspace=pymupdf.csGRAY)
        except Exception:
            continue
        g = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w)
        ls = [c for c in page_lines(g, cfg) if clean(c)]
        rng = np.random.default_rng(int(hashlib.md5(f.stem.encode()).hexdigest()[:8], 16))
        for i in rng.permutation(len(ls))[: cfg.lines_per_page]:
            crops.append(ls[i])
            docs.append(f.stem)
    log(f"{len(crops)} lines from {len(set(docs))} documents")
    E = embed(crops)
    keep = np.ones(len(crops), bool)
    if lab:  # handwriting filter from labelled lines; without labels every clean line is kept
        El, y = embed([x["crop"] for x in lab]), np.array([x["hw"] for x in lab])
        keep = y[np.argsort(-(E @ El.T), axis=1)[:, :5]].mean(1) >= cfg.knn_hw
    F = [feats(c) if k else None for c, k in zip(crops, keep)]
    idx = [i for i, f in enumerate(F) if f is not None]
    docs = np.array(docs)[idx]
    test = np.array([is_test(d, cfg.test_share) and d not in lab_docs for d in docs])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from .writerid import descriptor

    Hd = [descriptor(crops[i]) for i in idx]
    ok = [j for j, h in enumerate(Hd) if h is not None]
    idx, docs, test, Hd = [idx[j] for j in ok], docs[ok], test[ok], [Hd[j] for j in ok]
    scripts = np.array([doc_script(d) or "?" for d in docs])
    np.savez(
        out,
        E=E[idx],
        F=np.array([F[i] for i in idx]),
        H=np.array(Hd),
        doc=docs,
        test=test,
        script=scripts,
        source=np.array(["web"] * len(idx)),
    )
    sheet([crops[i] for i in idx], out.with_suffix(".png"))
    log(f"kept {len(idx)}: fit {int((~test).sum())}, test {int(test.sum())}")
    return out


LANG_SCRIPT = {
    "ru": "Cyrillic",
    "uk": "Cyrillic",
    "bg": "Cyrillic",
    "be": "Cyrillic",
    "mk": "Cyrillic",
    "el": "Greek",
    "he": "Hebrew",
    "ar": "Arabic",
    "ur": "Arabic",
    "fa": "Arabic",
    "ja": "Japanese",
    "zh": "Han",
    "ko": "Hangul",
    "hi": "Devanagari",
    "th": "Thai",
}
LATIN_LANGS = {
    "en",
    "de",
    "fr",
    "es",
    "it",
    "pt",
    "ca",
    "nl",
    "pl",
    "cs",
    "hu",
    "sv",
    "da",
    "no",
    "ro",
    "hr",
    "sl",
    "et",
    "cy",
    "latin",
}


def doc_script(name):
    """Script of a document from the language tag at the end of its file name (None if unknown or ambiguous)."""
    import re

    m = re.search(r"[._]([a-z]{2,5})(?:_p\d+)?$", name)
    tag = m.group(1) if m else ""
    return LANG_SCRIPT.get(tag) or ("Latin" if tag in LATIN_LANGS else None)


def build_image_refs(img_dir, out, source="iam", scale=0.5, cfg=RefsCfg(), log=print):
    """Reference set from page images (one writer per page, e.g. IAM handwriting forms): lines, embeddings, stroke
    features and hinge descriptors; split into fit and test by page (writer proxy)."""
    from .writerid import descriptor

    files = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".tif", ".tiff"))
    crops, docs = [], []
    for f in files:
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ls = [c for c in page_lines(g, cfg) if clean(c)]
        rng = np.random.default_rng(int(hashlib.md5(f.stem.encode()).hexdigest()[:8], 16))
        for i in rng.permutation(len(ls))[: cfg.lines_per_page]:
            crops.append(ls[i])
            docs.append(f.stem)
    log(f"{len(crops)} lines from {len(set(docs))} pages")
    F = [feats(c) for c in crops]
    Hd = [descriptor(c) for c in crops]
    idx = [i for i in range(len(crops)) if F[i] is not None and Hd[i] is not None]
    docs = np.array(docs)[idx]
    test = np.array([is_test(d.split("-")[0] + d.split("-")[1][:3] if "-" in d else d, cfg.test_share) for d in docs])
    E = embed([crops[i] for i in idx])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        E=E,
        F=np.array([F[i] for i in idx]),
        H=np.array([Hd[i] for i in idx]),
        doc=docs,
        test=test,
        script=np.array(["Latin"] * len(idx)),
        source=np.array([source] * len(idx)),
    )
    sheet([crops[i] for i in idx], out.with_suffix(".png"))
    log(f"kept {len(idx)}: fit {int((~test).sum())}, test {int(test.sum())}")
    return out


def sheet(crops, path, n=120, cols=3, h=40, w=600):
    rows = [cv2.resize(c, (min(int(c.shape[1] * h / c.shape[0]), w), h)) for c in crops[:n]]
    rows = [np.pad(r, ((0, 2), (0, w - r.shape[1])), constant_values=255) for r in rows]
    per = -(-len(rows) // cols)
    rows += [np.full((h + 2, w), 255, np.uint8)] * (per * cols - len(rows))
    cv2.imwrite(str(path), np.hstack([np.vstack(rows[i : i + per]) for i in range(0, len(rows), per)]))


class Realism:
    """Typicality among real lines of one split: kNN similarity of embeddings and kNN distance of standardised stroke
    features, each z-scored against the same statistic of the split's lines (leave-one-document-out)."""

    def __init__(self, split="fit", path=Paths().refs, k=5):
        r = np.load(path, allow_pickle=False)
        m = r["test"] if split == "test" else ~r["test"]
        self.E, F, self.doc = r["E"][m], r["F"][m], r["doc"][m]
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-9
        self.F = (F - self.mu) / self.sd
        self.k = k
        ref = np.vstack(
            [self._stats(self.E[i : i + 1], self.F[i : i + 1], self.doc != self.doc[i]) for i in range(len(F))]
        )
        self.m, self.s = ref.mean(0), ref.std(0) + 1e-9

    def _stats(self, E, Z, mask=None):
        Er, Fr = (self.E, self.F) if mask is None else (self.E[mask], self.F[mask])
        e = np.sort(E @ Er.T, axis=1)[:, -self.k :].mean(1)
        d = np.array([np.sort(np.linalg.norm(Fr - z, axis=1))[: self.k].mean() for z in Z])
        return np.c_[e, -d]

    def lines(self, E, F):
        """Realism of each line (mean of the two z-scores)."""
        return ((self._stats(E, (np.asarray(F) - self.mu) / self.sd) - self.m) / self.s).mean(1)

    def __call__(self, E, F):
        return float(self.lines(E, F).mean())
