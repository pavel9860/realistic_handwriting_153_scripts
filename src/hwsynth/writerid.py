"""Writer similarity.

Image (any handwriting, also real scans): contour-direction (f1) and contour-hinge (f2) distributions of the ink
contours of a line (Bulacu & Schomaker 2007), at several fragment lengths, chi-square distance.
Trajectory (generated handwriting, which has its pen paths): truncated path signatures (Chen; Lyons) of stroke
fragments, plus speed-independent shape statistics (turning and direction distributions), z-scored Euclidean
distance. Used to measure how distinguishable types, styles and individuals are, and to keep types apart during
evolution."""

import cv2
import numpy as np

H = 64  # lines are scaled to this height
FRAGS = (3, 5, 8)  # contour fragment lengths (px)
N_DIR = 12  # direction bins over [0, pi)
N_HINGE = 24  # hinge leg bins over [0, 2 pi)


def descriptor(crop):
    """f1 (12) and f2 (300) per fragment length, concatenated, each normalised to sum 1; None for empty lines."""
    g = cv2.resize(crop, (max(int(crop.shape[1] * H / crop.shape[0]), 1), H), interpolation=cv2.INTER_AREA)
    b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cs = [c[:, 0].astype(float) for c in cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
    out = []
    for frag in FRAGS:
        f1 = np.zeros(N_DIR)
        f2 = np.zeros((N_HINGE, N_HINGE))
        for c in cs:
            if len(c) < 2 * frag + 1:
                continue
            fwd = np.roll(c, -frag, 0) - c
            back = np.roll(c, frag, 0) - c
            a1 = np.mod(np.arctan2(-fwd[:, 1], fwd[:, 0]), 2 * np.pi)
            a2 = np.mod(np.arctan2(-back[:, 1], back[:, 0]), 2 * np.pi)
            f1 += np.bincount((np.mod(a1, np.pi) / np.pi * N_DIR).astype(int) % N_DIR, minlength=N_DIR)
            i1 = (a1 / (2 * np.pi) * N_HINGE).astype(int) % N_HINGE
            i2 = (a2 / (2 * np.pi) * N_HINGE).astype(int) % N_HINGE
            np.add.at(f2, (np.minimum(i1, i2), np.maximum(i1, i2)), 1)
        if f1.sum() == 0:
            return None
        f2 = f2[np.triu_indices(N_HINGE)]
        out += [f1 / f1.sum(), f2 / f2.sum()]
    return np.concatenate(out) / len(FRAGS)


def signature(path, depth=3):
    """Truncated path signature (levels 1-3: 2 + 4 + 8 terms) of a piecewise-linear 2D path (N, 2), vectorised
    over its segments with Chen's identity."""
    dx = np.diff(np.asarray(path, float), axis=0)
    s1 = np.cumsum(dx, 0)
    p1 = s1 - dx  # level 1 before each segment
    s2 = np.einsum("ni,nj->ij", p1, dx) + 0.5 * np.einsum("ni,nj->ij", dx, dx)
    out = [s1[-1], s2.ravel()]
    if depth >= 3:
        c2 = np.cumsum(np.einsum("ni,nj->nij", p1, dx) + 0.5 * np.einsum("ni,nj->nij", dx, dx), 0)
        p2 = c2 - (np.einsum("ni,nj->nij", p1, dx) + 0.5 * np.einsum("ni,nj->nij", dx, dx))
        s3 = (
            np.einsum("nij,nk->ijk", p2, dx)
            + 0.5 * np.einsum("ni,nj,nk->ijk", p1, dx, dx)
            + np.einsum("ni,nj,nk->ijk", dx, dx, dx) / 6
        )
        out.append(s3.ravel())
    return np.concatenate(out)


def trajectory_descriptor(strokes, xh, frag=0.6):
    """Mechanistic descriptor of a writer's pen paths (mm, scaled by the x-height): mean and spread of the path
    signatures of fragments frag x-heights long, turning-angle and direction distributions, pen-lift rate."""
    sigs, turn, dirs, length = [], [], [], 0.0
    for s in strokes:
        p = np.asarray(s.p if hasattr(s, "p") else s, float) / xh
        if len(p) < 3:
            continue
        seg = np.diff(p, axis=0)
        sl = np.hypot(*seg.T)
        length += sl.sum()
        ang = np.arctan2(-seg[:, 1], seg[:, 0])
        dirs.append(np.mod(ang, np.pi))
        turn.append(np.abs(np.angle(np.exp(1j * np.diff(ang)))) / (sl[1:] + 1e-6))
        arc = np.r_[0, np.cumsum(sl)]
        for a in np.arange(0, arc[-1] - frag, frag):
            m = (arc >= a) & (arc <= a + frag)
            if m.sum() >= 3:
                q = p[m] - p[m][0]
                sigs.append(signature(q))
    if not sigs:
        return None
    S = np.array(sigs)
    t = np.concatenate(turn) if turn else np.zeros(1)
    d = np.concatenate(dirs) if dirs else np.zeros(1)
    th = np.histogram(np.clip(t, 0, 20), 10, (0, 20))[0]
    dh = np.histogram(d, N_DIR, (0, np.pi))[0]
    return np.r_[S.mean(0), S.std(0), th / max(th.sum(), 1), dh / max(dh.sum(), 1), len(strokes) / max(length, 1e-6)]


def chi2(a, b):
    """Chi-square distances between rows of a (N, D) and b (M, D)."""
    a, b = np.atleast_2d(a)[:, None], np.atleast_2d(b)[None]
    return 0.5 * ((a - b) ** 2 / (a + b + 1e-12)).sum(-1)


def euclid(a, b):
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    return np.sqrt(np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None] - 2 * a @ b.T, 0))


def identification(desc, labels, groups=None, dist=chi2):
    """Top-1 accuracy of nearest-neighbour (chi-square) label identification; a sample is never matched to itself
    or to samples of its own group (e.g. the same text or document) when groups are given."""
    D = dist(desc, desc)
    np.fill_diagonal(D, np.inf)
    if groups is not None:
        g = np.asarray(groups)
        D[g[:, None] == g[None]] = np.inf
    labels = np.asarray(labels)
    return float((labels[D.argmin(1)] == labels).mean())


def separation(desc, own, others, dist=chi2):
    """Margin of a descriptor: distance to the nearest other-type centroid minus distance to its own."""
    return float(dist(desc, others).min() - dist(desc, own[None])[0, 0])
