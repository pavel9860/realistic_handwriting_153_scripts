"""Hand movement: strokes re-synthesised as sums of lognormal sub-movements (Kinematic Theory, Plamondon 1995;
Sigma-Lognormal model), and the wrist pivot arcs of a line.

A stroke is reduced to virtual targets (the turning points of its path). Each segment between two targets is one
sub-movement whose speed follows a lognormal impulse response; sub-movements start at roughly equal intervals
(isochrony) and overlap in time. The more they overlap (fast, fluent writing), the more corners are cut; with little
overlap (slow, careful writing, discrete-stroke scripts) the pen reaches every target. The speed of the synthesised
movement drives the pen (width and ink)."""

import cv2
import numpy as np
from scipy.linalg import solveh_banded

from .config import MotionCfg


def lognormal(t, t0, mu, sigma):
    x = np.maximum(t - t0, 1e-9)
    out = np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi) * x)
    return np.where(t > t0, out, 0.0)


def targets(p, tol):
    """Virtual targets of a path: Douglas-Peucker turning points (first and last point kept)."""
    if len(p) < 3:
        return p
    q = cv2.approxPolyDP(p.astype(np.float32)[:, None], tol, False)[:, 0].astype(float)
    return q if len(q) >= 2 else p[[0, -1]]


def synthesise(p, rng, overlap, mu, sigma, tol, step, cfg=MotionCfg()):
    """(path, speed) of a stroke re-synthesised from lognormal sub-movements; path sampled every step (same units
    as p), speed normalised to its maximum. overlap in (0, 1]: start interval of sub-movements over their duration."""
    tg = targets(p, tol)
    D = np.diff(tg, axis=0)
    k = len(D)
    if k == 0:
        return p, np.ones(len(p))
    peak = np.exp(mu - sigma**2)  # time of peak speed
    dur = np.exp(mu + 3 * sigma)
    mus = mu + rng.normal(0, cfg.mu_sd, k)
    sigs = np.clip(sigma + rng.normal(0, cfg.sigma_sd, k), 0.12, 0.6)
    t0 = np.r_[
        0, np.cumsum(np.clip(2 * overlap * peak * (1 + rng.normal(0, cfg.interval_sd, k - 1)), 0.1 * peak, None))
    ]
    t = np.linspace(0, t0[-1] + 1.5 * dur, max(cfg.samples, cfg.per_component * k))
    dt = t[1] - t[0]
    vel = np.zeros((len(t), 2))
    for j in range(k):  # each impulse on its own window
        a = int(t0[j] / dt)
        b = min(int((t0[j] + np.exp(mus[j] + 4 * sigs[j])) / dt) + 2, len(t))
        lam = lognormal(t[a:b], t0[j], mus[j], sigs[j])
        vel[a:b] += np.outer(lam / (lam.sum() * dt + 1e-12), D[j])  # integrates to the segment D[j]
    pos = tg[0] + np.cumsum(vel, 0) * dt
    pos += (tg[-1] - pos[-1]) * np.linspace(0, 1, len(pos))[:, None]  # numerical drift to the end target
    speed = np.hypot(*vel.T)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pos, axis=0).T))]
    if d[-1] < step:
        return p, np.ones(len(p))
    s = np.linspace(0, d[-1], int(d[-1] / step) + 1)
    keep = np.r_[True, np.diff(d) > 0]
    out = np.c_[np.interp(s, d[keep], pos[keep, 0]), np.interp(s, d[keep], pos[keep, 1])]
    v = np.interp(s, d[keep], speed[keep])
    return out, v / (v.max() + 1e-12)


def pivot(x, segments, radius, side):
    """Vertical offset (mm) of wrist pivot arcs: within each hand position (segment starts in segments) the pen
    follows an arc of the given radius; the hand is repositioned at every segment start."""
    starts = np.asarray(segments)
    x0 = starts[np.clip(np.searchsorted(starts, x, side="right") - 1, 0, len(starts) - 1)]
    return side * (x - x0) ** 2 / (2 * radius)


def effort_lambda(effort, cfg=MotionCfg()):
    """Weight of shape fidelity against jerk energy for an effort in [0, 1] (log-linear between the config ends)."""
    lo, hi = cfg.log_lambda
    return 10 ** (lo + (hi - lo) * effort)


def min_jerk(p, lam):
    """Path closest to minimum jerk energy for its shape: argmin_x lam * |x - p|^2 + |D3 x|^2 over equally spaced
    samples (D3: third difference), the end points held. Small lam: smooth economical movement that lets the shape
    go; large lam: the shape is kept at the cost of energy."""
    n = len(p)
    if n < 8 or lam >= 1e3:
        return p
    c = np.array([-1.0, 3.0, -3.0, 1.0])
    diag = [np.zeros(n) for _ in range(4)]  # D3^T D3 bands 0..3
    for r in range(n - 3):
        for a in range(4):
            for b in range(a, 4):
                diag[b - a][r + a] += c[a] * c[b]
    w = np.full(n, lam)
    w[[0, 1, -2, -1]] = 1e6  # end points held
    ab = np.zeros((4, n))
    ab[3] = diag[0] + w
    for d in (1, 2, 3):
        ab[3 - d, d:] = diag[d][: n - d]
    return solveh_banded(ab, w[:, None] * p)


def energy(p, xh):
    """Jerk energy of a path per unit length (squared third differences of x-height-scaled positions)."""
    q = np.asarray(p, float) / xh
    if len(q) < 4:
        return 0.0
    L = np.hypot(*np.diff(q, axis=0).T).sum()
    return float((np.diff(q, 3, axis=0) ** 2).sum() / max(L, 1e-9))


def turning_entropy(p, bins=16):
    """Shannon entropy (bits) of the turning angles along a path: how much shape detail the movement carries."""
    d = np.diff(np.asarray(p, float), axis=0)
    a = np.arctan2(d[:, 1], d[:, 0])
    t = np.angle(np.exp(1j * np.diff(a)))
    if len(t) < 2:
        return 0.0
    h = np.histogram(t, bins, (-np.pi / 2, np.pi / 2))[0].astype(float)
    h = h[h > 0] / h.sum()
    return float(-(h * np.log2(h)).sum())
