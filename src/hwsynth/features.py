"""Classical stroke features of a text line (from pages_generation tools/hwprint.py): stroke width and its
variation, component and gap statistics, slant spread, orientation entropy, junction and end densities, ink
variation and pitch regularity."""

import cv2
import numpy as np

NAMES = [
    "sw_cv",
    "sw_mean",
    "cc_h_cv",
    "base_res",
    "dens",
    "slant_sd",
    "cc_per_w",
    "cc_wh",
    "orient_ent",
    "ccw_cv",
    "gap_cv",
    "vert_frac",
    "solid_cv",
    "ends_per_len",
    "junc_per_len",
    "proj_reg",
    "vcont",
    "axis_frac",
    "straight",
    "ink_cv",
    "pitch_acf",
    "edge_sharp",
]


def feats(c):
    """22 stroke and layout statistics of a grayscale text-line crop (None when it has too little ink)."""
    from skimage.morphology import skeletonize

    h = 48
    c = cv2.resize(c, (max(int(c.shape[1] * h / c.shape[0]), h), h), interpolation=cv2.INTER_AREA)
    c = cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX)
    b = cv2.adaptiveThreshold(c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12) > 0
    if b.sum() < 30:
        return None
    dt = cv2.distanceTransform(b.astype(np.uint8), cv2.DIST_L2, 3)
    sk = skeletonize(b)
    w = 2 * dt[sk]
    if len(w) < 10:
        return None
    n, _, st, ce = cv2.connectedComponentsWithStats(b.astype(np.uint8), 8)
    k = [i for i in range(1, n) if st[i, 4] >= 8]
    if len(k) < 2:
        return None
    H = st[k, 3].astype(float)
    W = st[k, 2].astype(float)
    bot = st[k, 1] + st[k, 3]
    cx = ce[k, 0]
    res = np.std(bot - np.polyval(np.polyfit(cx, bot, 1), cx)) / np.median(H)
    gx = cv2.Sobel(c.astype(np.float32), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(c.astype(np.float32), cv2.CV_32F, 0, 1)
    m = np.hypot(gx, gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    hist = np.histogram(ang, 12, (0, np.pi), weights=m)[0]
    hist = hist / (hist.sum() + 1e-9)
    ent = -(hist * np.log(hist + 1e-9)).sum()
    a2 = ang[m > m.max() * 0.3]
    vf = ((np.abs(a2) < 0.35) | (np.abs(a2 - np.pi) < 0.35)).mean() if len(a2) else 0
    slant = np.std(a2[(a2 > 0.2) & (a2 < np.pi - 0.2)]) if len(a2) > 5 else 0
    sol = np.array([st[i, 4] / (st[i, 2] * st[i, 3]) for i in k])
    sk8 = sk.astype(np.uint8)
    nb = cv2.filter2D(sk8, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - sk8
    L = sk.sum()
    pr = b.sum(0).astype(float)
    pr = pr[pr > 0]
    ys, _ = np.nonzero(sk)
    rowc = np.bincount(ys, minlength=h).astype(float)
    axm = m[np.abs(np.sin(2 * ang)) < 0.35].sum() / (m.sum() + 1e-9)
    f = sk.astype(np.float32)
    yy, xx = np.mgrid[:h, : c.shape[1]].astype(np.float32)
    box = lambda a: cv2.boxFilter(a, -1, (9, 9), normalize=False)
    n9 = box(f) + 1e-6
    mx, my = box(f * xx) / n9, box(f * yy) / n9
    cxx, cyy, cxy = box(f * xx * xx) / n9 - mx * mx, box(f * yy * yy) / n9 - my * my, box(f * xx * yy) / n9 - mx * my
    dd = np.sqrt(np.maximum((cxx - cyy) ** 2 / 4 + cxy**2, 0))
    l1, l2 = (cxx + cyy) / 2 + dd, (cxx + cyy) / 2 - dd
    on = sk & (n9 > 4)
    stt = np.mean((1 - l2 / (l1 + 1e-6))[on]) if on.any() else 0
    inner = cv2.erode(b.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    icv = c[inner].std() / (c[inner].mean() + 1e-9) if inner.sum() > 10 else 0
    cp = b.sum(0).astype(float)
    cp -= cp.mean()
    ac = np.correlate(cp, cp, "full")[len(cp) - 1 :]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(h * 0.4), min(int(h * 1.6), len(ac) - 1)
    pk = ac[lo:hi].max() if hi > lo else 0
    g = np.diff(np.sort(cx))
    return [
        w.std() / w.mean(),
        w.mean() / h,
        H.std() / H.mean(),
        res,
        b.mean(),
        slant,
        len(k) / c.shape[1] * h,
        np.median(W / H),
        ent,
        W.std() / W.mean(),
        g.std() / (g.mean() + 1e-9),
        vf,
        sol.std() / (sol.mean() + 1e-9),
        ((nb == 1) & sk).sum() / L,
        ((nb >= 3) & sk).sum() / L,
        np.std(np.diff(pr)) / (pr.mean() + 1e-9),
        np.std(rowc) / (rowc.mean() + 1e-9),
        axm,
        stt,
        icv,
        pk,
        m[b].mean() / (w.mean() + 1e-9),
    ]
