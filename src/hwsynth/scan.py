"""Scanned-document look: ageing paper, fading ink, bleed-through and scanner artefacts.

Applied after the ink is composed on paper. Each effect is drawn at random with the probabilities of ScanCfg, so a
batch covers fresh notes on a phone scan as well as old letters from an archive scanner."""

import cv2
import numpy as np

from .config import ScanCfg


def fade(ink, rng, age, cfg=ScanCfg()):
    """Ink density lowered with age and unevenly along the page."""
    if age <= 0:
        return ink
    h, w = ink.shape
    patch = cv2.resize(
        rng.random((max(h // 80, 2), max(w // 80, 2))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC
    )
    return ink * np.clip(1 - cfg.fade * age * (0.6 + 0.8 * patch), 0.2, 1)


def age(rng, cfg=ScanCfg()):
    return float(rng.beta(*cfg.age_beta))


def degrade(img, ink, rng, dpi, age, cfg=ScanCfg()):
    """img: composed RGB float (H, W, 3) in [0, 1]; ink: its ink density; age in [0, 1]. Returns the scanned RGB."""
    img = img.astype(np.float32)
    h, w = ink.shape
    k = dpi / 25.4
    img = img * (1 - age * np.array(cfg.yellow))[None, None]  # yellowing
    for _ in range(rng.poisson(cfg.foxing * age)):  # foxing spots
        r = rng.uniform(0.5, 4) * k
        y, x = rng.uniform(0, h), rng.uniform(0, w)
        yy, xx = np.ogrid[:h, :w]
        spot = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * r * r)).astype(np.float32)
        img = img * (1 - rng.uniform(0.03, 0.15) * spot[..., None] * np.array((0.3, 0.6, 1.0)))
    if rng.random() < cfg.p_bleed:  # text of the other side showing through
        ghost = cv2.GaussianBlur(np.roll(ink[:, ::-1], int(rng.uniform(-5, 5) * k), 0), (0, 0), 0.6 * k / 4)
        img = img * (1 - rng.uniform(0.04, 0.12) * ghost[..., None])
    g = rng.uniform(0, cfg.shading)  # uneven light
    ys, xs = np.mgrid[:h, :w].astype(np.float32)
    a = rng.uniform(0, 2 * np.pi)
    plane = np.cos(a) * xs / max(w, 1) + np.sin(a) * ys / max(h, 1)
    vign = (xs / max(w, 1) - 0.5) ** 2 + (ys / max(h, 1) - 0.5) ** 2
    img = img * (1 - g * (0.6 * (plane - plane.min()) + 0.8 * vign))[..., None]
    if rng.random() < cfg.p_gray:
        img = np.repeat(img.mean(2, keepdims=True), 3, 2)
    gamma = rng.uniform(*cfg.gamma)
    img = np.clip(img, 0, 1) ** gamma
    img = img + rng.normal(0, rng.uniform(*cfg.noise), img.shape).astype(np.float32)
    for _ in range(rng.poisson(cfg.dust)):  # dust and specks
        cv2.circle(
            img,
            (int(rng.uniform(0, w)), int(rng.uniform(0, h))),
            int(rng.uniform(0, 2)),
            (rng.uniform(0.1, 0.5),) * 3,
            -1,
        )
    s = rng.uniform(0, cfg.blur)
    if s > 0.2:
        img = cv2.GaussianBlur(img, (0, 0), s)
    img = np.clip(img, 0, 1)
    if rng.random() < cfg.p_bitonal:
        g8 = (img.mean(2) * 255).astype(np.uint8)
        t, _ = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # adaptive: faint ink survives
        t = np.clip(t + rng.normal(0, 8), 30, 230)
        img = np.repeat((g8 > t).astype(np.float32)[..., None], 3, 2)
    if rng.random() < cfg.p_jpeg:
        q = int(rng.uniform(*cfg.jpeg))
        ok, buf = cv2.imencode(".jpg", (img * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32) / 255
    return img
