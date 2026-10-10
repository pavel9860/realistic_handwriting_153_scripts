"""Smooth random fields and 1D noise."""

import numpy as np


class Field:
    """Smooth random 2D displacement: sum of n plane waves of wavelength wl..3 wl."""

    def __init__(self, rng, amp, wl, n=4):
        a = rng.uniform(0, 2 * np.pi, n)
        self.k = np.c_[np.cos(a), np.sin(a)] * (2 * np.pi / rng.uniform(wl, 3 * wl, n))[:, None]
        self.ph = rng.uniform(0, 2 * np.pi, (n, 2))
        self.a = amp / np.sqrt(n)

    def __call__(self, p):
        t = p @ self.k.T
        return p + self.a * np.c_[np.sin(t + self.ph[:, 0]).sum(1), np.sin(t + self.ph[:, 1]).sum(1)]


def noise1d(rng, n, corr, amp):
    """n samples of zero-mean Gaussian noise smoothed over corr samples, standard deviation amp."""
    if n < 2:
        return np.zeros(n)
    sig = max(corr, 0.3)
    r = int(3 * sig) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    m = max(n, int(20 * sig) + 50)
    x = np.convolve(rng.normal(0, 1, m + 2 * r), k, "valid")
    return amp * x[:n] / (x.std() + 1e-9)
