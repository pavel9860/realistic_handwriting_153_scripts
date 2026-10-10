import numpy as np

from hwsynth.noise import Field, noise1d


def test_noise_scale_without_blowup():
    rng = np.random.default_rng(0)
    for n in (2, 3, 5, 50, 2000):
        for corr in (0.5, 5, 40):
            x = noise1d(rng, n, corr, 1.0)
            assert len(x) == n and np.abs(x).max() < 6
    assert abs(noise1d(rng, 5000, 3, 2.0).std() - 2.0) < 0.3


def test_field_is_smooth_and_bounded():
    f = Field(np.random.default_rng(1), 0.5, 10)
    p = np.random.default_rng(2).uniform(0, 100, (200, 2))
    assert np.abs(f(p) - p).max() <= 0.5 * 4 / 2 + 1e-9
    q = p + 0.01
    assert np.abs((f(q) - q) - (f(p) - p)).max() < 0.01
