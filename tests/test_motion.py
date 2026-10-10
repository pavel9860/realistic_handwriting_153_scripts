import numpy as np

from hwsynth.motion import lognormal, pivot, synthesise


def corner():
    return np.vstack([np.linspace((0, 0), (1, 0), 50), np.linspace((1, 0), (1, 1), 50)])


def test_lognormal_integrates_to_one():
    t = np.linspace(0, 3, 6000)
    assert abs(lognormal(t, 0.1, -1.6, 0.3).sum() * (t[1] - t[0]) - 1) < 1e-3


def test_ends_kept_and_speed_normalised():
    q, v = synthesise(corner(), np.random.default_rng(0), 0.5, -1.8, 0.3, 0.01, 0.02)
    assert np.allclose(q[0], (0, 0), atol=1e-6) and np.allclose(q[-1], (1, 1), atol=1e-6)
    assert abs(v.max() - 1) < 1e-9 and v.min() >= 0


def test_fluent_cuts_corners_careful_reaches_them():
    rng = np.random.default_rng(1)
    gap = lambda ov: np.hypot(*(synthesise(corner(), rng, ov, -1.8, 0.3, 0.01, 0.01)[0] - (1, 0)).T).min()
    assert gap(0.2) > 3 * gap(0.9)


def test_speed_dips_at_corner():
    q, v = synthesise(corner(), np.random.default_rng(2), 0.9, -1.8, 0.3, 0.01, 0.01)
    i = np.hypot(*(q - (1, 0)).T).argmin()
    assert v[i] < 0.5 * v.max()


def test_pivot_resets_per_hand_position():
    x = np.array([0.0, 10.0, 29.9, 30.0, 40.0])
    y = pivot(x, [-25.0, 30.0], 1000.0, -1.0)
    assert y[3] == 0 and y[2] < y[1] < 0 and abs(y[4] + 0.05) < 1e-9


def test_min_jerk_trades_shape_for_energy():
    from hwsynth.motion import energy, min_jerk

    t = np.linspace(0, 4 * np.pi, 400)
    p = np.c_[t, np.sign(np.sin(t))]  # square wave: high jerk at corners
    low, high = min_jerk(p, 1e-6), min_jerk(p, 1.0)
    assert energy(low, 1) < energy(high, 1) < energy(p, 1) + 1e-9
    assert np.abs(high - p).max() < np.abs(low - p).max()
    assert np.allclose(low[[0, -1]], p[[0, -1]], atol=1e-3)


def test_effort_orders_styles():
    import hwsynth as H
    from conftest import fonts_available
    from hwsynth.motion import energy

    if not fonts_available():
        return

    def style_energy(st):
        out = []
        for i, per in enumerate(("school_cursive", "print_script", "compact_print")):
            p, g = H.individual(per, i, bank=False)
            w = H.styled(p, g, st, i)
            out += [
                energy(s.plan, w.p["size"])
                for s in w.lines("minimum quick brown", 200).strokes
                if s.plan is not None and len(s.plan) > 8
            ]
        return np.median(out)

    e = [style_energy(s) for s in ("medical", "casual", "formal", "calligraphic")]
    assert e == sorted(e)
