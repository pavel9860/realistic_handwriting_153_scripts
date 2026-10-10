import numpy as np

from hwsynth.config import TOOLS
from hwsynth.pen import Pen, render
from hwsynth.writer import Stroke, decode


def genes(**kw):
    from hwsynth.config import GENES

    p = decode(np.full(len(GENES), 0.5))
    p.update(kw)
    return p


def pen(tool, **kw):
    t = TOOLS[tool]
    d = dict(
        tool=tool,
        width=np.mean(t.width),
        edge=np.mean(t.edge),
        angle=np.radians(45),
        density=np.mean(t.density),
        ink=(0.1, 0.1, 0.1),
        ink_name="black",
    )
    d.update(kw)
    return Pen(**d)


def ink(tool, stroke, p=None, **kw):
    rng = np.random.default_rng(0)
    C, _ = render([Stroke(stroke)], pen(tool, **kw), p or genes(size=4.0), 300, rng)
    return C


def test_density_range_and_nonempty():
    for t in TOOLS:
        C = ink(t, np.c_[np.linspace(0, 20, 50), np.zeros(50)])
        assert C.min() >= 0 and C.max() <= 1 and C.max() > 0.2, t


def test_broad_nib_contrast():
    h = np.c_[np.linspace(0, 20, 80), np.zeros(80)]
    v = np.c_[np.zeros(80), np.linspace(0, 20, 80)]
    width = lambda C, axis: (C > 0.3).sum(axis).max()
    a = ink("broad_nib", h, angle=0.0)
    b = ink("broad_nib", v, angle=0.0)
    assert width(b, 1) > 2 * width(a, 0)


def test_pencil_push_darkens():
    s = np.c_[np.linspace(0, 20, 80), np.zeros(80)]
    light = ink("pencil_hb", s, genes(size=4.0, press=0.0, press_sd=0.0)).sum()
    strong = ink("pencil_hb", s, genes(size=4.0, press=1.0, press_sd=0.0)).sum()
    assert strong > 1.2 * light


def test_brush_push_widens():
    s = np.c_[np.linspace(0, 20, 80), np.zeros(80)]
    light = (ink("brush_pen", s, genes(size=4.0, press=0.0, press_sd=0.0)) > 0.3).sum()
    strong = (ink("brush_pen", s, genes(size=4.0, press=1.0, press_sd=0.0)) > 0.3).sum()
    assert strong > 1.3 * light


def test_tool_sampling_prior():
    rng = np.random.default_rng(0)
    assert {Pen.sample(rng, {"gel": 1}).tool for _ in range(10)} == {"gel"}
    assert Pen.sample(rng, {"pencil_hb": 1}).ink_name == "graphite"
