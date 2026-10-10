import numpy as np

import hwsynth as H
from conftest import needs_fonts
from hwsynth.figures import FIGURES, shape
from hwsynth.math import chem, parse


def test_parse_scripts_fraction_root():
    n = parse(r"x^{2}+y_i=\frac{a}{b} -> \sqrt{x}")
    kinds = [x[0] for x in n]
    assert kinds == ["text", "sup", "text", "sub", "text", "frac", "space", "arrow", "space", "sqrt"]
    assert n[5][1] == [("text", "a")] and n[5][2] == [("text", "b")]


def test_symbols_and_benzene():
    assert "".join(x[1] for x in parse(r"\int\pi")) == "∫π"
    assert parse(r"\benzene") == [("benzene",)]


def test_chem_subscripts_charges_arrows():
    assert chem("2H2O") == [("text", "2H"), ("sub", [("text", "2")]), ("text", "O")]
    n = chem("SO4^{2-} + Na+ -> X")
    assert ("sup", [("text", "2-")]) in n and ("text", "+") in n and ("arrow", "right") in n
    i = n.index(("text", "Na"))
    assert n[i + 1] == ("sup", [("text", "+")])


def test_shapes_are_finite_paths():
    rng = np.random.default_rng(0)
    for k in FIGURES:
        paths = shape(k, 20.0, rng)
        assert paths and all(np.isfinite(p).all() and len(p) >= 2 for p in paths)


@needs_fonts
def test_inline_math_layout():
    w = H.sample_writer("school_cursive", "formal", seed=2, bank=False)
    lay = w.lines(r"area $\frac{\pi r^{2}}{2}$ and $\ce{H2O}$", 120)
    assert [x.text for x in lay.words] == ["area", r"\frac{\pi r^{2}}{2}", "and", r"\ce{H2O}"]
    assert all(x.box for x in lay.words)


@needs_fonts
def test_figure_sample():
    s = H.figure("circle", H.sample_writer(seed=1, bank=False), size=20, dpi=100)
    assert s.image.ndim == 3 and s.ink.max() > 0.3
