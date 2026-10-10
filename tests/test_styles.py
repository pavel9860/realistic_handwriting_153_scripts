import numpy as np

from hwsynth.config import GENE_INDEX, GENES, IDENTITY, PERSON_SPREAD, PERSONS, STYLES
from hwsynth.styles import apply_style, person_genes
from hwsynth.writer import encode


def test_person_within_spread():
    rng = np.random.default_rng(0)
    for name, p in PERSONS.items():
        for _ in range(5):
            g = person_genes(name, rng)
            for n, v in p.genes.items():
                assert abs(g[GENE_INDEX[n]] - encode({n: v})[GENE_INDEX[n]]) <= PERSON_SPREAD + 1e-9


def test_twenty_persons_ten_styles():
    assert len(PERSONS) == 20 and len(STYLES) == 10


def test_style_keeps_identity_and_bounds():
    g = np.random.default_rng(1).random(len(GENES))
    for name, st in STYLES.items():
        h = apply_style(g, name)
        assert ((h >= 0) & (h <= 1)).all()
        for n in IDENTITY:
            if n not in st.shift and n not in st.remap:
                assert h[GENE_INDEX[n]] == g[GENE_INDEX[n]]
        for n, (lo, hi) in st.remap.items():
            G = GENES[GENE_INDEX[n]]
            v = G.lo + (G.hi - G.lo) * h[GENE_INDEX[n]]
            assert lo - 1e-9 <= v <= hi + 1e-9


def test_style_keeps_order_of_people():
    a, b = np.full(len(GENES), 0.2), np.full(len(GENES), 0.8)
    for name in STYLES:
        assert (apply_style(a, name) <= apply_style(b, name) + 1e-9).all()


def test_rare_events_are_rare():
    rng = np.random.default_rng(2)
    g = np.array([person_genes(None, rng) for _ in range(500)])
    assert (g[:, GENE_INDEX["upper"]] < 0.5).all()
    assert np.median(g[:, GENE_INDEX["cross"]]) < 0.2
