"""Persons (handwriting types: identity) and styles (situations applied on top of a person)."""

import json

import numpy as np

from .config import (
    HABITS,
    CENTRED,
    GENE_INDEX,
    GENES,
    IDENTITY,
    NOISE_SKEW,
    PERSON_SPREAD,
    PERSONS,
    STYLES,
    QDCfg,
    Paths,
)
from .writer import Writer, encode


CHILDISH_TYPES = ("child",)  # only these may use fonts marked childish in the manifest
CHILDISH_STYLES = ("uneducated",)


def type_centre(person):
    """{gene index: centre in [0, 1]} of a type: its listed genes, and for every other identity gene a fixed centre
    drawn from the type name, so each type has one recognisable hand (font included)."""
    import zlib

    r = np.random.default_rng(zlib.crc32(person.encode()))
    out = {GENE_INDEX[n]: float(r.random()) for n in IDENTITY if n != "upper"}
    out.update({GENE_INDEX[n]: float(encode({n: v})[GENE_INDEX[n]]) for n, v in PERSONS[person].genes.items()})
    return out


def person_genes(person, rng):
    """Genes of a new individual of a handwriting type: type centres +- PERSON_SPREAD, other genes uniform (rare
    events skewed towards zero, capitals-only writing only where the type sets it)."""
    g = rng.random(len(GENES))
    for n, k in NOISE_SKEW.items():
        g[GENE_INDEX[n]] **= k
    for n in CENTRED:
        g[GENE_INDEX[n]] = 0.5 + 4 * (g[GENE_INDEX[n]] - 0.5) ** 3
    g[GENE_INDEX["upper"]] *= 0.4
    g[GENE_INDEX["caps"]] *= 0.15
    if person is None:
        return g
    for i, c in type_centre(person).items():
        g[i] = np.clip(c + rng.uniform(-PERSON_SPREAD, PERSON_SPREAD), 0, 1)
    return g


def apply_style(genes, style):
    """Style on top of a person: situational genes remapped into the style ranges (the person keeps its relative
    position), identity genes shifted; returns a new gene vector."""
    st = STYLES[style]
    g = np.array(genes, float)
    for n, (lo, hi) in st.remap.items():
        i = GENE_INDEX[n]
        G = GENES[i]
        a, b = (lo - G.lo) / (G.hi - G.lo), (hi - G.lo) / (G.hi - G.lo)
        g[i] = a + (b - a) * g[i]
    for n, d in st.shift.items():
        i = GENE_INDEX[n]
        g[i] = np.clip(g[i] + d / (GENES[i].hi - GENES[i].lo), 0, 1)
    return np.clip(g, 0, 1)


def font_class(person, style):
    cls = STYLES[style].fonts or (PERSONS[person].fonts if person else "any")
    return cls + ("+childish" if person in CHILDISH_TYPES or style in CHILDISH_STYLES else "")


def load_bank(person, paths=Paths()):
    """Writer bank of a type: the local one (hwsynth evolve) if present, else the bank shipped with the package."""
    from pathlib import Path

    for f in (paths.bank(person or "any"), Path(__file__).parent / "data" / "banks" / f"writers_{person}.json"):
        if f.exists():
            return [e for e in json.loads(f.read_text()) if len(e["genes"]) == len(GENES)]  # banks of this gene set
    return []


def personas(paths=Paths()):
    """Personas fitted to real writers: the local file (hwsynth fit) if present, else the one shipped."""
    from pathlib import Path

    for f in (paths.personas, Path(__file__).parent / "data" / "banks" / "personas_iam.json"):
        if f.exists():
            return [e for e in json.loads(f.read_text()) if len(e["genes"]) == len(GENES)]
    return []


def individual(person=None, seed=0, bank=True, paths=Paths(), cfg=QDCfg()):
    """(person, genes) of one individual: from the evolved bank of the type when there is one (with small gene
    noise), else freshly drawn. person None: a random type."""
    rng = np.random.default_rng(seed)
    if person is not None and person.startswith("iam"):
        ps = personas(paths)
        if not ps:
            raise ValueError("no fitted personas (run hwsynth fit)")
        e = next((x for x in ps if x["page"] == person[4:]), None) if ":" in person else ps[int(rng.integers(len(ps)))]
        if e is None:
            raise ValueError(f"unknown persona {person}")
        return f"iam:{e['page']}", np.array(e["genes"])
    if person is None:
        person = list(PERSONS)[int(rng.integers(len(PERSONS)))]
    entries = load_bank(person, paths) if bank else []
    if entries:
        g = np.array(entries[int(rng.integers(len(entries)))]["genes"])
        return person, np.clip(g + rng.normal(0, cfg.bank_sigma, len(g)), 0, 1)
    return person, person_genes(person, rng)


def _persona_type(person):
    """Handwriting type of a person name: the type itself, or the nearest type of a fitted persona."""
    if person and person.startswith("iam:"):
        e = next((x for x in personas() if x["page"] == person[4:]), None)
        return e["type"] if e else None
    return person


def styled(person, genes, style="casual", seed=0):
    """Writer of an individual in a style; seed drives everything situational (pen, jitter, layout)."""
    st = STYLES[style]
    kind = _persona_type(person)
    habits = dict(HABITS, **(PERSONS[kind].habits if kind in PERSONS else {}))
    return Writer(
        apply_style(genes, style),
        seed,
        font_class(kind, style),
        st.pens,
        name=f"{person}/{style}",
        paper=st.paper,
        habits=habits,
    )


def sample_writer(person=None, style="casual", seed=0, bank=True, paths=Paths(), cfg=QDCfg()):
    """A new writer: a new individual of the type, written in the style."""
    person, g = individual(person, seed, bank, paths, cfg)
    return styled(person, g, style, seed + 1)
