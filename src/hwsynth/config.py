"""All tunable numbers of hwsynth in one place: paths, glyph tracing, writer genes, pens, rendering, styles, evolution."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name, default):
    return Path(os.environ.get(name, default)).expanduser()


@dataclass(frozen=True)
class Paths:
    fonts: Path = _env_path("HWSYNTH_FONTS", "~/.cache/hwsynth/fonts")
    system_fonts: tuple = ("/usr/share/fonts",)  # extra fallback fonts (searched recursively)
    cache: Path = _env_path("HWSYNTH_CACHE", "~/.cache/hwsynth")
    data: Path = _env_path("HWSYNTH_DATA", "~/.cache/hwsynth/data")  # real-line references, writer banks, metrics

    def bank(self, style):
        return self.data / f"writers_{style}.json"

    @property
    def refs(self):
        """Primary real references (IAM handwriting forms)."""
        return self.data / "real_lines_iam.npz"

    @property
    def personas(self):
        """Personas fitted to real writers (hwsynth fit)."""
        return self.data / "personas_iam.json"

    @property
    def refs_web(self):
        """Secondary real references (handwritten web documents, all scripts; may be contaminated)."""
        return self.data / "real_lines.npz"


@dataclass(frozen=True)
class GlyphCfg:
    px: int = 160  # raster size of a glyph em for centreline tracing
    threshold: int = 110  # ink threshold of the glyph raster (0-255)
    spur: float = 0.05  # skeleton branches shorter than this (em) that end freely are pruned
    pre_step: float = 1.5  # resampling step (px) of traced paths before smoothing
    pre_smooth: float = 2.0  # Gaussian smoothing (samples) of traced paths
    step: float = 0.02  # final resampling step (em)
    join_turn: float = -0.5  # paths meeting at a junction continue one another when cos(angle) < this
    coverage: float = 0.9  # a font serves a script when it covers this share of the script's core letters


@dataclass(frozen=True)
class Gene:
    name: str
    lo: float
    hi: float
    doc: str


GENES = (
    Gene("size", 1.8, 3.4, "x-height (mm)"),
    Gene("xs", 0.75, 1.3, "lateral expansion: horizontal scale of letters"),
    Gene("asc", 0.75, 1.5, "ascender length (scale of the part above the x-height)"),
    Gene("desc", 0.75, 1.5, "descender length (scale of the part below the baseline)"),
    Gene("cap", 0.9, 1.5, "capital and first-letter size"),
    Gene("slant", -15, 32, "slant (deg, positive leans right)"),
    Gene("slant_sd", 0, 4, "slant jitter per letter (deg)"),
    Gene("size_sd", 0, 0.10, "size jitter per letter (share)"),
    Gene("rot_sd", 0, 2.5, "rotation jitter per letter (deg)"),
    Gene("base_sd", 0, 0.09, "baseline jitter per letter (x-heights)"),
    Gene("ls", -0.1, 0.3, "letter spacing (x-heights)"),
    Gene("ls_sd", 0, 0.08, "letter spacing jitter (x-heights)"),
    Gene("ws", 0.7, 2.2, "word spacing (x-heights)"),
    Gene("pitch", 2.5, 4.2, "line pitch (x-heights) on blank paper"),
    Gene("slope", -2.5, 2.5, "line slope on blank paper (deg)"),
    Gene("bow", -0.5, 0.5, "line bow (x-heights)"),
    Gene("wander", 0, 1, "slow baseline wander along the line"),
    Gene("elastic", 0, 1, "smooth shape drift over the page"),
    Gene("wobble", 0, 1, "low-frequency wobble along strokes"),
    Gene("tremor", 0, 1, "high-frequency tremor"),
    Gene("round", 0, 1, "corner rounding (garland, arcade movement)"),
    Gene("angular", 0, 1, "corner sharpening (angular movement)"),
    Gene("speed", 0, 1, "writing speed (rounding, ink skips, thin fast strokes)"),
    Gene("effort", 0, 1, "movement effort: energy spent to keep the letter shapes (minimum jerk trade-off)"),
    Gene("connect", 0, 1, "probability of joining letters"),
    Gene("trail", 0, 1, "letters shrink and flatten towards word ends (thread movement)"),
    Gene("flat", 0, 1, "vertical flattening of the x-height zone"),
    Gene("hook", 0, 1, "probability of entry hooks (commencements)"),
    Gene("flick", 0, 1, "probability of exit flicks (terminations)"),
    Gene("dot_dx", -0.1, 0.4, "dots and accents displaced in writing direction (x-heights)"),
    Gene("dot_len", 0, 1, "dots drawn as dashes"),
    Gene("ring", 0, 1, "dots drawn as small circles (above 0.5)"),
    Gene("press", 0, 1, "mean pen pressure: light to strong push"),
    Gene("press_sd", 0, 1, "pressure variation"),
    Gene("down", 0, 1, "extra pressure on downstrokes"),
    Gene("font", 0, 1, "allograph set: font choice"),
    Gene("allo", 0, 0.4, "share of letters written in a second allograph"),
    Gene("letters", 0, 1, "how far the writer's own letter shapes depart from the model letters"),
    Gene("hand", 0, 1, "writing hand: left above LEFT_HANDED"),
    Gene("cross", 0, 0.015, "probability of crossing out a misspelt word"),
    Gene("retrace", 0, 0.03, "probability of retracing a stroke"),
    Gene("caps", 0, 0.08, "probability of a capital letter inside a word"),
    Gene("upper", 0, 1, "writes everything in capitals when above 0.5"),
    Gene("drift", -0.25, 0.25, "size drift over the text"),
    Gene("margin_sd", 0, 1, "left margin wander (x-heights)"),
)
# Individual (person) characteristics: kept by a writer in every style. All other genes are situational.
IDENTITY = (
    "size",
    "xs",
    "asc",
    "desc",
    "cap",
    "slant",
    "ls",
    "ws",
    "pitch",
    "round",
    "angular",
    "connect",
    "hook",
    "flick",
    "dot_dx",
    "dot_len",
    "press",
    "font",
    "allo",
    "hand",
    "upper",
    "flat",
)
GENE_INDEX = {g.name: i for i, g in enumerate(GENES)}


@dataclass(frozen=True)
class Tool:
    """A writing tool. nib: round (width from the tip), broad (edge of width nib held at an angle: thick-thin by
    direction), brush (width grows with pressure, tapered ends). ink: solid (graphite: density follows force, paper
    grain), viscous (ballpoint paste: density falls with speed, skips), fluid (liquid ink: density nearly constant,
    width and spread grow with force and dwell)."""

    name: str
    nib: str
    ink: str
    width: tuple  # tip width range (mm)
    flex: float = 0.0  # width gain per unit pressure (share of width)
    edge: tuple = (0.0, 0.0)  # broad nib edge width range (mm)
    angle: tuple = (30.0, 45.0)  # broad nib angle to the baseline (deg)
    density: tuple = (0.85, 0.95)  # ink density range at normal pressure
    overlap: float = 0.25  # extra darkening where strokes cross
    blur: float = 0.02  # edge softness (mm)
    bleed: float = 0.0  # ink spread halo (share of density)
    grain: float = 0.0  # paper grain showing through (solid inks)
    skip: float = 0.0  # ink skips on fast strokes (viscous inks)
    max_xh: float = 0.15  # width cap (share of the x-height)


TOOLS = {
    t.name: t
    for t in (
        Tool("ballpoint_fine", "round", "viscous", (0.22, 0.3), 0.1, density=(0.75, 0.9), skip=1.0),
        Tool("ballpoint", "round", "viscous", (0.28, 0.4), 0.12, density=(0.78, 0.92), skip=0.8),
        Tool("ballpoint_bold", "round", "viscous", (0.4, 0.55), 0.12, density=(0.82, 0.95), skip=0.5),
        Tool("gel", "round", "fluid", (0.3, 0.55), 0.25, density=(0.9, 1.0), overlap=0.2),
        Tool("rollerball", "round", "fluid", (0.35, 0.6), 0.3, density=(0.88, 0.98), overlap=0.35, bleed=0.15),
        Tool("fineliner", "round", "fluid", (0.18, 0.35), 0.05, density=(0.9, 1.0), overlap=0.3, bleed=0.1),
        Tool(
            "felt_marker",
            "round",
            "fluid",
            (0.6, 1.1),
            0.15,
            density=(0.8, 0.95),
            overlap=0.6,
            blur=0.05,
            bleed=0.25,
            max_xh=0.3,
        ),
        Tool(
            "chisel_marker",
            "broad",
            "fluid",
            (0.25, 0.4),
            0.05,
            edge=(1.0, 2.0),
            angle=(25, 50),
            density=(0.75, 0.9),
            overlap=0.6,
            blur=0.05,
            bleed=0.2,
            max_xh=0.45,
        ),
        Tool(
            "fountain_fine",
            "round",
            "fluid",
            (0.12, 0.2),
            0.6,
            density=(0.65, 0.85),
            overlap=0.75,
            blur=0.025,
            bleed=0.3,
        ),
        Tool(
            "fountain_medium",
            "broad",
            "fluid",
            (0.12, 0.2),
            0.4,
            edge=(0.15, 0.35),
            angle=(30, 55),
            density=(0.65, 0.85),
            overlap=0.75,
            blur=0.025,
            bleed=0.3,
        ),
        Tool(
            "italic_nib",
            "broad",
            "fluid",
            (0.06, 0.1),
            0.1,
            edge=(0.5, 0.9),
            angle=(35, 50),
            density=(0.75, 0.9),
            overlap=0.75,
            blur=0.025,
            bleed=0.25,
            max_xh=0.3,
        ),
        Tool(
            "broad_nib",
            "broad",
            "fluid",
            (0.06, 0.12),
            0.1,
            edge=(0.8, 1.4),
            angle=(25, 35),
            density=(0.8, 0.95),
            overlap=0.75,
            blur=0.025,
            bleed=0.25,
            max_xh=0.4,
        ),
        Tool(
            "brush_pen",
            "brush",
            "fluid",
            (0.15, 0.3),
            2.5,
            density=(0.85, 0.98),
            overlap=0.5,
            blur=0.03,
            bleed=0.15,
            max_xh=0.35,
        ),
        Tool("pencil_hb", "round", "solid", (0.3, 0.5), 0.15, density=(0.45, 0.6), overlap=0.45, blur=0.03, grain=0.8),
        Tool("pencil_2b", "round", "solid", (0.35, 0.6), 0.2, density=(0.6, 0.78), overlap=0.45, blur=0.035, grain=0.6),
        Tool("pencil_2h", "round", "solid", (0.25, 0.4), 0.1, density=(0.4, 0.55), overlap=0.4, blur=0.025, grain=1.0),
        Tool("mechanical_pencil", "round", "solid", (0.2, 0.3), 0.05, density=(0.45, 0.62), overlap=0.4, grain=0.8),
    )
}
TOOL_PRIOR = {
    "ballpoint_fine": 2,
    "ballpoint": 5,
    "ballpoint_bold": 1.5,
    "gel": 3,
    "rollerball": 1,
    "fineliner": 1.5,
    "felt_marker": 0.5,
    "chisel_marker": 0.2,
    "fountain_fine": 0.6,
    "fountain_medium": 0.6,
    "italic_nib": 0.1,
    "broad_nib": 0.05,
    "brush_pen": 0.2,
    "pencil_hb": 1,
    "pencil_2b": 0.5,
    "pencil_2h": 0.3,
    "mechanical_pencil": 0.7,
}


@dataclass(frozen=True)
class InkCfg:
    inks: dict = field(
        default_factory=lambda: {  # RGB transmittance of the ink and its prior weight
            "blue": ((0.16, 0.24, 0.62), 0.42),
            "dark_blue": ((0.10, 0.13, 0.38), 0.12),
            "black": ((0.09, 0.09, 0.11), 0.36),
            "violet": ((0.30, 0.22, 0.55), 0.04),
            "red": ((0.72, 0.13, 0.13), 0.03),
            "green": ((0.12, 0.42, 0.24), 0.03),
        }
    )
    graphite: tuple = (0.27, 0.27, 0.29)


@dataclass(frozen=True)
class MotionCfg:
    samples: int = 600  # time samples per stroke (at least)
    per_component: int = 60  # time samples per sub-movement
    mu: tuple = (-1.9, -1.4)  # lognormal log-time delay range: slow writers at the top (log s)
    sigma: tuple = (0.22, 0.45)  # lognormal response time spread (log s): wider for slow, tremulous hands
    overlap: tuple = (0.25, 0.85)  # start interval of sub-movements over their duration: fluent ... careful
    mu_sd: float = 0.08  # per sub-movement variation
    sigma_sd: float = 0.04
    interval_sd: float = 0.15
    tol: float = 0.012  # virtual target tolerance (x-heights)
    log_lambda: tuple = (-6.0, 0.5)  # shape weight against jerk energy at effort 0 and 1 (log10)
    pivot_radius: tuple = (800.0, 2000.0)  # wrist pivot arc radius (mm)
    hand_shift: tuple = (25.0, 60.0)  # hand repositioned after this much writing (mm)
    # discrete-stroke scripts: less overlap between sub-movements (more pen-controlled strokes)
    script_overlap: dict = field(
        default_factory=lambda: {
            "Han": 0.15,
            "Hiragana": 0.1,
            "Katakana": 0.15,
            "Hangul": 0.15,
            "Devanagari": 0.05,
            "Bengali": 0.05,
            "Ethiopic": 0.1,
            "Cherokee": 0.1,
            "Canadian_Aboriginal": 0.1,
            "Tifinagh": 0.15,
        }
    )


@dataclass(frozen=True)
class ScanCfg:
    p: float = 0.5  # share of samples that look scanned (write(scan=None))
    age_beta: tuple = (1.2, 4.0)  # document age in [0, 1]: mostly recent
    yellow: tuple = (0.0, 0.06, 0.18)  # paper yellowing per unit age (R, G, B loss)
    foxing: float = 8.0  # expected foxing spots at age 1
    fade: float = 0.45  # ink density lost at age 1
    p_bleed: float = 0.2  # text of the other side showing through
    shading: float = 0.15  # uneven illumination (darkening at the far corner)
    p_gray: float = 0.4
    gamma: tuple = (0.8, 1.25)
    noise: tuple = (0.0, 0.025)
    dust: float = 3.0  # expected specks
    blur: float = 0.8  # px
    p_bitonal: float = 0.05
    p_jpeg: float = 0.6
    jpeg: tuple = (40, 95)


@dataclass(frozen=True)
class RenderCfg:
    supersample: int = 3
    step: float = 0.3  # sample spacing along strokes (share of the stroke radius, supersampled px)
    min_step: float = 0.25  # supersampled px
    min_radius: float = 0.35  # supersampled px


@dataclass(frozen=True)
class Person:
    """A handwriting type: centres (natural units) of identity genes and the font class; an individual of the type
    draws each centred gene within +-spread of the gene range around its centre, all other genes freely."""

    name: str
    genes: dict
    fonts: str = "any"  # any | cursive | print
    doc: str = ""
    habits: dict = field(default_factory=dict)  # symbol habit -> probability (overrides HABITS)


# Symbol habits: probability that a writer has the habit (drawn once per writer from its identity). Quote styles are
# a choice weighted by these shares.
HABITS = {"seven_bar": 0.25, "z_bar": 0.1, "zero_slash": 0.05, "one_flag": 0.35, "amp_plus": 0.15}
QUOTES = {"\u201c\u201d": 0.55, '""': 0.25, "\u201e\u201c": 0.1, "\u00ab\u00bb": 0.1}
EUROPEAN = dict(seven_bar=0.8, z_bar=0.5, one_flag=0.85)
DRAFTING = dict(zero_slash=0.6, seven_bar=0.4, z_bar=0.3, one_flag=0.2)


PERSON_SPREAD = 0.12  # share of a gene range
# Prior of genes a type leaves free: noise genes skewed towards zero (gene = u ** k: most adults vary narrowly to
# moderately), signed drifts concentrated around their centre (gene = 0.5 + 4 (u - 0.5) ** 3).
NOISE_SKEW = {
    "size_sd": 2.5,
    "slant_sd": 2.5,
    "rot_sd": 3.0,
    "base_sd": 2.5,
    "ls_sd": 2.5,
    "elastic": 3.0,
    "wobble": 3.0,
    "tremor": 3.0,
    "wander": 2.5,
    "margin_sd": 2.0,
    "cross": 3.0,
    "retrace": 3.0,
    "caps": 4.0,
}
CENTRED = ("slope", "bow", "drift")
LEFT_HANDED = 0.9  # hand gene above this: left-handed (about 10 % of writers)


PERSONS = {
    p.name: p
    for p in (
        Person(
            "school_cursive",
            dict(slant=14, round=0.8, angular=0, connect=0.9, asc=1.25, desc=1.25, xs=1.0, ls=0.0, speed=0.4),
            "cursive",
            "taught joined script: loops, right slant, garlands",
            habits=EUROPEAN,
        ),
        Person(
            "round_print",
            dict(slant=0, round=0.9, angular=0, connect=0.05, size=3.4, xs=1.15, ls=0.12, asc=1.0, dot_len=0.0),
            "print",
            "rounded upright print, open letters",
        ),
        Person(
            "compact_print",
            dict(slant=2, size=2.1, xs=0.85, ls=0.05, ws=1.0, connect=0.05, round=0.3, angular=0.4, size_sd=0.02),
            "print",
            "small neat narrow print",
            habits=DRAFTING,
        ),
        Person(
            "block_capitals",
            dict(upper=1.0, slant=0, xs=0.9, ls=0.18, ws=1.6, connect=0, angular=0.6, round=0.1, size=2.6),
            "print",
            "everything in capitals, drafting style",
            habits=DRAFTING,
        ),
        Person("backhand", dict(slant=-10, round=0.6, connect=0.3, xs=1.05, hook=0.6), "any", "left-leaning script"),
        Person(
            "angular_cursive",
            dict(slant=22, angular=0.9, round=0.0, connect=0.85, xs=0.85, asc=1.3, desc=1.3),
            "cursive",
            "sharp, narrow, strongly slanted joined script",
            habits=EUROPEAN,
        ),
        Person(
            "thready",
            dict(slant=18, trail=0.8, flat=0.5, connect=1.0, speed=0.85, round=0.7, angular=0.0, ls=-0.05),
            "cursive",
            "fast thread connections, letters dissolve",
        ),
        Person(
            "wide_garland",
            dict(xs=1.25, ls=0.1, ws=2.0, round=0.9, angular=0, slant=8, connect=0.7),
            "cursive",
            "wide open letters, generous spacing",
        ),
        Person(
            "narrow_tall",
            dict(xs=0.78, asc=1.45, desc=1.4, ls=-0.05, ws=0.9, slant=10, angular=0.5),
            "any",
            "narrow letters, long ascenders and descenders, tight spacing",
        ),
        Person(
            "micro",
            dict(size=1.9, xs=0.9, ls=0.0, ws=0.8, pitch=2.6, asc=1.1, desc=1.1),
            "any",
            "very small writing",
            habits=DRAFTING,
        ),
        Person(
            "expansive",
            dict(size=3.3, xs=1.2, ws=2.1, cap=1.5, asc=1.35, desc=1.35, pitch=3.8),
            "any",
            "large writing, big capitals, wide spacing",
        ),
        Person(
            "arcade",
            dict(slant=3, round=0.85, angular=0.0, connect=0.6, speed=0.2, flat=0.1),
            "cursive",
            "rounded tops, slow upright joined script",
        ),
        Person("print_script", dict(slant=8, connect=0.4, round=0.5, angular=0.2), "any", "print and script mixed"),
        Person(
            "elderly",
            dict(slant=12, connect=0.7, tremor=0.5, wobble=0.4, press_sd=0.8, slope=-1.5, size=3.3, speed=0.15),
            "cursive",
            "slow tremulous joined script",
            habits=EUROPEAN,
        ),
        Person(
            "child",
            dict(
                size=3.4,
                xs=1.15,
                ls=0.2,
                connect=0.0,
                base_sd=0.12,
                size_sd=0.12,
                press=0.9,
                round=0.6,
                slant=0,
                caps=0.05,
            ),
            "print",
            "large print, uneven baseline, heavy pressure",
        ),
        Person(
            "left_hooked",
            dict(slant=-4, hook=0.9, flick=0.1, connect=0.35, press=0.7, hand=1.0),
            "any",
            "entry hooks, slight left slant, pressure on downstrokes",
        ),
        Person(
            "copperplate",
            dict(slant=24, asc=1.4, desc=1.4, round=0.8, connect=1.0, cap=1.45, xs=0.95, speed=0.15),
            "cursive",
            "steep slant, long loops, ornate capitals",
            habits=EUROPEAN,
        ),
        Person(
            "sloppy_fast",
            dict(
                slant=15,
                speed=0.9,
                elastic=0.8,
                wobble=0.6,
                connect=0.75,
                trail=0.6,
                dot_dx=0.35,
                dot_len=0.9,
                flick=0.8,
            ),
            "cursive",
            "fluent but careless",
        ),
        Person(
            "ruled_neat",
            dict(
                slant=5,
                size_sd=0.01,
                base_sd=0.01,
                slant_sd=0.5,
                ls_sd=0.01,
                elastic=0.05,
                wobble=0.05,
                connect=0.5,
                round=0.5,
            ),
            "any",
            "very regular, even and aligned",
            habits=DRAFTING,
        ),
        Person(
            "rounded_neat",
            dict(
                slant=2, round=1.0, angular=0.0, xs=1.12, size=2.9, connect=0.35, ring=0.75, ls=0.06, asc=1.1, desc=1.1
            ),
            "print",
            "round, upright, regular and open: a rounded neat hand",
        ),
        Person(
            "bubbly_print",
            dict(slant=-3, round=1.0, angular=0.0, xs=1.25, size=3.2, connect=0.05, ring=1.0, ls=0.12, cap=1.3),
            "print",
            "bubbly rounded print with circle dots",
        ),
        Person(
            "looped_cursive",
            dict(slant=10, round=0.95, angular=0.0, connect=1.0, asc=1.45, desc=1.45, xs=1.05, ring=0.3, cap=1.3),
            "cursive",
            "rounded joined script with full ascender and descender loops",
            habits=EUROPEAN,
        ),
        Person(
            "soft_upright",
            dict(slant=0, round=0.85, angular=0.0, connect=0.55, xs=1.05, size=2.7, ring=0.55, flat=0.0),
            "any",
            "soft upright hand, rounded joins, even size",
        ),
        Person(
            "artistic",
            dict(cap=1.5, asc=1.45, desc=1.5, round=0.9, flick=0.9, hook=0.7, xs=1.1, slant=10),
            "cursive",
            "embellished capitals, exaggerated loops and terminals",
        ),
    )
}


@dataclass(frozen=True)
class Style:
    """A situation applied on top of a person: situational genes remapped into ranges (the writer's relative
    position in the global range is kept), identity genes shifted, optional font class and the pen prior."""

    name: str
    remap: dict = field(default_factory=dict)  # gene -> (lo, hi) natural units
    shift: dict = field(default_factory=dict)  # gene -> added natural units (identity genes)
    fonts: str = ""  # overrides the person's font class when set
    pens: dict = field(default_factory=dict)  # tool -> weight (empty: TOOL_PRIOR)
    paper: str = "any"  # any | ruled | blank
    doc: str = ""


_NEAT = dict(
    slant_sd=(0, 1.5),
    size_sd=(0, 0.04),
    rot_sd=(0, 1),
    base_sd=(0, 0.03),
    ls_sd=(0, 0.03),
    elastic=(0, 0.25),
    wobble=(0, 0.25),
    tremor=(0, 0.1),
    trail=(0, 0.15),
    cross=(0, 0),
    retrace=(0, 0.005),
    caps=(0, 0),
    margin_sd=(0, 0.2),
    wander=(0, 0.1),
    drift=(-0.05, 0.05),
)
STYLES = {
    s.name: s
    for s in (
        Style("casual", dict(effort=(0.4, 0.6)), doc="everyday notes: as the person writes"),
        Style(
            "formal",
            dict(_NEAT, effort=(0.65, 0.8), speed=(0.1, 0.5)),
            dict(connect=0.2),
            pens={"fountain_fine": 2, "fountain_medium": 2, "ballpoint": 3, "gel": 2, "rollerball": 1},
            doc="letters, applications: careful and even",
        ),
        Style(
            "accurate",
            dict(_NEAT, effort=(0.75, 0.9), speed=(0, 0.3), slant_sd=(0, 1)),
            dict(ls=0.05, connect=-0.4),
            "print",
            pens={"ballpoint": 3, "gel": 2, "fineliner": 2, "mechanical_pencil": 2, "pencil_hb": 1},
            doc="forms and labels: separated careful letters",
        ),
        Style(
            "calligraphic",
            dict(_NEAT, effort=(0.9, 1.0), speed=(0, 0.2), press=(0.5, 1), press_sd=(0.5, 1), size=(2.8, 3.4)),
            dict(connect=0.4, round=0.2),
            "cursive",
            pens={"broad_nib": 3, "italic_nib": 3, "brush_pen": 1, "chisel_marker": 0.5},
            doc="broad-nib pen calligraphy: slow, contrasted",
        ),
        Style(
            "professional",
            dict(
                effort=(0.3, 0.45),
                speed=(0.45, 0.8),
                elastic=(0, 0.45),
                wobble=(0, 0.4),
                tremor=(0, 0.2),
                trail=(0.1, 0.45),
                cross=(0, 0.01),
                caps=(0, 0.005),
            ),
            dict(size=-0.3, connect=0.15),
            pens={"ballpoint": 4, "gel": 3, "rollerball": 1, "fountain_fine": 1},
            doc="fast, fluent, consistent office notes",
        ),
        Style(
            "hurried",
            dict(
                effort=(0.1, 0.3),
                slant_sd=(1.5, 5),
                size_sd=(0.05, 0.14),
                base_sd=(0.05, 0.14),
                elastic=(0.2, 0.6),
                wobble=(0.2, 0.6),
                speed=(0.75, 1),
                trail=(0.5, 1),
                cross=(0.01, 0.05),
                margin_sd=(0.3, 1),
                slope=(-2.5, 2.5),
            ),
            dict(slant=5, connect=0.3, flat=0.2, dot_dx=0.1, flick=0.3, ls=-0.05),
            pens={"ballpoint": 5, "ballpoint_fine": 2, "gel": 2, "pencil_hb": 1, "felt_marker": 1},
            paper="blank",
            doc="written in a rush",
        ),
        Style(
            "medical",
            dict(
                effort=(0.0, 0.12),
                slant_sd=(2, 6),
                size_sd=(0.08, 0.14),
                base_sd=(0.06, 0.14),
                elastic=(0.3, 0.6),
                wobble=(0.3, 0.6),
                speed=(0.9, 1),
                trail=(0.75, 1),
            ),
            dict(slant=8, connect=0.5, flat=0.5, round=0.3, dot_dx=0.15, dot_len=0.5, flick=0.5, ls=-0.08, size=-0.4),
            "cursive",
            pens={"ballpoint": 6, "ballpoint_fine": 2, "gel": 2},
            paper="blank",
            doc="prescription scrawl",
        ),
        Style(
            "educated",
            dict(
                effort=(0.55, 0.7),
                speed=(0.35, 0.7),
                elastic=(0, 0.5),
                wobble=(0, 0.4),
                tremor=(0, 0.15),
                cross=(0, 0.01),
                caps=(0, 0),
            ),
            dict(connect=0.3),
            pens={"ballpoint": 4, "fountain_fine": 1.5, "fountain_medium": 1, "gel": 2},
            doc="fluent practised writing",
        ),
        Style(
            "uneducated",
            dict(
                effort=(0.5, 0.7),
                slant_sd=(2, 6),
                size_sd=(0.07, 0.14),
                rot_sd=(1.5, 4),
                base_sd=(0.06, 0.14),
                ls_sd=(0.05, 0.12),
                elastic=(0.2, 0.6),
                wobble=(0.2, 0.6),
                tremor=(0.2, 0.6),
                speed=(0, 0.3),
                press_sd=(0.4, 1),
                cross=(0.01, 0.05),
                retrace=(0.01, 0.03),
                caps=(0.02, 0.08),
                margin_sd=(0.4, 1),
                slope=(-2.5, 2.5),
            ),
            dict(connect=-0.6, ls=0.08, size=0.4, press=0.3, round=-0.2),
            "print",
            pens={"ballpoint": 5, "pencil_2b": 2, "pencil_hb": 1, "felt_marker": 1},
            doc="slow unpractised print, mixed capitals",
        ),
        Style(
            "unsupported",
            dict(
                effort=(0.4, 0.6),
                slant_sd=(2, 6),
                size_sd=(0.06, 0.14),
                rot_sd=(1.5, 4),
                base_sd=(0.06, 0.14),
                slope=(-2.5, 2.5),
                bow=(-0.5, 0.5),
                wander=(0.4, 0.8),
                elastic=(0.3, 0.6),
                wobble=(0.3, 0.6),
                tremor=(0.4, 1),
                press_sd=(0.5, 1),
                margin_sd=(0.5, 1),
            ),
            dict(size=0.3),
            pens={"ballpoint": 6, "gel": 2, "pencil_hb": 1, "felt_marker": 1},
            paper="blank",
            doc="on a sheet held in the hand, no support",
        ),
    )
}


@dataclass(frozen=True)
class QDCfg:
    init: int = 24  # random individuals per type to seed the archive
    iters: int = 12
    batch: int = 16  # new individuals per type and iteration
    iso_sd: float = 0.03  # iso-line variation: isotropic gene noise ...
    line_sd: float = 0.2  # ... plus noise along the line between two elites
    slant_edges: tuple = (-6.0, 0.0, 6.0, 12.0, 18.0, 24.0)  # measured slant (deg) cells
    lift_edges: tuple = (1.1, 1.3, 1.5, 1.7)  # pen lifts per letter cells
    # Baseline instability (word centres around the line, x-heights; qd.image_instability). On clean real lines of
    # the handwritten folder it runs from about 0.13 (steady) to 0.35 (wavy); higher values were extraction junk.
    inst_free: float = 0.15  # typical real variation: free
    inst_w: float = 4.0  # quality penalty per unit of instability above that
    inst_max: float = 0.35  # bank: never above the clean real upper range
    min_q_quantile: float = 0.5  # bank: realism at least this quantile of the re-evaluated elites
    final_seeds: int = 2
    bank_sigma: float = 0.03  # gene noise around a bank writer when sampling
    texts: tuple = (  # search texts (the metrics use other texts)
        "the quick brown fox jumps over the lazy dog and runs",
        "Съешь же ещё этих мягких французских булок",
        "Sphinx of black quartz, judge my vow 1984",
        "Le cœur déçu mais l'âme plutôt naïve",
    )


@dataclass(frozen=True)
class FitCfg:
    pages: int = 600  # real writers (fit-split pages) to fit
    init: int = 10  # random individuals of random types to start from
    gens: int = 5
    batch: int = 8
    elite: int = 4
    sigma: float = 0.12
    sigma_decay: float = 0.75
    feat_w: float = 0.02  # weight of the stroke-feature distance against the hinge chi-square
    seed: int = 0
    genes: tuple = (
        "size",
        "xs",
        "asc",
        "desc",
        "cap",
        "slant",
        "slant_sd",
        "size_sd",
        "base_sd",
        "ls",
        "ws",
        "round",
        "angular",
        "connect",
        "speed",
        "effort",
        "press",
        "font",
        "letters",
        "trail",
        "flat",
        "wobble",
    )
    texts: tuple = ("the quick brown fox jumps over the lazy dog", "Sphinx of black quartz, judge my vow 1984")


@dataclass(frozen=True)
class RefsCfg:
    dpi: int = 150
    lines_per_page: int = 8
    test_share: float = 0.3  # source documents held out for the metrics (split by document)
    min_h: int = 15
    max_h: int = 120
    min_aspect: float = 8.0
    knn_hw: float = 0.6  # kept when this share of the 5 nearest labelled lines are handwriting
