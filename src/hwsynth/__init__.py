"""hwsynth: natural synthetic handwriting in any Unicode script."""

from .config import GENES, PERSONS, STYLES
from .figures import FIGURES, figure
from .page import Sample, line_crop, write
from .styles import apply_style, individual, person_genes, sample_writer, styled
from .writer import Layout, Writer, decode, encode

__all__ = [
    "FIGURES",
    "figure",
    "GENES",
    "PERSONS",
    "STYLES",
    "Layout",
    "Sample",
    "Writer",
    "apply_style",
    "decode",
    "encode",
    "line_crop",
    "individual",
    "person_genes",
    "sample_writer",
    "styled",
    "write",
]
__version__ = "0.1.0"
