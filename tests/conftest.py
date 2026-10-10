import pytest

from hwsynth.config import Paths


def fonts_available():
    return Paths().fonts.exists() and any(Paths().fonts.glob("*.ttf"))


needs_fonts = pytest.mark.skipif(not fonts_available(), reason="fonts not fetched (scripts/fetch_fonts.py)")
