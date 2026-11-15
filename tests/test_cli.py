"""Command line: argument errors, the write command end to end, script coverage."""

import json

import cv2
import pytest
from conftest import needs_fonts

from hwsynth import cli


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        cli.main([])


@needs_fonts
def test_write_saves_the_image_and_its_words(tmp_path):
    out = tmp_path / "line.png"
    cli.main(["write", "Hello world", "-o", str(out), "--seed", "3", "--dpi", "100"])
    img = cv2.imread(str(out))
    meta = json.loads(out.with_suffix(".json").read_text())
    assert img is not None and img.shape[0] > 10 and (img < 128).any()
    assert len(meta["words"]) == 2


@needs_fonts
def test_coverage_lists_known_scripts_only():
    from hwsynth.scripts import all_scripts

    c = cli.coverage()
    assert c and set(c) <= set(all_scripts()) and set(c.values()) == {"hand"}
