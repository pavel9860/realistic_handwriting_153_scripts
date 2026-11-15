"""Stroke features of a line crop."""

import cv2
import numpy as np
import pytest

pytest.importorskip("skimage")
from hwsynth.features import NAMES, feats


def line(seed=0):
    img = np.full((60, 400), 255, np.uint8)
    rng = np.random.default_rng(seed)
    x = 10
    while x < 380:
        h = int(rng.integers(15, 35))
        cv2.ellipse(img, (x + 8, 40 - h // 2), (8, h // 2), float(rng.uniform(-20, 20)), 0, 360, 0, 2)
        x += 22
    return img


def test_one_value_per_name_and_finite():
    f = feats(line())
    assert f is not None and len(f) == len(NAMES) and np.isfinite(np.asarray(f, float)).all()


def test_blank_crop_has_no_features():
    assert feats(np.full((60, 400), 255, np.uint8)) is None


def test_features_tell_lines_apart():
    assert not np.allclose(feats(line(0)), feats(line(1)))
