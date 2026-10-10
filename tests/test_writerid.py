import cv2
import numpy as np

from hwsynth.writerid import chi2, descriptor, identification, signature, trajectory_descriptor


def line(slant):
    img = np.full((60, 400), 255, np.uint8)
    for x in range(10, 380, 25):
        cv2.line(img, (x, 50), (x + int(slant * 40), 10), 0, 3)
    return img


def test_descriptor_normalised():
    d = descriptor(line(0.0))
    assert d is not None and abs(d.sum() - 2) < 1e-6 and (d >= 0).all()


def test_hinge_tells_slants_apart():
    a, b, c = descriptor(line(0.0)), descriptor(line(0.02)), descriptor(line(0.6))
    assert chi2(a, b)[0, 0] < chi2(a, c)[0, 0]


def test_identification_respects_groups():
    X = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]) + 1e-3
    assert identification(X, [0, 0, 1, 1]) == 1.0
    assert identification(X, [0, 0, 1, 1], groups=[0, 0, 1, 1]) == 0.0


def test_signature_levels():
    s = signature(np.array([[0, 0], [1, 0], [1, 1]], float))
    assert np.allclose(s[:2], (1, 1))
    lvl2 = s[2:6].reshape(2, 2)
    assert np.isclose(lvl2[0, 1] - lvl2[1, 0], 1.0)  # twice the signed (Levy) area of the path


def test_trajectory_descriptor_shape():
    t = np.linspace(0, 2 * np.pi, 80)
    d = trajectory_descriptor([np.c_[t, np.sin(t)], np.c_[t, np.cos(t)]], 1.0)
    assert d is not None and np.isfinite(d).all()
