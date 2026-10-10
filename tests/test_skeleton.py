import cv2
import numpy as np

from hwsynth.skeleton import resample, smooth, trace


def canvas():
    return np.zeros((160, 160), np.uint8)


def lengths(strokes):
    return sorted(float(np.hypot(*np.diff(s, axis=0).T).sum()) for s in strokes)


def test_line_is_one_stroke():
    b = canvas()
    cv2.line(b, (20, 80), (140, 80), 255, 7)
    s = trace(b > 0)
    assert len(s) == 1 and lengths(s)[0] > 100


def test_ring_is_one_closed_stroke():
    b = canvas()
    cv2.circle(b, (80, 80), 50, 255, 7)
    s = trace(b > 0)
    assert len(s) == 1
    assert np.hypot(*(s[0][0] - s[0][-1])) < 3
    assert abs(lengths(s)[0] - 2 * np.pi * 50) < 25


def test_cross_continues_straight():
    b = canvas()
    cv2.line(b, (20, 80), (140, 80), 255, 7)
    cv2.line(b, (80, 20), (80, 140), 255, 7)
    s = trace(b > 0)
    assert len(s) == 2
    assert all(L > 100 for L in lengths(s))


def test_tee_keeps_bar_whole():
    b = canvas()
    cv2.line(b, (20, 30), (140, 30), 255, 7)
    cv2.line(b, (80, 30), (80, 140), 255, 7)
    s = trace(b > 0)
    assert len(s) == 2 and max(lengths(s)) > 100


def test_lone_dot_survives():
    b = canvas()
    cv2.circle(b, (80, 80), 3, 255, -1)
    assert len(trace(b > 0)) == 1


def test_resample_and_smooth():
    p = np.array([[0, 0], [10, 0], [10, 10]], float)
    q = resample(p, 0.5)
    d = np.hypot(*np.diff(q, axis=0).T)
    assert np.allclose(d, d.mean(), atol=1e-6) and abs(d.mean() - 0.5) < 0.01
    s = smooth(q, 3)
    assert np.allclose(s[0], q[0]) and np.allclose(s[-1], q[-1])
    t = np.linspace(0, 2 * np.pi, 100)
    ring = np.c_[np.cos(t), np.sin(t)]
    assert np.allclose(smooth(ring, 3)[0], smooth(ring, 3)[-1])
