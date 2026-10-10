import numpy as np

from hwsynth.metrics import cer, frechet, precision_recall
from hwsynth.realism import clean, is_test, page_lines


def test_frechet():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (400, 4))
    assert frechet(a, a) < 1e-6
    assert frechet(a, a + 3) > 30


def test_precision_recall():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (200, 3))
    assert precision_recall(a, a) == (1.0, 1.0)
    p, r = precision_recall(a, a + 10)
    assert p == 0 and r == 0


def test_cer():
    assert cer("abc", "abc") == 0
    assert cer("abc", "abd") == 1 / 3
    assert cer("abc", "") == 1


def test_split_by_document_is_stable():
    assert is_test("doc_a", 0.3) == is_test("doc_a", 0.3)
    share = np.mean([is_test(f"d{i}", 0.3) for i in range(5000)])
    assert 0.27 < share < 0.33


def test_page_lines_and_clean():
    import cv2

    page = np.full((600, 800), 250, np.uint8)
    for y in (100, 250, 400):
        for x in range(60, 700, 90):
            cv2.putText(page, "ab", (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, 20, 2)
    cv2.line(page, (0, 550), (799, 550), 0, 2)
    ls = page_lines(page)
    assert len(ls) == 3 and all(clean(c) for c in ls)
