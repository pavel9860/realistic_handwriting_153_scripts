import numpy as np

import hwsynth as H
from conftest import needs_fonts
from hwsynth.config import GENES


def test_decode_encode_roundtrip():
    g = np.random.default_rng(0).random(len(GENES))
    assert np.allclose(H.encode(H.decode(g)), g)


@needs_fonts
def test_deterministic():
    a = H.sample_writer("school_cursive", "casual", seed=3, bank=False).lines("Hello world again", 80)
    b = H.sample_writer("school_cursive", "casual", seed=3, bank=False).lines("Hello world again", 80)
    assert len(a.strokes) == len(b.strokes)
    assert all(np.allclose(x.p, y.p) for x, y in zip(a.strokes, b.strokes))


@needs_fonts
def test_wraps_within_width_and_boxes_words():
    text = "one two three four five six seven eight nine ten eleven twelve"
    w = H.sample_writer("compact_print", "accurate", seed=1, bank=False)
    lay = w.lines(text, 60)
    assert lay.lines > 1
    assert len([x for x in lay.words if not x.struck]) == len(text.split())
    xh = w.p["size"]
    for x in lay.words:
        assert x.box and x.box[2] <= 60 + 3 * xh and x.box[0] >= -3 * xh


@needs_fonts
def test_rtl_starts_at_right():
    w = H.sample_writer("print_script", "accurate", seed=2, bank=False)
    lay = w.lines("שלום עולם", 100)
    first, second = lay.words[0].box, lay.words[-1].box
    assert lay.rtl and first[0] > second[0]


@needs_fonts
def test_write_sample():
    s = H.write("Test 123", H.sample_writer(seed=4, bank=False), 60, 100)
    assert s.image.dtype == np.uint8 and s.image.ndim == 3
    assert 0 < s.ink.max() <= 1 and s.words


@needs_fonts
def test_left_hand_metadata_and_slant():
    from hwsynth.config import GENE_INDEX

    per, g = H.individual("print_script", 4, bank=False)
    g = g.copy()
    g[GENE_INDEX["hand"]] = 1.0
    w = H.styled(per, g, "casual", 1)
    assert w.left and w.lines("left hand", 50).meta["hand"] == "left"


@needs_fonts
def test_hand_movement_speed_attached():
    w = H.sample_writer("school_cursive", "casual", seed=5, bank=False)
    lay = w.lines("minimum", 80)
    moved = [s for s in lay.strokes if s.v is not None]
    assert moved and all(len(s.v) == len(s.p) for s in moved)


@needs_fonts
def test_scan_keeps_ink_and_range():
    for seed in range(4):
        s = H.write("scan test words", H.sample_writer(seed=seed, bank=False), 60, 100, False, scan=True)
        g = s.image.mean(2)
        assert s.image.dtype == np.uint8 and np.median(g) - g.min() > 50  # ink still visible after ageing and scanning


@needs_fonts
def test_personal_letters_stable_across_styles():
    from hwsynth.config import GENE_INDEX

    per, g = H.individual("print_script", 5, bank=False)
    g = g.copy()
    g[GENE_INDEX["letters"]] = 1.0
    assert len({H.styled(per, g, s, 1).identity for s in ("casual", "formal", "hurried", "medical")}) == 1
