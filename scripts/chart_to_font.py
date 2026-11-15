"""Glyph fonts from images of handwritten alphabet charts: letters are segmented, contours traced and written as a
TrueType font that hwsynth traces like any handwriting font.

python scripts/chart_to_font.py georgian IMAGE OUT.ttf [NUSKHURI_IMAGE]
        33 Mkhedruli letters, 3 rows of 11 (annotations in blue); optionally the 38 Nuskhuri letters (U+2D00..2D25)
python scripts/chart_to_font.py armenian IMAGE OUT.ttf    38 capital/small pairs, 5 columns, 39 cells
python scripts/chart_to_font.py tibetan  IMAGE OUT.ttf    30 consonants in the upper bands, 5 vowel cells
python scripts/chart_to_font.py lao CONSONANTS SYLLABLES OUT.ttf
        27 consonants (3 rows of 8, 9, 10) and the vowel signs cut from a chart of syllables on ກ (vowels in red)
python scripts/chart_to_font.py gurmukhi IMAGE OUT.ttf   35 letters of a handwritten manuscript page (7 rows of 5)
python scripts/chart_to_font.py geez IMAGE OUT.ttf       33 consonant rows of 7 orders (231 syllables) from a chart
python scripts/chart_to_font.py myanmar ARAKANESE VOWELS MEDIALS SYMBOLS OUT.ttf
        34 letters from the Mrauk-U column of the Arakanese chart (hand-drawn forms); vowel signs, medials, tones
        and numerals from typeset charts with dotted-circle placeholders (typeset forms, not handwriting)

The Armenian chart has one extra cell (index 36, the ու note) that is skipped; its letter order is the standard one.
Charts are inputs only: they are not part of the repository.
"""

import sys

import cv2
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

GEORGIAN = "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ"
ARMENIAN = [(chr(0x531 + i), chr(0x561 + i)) for i in range(38)]
TIBETAN = "ཀཁགངཅཆཇཉཏཐདནཔཕབམཙཚཛཝཞཟའཡརལཤསཧཨ"
TIBETAN_MARKS = (0x0F72, 0x0F74, 0x0F7A, 0x0F7C)
UP, PAD, XHEIGHT = 4, 3, 520


def load(path):
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw.shape[2] == 4:
        a = raw[..., 3:4].astype(float) / 255
        raw = (raw[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    return raw


def kmeans1(values, k):
    v = np.float32(values).reshape(-1, 1)
    _, lab, c = cv2.kmeans(
        v, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01), 10, cv2.KMEANS_PP_CENTERS
    )
    rank = np.empty(k, int)
    rank[np.argsort(c.ravel())] = np.arange(k)
    return rank[lab.ravel()]


def contours(mask):
    """Smoothed polygon contours (with holes) of a binary mask, in mask pixels."""
    big = cv2.GaussianBlur(cv2.resize(mask * 255, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC), (0, 0), UP * 0.6)
    _, big = cv2.threshold(big, 127, 255, cv2.THRESH_BINARY)
    cs, _ = cv2.findContours(big, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        c = cv2.approxPolyDP(c, 1.2, True)[:, 0, :].astype(float) / UP
        if len(c) >= 3:
            out.append(c)
    return out


def glyph(mask, x_ref, baseline, scale):
    """TT glyph of a full-image-size mask; x_ref: image x that becomes x 40, baseline: image y of the baseline."""
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min() - PAD, ys.min() - PAD, xs.max() + PAD + 1, ys.max() + PAD + 1
    pen = TTGlyphPen(None)
    h, w = mask.shape
    sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sub[sy0 - y0 : sy0 - y0 + min(y1, h) - sy0, sx0 - x0 : sx0 - x0 + min(x1, w) - sx0] = mask[
        sy0 : min(y1, h), sx0 : min(x1, w)
    ]
    for c in contours(sub):
        pts = [(round((px + x0 - x_ref) * scale + 40), round((baseline - (py + y0)) * scale)) for px, py in c]
        pen.moveTo(pts[0])
        for p in pts[1:]:
            pen.lineTo(p)
        pen.closePath()
    return pen.glyph()


def disc(cx, cy, r, n=14):
    return [(cx + r * np.cos(2 * np.pi * k / n), cy + r * np.sin(2 * np.pi * k / n)) for k in range(n)]


def poly_glyph(polys):
    """TT glyph of filled polygons (font units, y up)."""
    pen = TTGlyphPen(None)
    for pts in polys:
        pts = [(round(x), round(y)) for x, y in pts]
        pen.moveTo(pts[0])
        for q in pts[1:]:
            pen.lineTo(q)
        pen.closePath()
    return pen.glyph()


def bar(x, y0, y1, w=64, lean=0.0):
    return [(x - w / 2, y0), (x + w / 2, y0), (x + w / 2 + lean, y1), (x - w / 2 + lean, y1)]


def punctuation(kind):
    """Simple marks drawn as dots and bars: [(code point, glyph, advance)]."""
    r = 46
    if kind == "armenian":
        return [
            (0x589, poly_glyph([disc(70, 60, r), disc(70, 420, r)]), 200),
            (0x55D, poly_glyph([bar(80, 430, 640, lean=-60)]), 200),
        ]
    if kind == "tibetan":
        return [
            (0x0F0B, poly_glyph([disc(60, 260, r)]), 150),
            (0x0F0D, poly_glyph([bar(70, -40, 560)]), 200),
            (0x0F0E, poly_glyph([bar(60, -40, 560), bar(190, -40, 560)]), 320),
        ]
    if kind == "geez":
        return [
            (0x1361, poly_glyph([disc(70, 90, r), disc(70, 420, r)]), 200),
            (0x1362, poly_glyph([disc(70, 90, r), disc(70, 420, r), disc(250, 90, r), disc(250, 420, r)]), 380),
            (0x1363, poly_glyph([disc(70, 420, r), bar(70, 40, 330, lean=-50)]), 200),
            (0x1364, poly_glyph([disc(70, 420, r), disc(70, 160, r), bar(70, -90, 80, lean=-40)]), 200),
        ]
    return []


def ligature(first, second, overlap=70):
    """TT glyph of two glyphs side by side: (glyph, advance); first/second are (glyph, advance)."""
    from fontTools.pens.transformPen import TransformPen

    pen = TTGlyphPen(None)
    first[0].draw(pen, None)
    second[0].draw(TransformPen(pen, (1, 0, 0, 1, first[1] - overlap, 0)), None)
    return pen.glyph(), first[1] + second[1] - overlap


def write(path, entries, name):
    """entries: [(codepoint, glyph, advance)]."""
    glyphs, cmap, met, order = {".notdef": TTGlyphPen(None).glyph()}, {}, {".notdef": (500, 0)}, [".notdef"]
    for cp, g, adv in entries:
        nm = f"uni{cp:04X}"
        glyphs[nm], met[nm], cmap[cp] = g, (int(adv), 0), nm
        order.append(nm)
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(met)
    fb.setupHorizontalHeader(ascent=900, descent=-400)
    fb.setupNameTable({"familyName": name, "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()
    fb.save(path)


def comps_of(mask, min_area):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return lab, st, cen, [i for i in range(1, n) if st[i][4] >= min_area]


def nuskhuri(img):
    """38 Nuskhuri letters (U+2D00..2D25) of a stroke-order chart, rows of 7 (the last of 3): [(cp, glyph, advance)]."""
    r, g, b = [img[..., i].astype(int) for i in range(3)]
    mask = ((r < 120) & (g < 120) & (b < 120)).astype(np.uint8)
    lab, st, cen, ids = comps_of(mask, 20)
    assert len(ids) == 38, len(ids)
    row = kmeans1([cen[i][1] for i in ids], 6)
    key = sorted(zip(row, [st[i][0] for i in ids], ids))
    order = [i for _, _, i in key]
    rows = [r_ for r_, _, _ in key]
    scale = XHEIGHT * 1.4 / np.median([st[i][3] for i in order])
    base = {k: np.median([st[i][1] + st[i][3] for i, r_ in zip(order, rows) if r_ == k]) for k in set(rows)}
    return [
        (0x2D00 + k, glyph((lab == i).astype(np.uint8), st[i][0], base[r_], scale), st[i][2] * scale + 80)
        for k, (i, r_) in enumerate(zip(order, rows))
    ]


def georgian(img, out, old=None):
    r, g, b = [img[..., i].astype(int) for i in range(3)]
    mask = ((r < 120) & (g < 120) & (b < 120)).astype(np.uint8)
    lab, st, cen, ids = comps_of(mask, 40)
    assert len(ids) == 33, len(ids)
    row = kmeans1([cen[i][1] for i in ids], 3)
    order = [i for _, _, i in sorted(zip(row, [st[i][0] for i in ids], ids))]
    scale = XHEIGHT / np.median([st[i][3] for i in order])
    base = [np.median([st[order[r * 11 + j]][1] + st[order[r * 11 + j]][3] for j in range(11)]) for r in range(3)]
    ents = []
    for k, (ch, i) in enumerate(zip(GEORGIAN, order)):
        ents.append(
            (ord(ch), glyph((lab == i).astype(np.uint8), st[i][0], base[k // 11], scale), st[i][2] * scale + 80)
        )
    if old is not None:
        ents += nuskhuri(old)
    write(out, ents, "ChartGeorgian")


def armenian(img, out):
    mask = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 80).astype(np.uint8)
    lab, st, cen, ids = comps_of(mask, 60)
    row, col = kmeans1([cen[i][1] for i in ids], 8), kmeans1([cen[i][0] for i in ids], 5)
    cells = {}
    for i, r, c in zip(ids, row, col):
        cells.setdefault((r, c), []).append(i)
    keys = sorted(cells)
    assert len(keys) == 39, len(keys)
    keys.pop(36)
    cap_h = np.median([max(st[i][3] for i in cells[k]) for k in keys])
    scale = 700 / cap_h
    ents = []
    for (cap_ch, small_ch), k in zip(ARMENIAN, keys):
        cs = sorted(cells[k], key=lambda i: cen[i][0])
        gaps = [st[b][0] - (st[a][0] + st[a][2]) for a, b in zip(cs, cs[1:])]
        cut = int(np.argmax(gaps)) + 1
        left, right = cs[:cut], cs[cut:]
        base = max(st[i][1] + st[i][3] for i in left)
        for ch, grp in ((cap_ch, left), (small_ch, right)):
            m = np.isin(lab, grp).astype(np.uint8)
            x0 = min(st[i][0] for i in grp)
            x1 = max(st[i][0] + st[i][2] for i in grp)
            ents.append((ord(ch), glyph(m, x0, base, scale), (x1 - x0) * scale + 80))
    have = {cp: (g_, adv) for cp, g_, adv in ents}
    lig, adv = ligature(have[0x565], have[0x582])
    ents += [(0x587, lig, adv)] + punctuation("armenian")
    write(out, ents, "ChartArmenian")


def tibetan(img, out):
    mask = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 140).astype(np.uint8)
    lab, st, cen, ids = comps_of(mask, 15)
    bands = [(46, 92, 8), (241, 284, 8), (433, 479, 8), (633, 675, 6), (839, 898, 5)]
    cells = []
    for y0, y1, k in bands:
        c = [i for i in ids if y0 - 3 <= st[i][1] and st[i][1] + st[i][3] <= y1 + 3]
        cl = kmeans1([cen[i][0] for i in c], k)
        cells += [[i for i, l in zip(c, cl) if l == j] for j in range(k)]
    assert len(cells) == 35
    masks = [np.isin(lab, c).astype(np.uint8) for c in cells]
    boxes = [cv2.boundingRect(m) for m in masks]
    scale = XHEIGHT / np.median([b[3] for b in boxes[:30]])
    ents = []
    for ch, m, (x, y, w, h) in zip(TIBETAN, masks[:30], boxes[:30]):
        ents.append((ord(ch), glyph(m, x, y + h, scale), w * scale + 80))
    base_m, (bx, by, bw, bh) = masks[30], boxes[30]
    tpl = base_m[by : by + bh, bx : bx + bw]
    for cp, m, (x, y, w, h) in zip(TIBETAN_MARKS, masks[31:], boxes[31:]):
        pad = np.pad(m, 20)
        res = cv2.matchTemplate(pad.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCORR)
        _, _, _, (mx, my) = cv2.minMaxLoc(res)
        shifted = np.zeros_like(pad)
        shifted[my : my + bh, mx : mx + bw] = tpl
        mark = (pad & ~cv2.dilate(shifted, np.ones((5, 5), np.uint8)))[20:-20, 20:-20]
        n, l2, s2, _ = cv2.connectedComponentsWithStats(mark, connectivity=8)
        keep = np.isin(l2, [i for i in range(1, n) if s2[i][4] >= 20]).astype(np.uint8)
        if keep.any():
            ents.append((cp, glyph(keep, mx - 20 + x * 0 if False else mx - 20, my - 20 + bh, scale), 0))
    ents += [(cp, g_, adv) for cp, g_, adv in punctuation("tibetan")]
    write(out, ents, "ChartTibetan")


LAO = "ກຂຄງຈສຊຍດຕຖທນບປຜຝພຟມຢຣລວຫອຮ"
# (row, column) of the syllable chart -> (code point, kind): kind r right, l left (both take an advance), a above or
# below the consonant (zero advance, drawn over it)
LAO_MARKS = {
    (0, 0): (0x0EB0, "r"),
    (0, 1): (0x0EB1, "a"),
    (0, 2): (0x0EB4, "a"),
    (0, 3): (0x0EB6, "a"),
    (0, 4): (0x0EB8, "a"),
    (1, 2): (0x0EBB, "a"),
    (1, 6): (0x0EB2, "r"),
    (1, 7): (0x0EB5, "a"),
    (2, 0): (0x0EB7, "a"),
    (2, 1): (0x0EB9, "a"),
    (2, 2): (0x0EC0, "l"),
    (2, 3): (0x0EC1, "l"),
    (2, 4): (0x0EC2, "l"),
    (2, 5): (0x0ECD, "a"),
    (3, 5): (0x0EC4, "l"),
    (3, 6): (0x0EC3, "l"),
    (4, 1): (0x0EBD, "r"),
    (4, 6): (0x0EB3, "ar"),
    (4, 7): (0x0ECC, "a"),
}
SYLLABLE_ROWS = [(0, 45), (110, 185), (245, 320), (375, 455), (505, 590)]


def shifted(g, dx):
    """The TT glyph moved by dx font units."""
    if len(g.coordinates):
        for k in range(len(g.coordinates)):
            g.coordinates[k] = (g.coordinates[k][0] + dx, g.coordinates[k][1])
        g.recalcBounds(None)
    return g


def lao(consonants, syllables, out):
    g = cv2.cvtColor(consonants, cv2.COLOR_BGR2GRAY)
    d = ((cv2.GaussianBlur(g, (0, 0), 25).astype(int) - g.astype(int)) > 35).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    lab, st, cen, ids = comps_of(d, 30)
    ids = [i for i in ids if st[i][1] > 100]
    assert len(ids) == 27, len(ids)
    row = kmeans1([cen[i][1] for i in ids], 3)
    key = sorted(zip(row, [st[i][0] for i in ids], range(len(ids))))
    order = [ids[k] for _, _, k in key]
    rows = [row[k] for _, _, k in key]
    assert [rows.count(k) for k in range(3)] == [8, 9, 10]
    scale = XHEIGHT / np.median([st[i][3] for i in order])
    rowbase = [np.median([st[i][1] + st[i][3] for i, r in zip(order, rows) if r == k]) for k in range(3)]
    ents = {}
    for ch, i, r in zip(LAO, order, rows):
        ents[ord(ch)] = (glyph((lab == i).astype(np.uint8), st[i][0], rowbase[r], scale), st[i][2] * scale + 80)
    rr, gg, bb = syllables[..., 2].astype(int), syllables[..., 1].astype(int), syllables[..., 0].astype(int)
    red = ((rr - np.maximum(gg, bb)) > 60).astype(np.uint8)
    blk = (rr < 140).astype(np.uint8)
    bl, bst, bcen, bids = comps_of(blk, 40)
    rl, rst, rcen, rids = comps_of(red, 8)
    rids = [i for i in rids if not (rst[i][3] > 30 and rst[i][2] <= 3)]
    base = {}
    for r, (y0, y1) in enumerate(SYLLABLE_ROWS):
        bs = sorted((i for i in bids if y0 <= bcen[i][1] <= y1), key=lambda i: bcen[i][0])
        assert len(bs) == 8, (r, len(bs))
        base.update({(r, c): i for c, i in enumerate(bs)})
    k16 = scale * np.median([st[i][3] for i in order[:8]]) / np.median([bst[i][3] for i in base.values()])
    adv_ref = bst[base[(0, 0)]][2] * k16 + 80

    def parts(r, c):
        cx = bcen[base[(r, c)]][0]
        y0, y1 = SYLLABLE_ROWS[r]
        cy = bcen[base[(r, c)]][1]
        return [
            i
            for i in rids
            if abs(rcen[i][1] - cy) < 35
            and abs(rcen[i][0] - cx) < 45
            and abs(rcen[i][0] - cx) == min(abs(rcen[i][0] - bcen[base[(r, k)]][0]) for k in range(8))
        ]

    for (r, c), (cp, kind) in LAO_MARKS.items():
        near = parts(r, c)
        if not near:
            continue
        bx0, by0, bw, bh = bst[base[(r, c)]][:4]
        m = np.isin(rl, near).astype(np.uint8)
        x0, x1 = min(rst[i][0] for i in near), max(rst[i][0] + rst[i][2] for i in near)
        if "a" in kind:
            ents[cp] = (shifted(glyph(m, bx0, by0 + bh, k16), -adv_ref), 0)
        else:
            ents[cp] = (glyph(m, x0, by0 + bh, k16), (x1 - x0) * k16 + 80)
    write(out, [(cp, g_, a_) for cp, (g_, a_) in sorted(ents.items())], "ChartLao")


# components of the typeset charts, numbered as scripts' dark components (gray < 100, area >= 12) sorted by
# (round(cy / 40), cx); circles are the dotted placeholders of the base letter
MYANMAR_VOWELS = {
    0x102D: [50],
    0x102E: [51],
    0x102F: [59],
    0x1030: [60],
    0x1031: [54],
    0x102C: [118],
    0x102B: [121],
    0x1032: [116],
}
MYANMAR_SYMBOLS = {0x1037: [21], 0x1038: [15, 16], 0x103A: [12], 0x1036: [14]}
MYANMAR_DIGITS = {
    0x1040: [113],
    0x1041: [114],
    0x1042: [115],
    0x1043: [116],
    0x1044: [117],
    0x1045: [118],
    0x1046: [213],
    0x1047: [214],
    0x1048: [215],
    0x1049: [216],
}
MYANMAR_ORDER = "ကခဂဃငစဆဇဈညဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝသဟဠအ"


def numbered(img, thresh=100, min_area=12):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    n, lab, st, cen = cv2.connectedComponentsWithStats((g < thresh).astype(np.uint8), connectivity=8)
    big = sorted((i for i in range(1, n) if st[i][4] >= min_area), key=lambda i: (round(cen[i][1] / 40), cen[i][0]))
    return g, lab, st, cen, big


def circles(g, dark_lab):
    """Bounding boxes of the dotted placeholder circles: gray dots (small components) merged into rings; dark_lab
    marks the pixels of the real (large) components."""
    light = (g < 215).astype(np.uint8)
    light[cv2.dilate((dark_lab > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3) > 0] = 0
    ring = cv2.dilate(light, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(ring, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 22 <= w <= 40 and 22 <= h <= 40 and 0.75 < w / h < 1.33:
            inner = dark_lab[y + h // 4 : y + 3 * h // 4, x + w // 4 : x + 3 * w // 4]
            if not inner.any():
                out.append((x + 2, y + 2, x + w - 2, y + h - 2))
    return out


MYANMAR_ROWS = "ကခဂဃငစဆဇဈဉဋဌဍဎဏတထဒဓနပဖဗဘမယရလဝဿသဟဠအ"


def mraukU(img):
    """The 34 letters of the Mrauk-U column of the Arakanese chart (33 consonants in Brahmic order and the vowel a):
    list of (row mask, band top, band bottom, left)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = ((cv2.GaussianBlur(g, (0, 0), 20).astype(int) - g.astype(int)) > 40).astype(np.uint8)
    x0, x1, y0, y1 = 700, 790, 330, 1800
    prof = d[y0:y1, x0:x1].sum(1)
    bands, inside = [], False
    for y, v in enumerate(prof):
        if v > 0 and not inside:
            top, inside = y, True
        if v == 0 and inside:
            bands.append((top + y0, y + y0))
            inside = False
    assert len(bands) == 34, len(bands)
    out = []
    for t, b in bands:
        m = np.zeros_like(d)
        m[t:b, x0:x1] = d[t:b, x0:x1]
        xs = np.nonzero(m.any(0))[0]
        out.append((m, t, b, int(xs.min())))
    return out


def myanmar(c17, c12, c13, c15, out):
    letters_ = mraukU(c17)
    scale = XHEIGHT / np.median([b - t for _, t, b, _ in letters_])
    ents = {}
    for ch, (m, t, b, left) in zip(MYANMAR_ROWS, letters_):
        w = int(m.any(0).sum())
        ents[ord(ch)] = (glyph(m, left, b, scale), None)
    widths = {cp: np.ptp(np.nonzero(m.any(0))[0]) for (cp, _), (m, *_r) in zip(ents.items(), letters_)}
    ma_w = widths[ord("မ")] * scale
    adv_ref = float(np.percentile([w * scale for w in widths.values()], 85)) + 80
    ents = {cp: (g_, adv_ref) for cp, (g_, _) in ents.items()}

    def from_circles(img, spec, k_from="circle"):
        gg, lb, s2, c2, big = numbered(img)
        circ = circles(gg, np.isin(lb, big))
        assert circ, "no placeholder circles"
        dia = np.median([x1 - x0 for x0, _, x1, _ in circ])
        k = ma_w / dia
        for cp, idx in spec.items():
            comp = [big[j] for j in idx]
            m = np.isin(lb, comp).astype(np.uint8)
            cx = np.mean([c2[i][0] for i in comp])
            cy = np.mean([c2[i][1] for i in comp])
            cx0, cy0, cx1, cy1 = min(circ, key=lambda c: abs((c[0] + c[2]) / 2 - cx) + abs((c[1] + c[3]) / 2 - cy))
            right = cp in (0x102C, 0x102B)
            x0 = min(s2[i][0] for i in comp)
            x1 = max(s2[i][0] + s2[i][2] for i in comp)
            if right:
                ents[cp] = (glyph(m, x0, cy1, k), (x1 - x0) * k + 80)
            else:
                ents[cp] = (shifted(glyph(m, cx0, cy1, k), -adv_ref), 0)

    gg, lb, s3, c3, big3 = numbered(c13)
    ma13 = s3[big3[135]]
    k13 = ma_w / ma13[2]
    tpl = (lb[ma13[1] : ma13[1] + ma13[3], ma13[0] : ma13[0] + ma13[2]] == big3[135]).astype(np.uint8)
    for cp, idx in ((0x103D, [136]), (0x103E, [140])):
        comp = [big3[j] for j in idx]
        m = np.isin(lb, comp).astype(np.uint8)
        host = s3[big3[135 if cp == 0x103D else 139]]
        ents[cp] = (shifted(glyph(m, host[0], host[1] + host[3], k13), -adv_ref), 0)
    for cp, cell in ((0x103B, 38), (0x103C, 42)):
        cs = s3[big3[cell]]
        sub = np.zeros_like(lb, np.uint8)
        sub[cs[1] : cs[1] + cs[3], cs[0] : cs[0] + cs[2]] = (
            lb[cs[1] : cs[1] + cs[3], cs[0] : cs[0] + cs[2]] == big3[cell]
        )
        pad = np.pad(sub[cs[1] : cs[1] + cs[3], cs[0] : cs[0] + cs[2]], 20)
        res = cv2.matchTemplate(pad.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCORR)
        _, _, _, (mx, my) = cv2.minMaxLoc(res)
        host = np.zeros_like(pad)
        host[my : my + ma13[3], mx : mx + ma13[2]] = tpl
        mark = (pad & ~cv2.dilate(host, np.ones((5, 5), np.uint8)))[20:-20, 20:-20]
        full = np.zeros_like(lb, np.uint8)
        full[cs[1] : cs[1] + cs[3], cs[0] : cs[0] + cs[2]] = mark
        bx, by = cs[0] + mx - 20, cs[1] + my - 20
        ents[cp] = (shifted(glyph(full, bx, by + ma13[3], k13), -adv_ref), 0)
    from_circles(c12, MYANMAR_VOWELS)
    from_circles(c15, MYANMAR_SYMBOLS)
    gg, lb, s2, c2, big = numbered(c15)
    for cp, idx in MYANMAR_DIGITS.items():
        m = np.isin(lb, [big[j] for j in idx]).astype(np.uint8)
        x0, w = s2[big[idx[0]]][0], s2[big[idx[0]]][2]
        zero = s2[big[113]]
        k = XHEIGHT * 0.9 / zero[3]
        ents[cp] = (glyph(m, x0, zero[1] + zero[3], k), w * k + 80)
    write(out, [(cp, g_, a_) for cp, (g_, a_) in sorted(ents.items())], "ChartMyanmar")


GURMUKHI = "ੳਅੲਸਹਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਵੜ"


def gurmukhi(img, out):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = ((cv2.GaussianBlur(g, (0, 0), 25).astype(int) - g.astype(int)) > 28).astype(np.uint8)
    d[:740] = 0
    d[:, :200] = 0
    d[:, 1300:] = 0
    d[1900:] = 0
    lab, st, cen, ids = comps_of(d, 25)
    row = kmeans1([cen[i][1] for i in ids], 7)
    col = kmeans1([cen[i][0] for i in ids], 5)
    cells = {}
    for i, r, c in zip(ids, row, col):
        cells.setdefault((r, c), []).append(i)
    assert len(cells) == 35, len(cells)
    keys = sorted(cells)
    h = np.median([max(st[i][1] + st[i][3] for i in cells[k]) - min(st[i][1] for i in cells[k]) for k in keys])
    scale = 700 / h
    ents = []
    for ch, k in zip(GURMUKHI, keys):
        m = np.isin(lab, cells[k]).astype(np.uint8)
        x0 = min(st[i][0] for i in cells[k])
        x1 = max(st[i][0] + st[i][2] for i in cells[k])
        top = min(st[i][1] for i in cells[k])
        ents.append((ord(ch), glyph(m, x0, top + h, scale), (x1 - x0) * scale + 80))
    write(out, ents, "ChartGurmukhi")


GEEZ_BASES = [
    0x1200,
    0x1208,
    0x1210,
    0x1218,
    0x1220,
    0x1228,
    0x1230,
    0x1238,
    0x1240,
    0x1260,
    0x1270,
    0x1278,
    0x1280,
    0x1290,
    0x1298,
    0x12A0,
    0x12A8,
    0x12B8,
    0x12C8,
    0x12D0,
    0x12D8,
    0x12E0,
    0x12E8,
    0x12F0,
    0x1300,
    0x1308,
    0x1320,
    0x1328,
    0x1330,
    0x1338,
    0x1340,
    0x1348,
    0x1350,
]


def geez(img, out):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = (g < 140).astype(np.uint8)
    lab, st, cen, ids = comps_of(d, 12)
    glyphs_ = [i for i in ids if st[i][3] >= 12 and st[i][2] >= 6 and st[i][4] >= 25]
    ys = [cen[i][1] for i in glyphs_ if st[i][3] >= 17]
    rows = kmeans1(ys, 33)
    centers = [np.mean([y for y, r in zip(ys, rows) if r == k]) for k in range(33)]
    xs = [cen[i][0] for i in glyphs_ if st[i][3] >= 17]
    cols = kmeans1(xs, 7)
    cx = [np.mean([x for x, c in zip(xs, cols) if c == k]) for k in range(7)]
    cells = {}
    for i in glyphs_:
        r = int(np.argmin([abs(cen[i][1] - c) for c in centers]))
        c = int(np.argmin([abs(cen[i][0] - x) for x in cx]))
        if abs(cen[i][1] - centers[r]) < 18 and abs(cen[i][0] - cx[c]) < 22:
            cells.setdefault((r, c), []).append(i)
    for r in range(33):
        for c in range(7):
            if (r, c) not in cells:
                near = [i for i in ids if abs(cen[i][1] - centers[r]) < 18 and abs(cen[i][0] - cx[c]) < 22]
                if near:
                    cells[(r, c)] = near
    missing = [(r, c) for r in range(33) for c in range(7) if (r, c) not in cells]
    assert not missing, missing[:10]
    h = np.median([max(st[i][1] + st[i][3] for i in v) - min(st[i][1] for i in v) for v in cells.values()])
    scale = 650 / h
    rowbase = [np.median([max(st[i][1] + st[i][3] for i in cells[(r, c)]) for c in range(7)]) for r in range(33)]
    ents = []
    for r in range(33):
        for c in range(7):
            v = cells[(r, c)]
            m = np.isin(lab, v).astype(np.uint8)
            x0 = min(st[i][0] for i in v)
            x1 = max(st[i][0] + st[i][2] for i in v)
            ents.append((GEEZ_BASES[r] + c, glyph(m, x0, rowbase[r], scale), (x1 - x0) * scale + 80))
    ents += punctuation("geez")
    write(out, ents, "ChartGeez")


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "lao":
        lao(load(sys.argv[2]), load(sys.argv[3]), sys.argv[4])
    elif kind == "myanmar":
        myanmar(*[load(a) for a in sys.argv[2:6]], sys.argv[6])
    elif kind == "georgian" and len(sys.argv) > 4:
        georgian(load(sys.argv[2]), sys.argv[3], load(sys.argv[4]))
    else:
        {"georgian": georgian, "armenian": armenian, "tibetan": tibetan, "gurmukhi": gurmukhi, "geez": geez}[kind](
            load(sys.argv[2]), sys.argv[3]
        )
    print("written", sys.argv[-1])
