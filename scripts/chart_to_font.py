"""Glyph fonts from images of handwritten alphabet charts: letters are segmented, contours traced and written as a
TrueType font that hwsynth traces like any handwriting font.

python scripts/chart_to_font.py georgian IMAGE OUT.ttf    33 Mkhedruli letters, 3 rows of 11 (annotations in blue)
python scripts/chart_to_font.py armenian IMAGE OUT.ttf    38 capital/small pairs, 5 columns, 39 cells
python scripts/chart_to_font.py tibetan  IMAGE OUT.ttf    30 consonants in the upper bands, 5 vowel cells

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
    _, lab, c = cv2.kmeans(v, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01), 10, cv2.KMEANS_PP_CENTERS)
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
    sub[sy0 - y0:sy0 - y0 + min(y1, h) - sy0, sx0 - x0:sx0 - x0 + min(x1, w) - sx0] = mask[sy0:min(y1, h), sx0:min(x1, w)]
    for c in contours(sub):
        pts = [(round((px + x0 - x_ref) * scale + 40), round((baseline - (py + y0)) * scale)) for px, py in c]
        pen.moveTo(pts[0])
        for p in pts[1:]:
            pen.lineTo(p)
        pen.closePath()
    return pen.glyph()


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


def georgian(img, out):
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
        ents.append((ord(ch), glyph((lab == i).astype(np.uint8), st[i][0], base[k // 11], scale), st[i][2] * scale + 80))
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
    tpl = base_m[by:by + bh, bx:bx + bw]
    for cp, m, (x, y, w, h) in zip(TIBETAN_MARKS, masks[31:], boxes[31:]):
        pad = np.pad(m, 20)
        res = cv2.matchTemplate(pad.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCORR)
        _, _, _, (mx, my) = cv2.minMaxLoc(res)
        shifted = np.zeros_like(pad)
        shifted[my:my + bh, mx:mx + bw] = tpl
        mark = (pad & ~cv2.dilate(shifted, np.ones((5, 5), np.uint8)))[20:-20, 20:-20]
        n, l2, s2, _ = cv2.connectedComponentsWithStats(mark, connectivity=8)
        keep = np.isin(l2, [i for i in range(1, n) if s2[i][4] >= 20]).astype(np.uint8)
        if keep.any():
            ents.append((cp, glyph(keep, mx - 20 + x * 0 if False else mx - 20, my - 20 + bh, scale), 0))
    write(out, ents, "ChartTibetan")


if __name__ == "__main__":
    kind, image, out = sys.argv[1:4]
    {"georgian": georgian, "armenian": armenian, "tibetan": tibetan}[kind](load(image), out)
    print("written", out)
