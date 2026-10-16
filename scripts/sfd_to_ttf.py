import sys
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

src, out = sys.argv[1:3]
chars, cur = [], None
for line in open(src, encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    if line.startswith("StartChar:"):
        cur = dict(name=line.split(":", 1)[1].strip(), uni=-1, gid=len(chars), width=500, paths=[], refs=[])
        mode = None
    elif cur is None:
        continue
    elif line.startswith("Encoding:"):
        p = line.split()
        cur["uni"], cur["gid"] = int(p[1]), int(p[3])
    elif line.startswith("Width:"):
        cur["width"] = int(line.split()[1])
    elif line.startswith("Refer:"):
        p = line.split()
        cur["refs"].append((int(p[1]), tuple(map(float, p[4:10]))))
    elif line == "SplineSet":
        mode = "s"
    elif line == "EndSplineSet":
        mode = None
    elif line.startswith("EndChar"):
        chars.append(cur)
        cur = None
    elif mode == "s":
        p = line.split()
        if p and p[-2:-1] and p[-2] in ("m", "l", "c"):
            cur["paths"].append((p[-2], list(map(float, p[:-2]))))
by_gid = {c["gid"]: c for c in chars}

def draw(c, pen, depth=0):
    open_ = False
    for op, v in c["paths"]:
        if op == "m":
            if open_:
                pen.closePath()
            pen.moveTo((v[0], v[1])); open_ = True
        elif op == "l":
            pen.lineTo((v[0], v[1]))
        else:
            pen.curveTo((v[0], v[1]), (v[2], v[3]), (v[4], v[5]))
    if open_:
        pen.closePath()
    if depth < 4:
        for gid, m in c["refs"]:
            if gid in by_gid:
                draw(by_gid[gid], TransformPen(pen, (m[0], m[1], m[2], m[3], m[4], m[5])), depth + 1)

glyphs, cmap, metrics, order = {}, {}, {}, [".notdef"]
glyphs[".notdef"] = TTGlyphPen(None).glyph(); metrics[".notdef"] = (500, 0)
for c in chars:
    if c["uni"] < 0:
        continue
    name = f"uni{c['uni']:04X}" if c["uni"] <= 0xFFFF else f"u{c['uni']:X}"
    if name in glyphs:
        continue
    pen = TTGlyphPen(None)
    draw(c, Cu2QuPen(pen, 1.0, reverse_direction=True))
    glyphs[name] = pen.glyph(); metrics[name] = (c["width"], 0); cmap[c["uni"]] = name; order.append(name)
fb = FontBuilder(1000, isTTF=True)
fb.setupGlyphOrder(order)
fb.setupCharacterMap(cmap)
fb.setupGlyf(glyphs)
fb.setupHorizontalMetrics({n: (w, 0) for n, (w, _) in metrics.items()})
fb.setupHorizontalHeader(ascent=800, descent=-200)
fb.setupNameTable({"familyName": "Fontoshan Handwrite", "styleName": "Regular"})
fb.setupOS2(sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200)
fb.setupPost()
fb.save(out)
sin = [u for u in cmap if 0x0D80 <= u <= 0x0DFF]
print(len(cmap), "codepoints,", len(sin), "Sinhala:", f"U+{min(sin):04X}-U+{max(sin):04X}" if sin else "none")
