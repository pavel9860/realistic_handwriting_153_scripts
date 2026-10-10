"""Sheets of every script the fonts cover: real sentences where available, else pseudo-words of the script's letters.

python scripts/all_scripts.py [OUT_DIR]
"""

import sys
import cv2
import numpy as np
import hwsynth as H
from hwsynth.cli import coverage, SAMPLES, _label
from hwsynth.fonts import core_letters
from hwsynth.scripts import text_script

real = {text_script(t): t for t in SAMPLES.values()}
cov = coverage()
rows = []
for i, (scr, kind) in enumerate(sorted(cov.items())):
    rng = np.random.default_rng(i)
    if scr in real:
        text = real[scr]
    else:
        ls = list(core_letters(scr))
        text = " ".join("".join(rng.choice(ls, rng.integers(2, 6))) for _ in range(5))
    per, g = H.individual(None, 900 + i)
    try:
        im = H.write(text, H.styled(per, g, "casual", 950 + i), 120, 90, False).image
    except Exception as e:
        print(scr, e)
        continue
    im = cv2.resize(im, (im.shape[1], im.shape[0]))[:150]
    rows.append(_label(im, f"{scr} ({kind})"))
W = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 150 - r.shape[0] + 2), (0, W - r.shape[1] + 4), (0, 0)), constant_values=255) for r in rows]
cols = 4
per = -(-len(rows) // cols)
rows += [np.full_like(rows[0], 255)] * (per * cols - len(rows))
sheet = np.hstack([np.vstack(rows[c * per : (c + 1) * per]) for c in range(cols)])
OUT = sys.argv[1] if len(sys.argv) > 1 else "examples"
for k in range(3):
    part = sheet[k * sheet.shape[0] // 3 : (k + 1) * sheet.shape[0] // 3]
    cv2.imwrite(f"{OUT}/all_scripts_{k + 1}.jpg", part, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(len(rows))
