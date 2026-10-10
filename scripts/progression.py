"""Realism of handwriting as the models are switched on one after another (from font rendering to the full
pipeline), per style, against the held-out real lines; writes progression.json and a sample sheet.

python scripts/progression.py OUT_DIR [--writers 16]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import hwsynth as H
from hwsynth.config import GENE_INDEX
from hwsynth.features import feats
from hwsynth.metrics import TEXTS_EN, font_baseline
from hwsynth.page import line_crop
from hwsynth.pen import Pen
from hwsynth.realism import Realism, embed

STAGES = ("font", "writer", "variation", "hand movement", "pen and ink", "personal letters", "scan")
NOISE = ("elastic", "wobble", "tremor", "slant_sd", "size_sd", "rot_sd", "base_sd", "ls_sd", "wander")


def writer_at(stage, person, genes, style, seed):
    k = STAGES.index(stage)
    g = genes.copy()
    if k < 5:
        g[GENE_INDEX["letters"]] = 0.0
    w = H.styled(person, g, style, seed)
    if k < 2:
        for n in NOISE:
            w.p[n] = 0.0
    if k < 3:
        w._move = lambda strokes, xh: None
    if k < 4:
        w.pen = Pen.sample(np.random.default_rng(seed), {"fineliner": 1})
        w.p["press_sd"] = w.p["down"] = 0.0
    return w


def crop(stage, person, genes, style, seed, text):
    if stage == "font":
        return font_baseline(text, seed)
    return line_crop(writer_at(stage, person, genes, style, seed), text, scan=None if stage == "scan" else False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--writers", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    real = Realism("test")
    styles = ("casual", "formal", "hurried", "calligraphic")
    texts = TEXTS_EN[:2]
    res, sheet = {}, []
    for st in styles:
        res[st] = {}
        for stage in STAGES:
            crops, fs = [], []
            for i in range(a.writers):
                per, g = H.individual(list(H.PERSONS)[i % len(H.PERSONS)], 70_000 + i, bank=False)
                for t in texts:
                    c = crop(stage, per, g, st, 80_000 + i, t)
                    f = feats(c) if c is not None and c.shape[0] > 8 else None
                    if f is not None:
                        crops.append(c)
                        fs.append(f)
            q = real.lines(embed(crops), fs)
            res[st][stage] = dict(mean=float(q.mean()), sem=float(q.std() / np.sqrt(len(q))), n=len(q))
            print(st, stage, round(q.mean(), 3), flush=True)
    per, g = H.individual("school_cursive", 7, bank=False)
    for stage in STAGES:
        c = crop(stage, per, g, "casual", 5, "The quick brown fox jumps over the lazy dog")
        c = cv2.resize(c, (int(c.shape[1] * 60 / c.shape[0]), 60))
        row = np.full((60, 1100), 255, np.uint8)
        row[:, 170 : 170 + min(c.shape[1], 930)] = c[:, :930]
        cv2.putText(row, stage, (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 60, 1, cv2.LINE_AA)
        sheet.append(np.pad(row, ((0, 6), (0, 0)), constant_values=255))
    cv2.imwrite(str(out / "progression_samples.jpg"), np.vstack(sheet))
    (out / "progression.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
