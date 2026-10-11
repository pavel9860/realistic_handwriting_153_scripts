"""2D map of handwriting: real lines per script (targets) and generated lines per style as the models are added,
from font rendering (mechanical) to the full pipeline, the searched bank and the fitted personas.

The plane is the first two principal components of real lines only (ResNet18 embeddings and standardised stroke
features), so it is defined by real handwriting and generated lines are projected into it.

python scripts/map2d.py [--writers 12]   -> docs/map2d.json, docs/map2d.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import hwsynth as H
from hwsynth.config import Paths
from hwsynth.features import feats
from hwsynth.metrics import font_baseline
from hwsynth.page import line_crop
from hwsynth.realism import embed

sys.path.insert(0, str(Path(__file__).parent))
from progression import STAGES, writer_at  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STYLES = ("casual", "formal", "hurried", "calligraphic")
LATIN = ("Please send the invoice before Friday", "We met at the station near the old bridge")
OTHER = {
    "Cyrillic": "Завтра мы поедем на дачу к бабушке",
    "Japanese": "この手紙は自分の手で書きました",
    "Hebrew": "דג סקרן שט בים מאוכזב ולפתע מצא חברה",
    "Greek": "Ξεσκεπάζω την ψυχοφθόρα βδελυγμία",
    "Arabic": "صف خلق خود كمثل الشمس إذ بزغت",
}


def vec(E, F, mu, sd):
    return np.c_[E, 0.15 * (np.asarray(F) - mu) / sd]


def lines_of(stage, person, genes, style, seed, texts):
    out = []
    for t in texts:
        if stage == "font":
            c = font_baseline(t, seed)
        elif stage in ("searched bank", "fitted personas"):
            c = line_crop(H.styled(person, genes, style, seed), t, scan=None)
        else:
            c = line_crop(writer_at(stage, person, genes, style, seed), t, scan=None if stage == "scan" else False)
        f = feats(c) if c is not None and c.shape[0] > 8 else None
        if f is not None:
            out.append((c, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--writers", type=int, default=12)
    a = ap.parse_args()
    iam, web = np.load(Paths().refs), np.load(Paths().refs_web)
    E = np.vstack([iam["E"], web["E"]])
    F = np.vstack([iam["F"], web["F"]])
    label = np.r_[["Latin (IAM)"] * len(iam["E"]), [s if s != "Latin" else "Latin (web)" for s in web["script"]]]
    keep = label != "?"
    mu, sd = F[keep].mean(0), F[keep].std(0) + 1e-9
    X = vec(E[keep], F[keep], mu, sd)
    center = X.mean(0)
    _, _, vt = np.linalg.svd(X - center, full_matrices=False)
    P = vt[:2]
    proj = lambda e, f: (vec(e, f, mu, sd) - center) @ P.T
    real = {}
    for lab in sorted(set(label[keep])):
        m = label[keep] == lab
        real[lab] = dict(points=((X[m] - center) @ P.T).round(4).tolist(), n=int(m.sum()))
    stages = STAGES + ("searched bank", "fitted personas")
    paths = {}
    for st in STYLES:
        paths[st] = {}
        for stage in stages:
            got = []
            for i in range(a.writers):
                if stage == "fitted personas":
                    per, g = H.individual("iam", 90_000 + i)
                else:
                    per, g = H.individual(
                        list(H.PERSONS)[i % len(H.PERSONS)], 90_000 + i, bank=stage == "searched bank"
                    )
                got += lines_of(stage, per, g, st, 91_000 + i, LATIN)
            pts = proj(embed([c for c, _ in got]), [f for _, f in got])
            paths[st][stage] = dict(
                mean=pts.mean(0).round(4).tolist(), sem=(pts.std(0) / np.sqrt(len(pts))).round(4).tolist(), n=len(pts)
            )
            print(st, stage, paths[st][stage]["mean"], flush=True)
    scripts = {}
    for scr, text in OTHER.items():
        scripts[scr] = {}
        for stage in ("font", "writer", "hand movement", "scan", "searched bank"):
            got = []
            for i in range(a.writers):
                per, g = H.individual(list(H.PERSONS)[i % len(H.PERSONS)], 95_000 + i, bank=stage == "searched bank")
                got += lines_of(stage, per, g, "casual", 96_000 + i, (text,))
            if got:
                pts = proj(embed([c for c, _ in got]), [f for _, f in got])
                scripts[scr][stage] = dict(mean=pts.mean(0).round(4).tolist(), n=len(pts))
                print(scr, stage, scripts[scr][stage]["mean"], flush=True)
    (ROOT / "docs" / "map2d.json").write_text(json.dumps(dict(real=real, styles=paths, scripts=scripts)))


if __name__ == "__main__":
    main()
