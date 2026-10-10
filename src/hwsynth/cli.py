"""hwsynth command line.

hwsynth write "text" -o out.png [--person P --style S --seed N --width 150 --dpi 200]
hwsynth examples -o examples/                 persons, styles and scripts sheets
hwsynth coverage                              scripts that can be written (font coverage of Scripts.txt)
hwsynth refs PDF_DIR LABELS.pkl               real line references (fit / test split)
hwsynth evolve [--person P] [--gens N]        evolves the writer banks
hwsynth metrics -o metrics.json               metrics of all systems and styles
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one thread per process: the
    os.environ.setdefault(_v, "1")  # evolution runs one process per core
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .config import PERSONS, STYLES, EvolveCfg, Paths

SAMPLES = {
    "Latin (en)": "The quick brown fox jumps over the lazy dog.",
    "Latin (de)": "Zwölf Boxkämpfer jagen Viktor quer über den Sylter Deich.",
    "Latin (pl)": "Zażółć gęślą jaźń, napisz list do babci.",
    "Latin (vi)": "Tiếng Việt có dấu: Hà Nội mùa thu.",
    "Cyrillic (ru)": "Съешь же ещё этих мягких французских булок.",
    "Cyrillic (kk)": "Қазақ тілі: әдемі ғажайып өлке.",
    "Greek": "Ξεσκεπάζω την ψυχοφθόρα βδελυγμία.",
    "Armenian": "Բարև ձեզ, ես գրում եմ ձեռքով։",
    "Georgian": "გამარჯობა, ეს წერილი ხელით არის დაწერილი.",
    "Hebrew": "דג סקרן שט בים מאוכזב ולפתע מצא חברה.",
    "Arabic": "صِف خلق خود كمثل الشمس إذ بزغت.",
    "Persian": "این نامه را با دست نوشتم.",
    "Devanagari": "मैं यह पत्र अपने हाथ से लिख रहा हूँ।",
    "Bengali": "আমি নিজের হাতে এই চিঠি লিখছি।",
    "Tamil": "நான் இந்தக் கடிதத்தை என் கையால் எழுதுகிறேன்.",
    "Telugu": "నేను ఈ ఉత్తరాన్ని నా చేతితో రాస్తున్నాను.",
    "Gujarati": "હું આ પત્ર મારા હાથે લખું છું.",
    "Thai": "ฉันเขียนจดหมายฉบับนี้ด้วยมือของฉันเอง",
    "Han (zh)": "我用手写这封信，希望你一切都好。",
    "Japanese": "この手紙は自分の手で書きました。",
    "Hangul": "이 편지는 제 손으로 직접 썼습니다.",
    "Ethiopic": "ይህን ደብዳቤ በእጄ ጻፍኩት።",
    "Cherokee": "ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ ᎠᏍᎦᏯ ᎤᏬᏪᎳᏅ.",
    "Canadian syllabics": "ᐃᓄᒃᑎᑐᑦ ᐊᓪᓚᖑᐊᖅ ᐅᖃᐅᓯᖅ.",
    "Tifinagh": "ⵜⴰⵎⴰⵣⵉⵖⵜ ⵜⵓⵔⴰ ⵜⵜⵓⵔⴰ.",
    "Mongolian": "ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ",
    "Khmer": "ខ្ញុំសរសេរសំបុត្រនេះដោយដៃ",
    "Sinhala": "මම මේ ලිපිය අතින් ලිව්වා.",
    "Myanmar": "ကျွန်တော် ဒီစာကို လက်နဲ့ ရေးတယ်။",
    "Tibetan": "ངས་ཡི་གེ་འདི་ལག་པས་བྲིས།",
    "Syriac": "ܟܬܒܬ ܐܓܪܬܐ ܗܕܐ ܒܐܝܕܝ",
    "Thaana": "މި ސިޓީ އަތުން ލިޔުނީ",
}


def _stack(rows, gap=4):
    w = max(r.shape[1] for r in rows)
    return np.vstack([np.pad(r, ((0, gap), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows])


def _label(img, text):
    img = img[..., ::-1].copy()
    cv2.putText(img, text, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 190), 1, cv2.LINE_AA)
    return img


def cmd_write(a):
    from .page import write
    from .styles import sample_writer

    s = write(a.text, sample_writer(a.person, a.style, a.seed), a.width, a.dpi, a.ruled)
    cv2.imwrite(a.out, s.image[..., ::-1])
    Path(a.out).with_suffix(".json").write_text(json.dumps(dict(s.meta, words=s.words), ensure_ascii=False))


def cmd_examples(a):
    from .page import write
    from .styles import individual, styled

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    text = "The quick brown fox jumps over the lazy dog. Call me at 5:30!"
    rows = []
    for i, p in enumerate(PERSONS):
        per, g = individual(p, i, bank=not a.prior)
        rows.append(_label(write(text, styled(per, g, "casual", i), 130, a.dpi, False).image, p))
    cv2.imwrite(str(out / "persons.jpg"), _stack(rows))
    for p in a.persons.split(","):
        per, g = individual(p, 7, bank=not a.prior)
        rows = [
            _label(write(text, styled(per, g, s, 100 + j), 130, a.dpi, False).image, f"{p} / {s}")
            for j, s in enumerate(STYLES)
        ]
        cv2.imwrite(str(out / f"styles_{p}.jpg"), _stack(rows))
    rows = []
    for j, (name, t) in enumerate(SAMPLES.items()):
        per, g = individual(None, 500 + j, bank=not a.prior)
        rows.append(_label(write(t, styled(per, g, "casual", 600 + j), 150, a.dpi).image, name))
    cv2.imwrite(str(out / "scripts.jpg"), _stack(rows))
    from .config import GENE_INDEX, TOOLS
    from .pen import Pen

    rows = []
    per, g = individual("print_script", 11, bank=not a.prior)
    for j, t in enumerate(TOOLS):
        for push in (0.1, 0.9):
            gg = g.copy()
            gg[GENE_INDEX["press"]] = push
            w = styled(per, gg, "casual", 700 + j)
            w.pen = Pen.sample(np.random.default_rng(j), {t: 1})
            rows.append(
                _label(
                    write("Natural handwriting, every tool", w, 110, a.dpi, False).image,
                    f"{t}, {'strong' if push > 0.5 else 'light'} push",
                )
            )
    cv2.imwrite(
        str(out / "tools.jpg"), np.vstack([np.hstack(_pad_same(rows[i : i + 2])) for i in range(0, len(rows), 2)])
    )
    rows = []
    for hand, tool in ((0.0, "gel"), (1.0, "gel"), (1.0, "fountain_medium"), (1.0, "pencil_2b")):
        per, g = individual("print_script", 4, bank=not a.prior)
        g = g.copy()
        g[GENE_INDEX["hand"]] = hand
        w = styled(per, g, "casual", 9)
        w.pen = Pen.sample(np.random.default_rng(3), {tool: 1})
        rows.append(
            _label(
                write("The left hand writes this line; the right hand does not.", w, 120, a.dpi, False).image,
                f"{'left' if w.left else 'right'} hand, {tool}",
            )
        )
    cv2.imwrite(str(out / "hands.jpg"), _stack(rows))


def _pad_same(rows):
    h = max(r.shape[0] for r in rows)
    w = max(r.shape[1] for r in rows)
    return [np.pad(r, ((0, h - r.shape[0]), (0, w - r.shape[1] + 6), (0, 0)), constant_values=255) for r in rows]


def coverage():
    """{script: font kind} for scripts with a handwriting font ('hand') or only a fallback ('fallback')."""
    from .scripts import all_scripts
    from .writer import _index

    fi = _index()
    out = {}
    for s in all_scripts():
        c = fi.candidates(s)
        if c:
            out[s] = "hand" if c[0] in fi.hand else "fallback"
    return out


def cmd_coverage(a):
    c = coverage()
    from .scripts import all_scripts

    print(
        f"{len(c)} of {len(all_scripts())} Unicode scripts: {sum(v == 'hand' for v in c.values())} with "
        f"handwriting fonts, {sum(v == 'fallback' for v in c.values())} through fallback fonts"
    )
    if a.list:
        for s, v in sorted(c.items()):
            print(f"{s:28s} {v}")


def cmd_refs(a):
    from .realism import build_refs

    build_refs(a.pdf_dir, a.labels, pages=a.pages)


def cmd_evolve(a):
    from .evolve import evolve_all
    from .realism import Realism

    log = lambda m: print(m, flush=True)
    evolve_all(
        Realism("fit"),
        [a.person] if a.person else None,
        EvolveCfg(gens=a.gens, pop=a.pop),
        workers=a.workers,
        cap=a.cap,
        log=log,
    )


def cmd_metrics(a):
    from .metrics import SYSTEMS, distinguishability, evaluate, real_baseline
    from .realism import Realism

    r = Realism("test")
    rows = [real_baseline(r)]
    print(rows[-1], flush=True)
    for s in SYSTEMS:
        rows.append(evaluate(s, r, r, a.writers))
        print(rows[-1], flush=True)
    for st in STYLES:
        if st != "casual":
            rows.append(evaluate("bank", r, r, a.writers // 2, st))
            print(rows[-1], flush=True)
    for bank in (False, True):
        rows.append(dict(system="distinguishability", bank=bank, **distinguishability(bank=bank)))
        print(rows[-1], flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hwsynth", description="Natural synthetic handwriting in any Unicode script.")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("write")
    p.add_argument("text")
    p.add_argument("-o", "--out", default="handwriting.png")
    p.add_argument("--person", choices=list(PERSONS))
    p.add_argument("--style", choices=list(STYLES), default="casual")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=float, default=150.0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--ruled", action=argparse.BooleanOptionalAction, default=None)
    p = sp.add_parser("examples")
    p.add_argument("-o", "--out", default="examples")
    p.add_argument("--persons", default="school_cursive,block_capitals,elderly")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--prior", action="store_true", help="ignore the evolved banks")
    p = sp.add_parser("coverage")
    p.add_argument("--list", action="store_true")
    p = sp.add_parser("refs")
    p.add_argument("pdf_dir")
    p.add_argument("labels")
    p.add_argument("--pages", type=int)
    p = sp.add_parser("evolve")
    p.add_argument("--person", choices=list(PERSONS))
    p.add_argument("--gens", type=int, default=EvolveCfg().gens)
    p.add_argument("--pop", type=int, default=EvolveCfg().pop)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cap", type=int, default=200)
    p = sp.add_parser("metrics")
    p.add_argument("-o", "--out", default=str(Paths().data / "metrics.json"))
    p.add_argument("--writers", type=int, default=60)
    a = ap.parse_args(argv)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
