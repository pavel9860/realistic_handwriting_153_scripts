"""hwsynth command line.

hwsynth write "text" -o out.png [--person P --style S --seed N --width 150 --dpi 200]
hwsynth examples -o examples/                 persons, styles and scripts sheets
hwsynth coverage                              scripts that can be written (font coverage of Scripts.txt)
hwsynth refs PDF_DIR LABELS.pkl               real line references (fit / test split)
hwsynth search [--person P] [--iters N]       quality-diversity search of the writer banks
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

from .config import PERSONS, STYLES, QDCfg, Paths

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
    _math_figures(out, a.dpi, a.prior)


MATH_SAMPLES = (
    r"Solve $x^{2}-5x+6=0$: $x=\frac{5\pm 1}{2}$, so $x_1=3$, $x_2=2$.",
    r"Area $A=\pi r^{2}$, diagonal $d=\sqrt{a^{2}+b^{2}}$, and $\int_{0}^{1} x^{2} dx=\frac{1}{3}$",
    r"Combustion: $\ce{CH4 + 2O2 -> CO2 + 2H2O}$",
    r"Neutralisation: $\ce{H2SO4 + 2NaOH -> Na2SO4 + 2H2O}$, ions $\ce{SO4^{2-}}$ and $\ce{Na+}$",
    r"Equilibrium $\ce{N2 + 3H2 <=> 2NH3}$ and benzene $\benzene$ $\ce{C6H6}$",
)


def _math_figures(out, dpi, prior):
    from .figures import FIGURES, figure
    from .page import write
    from .styles import individual, styled

    rows = []
    for j, (t, p) in enumerate(
        zip(
            MATH_SAMPLES * 2,
            (
                "school_cursive",
                "rounded_neat",
                "compact_print",
                "elderly",
                "looped_cursive",
                "print_script",
                "bubbly_print",
                "ruled_neat",
                "sloppy_fast",
                "soft_upright",
            ),
        )
    ):
        per, g = individual(p, 300 + j, bank=not prior)
        rows.append(
            _label(write(t, styled(per, g, ("casual", "formal", "accurate")[j % 3], 310 + j), 150, dpi, False).image, p)
        )
    cv2.imwrite(str(out / "math_chemistry.jpg"), _stack(rows))
    rows = []
    for j, p in enumerate(("rounded_neat", "school_cursive", "child", "elderly")):
        per, g = individual(p, 400 + j, bank=not prior)
        cells = [
            figure(k, styled(per, g, "casual", 410 + 20 * j + i), size=45 if k == "axes" else 20, dpi=dpi).image
            for i, k in enumerate(FIGURES)
        ]
        cells = [cv2.resize(c, (int(c.shape[1] * 110 / c.shape[0]), 110)) for c in cells]
        rows.append(_label(np.hstack([np.pad(c, ((0, 0), (0, 6), (0, 0)), constant_values=255) for c in cells]), p))
    cv2.imwrite(str(out / "figures.jpg"), _stack(rows))


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


def cmd_search(a):
    from .qd import search
    from .realism import Realism

    search(
        Realism("fit"),
        [a.person] if a.person else None,
        QDCfg(iters=a.iters),
        workers=a.workers,
        log=lambda m: print(m, flush=True),
    )


def cmd_metrics(a):
    from .metrics import SYSTEMS, distinguishability, evaluate, real_baseline
    from .realism import Realism

    r = Realism("test")
    rows = []

    def add(row):  # saved after every row: a crash never loses finished work
        rows.append(row)
        print(row, flush=True)
        Path(a.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    add(real_baseline(r))
    for s in SYSTEMS:
        add(evaluate(s, r, r, a.writers))
    for st in STYLES:
        if st != "casual":
            add(evaluate("bank", r, r, a.writers // 2, st))
    for bank in (False, True):
        add(dict(system="distinguishability", bank=bank, **distinguishability(bank=bank)))


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
    p = sp.add_parser("search")
    p.add_argument("--person", choices=list(PERSONS))
    p.add_argument("--iters", type=int, default=QDCfg().iters)
    p.add_argument("--workers", type=int, default=8)
    p = sp.add_parser("metrics")
    p.add_argument("-o", "--out", default=str(Paths().data / "metrics.json"))
    p.add_argument("--writers", type=int, default=60)
    a = ap.parse_args(argv)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
