"""Writes the LaTeX inputs of paper/ from metrics.json, the configuration and the example sheets.

python scripts/paper_inputs.py METRICS.json
"""

import json
import shutil
import sys
from pathlib import Path

from hwsynth.config import PERSONS, STYLES, TOOLS

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
NAMES = {
    "font": "font rendering",
    "random": "random writers",
    "prior": "type priors",
    "bank": "evolved bank",
    "bank_no_pen": "bank, no pen model",
    "bank_no_noise": "bank, no shape variation",
    "bank_no_joins": "bank, no joins",
}


def f(x, d=3):
    return "--" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def main(path):
    rows = json.loads(Path(path).read_text())
    real = next(r for r in rows if r["system"].startswith("real"))
    sys_rows = [r for r in rows if r["system"] in NAMES and r["style"] == "casual"]
    style_rows = [r for r in rows if r["system"] == "bank"]
    dist = {r["bank"]: r for r in rows if r["system"] == "distinguishability"}
    bank = next(r for r in sys_rows if r["system"] == "bank")
    font = next(r for r in sys_rows if r["system"] == "font")
    d = dist.get(True, {})
    (PAPER / "abstract_numbers.tex").write_text(
        f"On held-out real handwriting the evolved bank reaches a Fr\\'echet embedding distance of "
        f"{bank['fd_embed']:.3f} against {font['fd_embed']:.3f} for font rendering and {real['fd_embed']:.3f} between "
        f"two halves of the real data, and its twenty types are identified with "
        f"{100 * d.get('pen_path_type', 0):.0f}\\,\\% top-1 accuracy (chance 5\\,\\%).\n"
    )
    out = [
        r"\subsection{Realism and diversity}",
        r"Table~\ref{tab:systems} compares the systems on the held-out test documents. Lower Fr\'echet distances "
        r"(FD) and higher precision and recall mean closer to real handwriting; the first row is the reference "
        r"level of real handwriting against other real handwriting.",
        r"\begin{table*}[t]\centering\small",
        r"\begin{tabular}{lrrrrrrrrr}\toprule",
        r"system & realism & FD emb. & FD feat. & precision & recall & diversity & writer id & CER & ms/line\\\midrule",
        f"real (other documents) & -- & {f(real['fd_embed'])} & {f(real['fd_feats'], 2)} & {f(real['precision'])} & "
        f"{f(real['recall'])} & -- & -- & -- & --\\\\",
    ]
    for r in sys_rows:
        out.append(
            f"{NAMES[r['system']]} & {f(r['realism'])} & {f(r['fd_embed'])} & {f(r['fd_feats'], 2)} & "
            f"{f(r['precision'])} & {f(r['recall'])} & {f(r['diversity'], 4)} & {f(r['writer_id'])} & "
            f"{f(r['cer'])} & {f(r['ms_per_line'], 0)}\\\\"
        )
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Systems against held-out real handwriting (casual style, texts disjoint from the fitness texts).}",
        r"\label{tab:systems}\end{table*}",
        "",
        r"\subsection{Styles and effort}",
        r"Table~\ref{tab:styles} lists the bank per style. Movement energy (jerk per unit length of the planned pen "
        r"path) and turning entropy rise with the effort a style spends on letter shapes, from medical scrawl to "
        r"calligraphy; legibility follows.",
        r"\begin{table}[t]\centering\small\setlength\tabcolsep{3pt}",
        r"\begin{tabular}{lrrrrr}\toprule style & realism & FD emb. & energy & entropy & CER\\\midrule",
    ]
    order = [
        "medical",
        "hurried",
        "professional",
        "casual",
        "unsupported",
        "educated",
        "uneducated",
        "formal",
        "accurate",
        "calligraphic",
    ]
    by = {r["style"]: r for r in style_rows}
    for s in order:
        if s in by:
            r = by[s]
            out.append(
                f"{s} & {f(r['realism'])} & {f(r['fd_embed'])} & {r['energy']:.2g} & {f(r['entropy'])} & {f(r['cer'])}\\\\"
            )
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Evolved bank per style, ordered by effort.}",
        r"\label{tab:styles}\end{table}",
        "",
        r"\subsection{Distinguishable types, individuals and styles}",
        r"Table~\ref{tab:dist} reports top-1 nearest-neighbour identification on held-out texts: a type must be "
        r"recognised from \emph{other} individuals of the type, an individual from its \emph{other} texts, a "
        r"style from \emph{other} individuals.",
        r"\begin{table}[t]\centering\small\setlength\tabcolsep{3pt}",
        r"\begin{tabular}{llrrr}\toprule writers & features & type & individual & style\\\midrule",
    ]
    for b, r in sorted(dist.items()):
        for feat, name in (("image", "contour hinge"), ("pen_path", "path signature")):
            out.append(
                f"{'bank' if b else 'prior'} & {name} & {f(r[feat + '_type'])} & {f(r[feat + '_individual'])} & "
                f"{f(r[feat + '_style'])}\\\\"
            )
    if dist:
        r = next(iter(dist.values()))
        out.append(f"chance & & {f(r['chance_type'])} & {f(r['chance_individual'], 4)} & {f(r['chance_style'])}\\\\")
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Identification accuracy (20 types, 5 styles).}",
        r"\label{tab:dist}\end{table}",
    ]
    (PAPER / "results.tex").write_text("\n".join(out) + "\n")
    t = [r"\begin{table}[h]\centering\small", r"\begin{tabular}{lll}\toprule type & fonts & description\\\midrule"]
    t += [f"{p.name.replace('_', ' ')} & {p.fonts} & {p.doc}\\\\" for p in PERSONS.values()]
    t += [r"\midrule style & tools & description\\\midrule"]
    t += [
        f"{s.name} & {', '.join(k.replace('_', ' ') for k in list(s.pens)[:3]) or 'all'} & {s.doc}\\\\"
        for s in STYLES.values()
    ]
    t += [
        r"\bottomrule\end{tabular}",
        r"\caption{The 20 handwriting types and 10 styles; "
        f"{len(TOOLS)} tools.}}",
        r"\label{tab:types}\end{table}",
    ]
    (PAPER / "types.tex").write_text("\n".join(t) + "\n")
    figs = PAPER / "figs"
    figs.mkdir(exist_ok=True)
    caps = {
        "persons": "One individual of each of the 20 types (casual style).",
        "styles_school_cursive": "One school-cursive individual in the 10 styles.",
        "scripts": "Random individuals writing 32 scripts.",
        "tools": "The 17 tools at light and strong push.",
        "hands": "Right hand and left hand (gel, fountain pen, pencil): upright slant and smudges.",
    }
    fx = []
    for name, cap in caps.items():
        src = ROOT / "examples" / f"{name}.jpg"
        if src.exists():
            shutil.copy(src, figs / src.name)
            fx += [
                r"\begin{figure}[h]\centering",
                f"\\includegraphics[width=0.95\\textwidth,height=0.9\\textheight,keepaspectratio]{{figs/{src.name}}}",
                f"\\caption{{{cap}}}\\end{{figure}}",
                r"\clearpage",
            ]
    (PAPER / "figures.tex").write_text("\n".join(fx) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
