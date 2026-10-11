"""Writes the LaTeX inputs of paper/ from docs/metrics.json, the references, the personas, the configuration and the
example sheets.

python scripts/paper_inputs.py
"""

import json
import shutil
from pathlib import Path

import numpy as np

from hwsynth.config import GENES, IDENTITY, PERSONS, STYLES, TOOLS, Paths
from hwsynth.styles import personas

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
NAMES = {
    "font": "font rendering",
    "random": "random writers",
    "prior": "type priors",
    "bank": "searched bank",
    "personas": "fitted personas",
    "bank_no_scan": "bank, no scan model",
    "bank_no_pen": "bank, no pen model",
    "bank_no_noise": "bank, no variation",
    "bank_no_joins": "bank, no joins",
}
ORDER = (
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
)
TEX = str.maketrans({"_": r"\_", "&": r"\&", "%": r"\%"})


def f(x, d=3):
    return "--" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def write(name, text):
    (PAPER / name).write_text(text if text.endswith("\n") else text + "\n")


def results(rows):
    real = next(r for r in rows if r["system"].startswith("real"))
    sysr = {r["system"]: r for r in rows if r["system"] in NAMES and r.get("style") == "casual"}
    by_style = {r["style"]: r for r in rows if r["system"] == "bank"}
    dist = {r["bank"]: r for r in rows if r["system"] == "distinguishability"}
    bank, font = sysr["bank"], sysr["font"]
    d = dist.get(True, {})
    gap = (font["fd_embed"] - bank["fd_embed"]) / (font["fd_embed"] - real["fd_embed"])
    write(
        "abstract_numbers.tex",
        f"On held-out IAM lines the searched bank reaches an embedding Fr\\'echet distance of {bank['fd_embed']:.3f}, "
        f"closing {100 * gap:.0f}\\,\\% of the gap between font rendering ({font['fd_embed']:.3f}) and real "
        f"handwriting of other forms ({real['fd_embed']:.3f}); its {d.get('types', len(PERSONS))} types are "
        f"identified with {100 * d.get('pen_path_type', 0):.0f}\\,\\% top-1 accuracy (chance "
        f"{100 * d.get('chance_type', 0):.0f}\\,\\%).",
    )
    out = [
        r"\begin{table*}[t]\centering\small",
        r"\begin{tabular}{lrrrrrrrrr}\toprule",
        r"system & realism & FD emb. & FD feat. & precision & recall & diversity & writer id & CER & s/line\\\midrule",
        f"real (other forms) & -- & {f(real['fd_embed'])} & {f(real['fd_feats'], 2)} & {f(real['precision'])} & "
        f"{f(real['recall'])} & -- & -- & -- & --\\\\",
    ]
    for k in NAMES:
        if k in sysr:
            r = sysr[k]
            out.append(
                f"{NAMES[k]} & {f(r['realism'])} & {f(r['fd_embed'])} & {f(r['fd_feats'], 2)} & "
                f"{f(r['precision'])} & {f(r['recall'])} & {f(r['diversity'], 4)} & {f(r['writer_id'])} & "
                f"{f(r['cer'])} & {r['ms_per_line'] / 1000:.2f}\\\\"
            )
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Systems against held-out IAM lines (casual style, texts disjoint from search and fitting). Lower "
        r"FD and higher precision and recall are closer to real handwriting.}",
        r"\label{tab:systems}\end{table*}",
        "",
        r"\begin{table}[t]\centering\small\setlength\tabcolsep{3pt}",
        r"\begin{tabular}{lrrrrr}\toprule style & realism & FD emb. & energy & entropy & CER\\\midrule",
    ]
    for s in ORDER:
        if s in by_style:
            r = by_style[s]
            out.append(
                f"{s} & {f(r['realism'])} & {f(r['fd_embed'])} & {r['energy']:.2g} & {f(r['entropy'])} & "
                f"{f(r['cer'])}\\\\"
            )
    out += [
        r"\bottomrule\end{tabular}",
        r"\caption{Searched bank per style, ordered by effort: movement energy (jerk per unit length of the planned "
        r"path), turning entropy (bits) and Tesseract CER.}",
        r"\label{tab:styles}\end{table}",
        "",
        r"\begin{table}[t]\centering\small\setlength\tabcolsep{3pt}",
        r"\begin{tabular}{llrrr}\toprule writers & features & type & individual & style\\\midrule",
    ]
    for b, r in sorted(dist.items()):
        for feat, name in (("image", "contour hinge"), ("pen_path", "path signature")):
            out.append(
                f"{'bank' if b else 'prior'} & {name} & {f(r[feat + '_type'])} & {f(r[feat + '_individual'])} "
                f"& {f(r[feat + '_style'])}\\\\"
            )
    if dist:
        r = next(iter(dist.values()))
        out.append(f"chance & & {f(r['chance_type'])} & {f(r['chance_individual'], 4)} & {f(r['chance_style'])}\\\\")
    para = (
        r"\paragraph{Results.} Table~\ref{tab:systems}: the searched bank closes "
        f"{100 * gap:.0f}\\,\\% of the embedding-FD gap between font rendering and real handwriting and raises "
        f"precision from {f(font['precision'], 2)} to {f(bank['precision'], 2)}. "
    )
    if "personas" in sysr:
        para += (
            f"The fitted personas reach FD {f(sysr['personas']['fd_embed'])} and precision "
            f"{f(sysr['personas']['precision'], 2)}. "
        )
    if "bank_no_scan" in sysr:
        para += (
            f"Without the scan model FD rises to {f(sysr['bank_no_scan']['fd_embed'])} and recall falls to "
            f"{f(sysr['bank_no_scan']['recall'])}, while realism, which treats scanner noise as atypical of the "
            f"clean IAM scans, moves up. "
        )
    abl = [
        (k, NAMES[k].split(", no ")[1].replace("pen model", "the pen model"))
        for k in ("bank_no_noise", "bank_no_joins", "bank_no_pen")
        if k in sysr
    ]
    worse = [n for k, n in abl if sysr[k]["realism"] < bank["realism"] and sysr[k]["precision"] < bank["precision"]]
    better_fd = [n for k, n in abl if sysr[k]["fd_embed"] < bank["fd_embed"]]
    if worse:
        names = ", ".join(worse[:-1]) + (" or " if len(worse) > 1 else "") + worse[-1]
        para += f"Removing {names} lowers both realism and precision"
        para += (
            f"; without {', '.join(better_fd)} the embedding FD is slightly lower, so FD alone does not "
            f"rank these ablations. "
            if better_fd
            else ". "
        )
    para += (
        r"Table~\ref{tab:styles}: movement energy "
        r"rises with effort from medical to calligraphic writing, entropy with it; accurate writing is the most "
        r"legible for Tesseract and medical writing the least. Table~\ref{tab:dist}: the search makes the types "
        r"much easier to tell apart while individuals stay identifiable."
    )
    out += [
        r"\bottomrule\end{tabular}",
        f"\\caption{{Top-1 identification on held-out texts ({d.get('types', len(PERSONS))} types, 5 styles): a "
        f"type from other individuals, an individual from its other texts, a style from other individuals.}}",
        r"\label{tab:dist}\end{table}",
        "",
        para,
    ]
    write("results.tex", "\n".join(out))


def numbers():
    iam = np.load(Paths().refs)
    write(
        "iam_numbers.tex",
        f"{len(iam['doc'])} lines from {len(set(iam['doc']))} forms "
        f"({int((~iam['test']).sum())} fit, {int(iam['test'].sum())} test).",
    )
    ps = personas()
    if not ps:
        write("persona_numbers.tex", "")
        return
    dd = np.array([p["distance"] for p in ps])
    types = {}
    for p in ps:
        types[p["type"]] = types.get(p["type"], 0) + 1
    top = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(types.items(), key=lambda x: -x[1])[:5])
    write(
        "persona_numbers.tex",
        f"{len(ps)} personas were fitted (median distance {np.median(dd):.3f}, interquartile range "
        f"{np.percentile(dd, 25):.3f}--{np.percentile(dd, 75):.3f}); their nearest types are spread over "
        f"{len(types)} of the {len(PERSONS)} types (most frequent: {top}).",
    )


def search_numbers():
    import re

    log = (ROOT / "docs" / "qd_log.txt").read_text()
    it = [
        tuple(float(x) for x in m.groups())
        for m in re.finditer(
            r"iter (\d+): added \d+, cells (\d+), quality median ([-\d.]+), instability median ([\d.]+)", log
        )
    ]
    thr = re.search(r"bank threshold: realism >= ([-\d.]+)", log)
    sizes = [int(x) for x in re.findall(r" (\d+)(?:,|$)", log.split("bank sizes:")[-1])] if "bank sizes" in log else []
    first, last = it[0], it[-1]
    text = (
        f"over {len(it)} iterations the median elite quality rises from {first[2]:.2f} to {last[2]:.2f}, "
        f"median baseline instability stays at {last[3]:.2f}, within the range of clean real lines, and the archive "
        f"fills {int(last[1])} cells. "
    )
    if thr and sizes:
        text += (
            f"The final re-evaluation (threshold: realism $\\ge {float(thr.group(1)):.2f}$, the median) banks "
            f"{sum(sizes)} individuals, {min(sizes)}--{max(sizes)} per type; types without banked individuals "
            f"fall back to their priors."
        )
    write("search_numbers.tex", text)


def tables():
    g = [
        r"\begin{table}[h]\centering\footnotesize",
        r"\begin{tabular}{lllp{0.55\textwidth}}\toprule",
        r"gene & range & kind & meaning\\\midrule",
    ]
    for gene in GENES:
        g.append(
            f"{gene.name.translate(TEX)} & {gene.lo:g}--{gene.hi:g} & "
            f"{'identity' if gene.name in IDENTITY else 'situation'} & {gene.doc.translate(TEX)}\\\\"
        )
    g += [
        r"\bottomrule\end{tabular}",
        f"\\caption{{The {len(GENES)} writer genes.}}",
        r"\label{tab:genes}\end{table}",
        r"\clearpage",
        r"\begin{table}[h]\centering\footnotesize",
        r"\begin{tabular}{lllp{0.6\textwidth}}\toprule type & fonts & habits & description\\\midrule",
    ]
    for p in PERSONS.values():
        hab = ", ".join(k.replace("_", " ") for k, v in p.habits.items() if v >= 0.5) or "--"
        g.append(f"{p.name.translate(TEX)} & {p.fonts} & {hab} & {p.doc.translate(TEX)}\\\\")
    g += [r"\midrule style & tools & paper & description\\\midrule"]
    for s in STYLES.values():
        tools = ", ".join(k.replace("_", " ") for k in list(s.pens)[:3]) or "all"
        g.append(f"{s.name} & {tools} & {s.paper} & {s.doc.translate(TEX)}\\\\")
    g += [
        r"\bottomrule\end{tabular}",
        f"\\caption{{The {len(PERSONS)} handwriting types and {len(STYLES)} styles.}}",
        r"\label{tab:types}\end{table}",
        r"\begin{table}[h]\centering\footnotesize",
        r"\begin{tabular}{llllll}\toprule tool & nib & ink & width (mm) & edge (mm) & density\\\midrule",
    ]
    for t in TOOLS.values():
        edge = f"{t.edge[0]:g}--{t.edge[1]:g}" if t.nib == "broad" else "--"
        g.append(
            f"{t.name.translate(TEX)} & {t.nib} & {t.ink} & {t.width[0]:g}--{t.width[1]:g} & {edge} & "
            f"{t.density[0]:g}--{t.density[1]:g}\\\\"
        )
    g += [r"\bottomrule\end{tabular}", f"\\caption{{The {len(TOOLS)} writing tools.}}", r"\label{tab:tools}\end{table}"]
    write("types.tex", "\n".join(g))


def figures():
    figs = PAPER / "figs"
    figs.mkdir(exist_ok=True)
    for name in ("map2d.png", "progression.png", "progression_samples.jpg", "search.png"):
        if (ROOT / "docs" / name).exists():
            shutil.copy(ROOT / "docs" / name, figs / name)
    caps = {
        "persons": "One individual of each of the 24 types (casual style).",
        "styles_school_cursive": "One school-cursive individual in the 10 styles.",
        "scripts": "Random individuals writing 32 scripts.",
        "math_chemistry": "Inline mathematics and chemistry.",
        "figures": "Hand-drawn figures by four types.",
        "tools": "The 17 tools at light and strong push.",
        "hands": "Right and left hand (gel, fountain pen, pencil): upright slant and smudges.",
    }
    fx = []
    for name, cap in caps.items():
        src = ROOT / "examples" / f"{name}.jpg"
        if src.exists():
            shutil.copy(src, figs / src.name)
            fx += [
                r"\begin{figure}[h]\centering",
                f"\\includegraphics[width=0.95\\textwidth,height=0.88\\textheight,keepaspectratio]{{figs/{src.name}}}",
                f"\\caption{{{cap}}}\\end{{figure}}",
                r"\clearpage",
            ]
    write("figures.tex", "\n".join(fx))


def main():
    results(json.loads((ROOT / "docs" / "metrics.json").read_text()))
    numbers()
    search_numbers()
    tables()
    figures()


if __name__ == "__main__":
    main()
