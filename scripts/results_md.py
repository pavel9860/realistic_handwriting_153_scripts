"""Writes docs/results.md from docs/metrics.json.

python scripts/results_md.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def f(x, d=3):
    return "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def main():
    R = json.loads((ROOT / "docs" / "metrics.json").read_text())
    L = [
        "# Results",
        "",
        "Measured with `hwsynth metrics` on held-out real handwriting (test documents), with texts "
        "disjoint from the search texts. Lower Fréchet distance (FD) and higher precision/recall are closer to real "
        "handwriting. Realism is a z-score against real lines; it saturates above 0 and treats scanner noise as "
        "atypical.",
        "",
        "![progression](progression.png)",
        "",
        "![search](search.png)",
        "",
        "## Systems (casual style)",
        "",
        "| system | realism | FD emb. | FD feat. | precision | recall | diversity | writer id | CER | ms/line |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in R:
        if r["system"].startswith("real"):
            L.append(
                f"| real, other documents | – | {f(r['fd_embed'])} | {f(r['fd_feats'], 2)} | {f(r['precision'])} | "
                f"{f(r['recall'])} | – | – | – | – |"
            )
        elif r["system"] != "distinguishability" and r["style"] == "casual":
            L.append(
                f"| {r['system']} | {f(r['realism'])} | {f(r['fd_embed'])} | {f(r['fd_feats'], 2)} | "
                f"{f(r['precision'])} | {f(r['recall'])} | {f(r['diversity'], 4)} | {f(r['writer_id'])} | "
                f"{f(r['cer'])} | {f(r['ms_per_line'], 0)} |"
            )
    L += [
        "",
        "## Styles (searched bank), ordered by effort",
        "",
        "| style | realism | FD emb. | movement energy | turning entropy | CER |",
        "|---|---|---|---|---|---|",
    ]
    by = {r["style"]: r for r in R if r["system"] == "bank"}
    for s in ORDER:
        if s in by:
            r = by[s]
            L.append(
                f"| {s} | {f(r['realism'])} | {f(r['fd_embed'])} | {r['energy']:.2g} | {f(r['entropy'])} | "
                f"{f(r['cer'])} |"
            )
    L += [
        "",
        "## Distinguishability (top-1 identification, held-out texts)",
        "",
        "| writers | features | type | individual | style |",
        "|---|---|---|---|---|",
    ]
    last = None
    for r in R:
        if r["system"] == "distinguishability":
            last = r
            for k, n in (("image", "contour hinge"), ("pen_path", "path signature")):
                L.append(
                    f"| {'bank' if r['bank'] else 'prior'} | {n} | {f(r[k + '_type'])} | "
                    f"{f(r[k + '_individual'])} | {f(r[k + '_style'])} |"
                )
    if last:
        L.append(
            f"| chance | | {f(last['chance_type'])} | {f(last['chance_individual'], 4)} | {f(last['chance_style'])} |"
        )
    L += [
        "",
        "CER: Tesseract (English, single line), a weak reader of any handwriting; useful for comparing styles.",
        "",
    ]
    (ROOT / "docs" / "results.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
