"""Adds Google Fonts (OFL) families to the font manifest and downloads them: handwriting-category families for the scripts that
have some, the most popular families (print class) for scripts that have none. Needs the Google Fonts metadata JSON.

python scripts/fetch_fonts.py GF_METADATA.json [--per-script 6] [--print-per-script 4]"""
import argparse
import json
import re
import urllib.request
from pathlib import Path

from hwsynth.config import Paths
from hwsynth.fonts import MANIFEST
from hwsynth.writer import _index

SUBSET = {"Cyrillic": "cyrillic", "Greek": "greek", "Hangul": "korean", "Hiragana": "japanese", "Thai": "thai",
          "Hebrew": "hebrew", "Devanagari": "devanagari", "Arabic": "arabic", "Malayalam": "malayalam",
          "Tamil": "tamil", "Telugu": "telugu", "Han": "chinese-simplified", "Armenian": "armenian", "Bengali": "bengali",
          "Ethiopic": "ethiopic", "Georgian": "georgian", "Gujarati": "gujarati", "Gurmukhi": "gurmukhi", "Khmer": "khmer",
          "Lao": "lao", "Myanmar": "myanmar", "Sinhala": "sinhala", "Tibetan": "tibetan", "Kannada": "kannada",
          "Oriya": "oriya", "Meetei_Mayek": "meetei-mayek"}  # Thaana has no Google Fonts subset
OWN = {"Hiragana": ["Han", "Hiragana", "Katakana"]}
RAW = "https://raw.githubusercontent.com/google/fonts/main/{lic}/{dir}/{name}"


def get(url):
    return urllib.request.urlopen(url, timeout=60).read()


def file_of(family):
    """(license dir, directory, regular-weight file name) from the family's METADATA.pb, or None."""
    d = re.sub(r"[^a-z0-9]", "", family.lower())
    for lic in ("ofl", "apache", "ufl"):
        try:
            meta = get(RAW.format(lic=lic, dir=d, name="METADATA.pb")).decode()
        except Exception:
            continue
        blocks = re.findall(r'fonts \{(.*?)\}', meta, re.S)
        files = [(re.search(r'style: "(\w+)"', b), re.search(r"weight: (\d+)", b), re.search(r'filename: "([^"]+)"', b))
                 for b in blocks]
        files = [(s.group(1), int(w.group(1)), f.group(1)) for s, w, f in files if s and w and f]
        pick = [f for s, w, f in files if s == "normal" and w == 400] or [f for s, w, f in files if s == "normal"]
        return (lic, d, pick[0]) if pick else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata")
    ap.add_argument("--per-script", type=int, default=6)
    ap.add_argument("--print-per-script", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="scripts to fetch for (default: all)")
    a = ap.parse_args()
    raw = Path(a.metadata).read_text()
    fams = json.loads(raw[raw.index("{"):])["familyMetadataList"]
    paths = Paths()
    paths.fonts.mkdir(parents=True, exist_ok=True)
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    have = {e["file"] for e in man["fonts"]}
    fi = _index()
    added = []
    for script, sub in SUBSET.items():
        if a.only and script not in a.only:
            continue
        n_have = len(fi.candidates(script))
        pool = [f for f in fams if sub in f["subsets"] and f["category"] not in ("Monospace", "Symbols")]
        hand = [f for f in pool if f["category"] == "Handwriting"]
        if hand:
            want, chosen, cls = max(0, a.per_script - n_have), hand, "cursive"
        else:
            want, chosen, cls = a.print_per_script, sorted(pool, key=lambda f: f["defaultSort"]), "print"
        for fonts, kind, need in ((chosen, cls, want), (sorted(pool, key=lambda f: f["defaultSort"]), "print",
                                                       max(0, a.print_per_script - n_have))):
            for f in fonts:
                if need <= 0:
                    break
                found = file_of(f["family"])
                if not found or found[2] in have:
                    continue
                lic, d, name = found
                try:
                    (paths.fonts / name).write_bytes(
                        get(RAW.format(lic=lic, dir=d, name=name.replace("[", "%5B").replace("]", "%5D"))))
                except Exception:
                    continue
                man["fonts"].append({"file": name, "url": f"https://github.com/google/fonts/raw/main/{lic}/{d}/{name}",
                                     "license": "OFL-1.1" if lic == "ofl" else lic, "class": kind,
                                     "own": OWN.get(script, [script])})
                have.add(name)
                added.append((script, f["family"], name))
                need -= 1
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for row in added:
        print(*row)
    print(len(added), "fonts added")


if __name__ == "__main__":
    main()
