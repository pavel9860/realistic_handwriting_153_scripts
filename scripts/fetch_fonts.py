"""Downloads the handwriting fonts of src/hwsynth/data/fonts.json into the font
directory (HWSYNTH_FONTS, default ~/.cache/hwsynth/fonts). Existing files are kept.

python scripts/fetch_fonts.py [--workers 8]
"""

import argparse
import json
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hwsynth.config import Paths

MANIFEST = Path(__file__).resolve().parents[1] / "src" / "hwsynth" / "data" / "fonts.json"


def fetch(entry, out):
    dst = out / entry["file"]
    if dst.exists() and dst.stat().st_size:
        return None
    if "local" in entry:
        shutil.copy(MANIFEST.parent / entry["local"], dst)
        return None
    tmp = dst.with_suffix(".part")
    try:
        with urllib.request.urlopen(urllib.parse.quote(entry["url"], safe=":/"), timeout=60) as r:
            tmp.write_bytes(r.read())
        tmp.rename(dst)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return f"{entry['file']}: {e}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = Paths().fonts
    out.mkdir(parents=True, exist_ok=True)
    entries = m["fonts"]
    with ThreadPoolExecutor(a.workers) as ex:
        errors = [e for e in ex.map(lambda x: fetch(x, out), entries) if e]
    print(f"{len(entries) - len(errors)}/{len(entries)} fonts in {out}")
    for e in errors:
        print("failed", e)


if __name__ == "__main__":
    main()
