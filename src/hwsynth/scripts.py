"""Unicode scripts (Scripts.txt), writing direction and how each script is segmented and sized."""

import bisect
import unicodedata
from functools import lru_cache
from pathlib import Path

import regex

# Scripts whose grapheme clusters are drawn one by one (each with its own jitter); all others are shaped and drawn
# word by word (joining and conjuncts come from the font).
CLUSTERED = frozenset(
    {
        "Latin",
        "Cyrillic",
        "Greek",
        "Armenian",
        "Georgian",
        "Hebrew",
        "Ethiopic",
        "Cherokee",
        "Canadian_Aboriginal",
        "Han",
        "Hiragana",
        "Katakana",
        "Hangul",
        "Bopomofo",
        "Yi",
        "Vai",
        "Tifinagh",
        "Runic",
        "Ogham",
        "Coptic",
        "Glagolitic",
        "Gothic",
        "Old_Italic",
        "Osage",
        "Deseret",
        "Shavian",
        "Bamum",
        "Adlam_unjoined",
        "Lisu",
        "Old_Persian",
        "Ugaritic",
        "Linear_B",
        "Cypriot",
        "Tai_Le",
        "Ol_Chiki",
        "Warang_Citi",
        "Bassa_Vah",
        "Mende_Kikakui",
    }
)
NOSPACE = frozenset(
    {
        "Han",
        "Hiragana",
        "Katakana",
        "Thai",
        "Lao",
        "Khmer",
        "Myanmar",
        "Tibetan",
        "Javanese",
        "Balinese",
        "Yi",
        "Tai_Tham",
        "New_Tai_Lue",
    }
)
JOINABLE = frozenset({"Latin", "Cyrillic", "Greek", "Armenian", "Georgian"})
CHARWISE = frozenset({"Han", "Hiragana", "Katakana", "Yi"})  # line breaks between any two characters
# Reference letter of a script and its height in x-heights (sets how large the script is written).
REF = {
    "Latin": ("x", 1.0),
    "Cyrillic": ("х", 1.0),
    "Greek": ("ο", 1.0),
    "Armenian": ("ս", 1.0),
    "Georgian": ("ა", 1.0),
    "Hebrew": ("ם", 1.15),
    "Arabic": ("ن", 1.3),
    "Devanagari": ("क", 1.5),
    "Bengali": ("ক", 1.5),
    "Gujarati": ("ક", 1.3),
    "Gurmukhi": ("ਕ", 1.5),
    "Tamil": ("க", 1.2),
    "Telugu": ("క", 1.4),
    "Kannada": ("ಕ", 1.4),
    "Malayalam": ("ക", 1.2),
    "Thai": ("ก", 1.15),
    "Han": ("国", 2.0),
    "Hiragana": ("あ", 1.8),
    "Katakana": ("カ", 1.8),
    "Hangul": ("한", 2.0),
    "Ethiopic": ("ሀ", 1.4),
    "Cherokee": ("Ꮳ", 1.6),
    "Canadian_Aboriginal": ("ᐃ", 1.3),
}
DEFAULT_REF = 1.4  # median letter of other scripts, in x-heights
UNIT = regex.compile(r"\X")


@lru_cache(1)
def _table():
    starts, ends, names = [], [], []
    for line in (Path(__file__).parent / "data" / "Scripts.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        rng, name = (x.strip() for x in line.split(";"))
        a, _, b = rng.partition("..")
        starts.append(int(a, 16))
        ends.append(int(b or a, 16))
        names.append(name)
    order = sorted(range(len(starts)), key=starts.__getitem__)
    return [starts[i] for i in order], [ends[i] for i in order], [names[i] for i in order]


@lru_cache(65536)
def script(ch):
    """Unicode script of a character ('Common' and 'Inherited' for shared characters and combining marks)."""
    starts, ends, names = _table()
    i = bisect.bisect_right(starts, ord(ch)) - 1
    return names[i] if i >= 0 and ord(ch) <= ends[i] else "Unknown"


def all_scripts():
    return sorted(set(_table()[2]) - {"Common", "Inherited", "Unknown"})


def letters(name, limit=None):
    """Letters (categories L and M) of a script, in code point order."""
    starts, ends, names = _table()
    out = [
        chr(c)
        for s, e, n in zip(starts, ends, names)
        if n == name
        for c in range(s, e + 1)
        if unicodedata.category(chr(c))[0] in "LM"
    ]
    return out[:limit] if limit else out


def text_script(text):
    """Most frequent script of the letters in text, 'Common' when there are none."""
    count = {}
    for ch in text:
        s = script(ch)
        if s not in ("Common", "Inherited", "Unknown"):
            count[s] = count.get(s, 0) + 1
    return max(count, key=count.get) if count else "Common"


def is_rtl(ch):
    return unicodedata.bidirectional(ch) in ("R", "AL")


def text_rtl(text):
    n = sum(map(is_rtl, text))
    return n > 0 and n >= sum(unicodedata.bidirectional(c) == "L" for c in text)


def units(word, scr):
    """Drawing units of a word: grapheme clusters for clustered scripts, else the whole word."""
    return UNIT.findall(word) if scr in CLUSTERED else [word]


def visual_units(word, scr):
    """Units in visual (left-to-right) order: right-to-left runs reversed, digits and Latin kept in order."""
    us = units(word, scr)
    if not any(is_rtl(c) for c in word) or len(us) == 1:
        return us
    runs, cur, cur_rtl = [], [], None
    for u in us:
        r = not any(unicodedata.bidirectional(c) in ("L", "EN", "AN") for c in u)
        if cur and r != cur_rtl:
            runs.append((cur_rtl, cur))
            cur = []
        cur.append(u)
        cur_rtl = r
    runs.append((cur_rtl, cur))
    return [u for r, run in runs[::-1] for u in (run[::-1] if r else run)]
