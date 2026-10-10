"""Font index: handwriting fonts (manifest) first, Noto and system fonts as fallback; per-script candidate lists."""

import json
import unicodedata
import pickle
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

from .config import GlyphCfg, Paths
from .scripts import REF, letters

MANIFEST = Path(__file__).parent / "data" / "fonts.json"
SUFFIXES = (".ttf", ".otf", ".ttc")


@lru_cache(1)
def manifest():
    """{file name: entry}; entry: class (cursive | print | brush), own (scripts the font is limited to),
    prefer ({script: weight}), skip (never used)."""
    return {e["file"]: e for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["fonts"]}


def _files(paths):
    out = sorted(p for p in paths.fonts.rglob("*") if p.suffix.lower() in SUFFIXES) if paths.fonts.exists() else []
    for d in paths.system_fonts:
        d = Path(d)
        if d.exists():
            out += sorted(p for p in d.rglob("*") if p.suffix.lower() in SUFFIXES and _system_ok(p.name))
    return out


def _system_ok(name):
    n = name.lower()
    return (
        "regular" in n
        and not any(k in n for k in ("mono", "display", "ui-", "ui.", "condensed"))
        or n in ("dejavusans.ttf",)
    )


class FontIndex:
    """Fonts with their character sets. kind: hand (in the manifest) or fallback."""

    def __init__(self, paths=Paths()):
        self.paths = paths
        files = _files(paths)
        cache = paths.cache / "font_index.pkl"
        key = [(str(f), f.stat().st_mtime) for f in files]
        data = None
        if cache.exists():
            data = pickle.loads(cache.read_bytes())
            data = data["fonts"] if data["key"] == key else None
        if data is None:
            from fontTools.ttLib import TTFont

            data = {}
            for f in files:
                try:
                    data[str(f)] = frozenset(TTFont(f, fontNumber=0, lazy=True).getBestCmap())
                except Exception:
                    continue
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps({"key": key, "fonts": data}))
        man = manifest()
        self.cmap = data
        self.meta = {f: man.get(Path(f).name, {}) for f in data}
        self.hand = [f for f in data if Path(f).name in man and not self.meta[f].get("skip")]
        self.fallback = [f for f in data if Path(f).name not in man]

    def covers(self, font, text):
        m = self.cmap[font]
        return all(ord(c) in m for c in text if not c.isspace())

    @lru_cache(4096)
    def candidates(self, script, cls="any"):
        """Handwriting fonts for a script (class-filtered when any match), else fallback fonts covering it."""
        core = core_letters(script)
        if not core:
            return ()
        need = GlyphCfg().coverage * len(core)

        def ok(f):
            own = self.meta[f].get("own")
            return (not own or script in own) and sum(ord(c) in self.cmap[f] for c in core) >= need

        childish = cls.endswith("+childish")
        cls = cls.removesuffix("+childish")
        hand = [f for f in self.hand if ok(f) and (childish or not self.meta[f].get("childish"))]
        if cls != "any":
            hand = [f for f in hand if self.meta[f].get("class") == cls] or hand
        return tuple(hand) or tuple(f for f in self.fallback if ok(f))

    def choose(self, script, u, cls="any"):
        """Font of a writer for a script; u in [0, 1) is the writer's font gene (shifted per script)."""
        cands = self.candidates(script, cls)
        if not cands:
            return None
        h = (u + (sum(map(ord, script)) % 97) * 0.6180339) % 1
        pref = [(f, self.meta[f].get("prefer", {}).get(script, 0)) for f in cands]
        share = max((w for _, w in pref), default=0)
        if share and h < share:
            top = [f for f, w in pref if w]
            return top[int(h / share * len(top))]
        return cands[int(h * len(cands))]

    @lru_cache(65536)
    def for_unit(self, font, unit):
        """font if it has every character of unit, else the first handwriting, then fallback font that has."""
        if font is not None and self.covers(font, unit):
            return font
        for f in self.hand + self.fallback:
            if self.covers(f, unit):
                return f
        return None


@lru_cache(512)
def core_letters(script):
    """Letters that a font must cover to serve a script: basic alphabets where the script has one, else up to
    48 letters spread over the script's range."""
    basic = {
        "Latin": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "Cyrillic": "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
        "Greek": "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
        "Han": "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可",
        "Hangul": "가나다라마바사아자차카타파하한글이는을를에서의고하다",
        "Arabic": "ابتثجحخدذرزسشصضطظعغفقكلمنهوي",
        "Hiragana": "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん",
        "Katakana": "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン",
        "Syriac": "ܐܒܓܕܗܘܙܚܛܝܟܠܡܢܣܥܦܨܩܪܫܬ",
    }
    if script in basic:
        return basic[script]
    ls = [c for c in letters(script) if c.isalpha() and not unicodedata.decomposition(c)]  # base letters only
    if script in REF and REF[script][0] not in ls:
        ls.insert(0, REF[script][0])
    if len(ls) > 48:
        ls = ls[:: len(ls) // 48][:48]
    return "".join(ls)


@lru_cache(256)
def pil_font(path, px):
    return ImageFont.truetype(path, px, layout_engine=ImageFont.Layout.RAQM)


@lru_cache(4096)
def em_per_xh(path, script, px):
    """Font em over the x-height the script is written at: height of the script's reference letter (or the median
    letter of the script) in em divided by that letter's height in x-heights."""
    from .scripts import DEFAULT_REF

    f = pil_font(path, px)
    ch, k = REF.get(script, (None, DEFAULT_REF))
    cands = [ch] if ch else [c for c in core_letters(script)[:16]]
    hs = []
    for c in cands:
        try:
            b = f.getbbox(c, anchor="ls")
        except Exception:
            continue
        if b[3] > b[1]:
            hs.append((b[3] - b[1]) / px)
    if not hs:
        return 1 / (f.getmetrics()[0] / px * 0.55)
    hs.sort()
    return k / max(hs[len(hs) // 2], 0.2)
