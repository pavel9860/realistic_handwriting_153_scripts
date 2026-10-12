# Local fonts

| File | Source | License |
|---|---|---|
| SinhalaHandwriteConverted-Regular.ttf | FontForge source of Fontoshan Handwrite (skyprolk/Fontoshan-Sinhala-English-Handwrite), converted | SIL OFL 1.1, see OFL-Fontoshan.txt |
| ChartGeorgian-Regular.ttf | traced from an image of a handwritten Mkhedruli chart with stroke-order arrows (33 letters) and a Nuskhuri stroke-order chart (38 letters, U+2D00..2D25) | provenance and licence of the image not known |
| ChartArmenian-Regular.ttf | traced from an image of a handwritten Armenian alphabet (38 capital/small pairs) carrying an Adobe Stock watermark | stock image; licence not known |
| ChartTibetan-Regular.ttf | traced from an image of a Tibetan cursive chart (30 consonants, vowel signs i u e o) | provenance and licence of the image not known |

The chart fonts are made by `scripts/chart_to_font.py`. They hold the letters of the charts only: no digits,
punctuation or conjuncts; characters without a glyph are skipped when writing.
| ChartLao-Regular.ttf | consonants traced from a photo of handwritten Lao letters; vowel signs cut from a handwritten syllable chart on ກ (vowels in red) | provenance and licence of the images not known |
| ChartMyanmar-Regular.ttf | 34 letters from the Mrauk-U column of an Arakanese script chart (hand-drawn historical forms; mapped by Brahmic order and checked on a labelled sheet); vowel signs, medials, tone marks and numerals from typeset Myanmar charts (typeset forms) | provenance and licence of the images not known |
| ChartGurmukhi-Regular.ttf | 35 letters traced from a handwritten manuscript alphabet page (Panjab Digital Library) | the library's terms apply; not checked |
| ChartGeez-Regular.ttf | 231 syllables (33 consonants x 7 orders) traced from a chart | provenance and licence of the image not known |

The Armenian ligature և (U+0587) is composed of the traced ե and ւ; the Armenian full stop and comma, the Tibetan tsheg and shad,
and the Ethiopic word space, full stop and comma are drawn as dots and bars by `scripts/chart_to_font.py` (not traced).
Latin digits and ASCII punctuation come from the Latin handwriting fonts (a unit without a glyph falls back to any handwriting
font that has it).
