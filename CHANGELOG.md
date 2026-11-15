# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions: [SemVer](https://semver.org/).

## 0.3.0 (2026-11-15)
### Changed
- Fonts: only Google Fonts of category Handwriting write; writer: digits and punctuation in the Latin hand, fallback
  fonts sized by their own script, optional per-letter variation (`lv_*`) and log-normal word gaps (`ws_ln`).
- Tunable numbers moved from module bodies into `config.py`; code formatted with ruff (PEP 8, line length 120), lint in CI.
### Removed
- Empty writer banks (`compact_print`, `micro`, `narrow_tall`, `school_cursive`, `thready`: they held no writer;
  those types use the prior, as before).
- `paper/figs/` (copies of `docs/` and `examples/` images; the paper includes those directly).

## 0.2.1 (2026-11-15)
- `pagegen` moved to ds-page-generation (`dspages.pagegen`): hwsynth writes handwriting lines only.

## 0.2.0 (2026-11-15)
- Apache-2.0.
