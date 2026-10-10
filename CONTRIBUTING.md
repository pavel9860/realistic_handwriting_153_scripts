# Contributing

Thanks for helping. Issues and pull requests are welcome.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,evolve]"
python scripts/fetch_fonts.py
pytest -q
```

## Guidelines

- **Numbers live in `src/hwsynth/config.py`.** Gene ranges, tools, persons, styles, motion, evolution: add a field
  there rather than a constant in code.
- **Keep it lean.** Small focused functions, no dead code, comments only where the reason is not obvious.
- **Determinism.** Everything random takes a `numpy.random.Generator`; the same seed must give the same output.
- **Tests.** Add a test for every behaviour you change (`tests/`); keep them fast (no long renders).
- **Fonts.** New fonts go into `src/hwsynth/data/fonts.json` with their URL and licence (OFL, Apache or similar);
  font files are never committed.
- **Data.** Do not commit scans or personal handwriting. Real-line references are built locally (`hwsynth refs`).

## Useful additions

- Handwriting fonts for scripts that only have Noto fallbacks (see `hwsynth coverage --list`).
- New handwriting types (`PERSONS`) or styles (`STYLES`) grounded in a source (forensic, palaeographic, pedagogical).
- Better writer-similarity measures (`writerid.py`) and realism references.

By contributing you agree that your contribution is licensed under CC BY-NC 4.0.
