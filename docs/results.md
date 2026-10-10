# Results

Measured with `hwsynth metrics` on held-out real handwriting (test documents, 1 181 lines), with texts disjoint from the fitness texts. Lower Fréchet distance (FD) and higher precision/recall are closer to real handwriting.

## Systems (casual style)

| system | realism | FD emb. | FD feat. | precision | recall | diversity | writer id | CER | ms/line |
|---|---|---|---|---|---|---|---|---|---|
| real, other documents | – | 0.065 | 3.70 | 0.746 | 0.658 | – | – | – | – |
| font | -0.651 | 0.453 | 30.61 | 0.244 | 0.000 | 0.0991 | 0.244 | 0.078 | 5 |
| random | 0.356 | 0.250 | 20.24 | 0.688 | 0.064 | 0.1189 | 0.588 | 0.524 | 205 |
| prior | 0.331 | 0.248 | 19.94 | 0.669 | 0.023 | 0.1322 | 0.544 | 0.403 | 158 |
| bank | 0.383 | 0.245 | 19.13 | 0.688 | 0.015 | 0.1353 | 0.550 | 0.464 | 160 |
| bank_no_pen | 0.372 | 0.259 | 19.43 | 0.625 | 0.013 | 0.1325 | 0.475 | 0.455 | 150 |
| bank_no_noise | 0.342 | 0.258 | 18.73 | 0.619 | 0.014 | 0.1376 | 0.544 | 0.430 | 149 |
| bank_no_joins | 0.364 | 0.249 | 18.93 | 0.613 | 0.020 | 0.1340 | 0.475 | 0.388 | 151 |

## Styles (evolved bank), ordered by effort

| style | realism | FD emb. | movement energy | turning entropy | CER |
|---|---|---|---|---|---|
| medical | 0.574 | 0.260 | 1.3e-07 | 0.882 | 0.690 |
| hurried | 0.528 | 0.254 | 2.7e-07 | 0.926 | 0.568 |
| professional | 0.391 | 0.276 | 6.8e-07 | 0.959 | 0.428 |
| casual | 0.383 | 0.245 | 1.3e-06 | 0.971 | 0.464 |
| unsupported | 0.504 | 0.254 | 1.2e-06 | 0.966 | 0.482 |
| educated | 0.384 | 0.269 | 3.3e-06 | 0.996 | 0.507 |
| uneducated | 0.452 | 0.284 | 1.5e-06 | 0.976 | 0.285 |
| formal | 0.269 | 0.284 | 5.9e-06 | 0.999 | 0.456 |
| accurate | 0.290 | 0.322 | 8.2e-06 | 0.997 | 0.209 |
| calligraphic | 0.280 | 0.252 | 3.1e-05 | 1.109 | 0.575 |

## Distinguishability (top-1 identification, held-out texts)

| writers | features | type | individual | style |
|---|---|---|---|---|
| prior | contour hinge | 0.367 | 0.302 | 0.446 |
| prior | path signature | 0.375 | 0.504 | 0.532 |
| bank | contour hinge | 0.485 | 0.306 | 0.444 |
| bank | path signature | 0.519 | 0.486 | 0.569 |
| chance | | 0.050 | 0.0125 | 0.200 |

CER: Tesseract (English, single line), a weak reader of any handwriting; useful for comparing styles, not as absolute legibility.
