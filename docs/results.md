# Results

Measured with `hwsynth metrics` on held-out real handwriting (test documents), with texts disjoint from the search texts. Lower Fréchet distance (FD) and higher precision/recall are closer to real handwriting. Realism is a z-score against real lines; it saturates above 0 and treats scanner noise as atypical.

![progression](progression.png)

![search](search.png)

## Systems (casual style)

| system | realism | FD emb. | FD feat. | precision | recall | diversity | writer id | CER | ms/line |
|---|---|---|---|---|---|---|---|---|---|
| real, other documents | – | 0.024 | 2.93 | 0.717 | 0.634 | – | – | – | – |
| font | -5.247 | 0.415 | 228.90 | 0.100 | 0.001 | 0.0941 | 0.106 | 0.048 | 11 |
| random | -1.984 | 0.152 | 21.89 | 0.081 | 0.350 | 0.1696 | 0.356 | 0.587 | 731 |
| prior | -2.159 | 0.146 | 22.05 | 0.069 | 0.303 | 0.1753 | 0.338 | 0.451 | 518 |
| bank | -2.024 | 0.139 | 37.78 | 0.138 | 0.281 | 0.1722 | 0.231 | 0.417 | 372 |
| personas | -2.129 | 0.129 | 109.67 | 0.125 | 0.325 | 0.1705 | 0.263 | 0.390 | 298 |
| bank_no_scan | -1.729 | 0.242 | 21.91 | 0.194 | 0.001 | 0.1215 | 0.356 | 0.363 | 352 |
| bank_no_pen | -2.030 | 0.150 | 32.57 | 0.106 | 0.278 | 0.1739 | 0.294 | 0.393 | 286 |
| bank_no_noise | -2.041 | 0.150 | 32.46 | 0.087 | 0.401 | 0.1745 | 0.225 | 0.441 | 315 |
| bank_no_joins | -2.723 | 0.128 | 216.11 | 0.119 | 0.201 | 0.1728 | 0.331 | 0.374 | 296 |

## Styles (searched bank), ordered by effort

| style | realism | FD emb. | movement energy | turning entropy | CER |
|---|---|---|---|---|---|
| medical | -2.355 | 0.189 | 1.6e-07 | 0.911 | 0.584 |
| hurried | -2.462 | 0.165 | 3.2e-07 | 0.924 | 0.522 |
| professional | -2.059 | 0.160 | 7.5e-07 | 0.961 | 0.370 |
| casual | -2.024 | 0.139 | 1.6e-06 | 0.974 | 0.417 |
| unsupported | -1.873 | 0.152 | 1.5e-06 | 0.970 | 0.441 |
| educated | -1.889 | 0.165 | 3.7e-06 | 0.998 | 0.383 |
| uneducated | -1.767 | 0.158 | 2.2e-06 | 0.978 | 0.234 |
| formal | -2.239 | 0.164 | 6.8e-06 | 1.000 | 0.408 |
| accurate | -1.776 | 0.177 | 9e-06 | 0.996 | 0.164 |
| calligraphic | -2.578 | 0.158 | 3.4e-05 | 1.159 | 0.569 |

## Distinguishability (top-1 identification, held-out texts)

| writers | features | type | individual | style |
|---|---|---|---|---|
| prior | contour hinge | 0.312 | 0.384 | 0.442 |
| prior | path signature | 0.292 | 0.542 | 0.484 |
| bank | contour hinge | 0.453 | 0.355 | 0.473 |
| bank | path signature | 0.522 | 0.447 | 0.541 |
| chance | | 0.042 | 0.0104 | 0.200 |

CER: Tesseract (English, single line), a weak reader of any handwriting; useful for comparing styles.
