# Results

Measured with `hwsynth metrics` on held-out real handwriting (test documents), with texts disjoint from the search texts. Lower Fréchet distance (FD) and higher precision/recall are closer to real handwriting. Realism is a z-score against real lines; it saturates above 0 and treats scanner noise as atypical.

![progression](progression.png)

![search](search.png)

## Systems (casual style)

| system | realism | FD emb. | FD feat. | precision | recall | diversity | writer id | CER | ms/line |
|---|---|---|---|---|---|---|---|---|---|
| real, other documents | – | 0.065 | 3.70 | 0.746 | 0.658 | – | – | – | – |
| font | -0.428 | 0.433 | 28.11 | 0.338 | 0.000 | 0.0941 | 0.106 | 0.048 | 10 |
| random | 0.242 | 0.180 | 18.36 | 0.519 | 0.070 | 0.1702 | 0.350 | 0.587 | 536 |
| prior | 0.280 | 0.160 | 17.13 | 0.594 | 0.082 | 0.1755 | 0.338 | 0.451 | 369 |
| bank | 0.402 | 0.158 | 16.45 | 0.688 | 0.064 | 0.1679 | 0.338 | 0.390 | 309 |
| bank_no_scan | 0.480 | 0.249 | 18.70 | 0.800 | 0.016 | 0.1202 | 0.350 | 0.385 | 461 |
| bank_no_pen | 0.403 | 0.171 | 16.38 | 0.662 | 0.047 | 0.1618 | 0.287 | 0.366 | 307 |
| bank_no_noise | 0.376 | 0.169 | 16.12 | 0.600 | 0.057 | 0.1702 | 0.325 | 0.396 | 445 |
| bank_no_joins | 0.346 | 0.157 | 16.32 | 0.606 | 0.065 | 0.1729 | 0.287 | 0.328 | 308 |

## Styles (searched bank), ordered by effort

| style | realism | FD emb. | movement energy | turning entropy | CER |
|---|---|---|---|---|---|
| medical | 0.239 | 0.201 | 1.7e-07 | 0.908 | 0.662 |
| hurried | 0.376 | 0.174 | 3.2e-07 | 0.931 | 0.544 |
| professional | 0.347 | 0.195 | 7.6e-07 | 0.950 | 0.415 |
| casual | 0.402 | 0.158 | 1.5e-06 | 0.974 | 0.390 |
| unsupported | 0.457 | 0.173 | 1.5e-06 | 0.966 | 0.394 |
| educated | 0.350 | 0.175 | 3.7e-06 | 0.996 | 0.407 |
| uneducated | 0.312 | 0.197 | 1.8e-06 | 0.968 | 0.249 |
| formal | 0.331 | 0.176 | 6.4e-06 | 1.000 | 0.348 |
| accurate | 0.423 | 0.203 | 9e-06 | 0.998 | 0.172 |
| calligraphic | 0.194 | 0.191 | 3.9e-05 | 1.203 | 0.548 |

## Distinguishability (top-1 identification, held-out texts)

| writers | features | type | individual | style |
|---|---|---|---|---|
| prior | contour hinge | 0.312 | 0.384 | 0.442 |
| prior | path signature | 0.292 | 0.542 | 0.484 |
| bank | contour hinge | 0.520 | 0.342 | 0.510 |
| bank | path signature | 0.582 | 0.454 | 0.604 |
| chance | | 0.042 | 0.0104 | 0.200 |

CER: Tesseract (English, single line), a weak reader of any handwriting; useful for comparing styles.
