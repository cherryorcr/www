# Master experiment report (2026-10-11)

This index separates the sealed primary comparison, construction-seed baseline confirmation, SAGE transfer, additional datasets, and the older topology-enabled link-prediction audit.

## Primary matched Prompt Student

| Dataset | Method | Final F1 | Accuracy | Old F1 | New F1 | Orders |
|---|---|---:|---:|---:|---:|---:|
| bookhis | Unified-Q (edge-free) | 41.97 | 80.34 | 41.39 | 48.38 | 3 |
| bookhis | GECC-hard TAG adapter | 41.23 | 79.97 | 40.81 | 45.83 | 3 |
| bookhis | CaT-CCTP adapter | 24.83 | 75.45 | 24.47 | 28.75 | 3 |
| bookhis | Adaptive coherent attribution | 23.47 | 74.61 | 23.16 | 26.81 | 3 |
| sportsfit | Unified-Q (edge-free) | 78.60 | 83.93 | 78.58 | 78.74 | 3 |
| sportsfit | GECC-hard TAG adapter | 77.32 | 85.52 | 77.61 | 75.70 | 3 |
| sportsfit | CaT-CCTP adapter | 73.75 | 81.94 | 73.54 | 74.95 | 3 |
| sportsfit | Adaptive coherent attribution | 74.54 | 82.98 | 74.24 | 76.17 | 3 |
| wikics | Unified-Q (edge-free) | 79.86 | 82.85 | 79.97 | 78.83 | 3 |
| wikics | GECC-hard TAG adapter | 79.08 | 82.22 | 79.06 | 79.21 | 3 |
| wikics | CaT-CCTP adapter | 80.35 | 83.44 | 80.51 | 78.88 | 3 |
| wikics | Adaptive coherent attribution | 80.47 | 83.61 | 80.60 | 79.27 | 3 |

## Matched external adapter seed confirmation

| Dataset | Adapter | Cells | Final F1 mean±SD | Accuracy mean±SD | Worst F1 |
|---|---|---:|---:|---:|---:|
| bookhis | cat_cctp | 9 | 24.91±4.55 | 75.29±4.62 | 18.95 |
| bookhis | gecc_hard | 9 | 40.31±3.31 | 79.38±2.33 | 36.74 |
| sportsfit | cat_cctp | 9 | 74.39±1.55 | 83.85±2.85 | 72.60 |
| sportsfit | gecc_hard | 9 | 77.60±1.47 | 85.46±4.58 | 74.82 |
| wikics | cat_cctp | 9 | 80.89±0.59 | 83.79±0.61 | 79.72 |
| wikics | gecc_hard | 9 | 79.12±0.98 | 82.29±0.85 | 77.63 |

## Prompt SAGE transfer

| Dataset | Method | Cells | Final F1 mean±SD | Accuracy mean±SD | Worst F1 |
|---|---|---:|---:|---:|---:|
| bookhis | cat_cctp | 9 | 29.24±4.05 | 63.26±14.92 | 23.46 |
| bookhis | gecc_hard | 9 | 35.50±4.32 | 69.45±8.08 | 28.36 |
| bookhis | uq_dm_edgefree | 9 | 36.83±4.22 | 71.23±6.48 | 29.33 |
| bookhis | uq_qmedoid | 9 | 35.58±4.57 | 70.17±7.13 | 28.45 |
| sportsfit | cat_cctp | 9 | 77.39±2.97 | 84.83±2.67 | 74.15 |
| sportsfit | gecc_hard | 9 | 74.91±1.93 | 81.86±3.44 | 73.10 |
| sportsfit | uq_dm_edgefree | 9 | 75.27±2.06 | 80.00±1.79 | 72.72 |
| sportsfit | uq_qmedoid | 9 | 75.29±2.81 | 80.56±2.66 | 71.05 |
| wikics | cat_cctp | 9 | 80.91±0.88 | 83.36±0.78 | 79.77 |
| wikics | gecc_hard | 9 | 78.74±1.48 | 80.80±1.63 | 77.39 |
| wikics | uq_dm_edgefree | 9 | 79.66±1.72 | 81.63±1.98 | 77.55 |
| wikics | uq_qmedoid | 9 | 79.42±1.53 | 81.51±1.62 | 77.92 |

## Additional datasets

| Dataset | Method | Orders | Final F1 mean±SD | Accuracy mean±SD |
|---|---|---:|---:|---:|
| arxiv | cat_cctp | 1 | 36.74±0.00 | 63.62±0.00 |
| arxiv | gecc_hard | 1 | 51.09±0.00 | 70.27±0.00 |
| arxiv | uq_dm_edgefree | 1 | 52.25±0.00 | 71.59±0.00 |
| cora | cat_cctp | 3 | 82.76±1.26 | 84.75±0.77 |
| cora | gecc_hard | 3 | 83.08±1.16 | 85.94±0.91 |
| cora | uq_dm_edgefree | 3 | 82.44±0.57 | 84.50±0.43 |

## Link prediction scope

The existing link-prediction artifact is the topology-enabled 2026-09-20 protocol and is not merged into the edge-free Unified-Q primary claim.

| Dataset | C0 AUC | Joint-kernel AUC | Degree-balanced AUC | Coverage AUC |
|---|---:|---:|---:|---:|
| cora | 85.90 | 84.68 | 87.11 | 87.57 |
| citeseer | 87.00 | 87.04 | 90.31 | 90.33 |

## Interpretation guardrails

- External methods are current-delta-only adapters, not official reimplementations unless explicitly marked.
- Prompt Student and Legacy Student numbers are separate; the Student main effect is not a condensation-method gain.
- Additional ArXiv/Cora rows are supplementary transfer checks and are not used to tune the primary method.
- The edge-free primary method does not support a direct link-prediction claim; the topology-enabled link-prediction artifact is reported as a separate scope audit.
