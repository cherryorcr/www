# CaT/GECC construction-seed confirmation

Same current-delta-only edge-free protocol, 3 orders × 3 construction seeds, Prompt GCN Student seeds 150/151/152.

| Dataset | Method | Cells | Final F1 mean±SD | Accuracy mean±SD | Stage F1 mean±SD | Worst F1 |
|---|---|---:|---:|---:|---:|---:|
| bookhis | cat_cctp | 9 | 24.91±4.55 | 75.29±4.62 | 30.55±7.23 | 18.95 |
| bookhis | gecc_hard | 9 | 40.31±3.31 | 79.38±2.33 | 48.13±2.52 | 36.74 |
| sportsfit | cat_cctp | 9 | 74.39±1.55 | 83.85±2.85 | 78.89±3.99 | 72.60 |
| sportsfit | gecc_hard | 9 | 77.60±1.47 | 85.46±4.58 | 82.15±4.20 | 74.82 |
| wikics | cat_cctp | 9 | 80.89±0.59 | 83.79±0.61 | 84.29±0.83 | 79.72 |
| wikics | gecc_hard | 9 | 79.12±0.98 | 82.29±0.85 | 82.43±0.97 | 77.63 |

The table is an adapter-stability confirmation, not an official reproduction of CaT or GECC.
