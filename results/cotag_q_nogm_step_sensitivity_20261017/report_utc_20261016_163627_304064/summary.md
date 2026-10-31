# no-GM construction convergence (report-dev only)

Completed sequences only. Values average Student seeds 150/151/152 within each construction seed.

| Dataset | Updates | Construction seeds | Final F1 | Accuracy | Time (s) | Peak allocated (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| arxiv | 30 | 3 | 51.702 | 70.957 | 225.6 | 11.09 |
| arxiv | 60 | 2 | 51.809 | 71.503 | 1525.3 | 16.31 |
| sportsfit | 30 | 3 | 79.980 | 85.969 | 153.5 | 23.77 |
| bookhis | 30 | 3 | 48.020 | 82.244 | 61.5 | 2.75 |

## Matched construction seeds

| Dataset | Updates | Seeds | Metric | no-GM30 | Longer run | Paired delta |
|---|---:|---:|---|---:|---:|---:|
| arxiv | 60 | 2 | final_f1 | 51.950 | 51.809 | -0.142 |
| arxiv | 60 | 2 | accuracy | 71.089 | 71.503 | +0.414 |

Paired deltas and construction-seed CIs: `paired_vs_30.csv`.
Per-stage objective traces, including finished stages of running cells: `stage_traces.csv`.
Pending entries: 25. Validation errors: 0.

- One class order; CI units are construction seeds, not individual Student runs.
- Partial sequences are excluded from final-cell means and paired differences.
- Loss reduction alone does not demonstrate downstream improvement or stationarity.
- Solver iteration counts are adaptive despite a common Sinkhorn configuration.
- no-GM with joint-DM disabled has no explicit text-loss-to-Q feedback.
- Current sweep protocol omits mask hash receipts; row counts and recorded fields are checked.
