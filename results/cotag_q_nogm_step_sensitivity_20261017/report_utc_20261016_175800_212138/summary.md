# no-GM construction convergence (report-dev only)

Completed sequences only. Values average Student seeds 150/151/152 within each construction seed.

| Dataset | Updates | Construction seeds | Final F1 | Accuracy | Time (s) | Peak allocated (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| arxiv | 30 | 3 | 51.702 | 70.957 | 225.6 | 11.09 |
| arxiv | 60 | 2 | 51.809 | 71.503 | 1525.3 | 16.31 |
| sportsfit | 30 | 3 | 79.980 | 85.969 | 153.5 | 23.77 |
| bookhis | 30 | 3 | 48.020 | 82.244 | 61.5 | 2.75 |
| bookhis | 60 | 2 | 45.685 | 82.343 | 282.0 | 5.35 |
| bookhis | 100 | 2 | 46.356 | 82.174 | 722.8 | 5.80 |

## Matched construction seeds

| Dataset | Updates | Seeds | Metric | no-GM30 | Longer run | Paired delta |
|---|---:|---:|---|---:|---:|---:|
| arxiv | 60 | 2 | final_f1 | 51.950 | 51.809 | -0.142 |
| arxiv | 60 | 2 | accuracy | 71.089 | 71.503 | +0.414 |
| bookhis | 60 | 2 | final_f1 | 47.934 | 45.685 | -2.249 |
| bookhis | 60 | 2 | accuracy | 82.306 | 82.343 | +0.036 |
| bookhis | 100 | 2 | final_f1 | 47.934 | 46.356 | -1.578 |
| bookhis | 100 | 2 | accuracy | 82.306 | 82.174 | -0.132 |

## Practical loss plateau

A stage is marked plateaued only when the last two consecutive 10-update windows each reduce the observed objective by less than 1%. This is a practical stopping rule, not a mathematical convergence proof.

Plateaued stage traces: 20/65.
The detailed 10-update reductions are recorded in `stage_traces.csv`.

Paired deltas and construction-seed CIs: `paired_vs_30.csv`.
Per-stage objective traces, including finished stages of running cells: `stage_traces.csv`.
Pending entries: 15. Failed cells: 6. Validation errors: 0.

## Failed executions

Queue exit records identify failures even when results.json still says running.

| Dataset | Updates | Construction seed | Failure |
|---|---:|---:|---|
| arxiv | 100 | 271006 | sinkhorn_tolerance |
| arxiv | 100 | 271109 | sinkhorn_tolerance |
| sportsfit | 60 | 271006 | oom |
| sportsfit | 100 | 271006 | oom |
| sportsfit | 60 | 271109 | oom |
| sportsfit | 100 | 271109 | oom |

Failure messages and log hashes: `execution_failures.csv`.

- One class order; CI units are construction seeds, not individual Student runs.
- Partial sequences are excluded from final-cell means and paired differences.
- Loss reduction alone does not demonstrate downstream improvement or stationarity.
- Solver iteration counts are adaptive despite a common Sinkhorn configuration.
- no-GM with joint-DM disabled has no explicit text-loss-to-Q feedback.
- Current sweep protocol omits mask hash receipts; row counts and recorded fields are checked.
