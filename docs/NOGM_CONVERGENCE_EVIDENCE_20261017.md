# Unified-Q construction-update evidence

## Recommended conclusion

We retain the original 30-update construction budget. The sensitivity study does not establish mathematical convergence at 30 updates for every dataset and stage. Instead, it shows that the objective decreases rapidly during early updates, enters a low-improvement regime on many later stage traces, and that extending the optimization to 60 or 100 updates does not provide a stable downstream Macro-F1 gain under the frozen protocol. Therefore, 30 updates are used as a practical accuracy-cost operating point rather than as a claim of exact convergence.

The study is a report-development diagnostic for the no-GM construction variant. It should not be used to claim that the full GM implementation has been proven converged.

## Evidence summary

| Dataset | Updates | Construction seeds | Final Macro-F1 | Accuracy | Construction time (s) | Peak memory (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| ArXiv | 30 | 3 | 51.702 | 70.957 | 225.6 | 11.09 |
| ArXiv | 60 | 2 | 51.809 | 71.503 | 1525.3 | 16.31 |
| SportsFit | 30 | 3 | 79.980 | 85.969 | 153.5 | 23.77 |
| BookHis | 30 | 3 | 48.020 | 82.244 | 61.5 | 2.75 |
| BookHis | 60 | 2 | 45.685 | 82.343 | 282.0 | 5.35 |
| BookHis | 100 | 2 | 46.356 | 82.174 | 722.8 | 5.80 |

Matched comparisons show no stable benefit from longer construction:

- ArXiv, 60 versus 30 updates: Macro-F1 `-0.142` percentage points and Accuracy `+0.414` points.
- BookHis, 60 versus 30 updates: Macro-F1 `-2.249` points and Accuracy `+0.036` points.
- BookHis, 100 versus 30 updates: Macro-F1 `-1.578` points and Accuracy `-0.132` points.

The longer SportsFit runs were not completed because the dense assignment computation exceeded the available GPU memory. The longer ArXiv runs terminated at the adaptive Sinkhorn tolerance check. These failures are retained as resource and numerical limitations, not treated as performance results.

## Plateau diagnostic

We mark a stage as practically plateaued when the observed construction objective decreases by less than 1% in each of two consecutive 10-update windows. This is an operational stopping rule, not a proof that the optimization gradient is zero.

Across all completed traces, 20 of 65 stage traces satisfy this criterion. The 100-update BookHis runs provide the clearest confirmation: all five stages have less than 1% relative loss reduction in both the 80--90 and 90--100 windows. However, some 30-update traces are still decreasing by more than 1% near the endpoint, so the paper should avoid claiming universal strict convergence at 30 updates.

## Suggested paper wording

> **Construction-update sensitivity.** We retain 30 construction updates in all main experiments. A report-development sensitivity study compares 30, 60, and 100 updates under the same data access, memory ratio, construction seeds, and Student evaluation protocol. The objective drops sharply during the first updates and becomes substantially flatter in later iterations. Extending the budget does not yield a consistent downstream improvement: on ArXiv, 60 updates change Macro-F1 by -0.14 points relative to 30 updates, while on BookHis, 60 and 100 updates reduce Macro-F1 by 2.25 and 1.58 points, respectively. We therefore use 30 updates as a fixed accuracy--cost operating point. This result supports the selected training budget empirically; it is not intended as a claim of exact optimization convergence for every stage.

## Artifacts

- Loss trajectories: `results/cotag_q_nogm_step_sensitivity_20261017/report_utc_20261016_175800_212138/loss_traces.png`
- Completed-cell cost and performance plot: `results/cotag_q_nogm_step_sensitivity_20261017/report_utc_20261016_175800_212138/completed_cells.png`
- Full report: `results/cotag_q_nogm_step_sensitivity_20261017/report_utc_20261016_175800_212138/summary.md`
- Per-stage traces and plateau flags: `results/cotag_q_nogm_step_sensitivity_20261017/report_utc_20261016_175800_212138/stage_traces.csv`
