# Baseline fidelity and chronological evaluation

This report is generated from result files. Incomplete chains are excluded from final comparisons.

## Coverage

| Output | Status | Completed stages |
|---|---|---|
| native_epochs/cora_outer5000/results.json | complete | 3 |
| precision_repair/temporal/uq_dm_edgefree_c271006/results.json | invalid/excluded | 1 |
| precision_repair/temporal/uq_dm_edgefree_c271108/results.json | invalid/excluded | 0 |
| precision_repair_v2/temporal/uq_dm_edgefree_c271006/results.json | complete | 5 |
| precision_repair_v2/temporal/uq_dm_edgefree_c271108/results.json | complete | 5 |
| precision_repair_v2/temporal/uq_dm_edgefree_c271109/results.json | complete | 5 |
| tagsam/arxiv_o9101_c271006/outer100_updates10/results.json | complete | 3 |
| tagsam/arxiv_o9101_c271006/outer5000_updates15/results.json | complete | 3 |
| tagsam/arxiv_o9101_c271006/outer500_updates10/results.json | complete | 3 |
| tagsam/bookhis_o9101_c271006/outer100_updates10/results.json | complete | 5 |
| tagsam/bookhis_o9101_c271006/outer5000_updates15/results.json | complete | 5 |
| tagsam/bookhis_o9101_c271006/outer500_updates10/results.json | complete | 5 |
| tagsam/sportsfit_o9101_c271006/outer100_updates10/results.json | complete | 5 |
| tagsam/sportsfit_o9101_c271006/outer5000_updates15/results.json | complete | 5 |
| tagsam/sportsfit_o9101_c271006/outer500_updates10/results.json | complete | 5 |
| teacher_budget2048/native_epochs/cora_outer5000/results.json | complete | 3 |
| temporal/cat_cctp_c271006/results.json | complete | 5 |
| temporal/cat_cctp_c271108/results.json | complete | 5 |
| temporal/cat_cctp_c271109/results.json | complete | 5 |
| temporal/gecc_hard_c271006/results.json | complete | 5 |
| temporal/gecc_hard_c271108/results.json | complete | 5 |
| temporal/gecc_hard_c271109/results.json | complete | 5 |
| temporal/uq_dm_edgefree_c271006/results.json | failed | 1 |
| temporal/uq_dm_edgefree_c271108/results.json | complete | 5 |
| temporal/uq_dm_edgefree_c271109/results.json | complete | 5 |

## Chronological ArXiv, Final Stage

Construction means average three student seeds. SD is across completed construction seeds.
Float64 denotes verified float64 normalization plus a solver cap of 16384, not the original implementation.

| Variant | Inference | Cohort | Split | n constructions | Macro-F1 | Accuracy |
|---|---|---|---|---:|---:|---:|
| uq_dm_edgefree_float64 | arrived_snapshot | cumulative | report_dev | 3 | 52.43 +/- 0.57 | 72.25 +/- 0.24 |
| uq_dm_edgefree_float64 | arrived_snapshot | current | report_dev | 3 | 49.88 +/- 0.06 | 72.02 +/- 0.22 |
| uq_dm_edgefree_float64 | arrived_snapshot | historical | report_dev | 3 | 52.84 +/- 0.75 | 72.34 +/- 0.25 |
| uq_dm_edgefree_float64 | arrived_snapshot | cumulative | historical_test | 3 | 53.83 +/- 0.29 | 72.07 +/- 0.01 |
| uq_dm_edgefree_float64 | arrived_snapshot | current | historical_test | 3 | 50.80 +/- 0.53 | 72.10 +/- 0.11 |
| uq_dm_edgefree_float64 | arrived_snapshot | historical | historical_test | 3 | 54.54 +/- 0.12 | 72.06 +/- 0.05 |
| uq_dm_edgefree_float64 | current_snapshot | current | report_dev | 3 | 49.46 +/- 0.31 | 70.20 +/- 0.19 |
| uq_dm_edgefree_float64 | current_snapshot | current | historical_test | 3 | 49.83 +/- 0.55 | 70.06 +/- 0.20 |
| cat_cctp | arrived_snapshot | cumulative | report_dev | 3 | 33.06 +/- 1.06 | 59.80 +/- 1.12 |
| cat_cctp | arrived_snapshot | current | report_dev | 3 | 31.67 +/- 0.56 | 62.41 +/- 0.94 |
| cat_cctp | arrived_snapshot | historical | report_dev | 3 | 33.17 +/- 1.21 | 58.76 +/- 1.27 |
| cat_cctp | arrived_snapshot | cumulative | historical_test | 3 | 33.66 +/- 1.18 | 59.12 +/- 1.06 |
| cat_cctp | arrived_snapshot | current | historical_test | 3 | 32.04 +/- 0.99 | 62.28 +/- 0.88 |
| cat_cctp | arrived_snapshot | historical | historical_test | 3 | 34.06 +/- 1.14 | 57.86 +/- 1.25 |
| cat_cctp | current_snapshot | current | report_dev | 3 | 37.39 +/- 0.36 | 65.33 +/- 0.70 |
| cat_cctp | current_snapshot | current | historical_test | 3 | 37.46 +/- 0.58 | 65.48 +/- 0.67 |
| gecc_hard | arrived_snapshot | cumulative | report_dev | 3 | 51.93 +/- 0.41 | 71.40 +/- 0.15 |
| gecc_hard | arrived_snapshot | current | report_dev | 3 | 49.61 +/- 0.46 | 70.97 +/- 0.19 |
| gecc_hard | arrived_snapshot | historical | report_dev | 3 | 52.34 +/- 0.45 | 71.57 +/- 0.16 |
| gecc_hard | arrived_snapshot | cumulative | historical_test | 3 | 53.07 +/- 0.32 | 71.22 +/- 0.31 |
| gecc_hard | arrived_snapshot | current | historical_test | 3 | 50.00 +/- 1.33 | 70.77 +/- 0.58 |
| gecc_hard | arrived_snapshot | historical | historical_test | 3 | 53.77 +/- 0.13 | 71.40 +/- 0.21 |
| gecc_hard | current_snapshot | current | report_dev | 3 | 48.41 +/- 0.92 | 69.30 +/- 0.50 |
| gecc_hard | current_snapshot | current | historical_test | 3 | 48.10 +/- 0.68 | 69.02 +/- 0.14 |
| uq_dm_edgefree | arrived_snapshot | cumulative | report_dev | 2 | 52.58 +/- 0.21 | 72.32 +/- 0.32 |
| uq_dm_edgefree | arrived_snapshot | current | report_dev | 2 | 49.72 +/- 0.03 | 72.09 +/- 0.58 |
| uq_dm_edgefree | arrived_snapshot | historical | report_dev | 2 | 53.05 +/- 0.16 | 72.41 +/- 0.22 |
| uq_dm_edgefree | arrived_snapshot | cumulative | historical_test | 2 | 53.59 +/- 0.34 | 72.07 +/- 0.22 |
| uq_dm_edgefree | arrived_snapshot | current | historical_test | 2 | 50.07 +/- 0.40 | 72.21 +/- 0.35 |
| uq_dm_edgefree | arrived_snapshot | historical | historical_test | 2 | 54.45 +/- 0.33 | 72.02 +/- 0.17 |
| uq_dm_edgefree | current_snapshot | current | report_dev | 2 | 49.72 +/- 0.98 | 70.08 +/- 0.28 |
| uq_dm_edgefree | current_snapshot | current | historical_test | 2 | 49.60 +/- 0.91 | 70.12 +/- 0.15 |

## Chronological ArXiv, Five-Stage Average

Unweighted mean of five stage scores within each construction, then mean and SD across constructions.
This descriptive trajectory summary is separate from final-stage scores.

| Variant | Inference | Cohort | Split | n constructions | Macro-F1 | Accuracy |
|---|---|---|---|---:|---:|---:|
| uq_dm_edgefree_float64 | arrived_snapshot | cumulative | report_dev | 3 | 52.36 +/- 0.13 | 70.96 +/- 0.07 |
| uq_dm_edgefree_float64 | arrived_snapshot | cumulative | historical_test | 3 | 53.71 +/- 0.50 | 70.84 +/- 0.23 |
| uq_dm_edgefree_float64 | current_snapshot | current | report_dev | 3 | 48.69 +/- 0.04 | 68.94 +/- 0.15 |
| uq_dm_edgefree_float64 | current_snapshot | current | historical_test | 3 | 49.15 +/- 0.62 | 68.27 +/- 0.23 |
| cat_cctp | arrived_snapshot | cumulative | report_dev | 3 | 39.00 +/- 1.16 | 63.42 +/- 1.11 |
| cat_cctp | arrived_snapshot | cumulative | historical_test | 3 | 39.97 +/- 1.34 | 63.15 +/- 1.08 |
| cat_cctp | current_snapshot | current | report_dev | 3 | 40.66 +/- 0.96 | 66.12 +/- 1.04 |
| cat_cctp | current_snapshot | current | historical_test | 3 | 41.24 +/- 1.69 | 65.32 +/- 1.05 |
| gecc_hard | arrived_snapshot | cumulative | report_dev | 3 | 51.93 +/- 0.12 | 70.53 +/- 0.19 |
| gecc_hard | arrived_snapshot | cumulative | historical_test | 3 | 52.74 +/- 0.14 | 70.31 +/- 0.19 |
| gecc_hard | current_snapshot | current | report_dev | 3 | 48.56 +/- 0.33 | 68.50 +/- 0.05 |
| gecc_hard | current_snapshot | current | historical_test | 3 | 48.72 +/- 0.06 | 67.75 +/- 0.18 |
| uq_dm_edgefree | arrived_snapshot | cumulative | report_dev | 2 | 52.36 +/- 0.12 | 70.98 +/- 0.10 |
| uq_dm_edgefree | arrived_snapshot | cumulative | historical_test | 2 | 53.56 +/- 0.22 | 70.76 +/- 0.06 |
| uq_dm_edgefree | current_snapshot | current | report_dev | 2 | 48.96 +/- 0.10 | 68.94 +/- 0.42 |
| uq_dm_edgefree | current_snapshot | current | historical_test | 2 | 48.97 +/- 0.31 | 68.35 +/- 0.05 |

## TAGSAM Complete-Chain Budget Diagnostic

These are minibatch-unroll adapters, not official full-epoch runs. SD is over students for one construction.

| Dataset | Outer updates | Split | Macro-F1 | Accuracy |
|---|---:|---|---:|---:|
| arxiv | 100 | report_dev | 39.44 +/- 1.70 | 54.60 +/- 2.73 |
| arxiv | 100 | historical_test | 40.19 +/- 1.52 | 55.06 +/- 2.71 |
| arxiv | 5000 | report_dev | 5.85 +/- 1.46 | 21.19 +/- 4.00 |
| arxiv | 5000 | historical_test | 5.75 +/- 1.51 | 20.55 +/- 4.15 |
| arxiv | 500 | report_dev | 26.91 +/- 3.89 | 40.73 +/- 3.50 |
| arxiv | 500 | historical_test | 27.85 +/- 3.02 | 41.15 +/- 3.59 |
| bookhis | 100 | report_dev | 10.83 +/- 0.73 | 60.55 +/- 0.97 |
| bookhis | 100 | historical_test | 10.80 +/- 1.14 | 59.18 +/- 1.17 |
| bookhis | 5000 | report_dev | 6.04 +/- 0.00 | 56.87 +/- 0.00 |
| bookhis | 5000 | historical_test | 5.93 +/- 0.00 | 55.28 +/- 0.00 |
| bookhis | 500 | report_dev | 6.11 +/- 0.11 | 56.88 +/- 0.01 |
| bookhis | 500 | historical_test | 5.95 +/- 0.03 | 55.29 +/- 0.01 |
| sportsfit | 100 | report_dev | 69.87 +/- 2.85 | 80.34 +/- 6.19 |
| sportsfit | 100 | historical_test | 70.13 +/- 3.58 | 80.08 +/- 6.16 |
| sportsfit | 5000 | report_dev | 1.39 +/- 0.80 | 6.54 +/- 0.34 |
| sportsfit | 5000 | historical_test | 1.34 +/- 0.70 | 6.56 +/- 0.21 |
| sportsfit | 500 | report_dev | 19.90 +/- 3.79 | 19.56 +/- 1.55 |
| sportsfit | 500 | historical_test | 20.12 +/- 3.85 | 19.88 +/- 1.67 |

## Native-Epoch Construction Adaptation

Native inner-epoch function, model, BERT, weighted BCE and schedule; common downstream student.
This is not an unchanged official end-to-end reproduction.

| Dataset | Teacher budget | Outer updates | Stage | Split | Macro-F1 | Accuracy |
|---|---|---:|---:|---|---:|---:|
| cora | 3_epochs | 5000 | 0 | report_dev | 45.53 +/- 5.35 | 68.50 +/- 2.08 |
| cora | 3_epochs | 5000 | 0 | historical_test | 39.83 +/- 2.56 | 62.85 +/- 1.17 |
| cora | 3_epochs | 5000 | 1 | report_dev | 40.37 +/- 5.52 | 55.20 +/- 2.17 |
| cora | 3_epochs | 5000 | 1 | historical_test | 35.30 +/- 4.58 | 51.24 +/- 1.63 |
| cora | 3_epochs | 5000 | 2 | report_dev | 42.36 +/- 3.46 | 57.20 +/- 3.15 |
| cora | 3_epochs | 5000 | 2 | historical_test | 41.50 +/- 2.05 | 53.51 +/- 1.69 |
| cora | 2048 | 5000 | 0 | report_dev | 83.36 +/- 0.76 | 86.61 +/- 0.79 |
| cora | 2048 | 5000 | 0 | historical_test | 82.56 +/- 1.38 | 84.99 +/- 0.88 |
| cora | 2048 | 5000 | 1 | report_dev | 60.94 +/- 1.13 | 74.73 +/- 0.54 |
| cora | 2048 | 5000 | 1 | historical_test | 59.80 +/- 1.47 | 74.65 +/- 1.34 |
| cora | 2048 | 5000 | 2 | report_dev | 64.68 +/- 4.44 | 72.45 +/- 2.25 |
| cora | 2048 | 5000 | 2 | historical_test | 63.80 +/- 0.43 | 73.19 +/- 0.56 |

## Separate Native-Student Diagnostic

Same final memory, BERT/native model, learned rates, fixed 15 epochs; not the common-student budget.

| Dataset | Teacher budget | Split | Macro-F1 | Accuracy | Reconstructed teacher F1 |
|---|---|---|---:|---:|---:|
| cora | 3_epochs | report_dev | 15.55 +/- 7.40 | 18.45 +/- 9.92 | 12.26 |
| cora | 3_epochs | historical_test | 15.70 +/- 7.57 | 17.22 +/- 8.72 | 14.13 |
| cora | 2048 | report_dev | 30.16 +/- 8.16 | 33.70 +/- 8.14 | 27.76 |
| cora | 2048 | historical_test | 29.39 +/- 6.96 | 30.87 +/- 7.33 | 23.20 |

## Interpretation Boundaries

- Publication time is observed; edge availability is assumed at the later endpoint arrival.
- The chronological stream has recurring classes. Arrival-cohort retention is not new-class forgetting.
- Arrived-snapshot inference can read historical nodes; constructors cannot. Current-snapshot inference cannot score old queries.
- Original numerical failures remain part of the evidence. No failed run is treated as a zero score.
- Historical masks have been inspected previously. These are protocol-frozen diagnostics, not new blind benchmarks.
- Co-located run timings are descriptive, not an isolated efficiency comparison.
- A small-graph native-core result does not close the large-graph official baseline fidelity gap.
