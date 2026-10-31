# Chronological citation subsets

Three constructions; three students averaged within each memory.
Sample SD across constructions. Test and report-dev remain separate.

## test

| Dataset | Method | Final cumulative F1 / Acc | Five-stage cumulative F1 | Final current-only F1 / Acc |
|---|---|---:|---:|---:|
| ACM | CoTAG-Q | 87.87 +/- 0.37 / 87.88 +/- 0.32 | 88.10 +/- 0.40 | 85.18 +/- 1.46 / 85.80 +/- 1.53 |
| ACM | GECC-hard adaptation | 88.38 +/- 0.15 / 88.25 +/- 0.12 | 88.26 +/- 0.09 | 85.10 +/- 1.58 / 85.31 +/- 1.62 |
| ACM | CaT adaptation | 86.62 +/- 0.75 / 86.88 +/- 0.56 | 87.39 +/- 0.92 | 85.14 +/- 2.02 / 86.18 +/- 1.59 |
| DBLP | CoTAG-Q | 81.50 +/- 0.56 / 82.55 +/- 0.62 | 82.13 +/- 0.35 | 73.75 +/- 1.22 / 77.00 +/- 1.16 |
| DBLP | GECC-hard adaptation | 82.24 +/- 0.58 / 83.36 +/- 0.35 | 82.85 +/- 0.23 | 75.30 +/- 0.42 / 78.03 +/- 0.59 |
| DBLP | CaT adaptation | 80.74 +/- 1.06 / 81.98 +/- 0.95 | 82.09 +/- 0.61 | 76.59 +/- 1.14 / 79.95 +/- 1.04 |

| Dataset | CoTAG-Q minus | Evaluation | F1 difference (pp) | Paired 95% CI | Positive pairs | Worst pair (pp) |
|---|---|---|---:|---:|---:|---:|
| ACM | GECC-hard adaptation | 4, arrived_snapshot | -0.51 | [-1.75, +0.74] | 1/3 | -0.94 |
| ACM | CaT adaptation | 4, arrived_snapshot | +1.24 | [-0.65, +3.14] | 3/3 | +0.37 |
| ACM | GECC-hard adaptation | 4, current_snapshot | +0.08 | [-0.37, +0.53] | 2/3 | -0.13 |
| ACM | CaT adaptation | 4, current_snapshot | +0.04 | [-4.14, +4.21] | 1/3 | -1.59 |
| ACM | GECC-hard adaptation | five_stage_mean, arrived_snapshot | -0.16 | [-0.95, +0.63] | 1/3 | -0.52 |
| ACM | CaT adaptation | five_stage_mean, arrived_snapshot | +0.71 | [-2.55, +3.97] | 2/3 | -0.75 |
| ACM | GECC-hard adaptation | five_stage_mean, current_snapshot | +0.46 | [-0.35, +1.27] | 3/3 | +0.12 |
| ACM | CaT adaptation | five_stage_mean, current_snapshot | -0.74 | [-1.96, +0.48] | 0/3 | -1.03 |
| DBLP | GECC-hard adaptation | 4, arrived_snapshot | -0.74 | [-3.54, +2.06] | 1/3 | -1.96 |
| DBLP | CaT adaptation | 4, arrived_snapshot | +0.76 | [-3.27, +4.79] | 2/3 | -0.94 |
| DBLP | GECC-hard adaptation | 4, current_snapshot | -1.55 | [-3.65, +0.55] | 0/3 | -2.29 |
| DBLP | CaT adaptation | 4, current_snapshot | -2.84 | [-8.00, +2.32] | 0/3 | -4.14 |
| DBLP | GECC-hard adaptation | five_stage_mean, arrived_snapshot | -0.72 | [-2.04, +0.61] | 0/3 | -1.31 |
| DBLP | CaT adaptation | five_stage_mean, arrived_snapshot | +0.04 | [-2.34, +2.43] | 2/3 | -0.90 |
| DBLP | GECC-hard adaptation | five_stage_mean, current_snapshot | -0.91 | [-1.57, -0.26] | 0/3 | -1.22 |
| DBLP | CaT adaptation | five_stage_mean, current_snapshot | -2.76 | [-4.74, -0.78] | 0/3 | -3.30 |

## report_dev

| Dataset | Method | Final cumulative F1 / Acc | Five-stage cumulative F1 | Final current-only F1 / Acc |
|---|---|---:|---:|---:|
| ACM | CoTAG-Q | 88.83 +/- 0.68 / 88.98 +/- 0.64 | 88.81 +/- 0.60 | 85.40 +/- 0.32 / 84.89 +/- 0.29 |
| ACM | GECC-hard adaptation | 88.55 +/- 0.10 / 88.54 +/- 0.13 | 88.47 +/- 0.25 | 83.31 +/- 0.62 / 82.47 +/- 0.70 |
| ACM | CaT adaptation | 86.78 +/- 1.39 / 87.30 +/- 1.19 | 87.77 +/- 0.74 | 83.22 +/- 1.63 / 83.68 +/- 1.24 |
| DBLP | CoTAG-Q | 81.04 +/- 0.74 / 82.01 +/- 0.77 | 81.35 +/- 0.38 | 74.77 +/- 0.92 / 76.92 +/- 0.93 |
| DBLP | GECC-hard adaptation | 82.08 +/- 0.44 / 83.09 +/- 0.24 | 82.14 +/- 0.07 | 75.38 +/- 1.06 / 77.41 +/- 0.64 |
| DBLP | CaT adaptation | 80.57 +/- 1.10 / 81.76 +/- 0.96 | 81.43 +/- 0.54 | 76.39 +/- 1.20 / 78.79 +/- 1.03 |

| Dataset | CoTAG-Q minus | Evaluation | F1 difference (pp) | Paired 95% CI | Positive pairs | Worst pair (pp) |
|---|---|---|---:|---:|---:|---:|
| ACM | GECC-hard adaptation | 4, arrived_snapshot | +0.28 | [-1.48, +2.04] | 2/3 | -0.35 |
| ACM | CaT adaptation | 4, arrived_snapshot | +2.05 | [-1.44, +5.53] | 3/3 | +0.86 |
| ACM | GECC-hard adaptation | 4, current_snapshot | +2.09 | [+1.12, +3.06] | 3/3 | +1.74 |
| ACM | CaT adaptation | 4, current_snapshot | +2.18 | [-2.23, +6.59] | 3/3 | +0.30 |
| ACM | GECC-hard adaptation | five_stage_mean, arrived_snapshot | +0.34 | [-0.95, +1.63] | 2/3 | -0.13 |
| ACM | CaT adaptation | five_stage_mean, arrived_snapshot | +1.04 | [-1.84, +3.92] | 2/3 | -0.22 |
| ACM | GECC-hard adaptation | five_stage_mean, current_snapshot | +1.24 | [+0.42, +2.05] | 3/3 | +0.90 |
| ACM | CaT adaptation | five_stage_mean, current_snapshot | +0.39 | [-0.21, +0.98] | 3/3 | +0.12 |
| DBLP | GECC-hard adaptation | 4, arrived_snapshot | -1.04 | [-3.98, +1.91] | 0/3 | -2.37 |
| DBLP | CaT adaptation | 4, arrived_snapshot | +0.48 | [-4.10, +5.06] | 2/3 | -1.60 |
| DBLP | GECC-hard adaptation | 4, current_snapshot | -0.60 | [-5.52, +4.31] | 1/3 | -2.77 |
| DBLP | CaT adaptation | 4, current_snapshot | -1.62 | [-6.62, +3.38] | 1/3 | -3.41 |
| DBLP | GECC-hard adaptation | five_stage_mean, arrived_snapshot | -0.79 | [-1.86, +0.27] | 0/3 | -1.23 |
| DBLP | CaT adaptation | five_stage_mean, arrived_snapshot | -0.08 | [-2.31, +2.15] | 2/3 | -1.05 |
| DBLP | GECC-hard adaptation | five_stage_mean, current_snapshot | -0.79 | [-2.84, +1.27] | 1/3 | -1.56 |
| DBLP | CaT adaptation | five_stage_mean, current_snapshot | -2.21 | [-4.62, +0.19] | 0/3 | -2.84 |

