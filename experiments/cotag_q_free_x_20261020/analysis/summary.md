# Independent feature optimization: audited results

Completed paired constructions: 9/9. No paper edits.

Means and sample SD are over construction seeds after averaging three Student seeds per memory.
ArXiv has 3 stages; SportsFit and BookHis have 5. Order seed is 9101.
Historical test results use a previously inspected mask and are not fresh blind evidence.

## report_dev

| Dataset | Student | Arm | Final Macro-F1 | Accuracy |
|---|---|---|---:|---:|
| ArXiv | GCN | CoTAG-Q | 51.48 +/- 0.14 | 70.78 +/- 0.45 |
| ArXiv | GCN | Independent X + learned text Q | 51.19 +/- 0.45 | 67.95 +/- 1.01 |
| ArXiv | GCN | Fixed Q (archive) | 39.48 +/- 0.08 | 55.63 +/- 0.34 |
| ArXiv | GraphSAGE | CoTAG-Q | 52.39 +/- 0.14 | 71.56 +/- 0.34 |
| ArXiv | GraphSAGE | Independent X + learned text Q | 50.93 +/- 0.35 | 68.94 +/- 0.55 |
| SportsFit | GCN | CoTAG-Q | 79.66 +/- 0.50 | 84.99 +/- 1.07 |
| SportsFit | GCN | Independent X + learned text Q | 75.08 +/- 0.29 | 80.45 +/- 1.01 |
| SportsFit | GCN | Fixed Q (archive) | 57.91 +/- 0.47 | 67.01 +/- 0.71 |
| SportsFit | GraphSAGE | CoTAG-Q | 77.80 +/- 0.61 | 82.63 +/- 0.49 |
| SportsFit | GraphSAGE | Independent X + learned text Q | 72.35 +/- 0.85 | 77.26 +/- 0.93 |
| BookHis | GCN | CoTAG-Q | 47.76 +/- 1.73 | 81.83 +/- 0.67 |
| BookHis | GCN | Independent X + learned text Q | 45.38 +/- 0.29 | 80.72 +/- 0.12 |
| BookHis | GCN | Fixed Q (archive) | 38.36 +/- 0.58 | 74.17 +/- 0.28 |
| BookHis | GraphSAGE | CoTAG-Q | 47.71 +/- 0.12 | 82.63 +/- 0.20 |
| BookHis | GraphSAGE | Independent X + learned text Q | 46.36 +/- 1.98 | 80.99 +/- 0.62 |

Paired differences in percentage points; positive favors CoTAG-Q.

| Dataset | Student | Control | Difference | 95% paired t CI | Positive constructions | Worst difference |
|---|---|---|---:|---|---:|---:|
| ArXiv | GCN | Independent X + learned text Q | +0.29 | [-0.48, 1.06] | 3/3 | +0.04 |
| ArXiv | GCN | Fixed Q (archive) | +12.00 | [11.49, 12.50] | 3/3 | +11.81 |
| ArXiv | GraphSAGE | Independent X + learned text Q | +1.46 | [0.82, 2.10] | 3/3 | +1.17 |
| SportsFit | GCN | Independent X + learned text Q | +4.58 | [3.80, 5.36] | 3/3 | +4.22 |
| SportsFit | GCN | Fixed Q (archive) | +21.75 | [19.89, 23.61] | 3/3 | +20.90 |
| SportsFit | GraphSAGE | Independent X + learned text Q | +5.45 | [4.34, 6.56] | 3/3 | +5.02 |
| BookHis | GCN | Independent X + learned text Q | +2.38 | [-2.62, 7.37] | 3/3 | +0.57 |
| BookHis | GCN | Fixed Q (archive) | +9.39 | [3.91, 14.88] | 3/3 | +7.74 |
| BookHis | GraphSAGE | Independent X + learned text Q | +1.35 | [-3.57, 6.27] | 2/3 | -0.03 |

## sealed_test

| Dataset | Student | Arm | Final Macro-F1 | Accuracy |
|---|---|---|---:|---:|
| ArXiv | GCN | CoTAG-Q | 52.96 +/- 0.21 | 70.64 +/- 0.41 |
| ArXiv | GCN | Independent X + learned text Q | 52.27 +/- 0.48 | 68.27 +/- 0.91 |
| ArXiv | GCN | Fixed Q (archive) | 40.20 +/- 0.20 | 55.80 +/- 0.34 |
| ArXiv | GraphSAGE | CoTAG-Q | 54.02 +/- 0.69 | 71.06 +/- 0.22 |
| ArXiv | GraphSAGE | Independent X + learned text Q | 51.74 +/- 0.49 | 68.86 +/- 0.58 |
| SportsFit | GCN | CoTAG-Q | 80.06 +/- 0.66 | 84.71 +/- 1.06 |
| SportsFit | GCN | Independent X + learned text Q | 75.03 +/- 0.47 | 80.12 +/- 1.01 |
| SportsFit | GCN | Fixed Q (archive) | 58.35 +/- 0.48 | 67.23 +/- 0.70 |
| SportsFit | GraphSAGE | CoTAG-Q | 77.96 +/- 0.67 | 82.44 +/- 0.53 |
| SportsFit | GraphSAGE | Independent X + learned text Q | 72.71 +/- 0.76 | 77.07 +/- 0.89 |
| BookHis | GCN | CoTAG-Q | 49.00 +/- 1.01 | 80.71 +/- 0.63 |
| BookHis | GCN | Independent X + learned text Q | 46.33 +/- 0.64 | 79.59 +/- 0.26 |
| BookHis | GCN | Fixed Q (archive) | 39.62 +/- 0.17 | 73.97 +/- 0.34 |
| BookHis | GraphSAGE | CoTAG-Q | 49.45 +/- 1.36 | 81.63 +/- 0.32 |
| BookHis | GraphSAGE | Independent X + learned text Q | 47.95 +/- 2.43 | 80.65 +/- 0.57 |

Paired differences in percentage points; positive favors CoTAG-Q.

| Dataset | Student | Control | Difference | 95% paired t CI | Positive constructions | Worst difference |
|---|---|---|---:|---|---:|---:|
| ArXiv | GCN | Independent X + learned text Q | +0.69 | [-0.38, 1.76] | 3/3 | +0.20 |
| ArXiv | GCN | Fixed Q (archive) | +12.76 | [11.78, 13.73] | 3/3 | +12.34 |
| ArXiv | GraphSAGE | Independent X + learned text Q | +2.29 | [0.14, 4.44] | 3/3 | +1.29 |
| SportsFit | GCN | Independent X + learned text Q | +5.03 | [4.48, 5.57] | 3/3 | +4.87 |
| SportsFit | GCN | Fixed Q (archive) | +21.71 | [19.66, 23.76] | 3/3 | +20.76 |
| SportsFit | GraphSAGE | Independent X + learned text Q | +5.26 | [3.93, 6.58] | 3/3 | +4.86 |
| BookHis | GCN | Independent X + learned text Q | +2.67 | [-1.10, 6.43] | 3/3 | +0.93 |
| BookHis | GCN | Fixed Q (archive) | +9.38 | [7.22, 11.54] | 3/3 | +8.52 |
| BookHis | GraphSAGE | Independent X + learned text Q | +1.50 | [-7.50, 10.50] | 2/3 | -2.22 |

## Interpretation limits

- Independent X removes the moving Q-conditioned feature construction, trust-radius projection and X anchor. It retains learned Q for text and propagation targets.
- This tests that constrained parameterization as a package, not shared Q alone or identical objectives.
- GraphSAGE uses the previously frozen self-loop mean implementation with a separate root mapping.
- Missing original full memories are reconstructed and compared using their actual new scores. Drift is listed separately.
- Three construction seeds in one class order give limited statistical power; nine Student fits are not nine independent memories.
- Per-stage construction peaks include the existing transductive evaluation cache and cannot establish physical streaming-memory savings.
- Training losses are diagnostic. A limited-budget difference is not a proof of asymptotic superiority.
