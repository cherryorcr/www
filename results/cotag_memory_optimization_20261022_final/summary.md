# Construction memory optimization

Report-development diagnostic. Three constructions; three students averaged per memory.
ArXiv uses the archived three-stage stream; SportsFit uses five stages.

## Four-way stage-zero pilot

| Dataset | Implementation | Peak GiB | Time s | Entry GiB | Parameters | F1 |
|---|---|---:|---:|---:|---:|---:|
| arxiv | cache_offload | 11.275 | 82.73 | 0.093 | 178,676,442 | 74.212 |
| arxiv | class_blocks | 4.973 | 79.40 | 0.381 | 34,428,898 | 73.543 |
| arxiv | combined | 4.680 | 79.58 | 0.093 | 34,428,898 | 73.737 |
| arxiv | original | 11.567 | 82.15 | 0.381 | 178,676,442 | 74.172 |
| sportsfit | cache_offload | 23.135 | 122.37 | 0.093 | 291,378,329 | 80.082 |
| sportsfit | class_blocks | 17.461 | 118.95 | 0.400 | 143,962,148 | 79.127 |
| sportsfit | combined | 17.154 | 118.68 | 0.093 | 143,962,148 | 79.825 |
| sportsfit | original | 23.442 | 122.66 | 0.400 | 291,378,329 | 80.425 |

## Complete streams

| Dataset | Implementation | Constructions | F1 mean +/- SD | Accuracy mean +/- SD | Peak GiB | Construction s |
|---|---|---:|---:|---:|---:|---:|
| arxiv | original | 3 | 51.640 +/- 0.148 | 70.939 +/- 0.470 | 11.537 | 240.89 |
| arxiv | combined | 3 | 51.660 +/- 0.285 | 70.903 +/- 0.466 | 5.252 | 243.26 |
| sportsfit | original | 3 | 79.681 +/- 0.585 | 85.464 +/- 0.662 | 23.802 | 176.67 |
| sportsfit | combined | 3 | 79.413 +/- 0.943 | 84.433 +/- 1.532 | 17.514 | 174.47 |

## Paired contrasts

Combined minus original; reductions use original as denominator. CI: two-sided 95% t, construction as unit.

| Dataset | Metric | n | Mean | CI |
|---|---|---:|---:|---|
| arxiv | f1 | 3 | 0.021 | [-0.557, 0.598] |
| arxiv | accuracy | 3 | -0.036 | [-0.330, 0.258] |
| arxiv | peak_gib | 3 | -6.285 | [-6.374, -6.195] |
| arxiv | seconds | 3 | 2.366 | [-29.257, 33.988] |
| arxiv | peak_gib_reduction_pct | 3 | 54.477 | [53.732, 55.223] |
| arxiv | seconds_reduction_pct | 3 | -1.092 | [-14.427, 12.244] |
| sportsfit | f1 | 3 | -0.268 | [-2.042, 1.506] |
| sportsfit | accuracy | 3 | -1.031 | [-4.059, 1.997] |
| sportsfit | peak_gib | 3 | -6.288 | [-6.292, -6.284] |
| sportsfit | seconds | 3 | -2.195 | [-10.590, 6.200] |
| sportsfit | peak_gib_reduction_pct | 3 | 26.422 | [25.519, 27.325] |
| sportsfit | seconds_reduction_pct | 3 | 1.242 | [-3.507, 5.991] |

## Interpretation boundaries

- Absolute allocated peak includes the resident encoder; reserved peak is separately reported.
- Construction time excludes evaluation-cache rebuild and student training; those are in evaluation.csv.
- Times from two concurrent GPU lanes are descriptive. CPU RSS is sampled every 20 ms.
- Losses and GM remain unchanged. Small GPU rounding differences can change hard text choices.
- Existing CaT/GECC results were not rerun; these tables do not establish lowest construction cost among methods.
- No sealed-test outcomes were used; no significance/equivalence claim from a single pilot.

Verified: 52 memories, 36 prefixes, 84 predictions, 168 independently recomputed scores.
