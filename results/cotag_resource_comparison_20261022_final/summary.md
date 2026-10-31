# Common construction resource comparison

Report-development only. ArXiv uses the archived three-stage class stream; SportsFit uses five stages.
Three constructions, each averaged over three student seeds. SD is across construction means.

| Dataset | Method | n | F1 mean +/- SD | Accuracy | Allocated GiB | Reserved GiB | Construction s | CPU RSS GiB | Student GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| arxiv | incremental_q | 3 | 51.660 +/- 0.285 | 70.903 | 5.252 | 5.816 | 243.26 | 3.365 | 0.183 |
| arxiv | full_history | 3 | 52.428 +/- 0.382 | 71.247 | 8.939 | 9.589 | 237.32 | 3.322 | 0.183 |
| arxiv | cat_cctp | 3 | 34.489 +/- 1.326 | 61.498 | 0.557 | 0.936 | 70.85 | 2.446 | 0.183 |
| arxiv | gecc_hard | 3 | 51.383 +/- 0.312 | 71.471 | 0.538 | 0.630 | 111.92 | 2.423 | 0.183 |
| sportsfit | incremental_q | 3 | 79.413 +/- 0.943 | 84.433 | 17.514 | 19.887 | 174.47 | 3.055 | 0.184 |
| sportsfit | full_history | 3 | 78.657 +/- 0.689 | 85.898 | 18.851 | 23.079 | 163.29 | 3.355 | 0.184 |
| sportsfit | cat_cctp | 3 | 74.929 +/- 2.376 | 86.273 | 0.738 | 1.186 | 128.98 | 2.450 | 0.185 |
| sportsfit | gecc_hard | 3 | 79.291 +/- 0.905 | 89.161 | 0.663 | 0.721 | 411.15 | 2.410 | 0.185 |

## Interpretation

- Full-history uses all final legal raw training nodes and edges. It is an access-relaxed reference.
- Full-history has one 30-update call; incremental Q has three/five 30-update calls. Total compute is not matched.
- Incremental Q reuses the six verified combined runs. Baselines retain CaT 3000 updates and GECC 20 restarts.
- Construction includes the resident text encoder, excludes original-graph GPU evaluation caches, extraction and serialization.
- CPU RSS includes the CPU source loader. This experiment does not establish lower total host memory for streaming.
- Student and original-graph inference are measured separately. Their peaks must not be presented as construction peaks.
- Times are descriptive across batches and GPU lanes; no isolated timing speedup claim.
- Both baselines are mechanism adaptations, not unchanged official reproductions.
- Failure is missing evidence, not zero performance. Paired intervals use construction seeds, not nine independent students.

See paired.csv for paired 95% t intervals, sign counts and worst paired outcome.
Verified 78 memories, 54 prefixes, 126 predictions and 252 independently recomputed scores.
