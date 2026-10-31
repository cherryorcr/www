# Additional current-protocol datasets

All rows use current-delta-only construction, edge-free memory, the Prompt GraphText Student, and report-dev evaluation. These rows are supplementary and are not used to retune the primary method.

| Dataset | Method | Orders | Final F1 mean±SD | Accuracy mean±SD | Old F1 | New F1 | Build s |
|---|---|---:|---:|---:|---:|---:|---:|
| arxiv | cat_cctp | 1 | 36.74±0.00 | 63.62±0.00 | 35.36 | 39.59 | 82.3 |
| arxiv | gecc_hard | 1 | 51.09±0.00 | 70.27±0.00 | 49.93 | 53.50 | 203.4 |
| arxiv | uq_dm_edgefree | 1 | 52.25±0.00 | 71.59±0.00 | 51.04 | 54.77 | 263.7 |
| cora | cat_cctp | 3 | 82.76±1.26 | 84.75±0.77 | 82.48 | 83.13 | 13.7 |
| cora | gecc_hard | 3 | 83.08±1.16 | 85.94±0.91 | 82.02 | 84.50 | 0.4 |
| cora | uq_dm_edgefree | 3 | 82.44±0.57 | 84.50±0.43 | 81.81 | 83.29 | 38.6 |

ArXiv is an additional large-scale transfer check; Cora is an additional standard small-graph check. The Cora stream uses the three predeclared order manifests extracted from the existing current Prompt Student protocol, but the memory is rebuilt by the current edge-free runner.

No link-prediction result is merged here: the primary Unified-Q memory is edge-free. The existing link-prediction artifact uses a separate topology-enabled protocol and remains a supplementary structural audit, not evidence for the primary method.

Status: `complete`; complete cells: `12`.
