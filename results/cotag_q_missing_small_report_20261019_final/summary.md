# Small-graph baseline completion

Completed: 10/10 method-dataset entries.

Each entry uses one class order and one construction seed, with Student seeds 150/151/152.
Values are mean +/- sample SD across Students, not nine-order/construction statistics.
Common-Student references use the historical test mask (not a new blind test).
Inspired references use report-dev and a 1024-step legacy Student; do not rank them with the common Student.

| Dataset | Method | Split | Macro-F1 | Accuracy | Build seconds | Peak MiB |
|---|---|---|---:|---:|---:|---:|
| cora | TAGSAM bounded core | sealed_test | 60.55 +/- 3.22 | 67.28 +/- 1.49 | 50.02 | 4630.6 |
| cora | STAD local-Qwen | sealed_test | 69.44 +/- 2.40 | 68.88 +/- 3.79 | 182.36 | 5324.7 |
| cora | DANCE-inspired | report_dev | 39.83 +/- 0.55 | 47.11 +/- 1.54 | 0.16 | 161.3 |
| cora | SR-GM-inspired | report_dev | 28.41 +/- 4.38 | 31.12 +/- 6.07 | 0.32 | 149.3 |
| cora | STAD-inspired | report_dev | 29.84 +/- 1.79 | 33.95 +/- 2.88 | 0.14 | 167.8 |
| citeseer | TAGSAM bounded core | sealed_test | 65.33 +/- 0.88 | 72.20 +/- 1.27 | 46.39 | 4410.6 |
| citeseer | STAD local-Qwen | sealed_test | 67.10 +/- 0.66 | 71.89 +/- 0.48 | 185.86 | 5294.8 |
| citeseer | DANCE-inspired | report_dev | 43.63 +/- 1.42 | 49.84 +/- 1.44 | 0.14 | 156.3 |
| citeseer | SR-GM-inspired | report_dev | 48.94 +/- 3.02 | 53.50 +/- 3.63 | 0.33 | 192.9 |
| citeseer | STAD-inspired | report_dev | 46.08 +/- 1.66 | 52.56 +/- 1.99 | 0.14 | 162.7 |

TAGSAM keeps the native PMI selector with a bounded 100-by-10 response loop.
STAD uses the pinned native core with local Qwen2.5-1.5B and internal candidate validation.
Both change the original access/training interface and are qualified adaptations, not official reproductions.
SR-GM-inspired projects conflicts in shared 128-D latent space because raw feature/text dimensions differ.
Budget, stream and mask hashes match the order-9101 primary records. Failed outputs are retained.
TAGSAM stage zero was migrated only after checking its protocol, memory hash and completed Student scores.
Construction times include the reused original stage zero; peak is max allocated CUDA memory across stages.
