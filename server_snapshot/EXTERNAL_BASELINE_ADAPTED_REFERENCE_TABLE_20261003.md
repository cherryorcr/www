# Adapted/Reference Results Outside the Strict Baseline Table (2026-10-03)

TAGSAM and STAD source runs use Student seeds 100-105, while the frozen strict comparison requires 150-155. These results are retained as supplementary adapted/reference evidence and are not ranked with the strict table.

| Method | Dataset | Order | Stage | Macro-F1 | Accuracy | Old F1 | New F1 | Worst-stage F1 | Student seeds |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| STAD | citeseer | order-unspecified | 2 | 28.9085 | 34.4828 | 39.0430 | 8.6393 | 22.3482 | 100, 101, 102, 103, 104, 105 |
| STAD | cora | order-unspecified | 2 | 33.9941 | 42.9889 | 45.1248 | 19.1530 | 29.5193 | 100, 101, 102, 103, 104, 105 |
| TAGSAM | citeseer | order-unspecified | 2 | 56.1442 | 61.0240 | 64.0506 | 40.3314 | 30.4895 | 100, 101, 102, 103, 104, 105 |
| TAGSAM | cora | order-unspecified | 2 | 49.8096 | 59.4096 | 67.7595 | 25.8764 | 32.8426 | 100, 101, 102, 103, 104, 105 |
