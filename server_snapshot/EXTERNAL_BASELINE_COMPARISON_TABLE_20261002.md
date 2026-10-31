# External Baseline Comparison Table (2026-10-02)

Strict continual comparison: final stage of each available dataset/order/method group, restricted to Student seeds 150-155. Blank costs mean the source manifest did not expose that quantity.

| Method | Dataset | Order | Stage | Macro-F1 | Accuracy | Old F1 | New F1 | Worst-stage F1 | Memory rows | Student s | Construction s | Warning |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CaT-MMD-TAG | citeseer | order19611 | 2 | 56.9341 | 59.8746 | 64.2764 | 42.2494 | 56.6011 | 111 | 25.176 | 4.745 |  |
| CaT-MMD-TAG | citeseer | order19612 | 2 | 54.9264 | 58.8819 | 52.4984 | 59.7823 | 38.4252 | 111 | 24.902 | 4.540 |  |
| CaT-MMD-TAG | citeseer | order19613 | 2 | 60.0944 | 64.1066 | 53.7649 | 72.7534 | 49.1440 | 111 | 22.026 | 4.094 |  |
| CaT-MMD-TAG | citeseer | order19614 | 2 | 56.1982 | 59.4044 | 49.2439 | 70.1070 | 45.8339 | 111 | 22.362 | 4.219 |  |
| CaT-MMD-TAG | citeseer | order19615 | 2 | 57.0964 | 60.7106 | 63.7402 | 43.8088 | 57.0964 | 111 | 24.991 | 4.668 |  |
| CaT-MMD-TAG | cora | order19611 | 2 | 55.6425 | 60.7626 | 56.6979 | 54.2352 | 45.3379 | 94 | 21.177 | 5.446 |  |
| CaT-MMD-TAG | cora | order19612 | 2 | 50.9772 | 60.3321 | 38.5618 | 67.5312 | 32.7002 | 94 | 21.144 | 5.521 |  |
| CaT-MMD-TAG | cora | order19613 | 2 | 54.0293 | 61.6851 | 42.4801 | 69.4283 | 40.8225 | 94 | 19.029 | 5.197 |  |
| CaT-MMD-TAG | cora | order19614 | 2 | 54.6388 | 61.3161 | 59.1774 | 48.5874 | 45.6052 | 94 | 18.860 | 5.124 |  |
| CaT-MMD-TAG | cora | order19615 | 2 | 52.9773 | 62.3001 | 51.8753 | 54.4466 | 37.5968 | 94 | 21.117 | 5.530 |  |
| GECC-hard-TAG | citeseer | order19611 | 2 | 48.7244 | 50.5747 | 52.2628 | 41.6476 | 40.5566 | 111 | 25.084 | 0.109 |  |
| GECC-hard-TAG | citeseer | order19612 | 2 | 52.7062 | 55.7994 | 52.2760 | 53.5666 | 36.5993 | 111 | 25.037 | 0.131 |  |
| GECC-hard-TAG | citeseer | order19613 | 2 | 48.8412 | 53.2393 | 43.6591 | 59.2053 | 32.9582 | 111 | 22.075 | 0.144 |  |
| GECC-hard-TAG | cora | order19611 | 2 | 47.7764 | 53.3210 | 47.3025 | 48.4083 | 47.7764 | 94 | 21.258 | 0.125 |  |
| GECC-hard-TAG | cora | order19612 | 2 | 45.5756 | 50.4305 | 34.6181 | 60.1856 | 29.3603 | 94 | 21.174 | 0.132 |  |
| GECC-hard-TAG | cora | order19613 | 2 | 49.1539 | 53.3825 | 41.8619 | 58.8765 | 49.0834 | 94 | 20.657 | 0.159 |  |
| GECC-hard-TAG | cora | order19614 | 2 | 46.7992 | 52.0295 | 45.8833 | 48.0205 | 46.7992 | 94 | 18.892 | 0.141 |  |
| PUMA-inspired | citeseer | order19611 | 2 | 16.9030 | 20.2194 | 19.7909 | 11.1271 | 12.4912 | 111 | 23.980 |  |  |
| PUMA-inspired | citeseer | order19612 | 2 | 12.5261 | 14.8903 | 11.5650 | 14.4485 | 12.5261 | 111 | 30.910 |  |  |
| PUMA-inspired | cora | order19611 | 2 | 17.4073 | 17.4047 | 17.1139 | 17.7985 | 8.3044 | 94 | 25.061 |  |  |
| PUMA-inspired | cora | order19612 | 2 | 9.5389 | 10.0246 | 12.0781 | 6.1534 | 8.0079 | 94 | 20.090 |  |  |
| k-center/FPS | citeseer | order19611 | 2 | 51.7177 | 57.5235 | 60.3784 | 34.3963 | 51.7177 | 111 | 25.193 | 0.006 |  |
| k-center/FPS | citeseer | order19612 | 2 | 59.8736 | 66.3532 | 56.4972 | 66.6265 | 59.8736 | 111 | 25.380 | 0.007 |  |
| k-center/FPS | citeseer | order19613 | 2 | 52.3714 | 59.0909 | 44.2798 | 68.5544 | 46.9270 | 111 | 22.288 | 0.006 |  |
| k-center/FPS | citeseer | order19614 | 2 | 51.1899 | 58.0982 | 42.6527 | 68.2643 | 51.1899 | 111 | 22.479 | 0.006 |  |
| k-center/FPS | citeseer | order19615 | 2 | 53.1049 | 58.7774 | 62.2292 | 34.8564 | 53.1049 | 111 | 25.274 | 0.006 |  |
| k-center/FPS | cora | order19611 | 2 | 54.5393 | 59.7786 | 56.6386 | 51.7403 | 54.5393 | 94 | 21.261 | 0.005 |  |
| k-center/FPS | cora | order19612 | 2 | 50.6047 | 54.0590 | 38.6141 | 66.5922 | 8.4075 | 94 | 21.389 | 0.004 |  |
| k-center/FPS | cora | order19613 | 2 | 43.9815 | 47.1710 | 30.3781 | 62.1193 | 27.9653 | 94 | 19.047 | 0.008 |  |
| k-center/FPS | cora | order19614 | 2 | 53.3199 | 61.3776 | 59.3131 | 45.3290 | 53.3199 | 94 | 18.691 | 0.006 |  |
| k-center/FPS | cora | order19615 | 2 | 41.9008 | 43.9114 | 42.3493 | 41.3028 | 41.9008 | 94 | 21.253 | 0.010 |  |
| medoid | citeseer | order19611 | 2 | 47.3321 | 51.8809 | 54.7652 | 32.4659 | 47.3321 | 111 | 25.282 | 0.040 |  |
| medoid | citeseer | order19612 | 2 | 48.9570 | 53.5528 | 47.6105 | 51.6500 | 36.0937 | 111 | 25.350 | 0.055 |  |
| medoid | citeseer | order19613 | 2 | 53.7274 | 57.3145 | 48.9730 | 63.2364 | 31.9292 | 111 | 22.359 | 0.076 |  |
| medoid | cora | order19611 | 2 | 46.3926 | 51.7835 | 50.8938 | 40.3911 | 32.3828 | 94 | 21.261 | 0.045 |  |
| medoid | cora | order19612 | 2 | 46.5837 | 55.2276 | 34.6670 | 62.4726 | 15.1226 | 94 | 21.311 | 0.037 |  |
| medoid | cora | order19613 | 2 | 53.6659 | 60.7011 | 45.6999 | 64.2873 | 43.2138 | 94 | 18.906 | 0.072 |  |
| medoid | cora | order19614 | 2 | 51.0604 | 57.0726 | 52.4622 | 49.1913 | 32.3828 | 94 | 18.702 | 0.054 |  |
| semantic-recursive | citeseer | order19611 | 2 | 57.7037 | 62.3302 | 66.3870 | 40.3371 | 34.4598 | 111 | 25.193 |  |  |
| semantic-recursive | citeseer | order19612 | 2 | 54.0072 | 61.4420 | 52.3956 | 57.2305 | 54.0072 | 111 | 25.255 |  |  |
| semantic-recursive | cora | order19611 | 2 | 50.2142 | 56.3961 | 49.4189 | 51.2746 | 48.7806 | 94 | 21.247 |  |  |
| semantic-recursive | cora | order19612 | 2 | 50.1366 | 58.1796 | 41.0960 | 62.1908 | 17.5863 | 94 | 21.288 |  |  |

TAGSAM and STAD use Student seeds 100-105 and are reported separately in `EXTERNAL_BASELINE_ADAPTED_REFERENCE_TABLE_20261003.md`. Reference-only methods are listed in `EXTERNAL_BASELINE_SUMMARY_20261002.md`.
