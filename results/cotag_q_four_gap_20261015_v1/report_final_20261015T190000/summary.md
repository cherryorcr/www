# CoTAG-Q four-gap evidence

Batch status: complete; pending cells: 0.

All current test-mask metrics below are post-hoc. No method is selected from these results.

## Text initialization and pair feedback

Three construction seeds, one order, three Student seeds per construction; F1 is a percentage.

| Dataset | Arm | F1 mean | Construction SD |
| --- | --- | --- | --- |
| arxiv | full | 52.96 | 0.21 |
| arxiv | no_text_init | 53.52 | 0.79 |
| arxiv | no_pair_feedback | 53.13 | 0.28 |
| arxiv | both_off | 53.32 | 0.41 |
| sportsfit | full | 80.02 | 0.30 |
| sportsfit | no_text_init | 77.50 | 0.83 |
| sportsfit | no_pair_feedback | 80.30 | 0.98 |
| sportsfit | both_off | 77.64 | 0.69 |
| bookhis | full | 48.53 | 0.85 |
| bookhis | no_text_init | 46.79 | 1.21 |
| bookhis | no_pair_feedback | 48.14 | 2.04 |
| bookhis | both_off | 47.87 | 0.63 |

| Dataset | Full minus | F1 difference | 95% paired CI | Positive | Worst |
| --- | --- | --- | --- | --- | --- |
| arxiv | no_text_init | -0.56 | [-2.05, 0.93] | 0/3 | -1.23 |
| arxiv | no_pair_feedback | -0.16 | [-1.36, 1.03] | 1/3 | -0.46 |
| arxiv | both_off | -0.36 | [-0.89, 0.18] | 0/3 | -0.60 |
| sportsfit | no_text_init | 2.51 | [0.48, 4.55] | 3/3 | 1.74 |
| sportsfit | no_pair_feedback | -0.28 | [-3.47, 2.91] | 1/3 | -1.30 |
| sportsfit | both_off | 2.38 | [0.51, 4.25] | 3/3 | 1.52 |
| bookhis | no_text_init | 1.74 | [-3.32, 6.80] | 2/3 | -0.60 |
| bookhis | no_pair_feedback | 0.39 | [-2.68, 3.46] | 1/3 | -0.51 |
| bookhis | both_off | 0.66 | [-2.37, 3.68] | 2/3 | -0.72 |

Initialization intervention removes the text term from both prototype selection and initial logits. 
Pair intervention removes the node-payload pair term on both real and synthetic proxy losses; prompt supervision remains.
The interaction is full minus no-init minus no-pair plus both-off. All effects are conditional on the frozen constructor.

## Full-history matched reference

Same legal pool, order, masks, text preprocessing, encoder, Student and 2048-update budget. 
Full topology means edges induced by legal training nodes, not a fully supervised entire-graph upper bound.

| Dataset | Reference | Metric | Ours | Full | Ratio of means (%) | Mean paired ratio (%) | Loss-row coverage (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arxiv | full_ef | final_f1 | 52.96 | 50.93 | 103.99 | 104.01 | 33.91 |
| arxiv | full_ef | accuracy | 70.64 | 71.11 | 99.34 | 99.35 | 33.91 |
| arxiv | full_topology | final_f1 | 52.96 | 53.47 | 99.04 | 99.07 | 33.91 |
| arxiv | full_topology | accuracy | 70.64 | 72.37 | 97.60 | 97.61 | 33.91 |
| sportsfit | full_ef | final_f1 | 80.02 | 79.98 | 100.05 | 100.05 | 33.30 |
| sportsfit | full_ef | accuracy | 85.38 | 88.29 | 96.71 | 96.71 | 33.30 |
| sportsfit | full_topology | final_f1 | 80.02 | 81.46 | 98.23 | 98.24 | 33.30 |
| sportsfit | full_topology | accuracy | 85.38 | 89.33 | 95.58 | 95.58 | 33.30 |
| bookhis | full_ef | final_f1 | 48.53 | 40.16 | 120.84 | 122.14 | 81.76 |
| bookhis | full_ef | accuracy | 81.38 | 81.48 | 99.88 | 99.88 | 81.76 |
| bookhis | full_topology | final_f1 | 48.53 | 46.96 | 103.33 | 104.51 | 81.76 |
| bookhis | full_topology | accuracy | 81.38 | 82.09 | 99.14 | 99.14 | 81.76 |

Loss-row coverage counts rows sampled for supervision, not graph-forward exposure. 
All nodes participate in the full graph forward; training-pool neighbors can receive indirect gradients.

## TAGSAM last-stage response budget

Same fixed prefix, PMI initialization and teacher; 100 versus 500 outer updates, 10 inner steps. 
This is a last-stage diagnostic, not all-stage retraining or official 5000x15 reproduction.

| Dataset | Outer steps | F1 | Accuracy | Response seconds |
| --- | --- | --- | --- | --- |
| arxiv | 100 | 42.22 | 57.64 | 7.26 |
| arxiv | 500 | 38.68 | 59.00 | 36.87 |
| sportsfit | 100 | 69.10 | 81.36 | 6.76 |
| sportsfit | 500 | 61.13 | 74.27 | 33.28 |
| bookhis | 100 | 11.03 | 59.38 | 5.78 |
| bookhis | 500 | 10.48 | 58.85 | 28.69 |

## Identity-operator resource diagnostic

Stage zero, one seed, one dense then sparse run per graph. Timing is descriptive, without repeated timing inference.

| Dataset | Operator | Seconds | Allocated GiB | Reserved GiB | X max error | Same text |
| --- | --- | --- | --- | --- | --- | --- |
| arxiv | dense_identity | 82.02 | 11.56 | 12.23 | 0.000941 | False |
| arxiv | sparse_identity | 82.23 | 11.50 | 12.17 | 0.000941 | False |
| sportsfit | dense_identity | 123.08 | 23.44 | 26.10 | 0.00267 | False |
| sportsfit | sparse_identity | 123.15 | 23.33 | 25.62 | 0.00267 | False |
| bookhis | dense_identity | 20.00 | 1.17 | 1.57 | 0.000336 | True |
| bookhis | sparse_identity | 19.65 | 1.17 | 1.57 | 0.000336 | True |
