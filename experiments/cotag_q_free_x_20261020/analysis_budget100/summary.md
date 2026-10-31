# Longer-budget diagnostic

One construction (271006), one order (9101), three Student fits per memory.
The 30-update rows below use that same construction, not the three-construction primary mean.
100-update runs use an amended Sinkhorn tolerance of 2e-4 instead of 1e-4.
Therefore this is not a pure step-only contrast. No construction-level CI can be estimated here.
Test-mask observations were inspected before; these are not fresh blind evidence.

## report_dev

| Dataset | Student | Arm | Updates | F1 | Accuracy |
|---|---|---|---:|---:|---:|
| ArXiv | GCN | Independent X + learned text Q | 30 | 51.30 | 67.68 |
| ArXiv | GCN | Independent X + learned text Q | 100 | 48.85 | 67.33 |
| ArXiv | GCN | CoTAG-Q | 30 | 51.50 | 71.24 |
| ArXiv | GCN | CoTAG-Q | 100 | 51.54 | 71.86 |
| ArXiv | GraphSAGE | Independent X + learned text Q | 30 | 51.26 | 69.27 |
| ArXiv | GraphSAGE | Independent X + learned text Q | 100 | 50.99 | 70.53 |
| ArXiv | GraphSAGE | CoTAG-Q | 30 | 52.43 | 71.91 |
| ArXiv | GraphSAGE | CoTAG-Q | 100 | 52.57 | 72.34 |
| SportsFit | GCN | Independent X + learned text Q | 30 | 75.41 | 81.61 |
| SportsFit | GCN | CoTAG-Q | 30 | 80.15 | 86.02 |
| SportsFit | GraphSAGE | Independent X + learned text Q | 30 | 72.02 | 77.08 |
| SportsFit | GraphSAGE | CoTAG-Q | 30 | 77.93 | 82.63 |
| BookHis | GCN | Independent X + learned text Q | 30 | 45.39 | 80.69 |
| BookHis | GCN | Independent X + learned text Q | 100 | 43.88 | 80.08 |
| BookHis | GCN | CoTAG-Q | 30 | 47.41 | 81.18 |
| BookHis | GCN | CoTAG-Q | 100 | 45.83 | 82.62 |
| BookHis | GraphSAGE | Independent X + learned text Q | 30 | 47.63 | 81.08 |
| BookHis | GraphSAGE | Independent X + learned text Q | 100 | 46.20 | 80.43 |
| BookHis | GraphSAGE | CoTAG-Q | 30 | 47.60 | 82.84 |
| BookHis | GraphSAGE | CoTAG-Q | 100 | 47.15 | 82.73 |

| Dataset | Student | Arm | F1 change, 100 minus 30 |
|---|---|---|---:|
| ArXiv | GCN | Independent X + learned text Q | -2.46 |
| ArXiv | GCN | CoTAG-Q | +0.04 |
| ArXiv | GraphSAGE | Independent X + learned text Q | -0.27 |
| ArXiv | GraphSAGE | CoTAG-Q | +0.14 |
| BookHis | GCN | Independent X + learned text Q | -1.51 |
| BookHis | GCN | CoTAG-Q | -1.58 |
| BookHis | GraphSAGE | Independent X + learned text Q | -1.43 |
| BookHis | GraphSAGE | CoTAG-Q | -0.45 |

| Dataset | Student | Updates | F1 difference, full minus independent X |
|---|---|---:|---:|
| ArXiv | GCN | 30 | +0.20 |
| ArXiv | GCN | 100 | +2.70 |
| ArXiv | GraphSAGE | 30 | +1.17 |
| ArXiv | GraphSAGE | 100 | +1.59 |
| SportsFit | GCN | 30 | +4.74 |
| SportsFit | GraphSAGE | 30 | +5.91 |
| BookHis | GCN | 30 | +2.01 |
| BookHis | GCN | 100 | +1.94 |
| BookHis | GraphSAGE | 30 | -0.03 |
| BookHis | GraphSAGE | 100 | +0.95 |

## sealed_test

| Dataset | Student | Arm | Updates | F1 | Accuracy |
|---|---|---|---:|---:|---:|
| ArXiv | GCN | Independent X + learned text Q | 30 | 52.35 | 68.10 |
| ArXiv | GCN | Independent X + learned text Q | 100 | 49.09 | 67.17 |
| ArXiv | GCN | CoTAG-Q | 30 | 53.20 | 71.05 |
| ArXiv | GCN | CoTAG-Q | 100 | 53.10 | 71.58 |
| ArXiv | GraphSAGE | Independent X + learned text Q | 30 | 52.10 | 69.23 |
| ArXiv | GraphSAGE | Independent X + learned text Q | 100 | 51.37 | 69.78 |
| ArXiv | GraphSAGE | CoTAG-Q | 30 | 53.40 | 71.15 |
| ArXiv | GraphSAGE | CoTAG-Q | 100 | 53.16 | 71.63 |
| SportsFit | GCN | Independent X + learned text Q | 30 | 75.50 | 81.29 |
| SportsFit | GCN | CoTAG-Q | 30 | 80.78 | 85.68 |
| SportsFit | GraphSAGE | Independent X + learned text Q | 30 | 72.40 | 76.89 |
| SportsFit | GraphSAGE | CoTAG-Q | 30 | 78.26 | 82.59 |
| BookHis | GCN | Independent X + learned text Q | 30 | 46.24 | 79.86 |
| BookHis | GCN | Independent X + learned text Q | 100 | 43.96 | 79.81 |
| BookHis | GCN | CoTAG-Q | 30 | 49.95 | 79.99 |
| BookHis | GCN | CoTAG-Q | 100 | 47.73 | 81.88 |
| BookHis | GraphSAGE | Independent X + learned text Q | 30 | 48.42 | 80.42 |
| BookHis | GraphSAGE | Independent X + learned text Q | 100 | 46.18 | 79.95 |
| BookHis | GraphSAGE | CoTAG-Q | 30 | 50.13 | 81.99 |
| BookHis | GraphSAGE | CoTAG-Q | 100 | 49.18 | 82.09 |

| Dataset | Student | Arm | F1 change, 100 minus 30 |
|---|---|---|---:|
| ArXiv | GCN | Independent X + learned text Q | -3.26 |
| ArXiv | GCN | CoTAG-Q | -0.09 |
| ArXiv | GraphSAGE | Independent X + learned text Q | -0.74 |
| ArXiv | GraphSAGE | CoTAG-Q | -0.24 |
| BookHis | GCN | Independent X + learned text Q | -2.28 |
| BookHis | GCN | CoTAG-Q | -2.22 |
| BookHis | GraphSAGE | Independent X + learned text Q | -2.24 |
| BookHis | GraphSAGE | CoTAG-Q | -0.95 |

| Dataset | Student | Updates | F1 difference, full minus independent X |
|---|---|---:|---:|
| ArXiv | GCN | 30 | +0.84 |
| ArXiv | GCN | 100 | +4.01 |
| ArXiv | GraphSAGE | 30 | +1.29 |
| ArXiv | GraphSAGE | 100 | +1.79 |
| SportsFit | GCN | 30 | +5.28 |
| SportsFit | GraphSAGE | 30 | +5.86 |
| BookHis | GCN | 30 | +3.71 |
| BookHis | GCN | 100 | +3.77 |
| BookHis | GraphSAGE | 30 | +1.70 |
| BookHis | GraphSAGE | 100 | +3.00 |

## Loss stability

Two adjacent changes between the final three ten-update means must both be below 1%.
Each objective is examined within its own run; their absolute minima are not equivalent.

| Dataset | Arm | Stage | Component | Last loss | Window changes (%) | Plateau |
|---|---|---:|---|---:|---|---|
| ArXiv | CoTAG-Q | 0 | loss | 0.071653 | 0.43, 0.56 | True |
| ArXiv | CoTAG-Q | 0 | gradient_match | 0.002169 | 0.72, 3.33 | False |
| ArXiv | CoTAG-Q | 0 | distribution_match | 0.063409 | 0.49, 0.46 | True |
| ArXiv | CoTAG-Q | 0 | propagation | 0.012092 | 0.42, 0.29 | True |
| ArXiv | CoTAG-Q | 1 | loss | 0.304286 | 0.35, 0.15 | True |
| ArXiv | CoTAG-Q | 1 | gradient_match | 0.006850 | 5.72, 1.47 | False |
| ArXiv | CoTAG-Q | 1 | distribution_match | 0.291045 | 0.22, 0.19 | True |
| ArXiv | CoTAG-Q | 1 | propagation | 0.012575 | 0.09, 0.14 | True |
| ArXiv | CoTAG-Q | 2 | loss | 0.120577 | 0.23, 0.57 | True |
| ArXiv | CoTAG-Q | 2 | gradient_match | 0.004230 | 0.97, 7.66 | False |
| ArXiv | CoTAG-Q | 2 | distribution_match | 0.110924 | 0.27, 0.25 | True |
| ArXiv | CoTAG-Q | 2 | propagation | 0.010759 | 0.48, 0.38 | True |
| ArXiv | Independent X + learned text Q | 0 | loss | 0.018396 | 7.66, 5.77 | False |
| ArXiv | Independent X + learned text Q | 0 | gradient_match | 0.002976 | 3.29, 0.42 | False |
| ArXiv | Independent X + learned text Q | 0 | distribution_match | 0.001102 | 11.70, 4.61 | False |
| ArXiv | Independent X + learned text Q | 0 | propagation | 0.028581 | 8.30, 7.27 | False |
| ArXiv | Independent X + learned text Q | 1 | loss | 0.048201 | 4.72, 3.44 | False |
| ArXiv | Independent X + learned text Q | 1 | gradient_match | 0.003049 | 3.17, 1.46 | False |
| ArXiv | Independent X + learned text Q | 1 | distribution_match | 0.023541 | 3.70, 2.96 | False |
| ArXiv | Independent X + learned text Q | 1 | propagation | 0.043038 | 6.04, 4.71 | False |
| ArXiv | Independent X + learned text Q | 2 | loss | 0.019176 | 8.00, 6.30 | False |
| ArXiv | Independent X + learned text Q | 2 | gradient_match | 0.002512 | 10.32, 6.27 | False |
| ArXiv | Independent X + learned text Q | 2 | distribution_match | 0.001264 | 11.80, 6.89 | False |
| ArXiv | Independent X + learned text Q | 2 | propagation | 0.030720 | 7.31, 6.28 | False |
| BookHis | CoTAG-Q | 0 | loss | 0.088539 | 0.13, 0.12 | True |
| BookHis | CoTAG-Q | 0 | gradient_match | 0.002122 | 0.81, 1.53 | False |
| BookHis | CoTAG-Q | 0 | distribution_match | 0.077117 | 0.09, 0.07 | True |
| BookHis | CoTAG-Q | 0 | propagation | 0.018296 | 0.25, 0.16 | True |
| BookHis | CoTAG-Q | 1 | loss | 0.103429 | 0.09, 0.15 | True |
| BookHis | CoTAG-Q | 1 | gradient_match | 0.007911 | 0.66, 1.26 | False |
| BookHis | CoTAG-Q | 1 | distribution_match | 0.088645 | 0.03, 0.08 | True |
| BookHis | CoTAG-Q | 1 | propagation | 0.004314 | 0.91, 0.49 | True |
| BookHis | CoTAG-Q | 2 | loss | 0.121537 | 0.05, 0.01 | True |
| BookHis | CoTAG-Q | 2 | gradient_match | 0.017423 | 0.55, 0.04 | True |
| BookHis | CoTAG-Q | 2 | distribution_match | 0.093172 | 0.06, 0.00 | True |
| BookHis | CoTAG-Q | 2 | propagation | 0.007638 | 0.26, 0.02 | True |
| BookHis | CoTAG-Q | 3 | loss | 0.050527 | 0.08, 0.06 | True |
| BookHis | CoTAG-Q | 3 | gradient_match | 0.004344 | 1.85, 0.09 | False |
| BookHis | CoTAG-Q | 3 | distribution_match | 0.042344 | 0.06, 0.03 | True |
| BookHis | CoTAG-Q | 3 | propagation | 0.007470 | 0.38, 0.34 | True |
| BookHis | CoTAG-Q | 4 | loss | 0.023681 | 0.17, 0.29 | True |
| BookHis | CoTAG-Q | 4 | gradient_match | 0.002621 | 0.62, 1.21 | False |
| BookHis | CoTAG-Q | 4 | distribution_match | 0.013834 | 0.43, 0.00 | True |
| BookHis | CoTAG-Q | 4 | propagation | 0.008596 | 0.66, 0.68 | True |
| BookHis | Independent X + learned text Q | 0 | loss | 0.019433 | 5.23, 5.22 | False |
| BookHis | Independent X + learned text Q | 0 | gradient_match | 0.001833 | 6.81, 12.93 | False |
| BookHis | Independent X + learned text Q | 0 | distribution_match | 0.006180 | 4.89, 3.74 | False |
| BookHis | Independent X + learned text Q | 0 | propagation | 0.022540 | 5.09, 4.09 | False |
| BookHis | Independent X + learned text Q | 1 | loss | 0.042509 | 1.90, 0.57 | False |
| BookHis | Independent X + learned text Q | 1 | gradient_match | 0.002404 | 10.66, 7.17 | False |
| BookHis | Independent X + learned text Q | 1 | distribution_match | 0.013270 | 1.76, 0.91 | False |
| BookHis | Independent X + learned text Q | 1 | propagation | 0.045327 | 1.84, 1.77 | False |
| BookHis | Independent X + learned text Q | 2 | loss | 0.102013 | 0.14, 0.13 | True |
| BookHis | Independent X + learned text Q | 2 | gradient_match | 0.018094 | 0.75, 0.58 | True |
| BookHis | Independent X + learned text Q | 2 | distribution_match | 0.064951 | 0.27, 0.03 | True |
| BookHis | Independent X + learned text Q | 2 | propagation | 0.026635 | 1.15, 0.55 | False |
| BookHis | Independent X + learned text Q | 3 | loss | 0.010276 | 3.85, 1.66 | False |
| BookHis | Independent X + learned text Q | 3 | gradient_match | 0.001504 | 8.80, 28.36 | False |
| BookHis | Independent X + learned text Q | 3 | distribution_match | 0.002528 | 4.49, 1.76 | False |
| BookHis | Independent X + learned text Q | 3 | propagation | 0.012265 | 2.73, 1.77 | False |
| BookHis | Independent X + learned text Q | 4 | loss | 0.017293 | 0.89, 0.65 | True |
| BookHis | Independent X + learned text Q | 4 | gradient_match | 0.001663 | 2.40, 6.83 | False |
| BookHis | Independent X + learned text Q | 4 | distribution_match | 0.005107 | 0.62, 0.12 | True |
| BookHis | Independent X + learned text Q | 4 | propagation | 0.015486 | 2.23, 0.11 | False |

## Attempts and failures

Missing or failed outcomes are not zero scores. Failed attempts are not independent repetitions.

- cotag_q_free_x_budget100_20261020\arxiv_o9101_c271006\results.json: interrupted (process absence verified); 0 saved stage rows; TerminatedDuringGPUConflict: first attempt overlapped the primary queue and was stopped; immutable raw file still says running
- cotag_q_free_x_budget100_20261020_v2\arxiv_o9101_c271006\results.json: failed; 0 saved stage rows; RuntimeError: Sinkhorn failed tolerance 0.0001: row=0.00010293722152709961, col=2.902775406710134e-07
- cotag_q_free_x_budget100_20261020_v3\arxiv_o9101_c271006\results.json: failed; 0 saved stage rows; AssertionError: {'budget_ok': True, 'prefix_ok': True, 'finite_ok': True, 'sinkhorn_feasible': False, 'sinkhorn_block_error': 0.00019997358322143555, 'sinkhorn_full_error': 0.00019997358322143555, 'sinkhorn_solver_tolerance': 0.0001, 'full_matrix_rounding_allowance': 9.5367431640625e-07, 'numerical_warning': True}
- cotag_q_free_x_budget100_20261020_v4\arxiv_o9101_c271006\results.json: failed; 2 saved stage rows; BrokenPipeError: [Errno 32] Broken pipe
- cotag_q_free_x_budget100_20261020_v4\sportsfit_o9101_c271006\results.json: failed; 0 saved stage rows; OutOfMemoryError: CUDA out of memory. Tried to allocate 496.00 MiB. GPU 0 has a total capacity of 47.37 GiB of which 75.25 MiB is free. Including non-PyTorch memory, this process has 47.28 GiB memory in use. Of the allocated memory 44.37 GiB is allocated by PyTorch, and 2.45 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- cotag_q_free_x_budget100_single_arm_20261021\sportsfit_independent_x_o9101_c271006\results.json: failed; 0 saved stage rows; OutOfMemoryError: CUDA out of memory. Tried to allocate 496.00 MiB. GPU 0 has a total capacity of 47.37 GiB of which 187.25 MiB is free. Including non-PyTorch memory, this process has 47.17 GiB memory in use. Of the allocated memory 45.07 GiB is allocated by PyTorch, and 1.64 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
