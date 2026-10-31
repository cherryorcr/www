# Unified-Q controlled diagnostics and v3 preregistration

## Frozen scope

Do not delete files, alter previous results, submit git commits, or select a
configuration from final test scores. Use report-dev for development only.
Main setting: current-delta-only, own synthetic parent, 5% cumulative rate
with recorded class-floor exceptions. The inherited fixed schedule is a
separate stress test, not the same protocol. Student: GraphTextPromptStudent,
2048 steps, batch 24, seeds 150/151/152. Frozen MiniLM and 320-character
materializable payloads. Five class-incremental stages, reference order 9101.

## Execution order

1. Reconstruct each saved frozen-prefix memory from the final checkpoint;
   require exact equality to every archived stage memory hash. Evaluate both
   legacy row-wise InfoNCE and Prompt Student on that SAME memory. Two datasets
   (WikiCS, BookHis), both budget settings, CaT-CCTP and Unified-Q. Keep common
   seeds, steps, batches, masks and metric definitions. Record the separate
   propagation operator hashes because legacy normalization is part of the
   legacy Student, not a constructor change. Do not interpret this as a
   single-loss ablation or strict unseen-class zero-shot evaluation.
2. Numerical audit: fixed 30/64/128 Sinkhorn steps versus a predeclared
   converged solver. Tolerance 1e-4 for absolute row error and relative column
   error. Maximum 512 iterations; failure is explicit, never an F1-based
   fallback. Check initialization and final trained assignments, not only a
   random toy matrix. Class mask and nonnegative transport must be retained.
3. Component attribution: no outer update; frozen Q/train R; train Q/frozen R;
   no text gradient with frozen initial payload; text learning without new
   topology; complete strict solver. The no-text-gradient control still saves
   the same initial payload for the common evaluation Student; it is NOT a
   claim of a text-free downstream model.
4. Proxy alignment: strict solver with 384/384 proxy, other losses unchanged.
   Then correct propagation scaling separately: mean squared difference of
   unit vectors currently divides by feature dimension. Compare mean cosine
   distance without adding a second propagation loss.
5. Topology: bounded degree-preserving edge swaps after Q-induced top-k,
   accepting only reductions in the legal current-delta propagation target.
   Retain total edges, per-node degree and frozen old-old edges. Use the same
   hard materialization in the outer forward and final saved graph. This is
   a constrained numerical sparsification step, not exact discrete-gradient
   optimization. Preregister the full candidate before looking at its F1.
6. Old-feature adaptation: frozen vs 1% per-stage trust region vs no hard
   radius, same anchor objective otherwise. Do not update old text/edges or
   read old raw. Report separately from the frozen-prefix protocol.
7. Validate the preregistered full candidate and CaT-CCTP on a third dataset
   (SportsFit), then independent orders 9102/9103 and construction seeds
   271007/271008. Report the 3x3 crossed cells on WikiCS/BookHis separately
   from the single-order third-dataset transfer. Do not promote the best arm
   retrospectively or pool budget protocols.

## Artifacts and limits

Use a new immutable remote code snapshot and a new results directory.
Save scalar audits, hashes, per-seed metrics and at most the latest compact
memory per method, not all raw graphs or dense Q tensors. Available disk was
5.36 GiB before this round; do not clean user data to make experiments fit.
Report Final and stage-average Macro-F1/Accuracy, Old/New F1, worst stage,
paired order-level uncertainty, memory rows, construction/Student time,
peak GPU allocation, Sinkhorn violations and actual update counts.

Existing constraints remain: no boundary sketch in this main round; no EM,
selector, dynamic budget or trainable text encoder. Reuse the existing
propagation loss rather than falsely claiming it as a new component.

## Execution amendments (numerical/resource failures only)

- Strict Sinkhorn exceeded the original 512-iteration ceiling. A separate
  retry snapshot raised only the ceiling to 2048; tolerance stayed 1e-4.
- Both original SportsFit UQ arms exhausted a 48-GiB GPU during stage zero.
  The `memscale` snapshot checkpoints large class blocks (at least one
  million entries) in eight-iteration chunks, including solves of 64 or
  fewer iterations. Data, budgets, losses, precision, iterations and seeds
  are unchanged. Forward and first/second derivative equivalence is tested.
- Original failed/partial artifacts are retained. Successful `retry_` and
  `memretry_` results resolve tasks without pooling failed attempts into
  method means. See the final results document and completion audit.
