# Resource comparison and full-history reference

Registered 2026-10-22 before the new outcomes. No paper edits or test-based selection.

## Questions and frozen scope

1. Compare the same class-block CoTAG-Q constructor on current-only increments
   with a single full-history construction on all final legal training nodes.
2. Compare optimized CoTAG-Q, the archived CaT adapter, and the archived GECC-hard
   TAG adapter under the same measurement boundaries, final memory budgets and
   common student. These remain adaptations, not unchanged official reproductions.

ArXiv and SportsFit are the two datasets with the largest prior construction
peaks. Reuse their completed combined-implementation streams from
`cotag_memory_optimization_20261022_v2`, checking hashes and protocol. Do not rerun
those 6 completed streams. New runs: 2 datasets x 3 constructions x 3 arms
(`full_history`, `cat_cctp`, `gecc_hard`), 18 total. Seeds 271006/271108/271109,
order 9101, students 150/151/152, 2048 updates, batch 24. All scores are report-dev.
ArXiv is the archived THREE-stage class protocol, SportsFit FIVE. Neither is a
chronological experiment or an ArXiv base+4 replacement.

## Full-history reference

Use the union of exactly the final legal training pool, with its induced graph,
parent=None, and the exact concatenation of incremental class/slot quotas. The
full-history optimizer sees original historical nodes and cross-stage training
edges. It is an access-relaxed reference, not a feasible baseline under the
current-only access restriction. Final student evaluation uses the identical
cohort, classes and transductive graph as the incremental run.

Both constructors use legal class blocks, the same complete objective, GM, DM,
propagation, trust radius, float32 Sinkhorn settings and 30 outer updates per
construction call. The full reference makes one call; the incremental procedure
makes T calls. Report 30 versus 30*T updates explicitly. This matches the optimizer
configuration per invocation, not total compute or optimization opportunity.
Do not claim that a peak difference proves a speedup or an accuracy advantage at
matched total compute. Full-history stage RNG is stage0 and the original seed.

## Common measurement

Remove original-graph GPU inference caches before every construction for all
methods; retain the same MiniLM encoder and CPU data loader. Include encoding or
selection done inside each native constructor. Exclude common preloaded data,
delta extraction, serialization, and student/inference from construction timing.
Record allocated and reserved GPU peaks, CPU RSS sampled at20ms, wall and CPU
process time, stage dimensions, payload bytes, and the common student phase.

CPU threads4 and device cuda:0 within each subprocess. One sequential lane per
dataset, as in the reused CoTAG batch, on the same two GPUs. Times are descriptive:
new versus reused observations and co-located lanes are not isolated speed trials.
All methods keep their frozen construction budgets: CaT3000 updates, GECC20
restarts; do not equalize incomparable iteration counts or weaken a baseline.

Preserve every stage memory and prediction and every failed run. A failed attempt
is not scored as zero. No automatic retry, precision/cap/step change or new tuning
after an OOM/numerical failure. Leave at least8GiB free disk. A complete batch means
each registered attempt has a recorded outcome; failures remain failures.

## Analysis

First average three students within each construction, then summarize the three
construction seeds and paired95% t intervals. Independently recompute F1/Accuracy
from predictions and verify memory hashes, exact quotas, frozen prefixes, pool
and report hashes. Distinguish allocated/reserved memory, CPU memory, time, and
prediction quality; no blanket efficiency ranking from only one dimension.

The four-issue writing synthesis must distinguish shared versus independent Q
from free-X optimization, native-core Cora TAGSAM from large bounded adapters,
class increments from publication-year streams, and lower student costs from
construction costs. It must preserve negative results and scope limitations.
