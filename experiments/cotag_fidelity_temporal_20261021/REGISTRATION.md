# Baseline fidelity and chronological evaluation

Registered before this batch's downstream outcomes on 2026-10-21.
No paper changes. No primary, sealed, or failed artifact is overwritten.

## Chronological extension

Use original ArXiv publication years. Five stages end in 2015, 2016, 2017,
2018, and 2020. Do not split a year to balance node counts. This tests
recurring-class temporal arrivals, not a second class-incremental order.
The last stage contains 2019 and 2020. Category prompts are known taxonomy
metadata, not future node supervision.

Constructors receive only the current-stage legal training nodes, their
induced edges, and their own frozen synthetic prefix. Keep the existing
ID-based pool/dev/test masks, 5% cumulative training-pool budget, frozen
MiniLM, 30 CoTAG-Q construction updates, and common 2048-update prompt
student (batch 24; seeds 150/151/152). Use construction seeds
271006/271108/271109 for CoTAG-Q, CaT, and GECC-hard.

Evaluate the same trained model on (a) all nodes that have arrived by the
current stage and (b) current-stage nodes only. The latter cannot evaluate
old queries and must not be labeled old-class retention. In the first mode
the serving graph, but not the constructor, can access historical nodes.
Neither graph contains future nodes. Edges come from the final citation
graph restricted to available endpoints. This is an explicit edge-time
assumption, not a record of real edge events. CPU data are held by the
offline harness; physical erasure of historical data is not demonstrated.
Report all-40-class Macro-F1, Accuracy, arrival-cohort metrics, actual graph
sizes, training resources, and per-stage construction resources. No
test-based tuning or stopping.

## TAGSAM fidelity

Audit the pinned upstream source before choosing claims. Its default 5000
outer iterations and 15 inner loops mean 15 full synthetic-data epochs,
not 15 individual minibatch updates. The historical adapter used 100 outer
iterations, 10 minibatch updates, a fixed 256-row response subset, changed
encoders and a common downstream optimizer. It is not an official run.

First compare complete class-incremental chains with 100x10, 500x10 and
5000x15 **minibatch updates** on ArXiv, SportsFit and BookHis, with one
construction seed 271006, order 9101, and three common student seeds. Every
arm has its own stage-zero-to-final prefix and teacher; selector initial
blocks may be reused because they do not depend on the historical prefix.
The bounded reference is rerun with the same implementation for consistency.
This only tests adapter optimization budget, not official-epoch equivalence.

Separately audit full-epoch feasibility and retained mechanisms. Any
full-epoch native-core experiment needs its own manifest specifying batch
size, number of complete passes, response resampling and downstream rates.
Do not label a large minibatch-count run as a high-fidelity official
reproduction. Local STAD generator/supervision changes remain explicit;
more rows or student repeats do not remove these fidelity differences.

### Native-epoch follow-up, registered before its outcomes

Run one complete Cora chain with the native 5000 outer updates and 15
synthetic-data epochs per update (batch 20, dropping incomplete tails as
the upstream function does). Execute the pinned upstream CLIP, GCN,
residual text projection, ReparamModule, inner epoch function and weighted
BCE directly. Restore frozen BERT CLS embeddings inside condensation,
SGD with learned inner rates, the original StepLR(200, 0.95), and a newly
sampled response set capped at 2000 per update. Retain native teacher
defaults (3 epochs, batch 1024, Adam 2e-5), but use the final teacher
instead of selecting by external evaluation accuracy. A CPU miniature test
checks literal epoch execution and prefix freezing before launching.

For the access-controlled adaptation use the full legal source graph for
message passing rather than NeighborSampler, and optimize only the new
rows. Keep the same cumulative 5% budget and stored 320-character text cap.
Re-encode stored text with common MiniLM for the common final student.
This is a higher-fidelity **construction adaptation**, not unchanged native
end-to-end reproduction or a large-graph fidelity claim. A native student
using distilled rates has a different contract and is not silently mixed
into the primary comparison. Save failures if the native epoch semantics
are infeasible at the registered budget.

The native Cora diagnostic may run on the same GPU as the bounded adapter
queue. Preserve process-specific allocated peaks, but do not use these
co-located construction timings as a controlled efficiency comparison.

## Reporting

### Native-student transfer diagnostic

After the Cora native-core construction finishes, evaluate the same final
memory with the native CLIP model, frozen BERT CLS, learned graph/text rates,
SGD momentum .9 and weight decay 5e-4. Train exactly 15 full epochs at batch
32, retaining incomplete batches, and report the last epoch for seeds
150/151/152. Use batch-local self-loop computation, equivalent to sampling
the self-loop synthetic graph. Keep the common class prompts and all-seen
category evaluation on the same masks. Do not select the best test epoch
or reproduce the upstream five-way grouping. This is an explicitly separate
native-student diagnostic, not a common-budget comparison or unchanged
official end-to-end reproduction. It can expose a transfer limitation but
cannot isolate the encoder, loss, optimizer, and budget changes individually.
Also reconstruct the final-stage teacher from the same legal delta, frozen
prefix, seed and fixed 3-epoch budget. Evaluate it once on the same masks and
compare its loss trace to the saved teacher trace. This explanatory audit
does not tune the teacher or supply a new performance upper bound.

### Teacher-budget follow-up after the default Cora diagnostic

The Cora native-core run completed with low common/native-student utility.
Reconstruction confirmed that the transferred 3-epoch default produced only
three teacher updates for the final legal reference of 745 rows. Its
historical-test Macro-F1 was 14.13. This motivates one fixed-budget teacher
diagnostic, not a search over test outcomes. Run a new complete Cora chain
with exactly 2048 native teacher updates per stage, keeping Adam 2e-5,
batch 1024, all 5000 native outer updates, 15 full inner epochs and native
SGD/StepLR unchanged. Retain independent prefixes. Reuse only the identical
prefix-independent PMI selector initial blocks after hash checks. Store it
under `teacher_budget2048/`, never overwrite the default run. Evaluate both
common and native students as above and report both budgets. This changes
the official teacher default and remains a construction adaptation. No
additional teacher-budget search is registered.

### Numerical repair after the first temporal failure

The original CoTAG-Q construction at seed 271006 completed stage 0, but
stage 1 failed the frozen 1e-4 Sinkhorn tolerance after 4096 iterations
(row error 0.00010639429092407227). Preserve this failed run. Register a
separate float64-normalization run for every construction seed, returning
float32 Q for the remaining computation. Keep the loss, initialization,
30-update budget, tolerance, iteration cap and data boundaries unchanged.
Do not select a solver based on downstream scores. This is a numerical
repair, not an unchanged implementation result. The independent comparison
must distinguish the original failure rate from the repaired variant.

The first wrapper was invalidated: the frozen solver internally coerces
scores to float32. Its files and termination receipt remain under
`precision_repair/INVALID_ATTEMPT.json` and must not be analyzed as float64.
The corrected `precision_repair_v2` changes the single internal cast via an
audited in-memory AST clone, asserts the intermediate return dtype, and
passes a strict 1e-12 miniature marginal test. It also increases the solver
iteration cap from 4096 to 16384; the stopping tolerance stays 1e-4. This
is a **combined numerical repair**, not a precision-only causal experiment.
All 30 model updates, losses and data boundaries are unchanged. No further
outcome-driven solver search is registered. Original and repaired jobs may
share a GPU; their times are not isolated resource comparisons.

Keep failed attempts and logs. Save stage memory checkpoints and hashes,
source hashes, access assertions and full loss traces. Stop at the registered
budget, never at the best test outcome. Require 8 GiB free disk. Means and
sample SD across student seeds describe one memory; temporal construction
means are the units for construction SD and paired confidence intervals.
