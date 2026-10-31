# Single-source text attachment for chronological GECC

Registered on 2026-10-22 before any outcome from this new variant. This is a
post-hoc sensitivity experiment motivated by already inspected chronological
results, not a new blind test or an outcome-independent baseline selection.
Preserve and report the original nearest-5 GECC, CoTAG-Q and CaT results.
Do not edit the manuscript or push Overleaf in this batch.

## Intervention

Reuse the exact saved GECC-hard numeric features, labels, mass, stage/depth
metadata and edge-free topology at every stage. Do not rerun clustering, change
its 20 restarts/300-iteration limit, propagation weights or balance score.
Only replace nearest-5 text by nearest-1 text. The 320-character total text cap
and frozen MiniLM stay unchanged. The selector uses the current training pool,
same-class propagated-feature cosine similarity, float32 arithmetic and stable
tie ordering from the frozen original adapter. It is not a learned selector.

Before evaluating each stage, reconstruct the original nearest-5 attachment
from its saved features and the legal current increment. Require exact text
agreement with the archived five-source memory and confirm encoded vectors
within absolute tolerance 1e-6, relative tolerance 1e-5. The new source ID must
equal the first source ID of this verified nearest-5 selection. Stop and retain
failure evidence if this provenance gate fails; do not silently change the
selector, numeric precision or candidate ordering to obtain favorable results.

At stage zero create the single-source block from the saved centers. At later
stages append only the new single-source block to this variant's own frozen
prefix. Never rewrite its prior texts or access historical raw data to update
old entries. The offline loader still retains source files on CPU for slicing.

## Fixed matrix

Datasets: all three existing chronological streams, ArXiv, ACM temporal domain
subset and DBLP temporal domain subset. Reuse their exact five stages, masks,
legal pools, class prompts, cumulative 5% rounded budgets and inference modes.
Construction IDs 271006/271108/271109 identify the reused original memories;
there are no new independent clustering runs. Student seeds 150/151/152,
2048 updates, batch24, learning rate .001 and all losses remain unchanged.

Source GECC directories:
- results/cotag_fidelity_temporal_20261021/temporal/gecc_hard_c{seed}
- results/cotag_temporal_bibliographic_20261022/runs/{acm,dblp}/gecc_hard_c{seed}

ArXiv's old historical_test mask will be called test in the new common saved
schema, with mask hashes checked exactly. It remains the previously inspected
mask, not a new split. ACM/DBLP masks have also been inspected before this
sensitivity request. No results-based variant selection, early stopping,
hyperparameter changes or dataset exclusion are allowed.

## Reporting and safeguards

Report both final and equal-weight five-stage Macro-F1/Accuracy, arrived/current
snapshot modes and cumulative/current/historical cohorts. Compare nearest-1
against nearest-5 using matched construction IDs after averaging three students
per memory. Report sample construction SD, unadjusted paired t95% interval,
positive-pair ratio and worst paired difference. Retain the same CoTAG-Q and CaT
reference scores. Save predictions for independent metric recomputation.

Report text length, number of sources, unique-text ratio and the fraction of
memory texts changed. No improvement is presumed. The intervention changes
both source diversity and the amount of text retained from each source, so it
is not a pure source-count effect at fixed per-source content.

Record attachment validation, attachment and student costs separately. Since
clustering is reused, this is not a complete GECC construction-cost benchmark.
Keep an 8 GiB remote disk floor, never overwrite prior completed/failed outputs,
stop on unexpected provenance or numeric failure, and reuse verified checkpoints.
