# Additional chronological TAG evaluation

Registered on 2026-10-22 before downstream model outcomes. The user authorized
replacing SportsFit/BookHis for this extension because their published artifacts
have no verified node timestamps. Do not edit the paper in this batch.

## Sources and tasks

Use AMiner's versioned ACM v9 and DBLP v10 citation releases:

- https://lfs.aminer.cn/lab-datasets/citation/acm.v9.zip (686542584 bytes).
- https://lfs.aminer.cn/lab-datasets/citation/dblp.v10.zip (1848832450 bytes).

These are publication-year sequences, not observed website ingestion events.
Use title and available abstract as text, publication year as the node time,
and references between retained papers as edges. Never expose venue, author,
ID, citation count, or year inside model text. Labels are venue-derived research
communities, not independent article-level expert topic annotations. Call these
**ACM temporal domain subset** and **DBLP temporal domain subset**, not unchanged
official benchmarks or two independent Web application domains.

ACM communities: databases, data mining, computer networks. DBLP communities:
artificial intelligence, computer vision, natural language processing, machine
learning. Exact venue rules are versioned in prepare_data.py before scoring.
Profile coverage without inspecting any downstream score. Retain 1990-2015
inclusive, nonempty substantive titles, and available abstracts. Exclude generic
index/front-matter records by fixed title rules, ambiguous venue matches, and
ambiguous same-title/year/label duplicates. Merge equivalent same-title records
with identical year/label using the lexicographically smallest source ID. Remove
DBLP titles already retained in ACM and report the overlap. No score-dependent
choice of papers, classes or boundaries is allowed.

If eligible nodes exceed 60000, retain the 60000 smallest SHA256 hashes of
dataset name plus source ID, independent of labels, time, edges, and outcomes.
Keep isolated nodes. This is a compute-bounded induced subgraph; disclose all
filtering and do not report it as the complete archive.

Split by a stable source-ID hash modulo 10: 1 selector-dev, 3 report-dev, 2 test,
all other values train. No task or parameter selection uses held-out labels.
From training-node years only, choose the nearest-rank 40/55/70/85/100 percentiles
as five inclusive upper boundaries. Never split the same year; abort if the five
bins are not distinct or any training bin is empty. Categories are fixed known
taxonomy metadata, not future node supervision. Report class support in each bin.

Keep references only within retained nodes. Drop self references and citations
whose referenced publication year exceeds the citing year; report the counts.
An edge is available when the citing publication arrives. Reference lists and
texts still come from a released snapshot, not historical content versions.
Symmetrize for the common GCN. Do not claim measured edge-event timestamps.

## Frozen comparison

Reuse the audited construction code from uq_final_evidence_20261012, the common
MiniLM, 30 construction updates, current-increment-only source access, frozen
synthetic prefix, cumulative 5% legal-pool budget, and 2048-update prompt student
with batch 24. Compare CoTAG-Q, CaT adaptation and GECC-hard adaptation. Use
construction seeds 271006/271108/271109 and student seeds 150/151/152. Do not
substitute nine student repetitions for three independent constructions.

Use the previously verified float64 Sinkhorn normalization with cap 16384 and
unchanged 1e-4 marginal tolerance for CoTAG-Q. Explicitly label this numerical
repair; it is not an unchanged-original solver experiment. No other method
parameters are retuned. MiniLM encoding is pointwise and frozen. Raw numerical
features use this same encoder; held-out features are stored on CPU for inference
and are never passed to constructors. No full future graph GPU cache is built.

## Inference, reporting and failure policy

At each stage evaluate the same trained student on arrived and current-only
snapshots. Report fixed-taxonomy Macro-F1, Accuracy, cumulative/current/historical
arrival cohorts, five-stage means and final stage. Compare inference access on
identical current queries. Current-only inference cannot evaluate past queries.
Save raw predictions for metric verification. Average three students within a
memory; compute construction sample SD, paired t intervals, positive-pair ratio
and worst paired difference across three matched construction seeds.

Record construction, synthetic student training and inference resources
separately. The CPU harness holds archives for slicing, not physical erasure.
Keep an 8 GiB disk floor, never overwrite sealed or failed results, skip verified
completed runs, and stop on unexpected protocol or numerical failures. All source,
dataset, memory, code and protocol hashes must be saved. Failures remain evidence,
not a reason to change datasets after seeing performance. Data-only feasibility
failures may require a separately documented amendment before model evaluation.
