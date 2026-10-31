# BookHis benefit decomposition

Registered before generating per-class predictions on 2026-10-22.

## Question

Locate the observed CoTAG-Q versus GECC-hard Macro-F1 difference by class
frequency, introduction stage and evaluation stage. This is a descriptive,
post-hoc diagnostic, not a causal test of long-tail learning or blind testing.

## Fixed comparisons

- Historical trajectory: all complete BookHis primary runs in
  `uq_final_evidence_20261012/confirmation`, three class orders and three
  construction seeds. Reuse saved aggregate scores without rerunning them.
- Prediction diagnostic: order 9101; constructions 271006, 271108, 271109;
  students 150, 151, 152; all five stages; both methods. Reuse the saved
  30-update complete-objective CoTAG-Q memories from the independent-X batch.
  Slice each final memory only after matching every stage hash to that batch.
  The primary BookHis memories were not retained. Reconstruct GECC from the
  frozen primary implementation, record memory/score drift, and evaluate
  both methods afresh. Never substitute this diagnostic for the primary table.
- Keep the primary current-increment construction, frozen prefix, cumulative
  5% item budget, MiniLM encoder, full-taxonomy prompt GCN, 2048 updates,
  batch 24, learning rate 0.001 and original disjoint evaluation masks.
- Historical raw graph access at inference follows the original transductive
  protocol. This is not the separate chronological evaluation.

## Analysis fixed before granular results

- Rank all 12 classes by construction-pool node count, ties by numeric class
  ID. Split into low/middle/high frequency groups of four classes each.
  Include every class and group, irrespective of effect sign.
- Compute precision, recall and F1 using the full seen-class evaluation
  cohort. Old/New mean class F1 over old/new labels on that same cohort.
  Keep predictions of unseen labels in the confusion matrix.
- Exact final Macro-F1 decomposition: each class contributes its F1
  difference divided by 12. Accuracy decomposition: each class contributes
  its test prevalence times its recall difference. Check both identities.
- Save raw predictions, evaluation masks, labels, metadata, memory quotas,
  per-student metrics and all GECC stage memories.
- Average the three student seeds within each constructed memory. The new
  diagnostic has three paired constructions, not nine independent runs.
  Report paired t intervals (df=2), all signs and per-construction values.
  Per-class intervals are descriptive, not multiple-testing discoveries.
- For the historical 3x3 trajectory also report a crossed order/construction
  bootstrap, resampling the two factors independently (20,000 repetitions,
  seed 271022). Treat stages as repeated observations, not independent runs.
- A train-frequency association or arithmetic contribution is not evidence
  that Q caused a general long-tail benefit. No tuning from these results.

## Safety and outputs

Use a new result directory; refuse overwrites and partial-run reuse. Keep an
8 GiB free-disk floor. Existing completed experiments and sealed tables are
read-only. This batch makes no changes to paper sources or Overleaf.
