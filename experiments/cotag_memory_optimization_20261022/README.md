# Construction memory optimization audit

Registered before GPU outcomes, 2026-10-22. No paper or sealed-result changes.

## Scope

- Four implementations: `original`, `cache_offload`, `class_blocks`, `combined`.
- `cache_offload` removes only the full-graph inference GPU cache during construction.
- `class_blocks` stores legal class blocks of logits and Q, including optimizer states,
  and uses blockwise feature/text/message multiplication. It preserves dense-mean
  Q-anchor normalization, balanced capacities, full GM/DM/propagation losses,
  initialization, trust projection, text selection and 30 outer updates.
- Main code defaults to dense storage. This opt-in path is edge-free and q-max only.
- Float32 Sinkhorn and its existing checkpoint/tolerance/cap are unchanged.
- CPU graph/text archives and the text encoder are not offloaded by this experiment.

## Frozen comparisons

Datasets ArXiv and SportsFit, order 9101, construction seeds 271006/271108/271109,
student seeds 150/151/152, 2048 student updates, batch 24. Reuse archived primary
groups and exact cumulative 5% budgets to isolate implementation effects. In
this resource audit ArXiv has the archived THREE stages, SportsFit FIVE; these
results must not be relabeled as ArXiv base+4 or chronological results.

First verify forward values, first/second derivatives, materialization ties and
two-stage frozen prefixes with unit tests. Run stage zero with all four variants
for construction 271006. Then complete the original and combined streams for all
three constructions, reusing their existing stage-zero checkpoints. Do not rerun
completed stages. Failures and partial runs are preserved and reported, not scored
as zero. No automatic precision, solver-cap, learning-rate or objective changes.

Measure absolute peak allocated/reserved CUDA memory, construction wall time,
entry/exit allocated memory, process CPU RSS (sampled), evaluation-cache rebuild
time, and student/inference phases separately. GPU timings on two simultaneously
active GPUs are descriptive, not isolated-runtime speedup claims. CPU threads are
fixed to four. An 8 GiB disk floor is enforced before each stage.

Run final-stage student evaluation, and stage-zero evaluation for the four-way
pilot. Save predictions and memories. Use only report-dev scores for this audit;
do not open sealed-test outcomes or choose configurations by scores. Compare
floating feature drift, selected-source agreement, prefix consistency and final
F1/Accuracy. Exact operator mathematics does not imply bitwise identical Adam
trajectories or hard-source choices on GPU.

Report mean and sample SD across three constructions after averaging the three
students per memory. Paired 95% t intervals use constructions as the independent
units. Predeclared quality reference is a one-percentage-point F1 loss margin;
report whether the lower paired interval exceeds -1, without tuning/retrying to
make it pass. A pilot is not a replacement for the complete-stream check.

This measures construction implementation savings, not superiority over all
baselines, end-to-end amortization, or purely physical streaming ingestion.
