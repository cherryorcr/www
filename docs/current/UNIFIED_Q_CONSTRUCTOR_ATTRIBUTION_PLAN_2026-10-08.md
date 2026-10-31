# Constructor-Side Attribution: Registered Plan

Registered before running new controls. No parameter search, no test-set use,
no core constructor/Student changes, no deletion of historical data.

## Scope

- WikiCS, SportsFit, BookHis; orders 9101/9102/9103; construction seed 271006.
- Five stages, the existing fixed budget schedule; Student seeds 150/151/152,
  2048 steps, batch 24; unchanged Prompt Student, MiniLM and 320 characters.
- Each arm constructs its own complete continual chain from current delta and
  its own frozen synthetic prefix. No historical raw reread by the constructor.
- Reuse archived CaT-nearest5 strings and existing fixed-Q/no-GM evidence.
  Reconstruct the unchanged full arm in this batch for paired comparisons and
  quantify its numerical replay drift; do not pool reruns as independent seeds.

## Controls

| Arm | Intervention | Orders |
|---|---|---|
| uq_dm_edgefree | Frozen full constructor | All three |
| no_node_text | text_weight=0 AND pair_weight=0 | All three |
| no_residual | train_residual=False | All three |
| no_distribution | lambda_distribution=0 | All three |
| free_x | Independent X parameter with original objective and conditional bound | All three |
| no_text_init | Only text_weight=0 | Reference 9101 |
| no_payload_gm | Only pair_weight=0 | Reference 9101 |
| stop_text_gradient | Current hard routing, detach its direct text-to-Q gradient | Reference 9101 |

The text controls retain class prompts and dataset-provided X, which may be
text-derived. They remove NODE PAYLOAD information, not all possible text
information. Setting pair_weight=0 retains graph-to-class-prompt GM; unlike
no-GM, it does not remove all gradient matching.

Free-X starts at the same pool(Q0). It minimizes the same losses as functions
of (Q,X), including ||(X-pool(Q))/||pool(Q0)||||^2, and is projected onto
||X_j-pool(Q)_j|| <= 0.05 ||pool(Q)_j|| (same numerical floor as the full arm).
Q still controls text, propagation targets and the moving feasible set.
Thus it tests the optimizer parameterization, not unbounded free-X/CaT and
not removal of every X-Q dependency. Its code is isolated from the main file.

## Readouts

- Native: every stage, using the arm's own materialized text.
- Common CaT payload: final stage only, replace each arm's text with the exact
  same archived CaT-nearest5 strings/embeddings, by identical slot labels/order.
  Keep X/mass/depth/stages/edges unchanged. This is a posthoc diagnostic; it
  never feeds back into construction and is not a new continual method.
- 54 five-stage chains, 270 native + 54 common-payload records, 972 Student
  trainings. This batch has one construction seed, not three.

## Analysis and Audit

Use within-order, same-Student-seed paired differences; summarize order means,
order SD and paired t intervals (n=3, no equivalence claim). Reference-only
text factorial/stop-gradient diagnostics remain conditional on one order.
Report F1, Accuracy, Old/New, stage-average, cost and worst-order deltas.

Archive incremental blocks. Verify budgets, exact prefix, source IDs, class
mask/capacity tolerances, text re-encoding, feature radius and identical
common payload. Record original source hashes and actual input mask/stage IDs.
Disk guards halt on low space; they do not remove files. Numerical same-seed
drift remains a limitation for very small differences.

## Harness Repair Before Formal Run

The initial queue stopped after stage 0 because the archived CaT block has no
Q-specific support_ids and the harness used the Q-only append helper. Replaced
that with the existing generic edge-free append helper; added a five-block
regression test. No optimization, Student or budget parameter changed. Partial
startup records remain in unified_q_constructor_attribution_20261008; formal
results use unified_q_constructor_attribution_20261008_v2, without pooling the
failed startup into any final score.
