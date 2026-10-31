# Independent feature optimization control

Registered on 2026-10-20 before new outcome inspection. No paper edits.

## Question

Does differentiable Q-based numerical feature construction provide useful
inductive bias compared with independently optimizing numerical features,
while retaining learned Q-based readable text selection?

## Frozen comparison

- Datasets: ArXiv, SportsFit, BookHis.
- Order seed: 9101. Construction seeds: 271006, 271108, 271109.
- Student seeds: 150, 151, 152; 2048 updates, batch 24.
- Source: uq_final_evidence_20261012/code, verified against primary results.
- Fixed condensation rate: 0.05 of the current legal cumulative pool.
- Current increment only; each arm has its own frozen historical memory.
- ArXiv retains the original three-stage primary protocol. SportsFit and
  BookHis have five stages. These are not relabeled as a uniform base+4 run.

Full CoTAG-Q uses pooled Q features plus a bounded residual. The new control
initializes X from the same pooled Q0, then updates independent X and text Q.
It removes both the moving Q-conditioned feature projection and the
X-to-pool(Q) anchor penalty. GM, DM, propagation, Q anchor, initial state,
text materialization, proxy seeds and all update budgets are retained.
Propagation targets still depend on Q. This is not a completely Q-free arm,
nor an identical constrained optimization problem.

Existing full-arm metrics are reused after provenance checks. Available
full-arm memory archives are verified by hash. Missing archives are
reconstructed solely for same-memory GraphSAGE evaluation, with any hash
or metric discrepancy reported. Reconstructions are not independent seeds.
Matching fixed-Q results are reused rather than reconstructed without need.

Both final memories are evaluated with GCN and the previously frozen
self-loop mean GraphSAGE variant, with no architecture-specific tuning.
GraphSAGE includes self in the neighborhood and a separate root transform;
it is not canonical loop-free GraphSAGE.

## Analysis and safeguards

Primary construction budget is 30 updates for both arms. Record per-step
control losses and components, gradient norms, text coverage/diversity,
soft/hard cosine gap, cost, and maximum displacement from pooled Q features.
Do not infer a fundamental disadvantage from an underoptimized control.
Any extension is a separately recorded matched-budget development check.

Average three student seeds within each constructed memory, then report
mean and sample SD over the three construction seeds. Pair by construction
seed; report paired differences and t confidence intervals with their small
sample limitation. Keep report-dev and historical test-mask results separate.
The historical test mask has been inspected previously; it is not fresh blind
evidence. No test-based configuration selection.

Write new results separately, preserve failed runs, never overwrite sealed
results, and stop launching work below 8 GiB free disk space. No dense Q is
persisted. Existing unrelated jobs and files are left alone.
