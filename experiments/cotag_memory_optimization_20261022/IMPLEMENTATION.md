# Mathematical and implementation audit

For current raw rows I_c and new slots J_c of class c, the original masked
assignment has Q[I_c,J_d] = 0 whenever c != d. The new representation stores
only Q_c = Q[I_c,J_c]. It does not prune any previously legal candidate.

For features X, the pooled slot rows are computed as Q_c.T @ X[I_c] and scattered
back into the original J_c order. The same operation is used for text embeddings
and propagated graph features. Column masses come from Q_c.sum(0). The existing
Sinkhorn routine, temperature, capacity and stopping test are applied independently
to exactly the same class scores as before. Both original and optimized versions
retain the existing activation checkpoint implementation.

The Q-anchor is the sum of squared changes over legal blocks divided by n*m,
NOT divided by the number of stored legal entries. Cross-class zero contributions
therefore have the same meaning as the original dense mean. The entropy diagnostic
also includes the tiny implicit-zero contribution from the original 1e-12 clamp.
Argmax is mapped to original row/slot IDs with the first-index tie rule. Materialized
strings and original source IDs are saved for independent comparison.

Parameter gradients, Adam states and initialization are stored only for legal
blocks. No weight decay was used in the original Adam. Illegal logits receive
zero gradients in the dense implementation and do not influence its legal updates.
Global gradient clipping still includes all legal logits and the feature residual.
No GM/DM/propagation objective is removed, detached or reweighted.

This establishes algebraic correspondence, not bitwise numerical correspondence.
Blockwise matrix multiplication changes floating-point reduction order; gradient
norm reductions also change grouping. Tiny numerical changes can change hard
argmax text choices and subsequent optimization. GPU sparse propagation can also
produce run-to-run differences in the dense path. Tests check operator values and
first/second derivatives; saved memories separately measure actual trajectory
drift and student performance. Do not call complete memories numerically identical
unless their hashes match.

The cache treatment discards only data['x_device'] and data['real_adj'] before
construction, then rebuilds them for inference. CPU source tensors and MiniLM
remain resident. All four arms measure student training without the original-graph
cache, so their reported student peaks use the same phase boundaries.

Implementation scope is explicitly the primary edge-free, q-max branch. Optional
topology, boundary, random-Q and joint-distribution configurations are rejected
for blocked storage, not silently approximated. Default storage remains dense.

An existing development-branch uninitialized `initial_hard` variable was fixed
for edge-free DM before this batch. The archived constructor already handles that
case. A fresh-interpreter two-stage regression fixture checks exact output hashes
and traces between the archived implementation and the updated default dense path.
