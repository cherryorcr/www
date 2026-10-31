# Overnight Unified-Q-EF Evidence

The auditable aggregation is stored in `results/unified_q_overnight_20261012/`.

The attribution audit is stored in `results/unified_q_overnight_20261012/q_attribution_audit.md`. It separates shared-Q versus independent-Q, fixed-Q, no-GM, no-DM, text-path controls, hard/soft materialization diagnostics, and slot-level traces. A literal random-Q control is now complete on the three large datasets for one registered reference order and construction seed; it is reported as a reference-order attribution result, not as a full factorial stability result. The older random real-node coreset remains a separate control. Likewise, the hard/soft result verifies straight-through consistency and does not establish that a purely soft stored payload is better.

## Construction/order stability

The paired panel uses three orders and three construction seeds, with Student seeds 150, 151, and 152 averaged inside each cell. Relative to GECC-hard-TAG, Unified-Q-EF improves final Macro-F1 by `+1.25` on SportsFit, `+1.66` on BookHis, and `+1.25` on WikiCS. Relative to CaT-CCTP, the corresponding differences are `+4.46`, `+17.06`, and `-0.51` points.

The SportsFit/order 9103/construction 271108 cell is explicitly marked as a numerical Sinkhorn recovery using a 4096 iteration cap. It is useful for completeness and is not merged into the sealed primary table.

## Literal random-Q attribution

The random-Q control keeps class legality and balanced Sinkhorn capacity but replaces the structured feature/message/text initialization with random legal logits, freezes (Q), and disables the residual update. At the final stage, Unified-Q-EF exceeds random-Q by `+9.21` Macro-F1 points on SportsFit, `+6.18` on BookHis, and `+8.74` on WikiCS. This supports the meaningfulness of the learned assignment and initialization, but is not a universal-SOTA claim.

## Cross-architecture transfer

The Prompt GraphSAGE batch reuses frozen memories from the registered construction matrix and does not re-optimize the constructor. Unified-Q-EF is strongest on BookHis (`36.83 +/- 4.22` Macro-F1), while CaT remains stronger on SportsFit and WikiCS. This supports transfer beyond a single GCN Student, not universal downstream dominance.

## Paper scope

The paper calls the primary edge-free method `Unified-Q-EF`. The topology-enabled branch is named `Unified-Q-T` and remains a separate extension for structural downstream tasks. The current evidence does not support combining their claims.
