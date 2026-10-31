# Frozen P0/P1 Attribution Plan

Registered before evaluating the new controls. No parameter search, no main
constructor change, no old-result deletion and no sealed-test selection.

## Scope

- Datasets: WikiCS, SportsFit, BookHis.
- Primary: the unchanged `uq_dm_edgefree` configuration (30 outer steps).
- Same current pool/report split, base+4 groups, reference memory schedule,
  MiniLM, 320-character maximum, Prompt Student, 2048 steps, batch 24.
- Construction seed: 271006. Student seeds: 150, 151, 152.
- P0: orders 9101/9102/9103, CaT-nearest5, Adaptive coherent and In-loop CaT-Q.
- P1: order 9101, fixed-X text controls, independently on all three datasets.
- Student-loss replay: order 9101, final stage, the exact same three primary
  memories evaluated using the historical row-wise InfoNCE trainer and the
  current Prompt Student. Batch, steps, seeds, graph operator and metrics match.
  This isolates the training-target family; it is not a full historical-protocol
  reproduction and cannot explain every old/new result difference.

## Method Arms

| Arm | X | Text | Role |
|---|---|---|---|
| cat_cctp | CaT class-wise MMD | Graph-nearest top5 | External CaT mechanism adapter |
| adaptive_coherent | Exact same CaT X | Historical adaptive coherent routing | Internal historical candidate |
| uq_dm_edgefree | Unchanged learned Q + bounded residual | Learned Q top1 | Primary candidate |
| cat_nearest1 | Exact same CaT X | Graph-nearest top1 | Text attribution |
| cat_random5 | Exact same CaT X | Same-class random top5 | Text attribution |
| uq_nearest1 | Exact same primary X | Graph-nearest top1 | Text attribution |
| uq_nearest5 | Exact same primary X | Graph-nearest top5 | Text attribution |

The UQ nearest controls reuse its candidate text cleaning/truncation before
selection. Thus UQ-top1 versus UQ-nearest1 has the same candidate strings,
encoder and character budget. Top5 is a concatenation control, not a learned
Q-top5 method. Nearest ranking uses current-delta one-hop graph features.

Text variants are post-construction diagnostics. Their modified parent text
does not feed back into the next primary X construction. All variant streams
nevertheless preserve their own old synthetic prefix exactly. Adaptive coherent
does not modify CaT X, so it shares one deterministic CaT construction per stage.

## Audit and Persistence

- Exact X, labels, mass, depth, stage and edge equality for text-only controls.
- Same-class, current-delta source IDs for every materialized string.
- Re-encode stored strings and verify agreement with stored text embeddings.
- Budget and prefix checks at every stage; no synthetic non-self-loop edges.
- Report/pool/stage hashes and constructor/Student source hashes recorded.
- Archive lossless new blocks and text sidecars, not duplicate cumulative X.
- Resume verifies archives against evaluated memory hashes; no partial arm is
  silently pooled with a different reconstruction.
- Auxiliary work preserves Python/NumPy/Torch RNG states.
- A disk guard halts new writes; it never deletes historical artifacts.

## Analysis

Report F1 and Accuracy together, Old/New F1 and stage means. P0 intervals use
paired order means (three independent order cells, not nine independent Student
seeds). P1 and Student-loss replay are conditional on one construction/order;
Student variability is not construction generalization. Compare the historical
candidate with both CaT and the current primary without promoting a winner
based on a single dataset. Existing matrix results are separate evidence, not
extra independent repeats of the same construction cell.

Subsequent P2-P5 work remains pending: higher-fidelity public baselines and
construction seeds; key mechanism confirmation; downstream transfer; budgets,
mixed/repeated arrivals, efficiency and sealed-test evaluation.
