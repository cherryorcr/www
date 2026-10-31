# Native Continual TAG v1

This directory implements a TAGSAM-derived development protocol, not an exact
reproduction of the paper. The downstream model is a two-layer GCN with frozen
BERT CLS inputs and a trainable text projection. Independent students train
from scratch on exported compact TAGs.

The current Chinese result report is
[NATIVE_TAG_V1_FEASIBILITY_ATTRIBUTION_AND_PROGRESS_ZH_2026-09-07.md](../../docs/current/NATIVE_TAG_V1_FEASIBILITY_ATTRIBUTION_AND_PROGRESS_ZH_2026-09-07.md).
The prototype runs end to end, but its advantage over Append is not established.

The consolidated progress summary through 2026-09-08 is
[CURRENT_PROGRESS_SUMMARY_ZH_2026-09-08.md](../../docs/current/CURRENT_PROGRESS_SUMMARY_ZH_2026-09-08.md).

The static-baseline and evaluation clarification is recorded in
[STATIC_TAGSAM_AND_CONTINUAL_PROTOCOL_CLARIFICATION_ZH_2026-09-08.md](../../docs/current/STATIC_TAGSAM_AND_CONTINUAL_PROTOCOL_CLARIFICATION_ZH_2026-09-08.md).

The latest field-level X and fixed-outer-step attribution is recorded in
[DIAGNOSTIC_X_OUTER_TRAJECTORY_2026-09-08.md](../../docs/current/DIAGNOSTIC_X_OUTER_TRAJECTORY_2026-09-08.md).

## Files

| File | Role |
|---|---|
| native_core.py | Model, label-free InfoNCE, differentiable ASM, reference sampling, protected X rows |
| protocol_data.py | Induced/disjoint graphs and joint representative initialization |
| topology.py | Temporary source link predictor and fixed synthetic topology |
| prepare_native_data.py | Signed BERT cache, class stream, train/checkpoint/confirmation split |
| run_native_experiments.py | Static and recursive baselines, step-0/endpoint curves, exported student models |
| run_native_attribution.py | Paired old-memory diagnostics and joint/protected recursive prototypes |
| run_native_stage2_attribution.py | Stage-2 fixed-memory composition diagnostics (raw vs matched raw vs compact) |
| run_fine_attribution.py | Paired old-X update and new-block quality diagnostics with row-drift snapshots |
| run_outer_trajectory.py | Fixed outer-step (0/20/40/60/80/100) multi-student response curves |
| analyze_text_alignment.py | Label-free graph/text alignment for every saved outer snapshot |
| select_trajectory_artifact.py | Development-only outer selection and recursive provenance export |
| train_saved_artifact_students.py | Independent student retraining from one frozen artifact |
| summarize_recursive_text_anchor.py | Strict stage-1 to stage-2 recursive aggregate and seed decomposition |
| build_text_anchor_report_artifact.py | Canonical portable technical-report data and artifact builder |
| FINE_GRAINED_CONTINUAL_TAG_ATTRIBUTION_2026-09-07.md | Results and evidence-based attribution of stage-2 loss |
| TEXT_ANCHORED_ASM_TECHNICAL_REPORT_ZH_2026-09-08.md | Text drift attribution, first repair, fair comparisons and strict recursion |
| confirm_native_artifacts.py | Freeze selected artifacts/models, replay checkpoint scores, evaluate confirmation |
| summarize_native_evidence.py | Verify saved hashes and export development metrics/audits |

## Environment

Validated remotely with Python from
/home/pc/miniconda3/envs/tagsam/bin/python, PyTorch 2.4.1+cu121 and RTX 4090.
The core uses PyTorch, NumPy and scikit-learn. Preparation additionally uses
transformers and huggingface-hub. Tests for preparation need those dependencies.

The signed prepared dataset is at:

    /home/pc/research/TAGSAM/problem_validation_results/native_tag_v1_2026_09_07/prepared_v2

Local results_mirror contains JSON logs, compact evidence/confirmation summaries,
and the final compact artifacts for the main three-stage runs. Complete model
checkpoints and source snapshots remain in the corresponding remote run folders.

## Run

From the code directory, a new 5% static run can be started with:

~~~bash
python run_native_experiments.py \
  --prepared /path/to/prepared_v2 --output /path/to/new_static_run \
  --phase static --stages 1 --rate .05 --seed 44 --model-seeds 100 101 \
  --arms full random coreset frozen offline_asm --device cuda:0 \
  --student-steps 512 --teacher-steps 800 --validation-steps 512 \
  --check-every 32 --link-steps 160 --outer-steps 100 \
  --outer-eval-every 20 --inner-steps 10 --inner-lr .005 \
  --outer-lr .01 --reference-size 128
~~~

The saved stage-0 selected ASM artifact is a shared anchor for diagnostics:

~~~bash
python run_native_attribution.py \
  --base-run /path/to/completed_static_run \
  --output /path/to/new_source_diagnostic --mode source

python run_native_attribution.py \
  --base-run /path/to/completed_static_run \
  --output /path/to/new_recondensation_diagnostic \
  --mode recondense --reference-probe

python run_native_attribution.py \
  --base-run /path/to/completed_static_run \
  --output /path/to/new_recursive_run --mode preserve --stages 3
~~~

The preserve mode compares Union, Append, Append-joint and Protected-joint.
Protected-joint retains the old graph/text pairs exactly and optimizes new X
rows through joint student training on old/new compact data. Source-identical
branches reuse one teacher instance; the runner records which instance is used.
Every output directory must be new.

For strict recursive Text-Anchored ASM, generate fixed snapshots for stage 1,
export only the development-selected artifact, then pass that export as the
sole old memory to stage 2:

~~~bash
python run_outer_trajectory.py \
  --base-config /path/to/recursive/run_config.json \
  --prepared /path/to/prepared_v2 --old-artifact /path/to/stage0_artifact.pt \
  --output /path/to/stage1_trajectory --stage 1 --variant joint \
  --model-seeds 100 101 --outer-steps 100 --outer-eval-every 20 \
  --text-anchor-weight .01 --text-anchor-scope all

python select_trajectory_artifact.py --run-dir /path/to/stage1_trajectory

python run_outer_trajectory.py \
  --base-config /path/to/recursive/run_config.json \
  --prepared /path/to/prepared_v2 \
  --old-artifact /path/to/stage1_trajectory/selected_dev_artifact.pt \
  --output /path/to/stage2_trajectory --stage 2 --variant joint \
  --model-seeds 100 101 --outer-steps 100 --outer-eval-every 20 \
  --text-anchor-weight .01 --text-anchor-scope all
~~~

`select_trajectory_artifact.py` refuses confirmation-accessed trajectories,
uses cumulative development Macro-F1 only, breaks exact ties toward the lower
outer step, and restores signed manifest provenance for the next stage.

The stage-2 composition diagnostic is recorded in
`STAGE2_SOURCE_COMPOSITION_ATTRIBUTION_2026-09-07.md`. It is a
development-only attribution run and does not add a new continual method.

Only after candidates are fixed:

~~~bash
python confirm_native_artifacts.py \
  --runs /path/to/completed_static_run /path/to/completed_recursive_run \
  --output /path/to/new_confirmation_record

python summarize_native_evidence.py \
  --root /path/to/runs --output /path/to/new_evidence_record

python -m unittest discover -s . -p 'test_*.py'
~~~

## Interpretation

- Cora is development evidence, including the confirmation partition accessed
  on 2026-09-07. It is not a pristine final test.
- The class stream is label-defined; training/condensation losses use no node
  class labels. Checkpoint classification labels select teachers, students and
  synthetic artifacts.
- Source labels are excluded from training graph dictionaries. Evaluation
  graphs include held-out node inputs for message passing, so this is not
  strict node-inductive evaluation.
- Uniform/reference-quota comparison changes reference positive and negative
  composition. Minibatch query weighting does not restore global mass sampling.
- Representative stage IDs are text provenance, not mixed-cluster lineage.
- Budgets grow with cumulative train roots. Protected-joint does not implement
  eviction under a fixed total memory capacity.
- Equal optimizer steps and batch limits do not equal identical examples_seen
  or tail-batch negative counts.
- A lower ASM outer learning rate made development curves smoother in one
  diagnostic, but both frozen confirmation comparisons still lost; it is not
  the selected method configuration.
- Early attribution configs retained the base static arms list. The actual
  results.json rows and evidence audit identify those historical cases.

See EXECUTION_PROTOCOL_2026-09-07.md and ATTRIBUTION_ADDENDUM_2026-09-07.md
for development decisions, including the teacher-sharing correction.
