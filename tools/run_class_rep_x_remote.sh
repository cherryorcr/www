#!/usr/bin/env bash
set -euo pipefail
cd /home/pc/research/TAG_condensation
export CUDA_VISIBLE_DEVICES=0
PY=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation/results/direct_student_outer_20260928

run_one() {
  local name="$1" old="$2"
  "$PY" code/v2_llm_fusion/run_direct_student_outer.py \
    --base "$ROOT/control_${name}/memory.pt" \
    --support "$ROOT/prep/${name}/support.pt" \
    --output "$ROOT/class_rep_x_v1_${name}" \
    --old-rows "$old" --mode x-class-rep --row-selection new \
    --device cuda:0 --inner-steps 2 --inner-batch 16 \
    --outer-steps 2 --outer-lr 0.01 --support-batch 16 \
    --old-weight 0.25 --new-weight 1.0 --x-new-radius 0.20 \
    --geometry-anchor-weight 0.50 --geometry-smoothness-weight 0.25 \
    --class-representation-weight 1.0 \
    --base-slot-class-id "$ROOT/class_slot_map_${name}/base_slot_class_id.pt" \
    --support-class-id "$ROOT/prompt_support_v4_${name}/support_class_id.pt" \
    --train-seeds 9300 --validation-seeds 9310 --plan-seed 9320
}

run_one cora_canonical 25
run_one cora_order19102 35
run_one citeseer_canonical 38
run_one citeseer_order19102 40
echo CLASS_REP_X_V1_COMPLETE
