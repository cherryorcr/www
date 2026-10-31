#!/usr/bin/env bash
set -euo pipefail

cd /home/pc/research/TAG_condensation
export CUDA_VISIBLE_DEVICES=0
PY=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation/results/direct_student_outer_20260928
MINILM=/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip
CACHE=/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919
EVAL=code/v2_llm_fusion/run_gc_sr_stage_eval.py

evaluate_one() {
  local name="$1" dataset="$2" order="$3"
  local output="$ROOT/eval_geometry_x_v2_${name}.json"
  if [[ -f "$output" ]]; then
    echo "SKIP existing immutable result: $output"
    return
  fi
  PYTHONPATH="code/v2_llm_fusion:code/frozen_v1_2026_09_09" \
    "$PY" -u "$EVAL" \
      --dataset "$dataset" --order "$order" --stage 1 \
      --base "$ROOT/control_${name}/memory.pt" \
      --candidate "$ROOT/geometry_x_v2_${name}/memory.pt" \
      --output "$output" --minilm "$MINILM" --cache-root "$CACHE" \
      --device cuda:0 --seeds 150 151 152 153 154 155 \
      > "$ROOT/eval_geometry_x_v2_${name}.log" 2>&1
}

evaluate_one cora_canonical cora canonical
evaluate_one cora_order19102 cora order19102
evaluate_one citeseer_canonical citeseer canonical
evaluate_one citeseer_order19102 citeseer order19102
echo GEOMETRY_X_V2_EVAL_COMPLETE
