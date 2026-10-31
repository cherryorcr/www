#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation
WORK=$ROOT/work_runs/phase12_20260920_snapshot
R10_RUNNER=$WORK/code/v2_llm_fusion/run_tgc_r10_independent_20260925.py
R11_RUNNER=$WORK/code/v2_llm_fusion/run_tgc_r11_independent_20260925.py
COMBO_RUNNER=$WORK/code/v2_llm_fusion/run_tgc_p_content_semantic_perm.py
R10_OUT=$ROOT/results/tgc_r10_joint_confirmation_independent_20260925
R11_OUT=$ROOT/results/tgc_r11_c0_anchor_independent_20260925
MAP1=$ROOT/results/tgc_two_prototype_role_permutation_independent_20260925
MAP2=$MAP1
CACHE=$ROOT/results/p5_rate_curve_small_20260919
MINILM=/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip
OUT=$ROOT/results/tgc_p_content_two_proto_independent_20260925e
PYTHONPATH_VALUE=$ROOT/code/v2_llm_fusion:$WORK/code/v2_llm_fusion

mkdir -p "$R10_OUT" "$R11_OUT" "$OUT/logs"

run_r10_prepare() {
  local dataset="$1"; local device="$2"
  CUDA_VISIBLE_DEVICES="$device" PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" -u "$R10_RUNNER" \
    --dataset "$dataset" --phase prepare --output "$R10_OUT" --device cuda:0 \
    > "$OUT/logs/r10_${dataset}.log" 2>&1
}

run_r11_prepare() {
  local dataset="$1"; local device="$2"
  CUDA_VISIBLE_DEVICES="$device" PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" -u "$R11_RUNNER" \
    --dataset "$dataset" --phase prepare --output "$R11_OUT" --device cuda:0 \
    > "$OUT/logs/r11_prepare_${dataset}.log" 2>&1
}

run_r11_evaluate() {
  local dataset="$1"; local device="$2"
  CUDA_VISIBLE_DEVICES="$device" PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" -u "$R11_RUNNER" \
    --dataset "$dataset" --phase evaluate --output "$R11_OUT" --device cuda:0 \
    > "$OUT/logs/r11_evaluate_${dataset}.log" 2>&1
}

run_combo() {
  local dataset="$1"; local device="$2"
  for order in order19605 order19606 order19608; do
    local log="$OUT/logs/combo_${dataset}_${order}.log"
    CUDA_VISIBLE_DEVICES="$device" PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" -u "$COMBO_RUNNER" \
      --dataset "$dataset" --order "$order" \
      --r11-source "$R11_OUT" \
      --mapping-root-stage1 "$MAP1" --mapping-root-stage2 "$MAP2" \
      --output "$OUT" --cache-root "$CACHE" --minilm "$MINILM" --device cuda:0 \
      --stages 0 1 2 --report-seeds 150 151 152 153 154 155 \
      > "$log" 2>&1
  done
}

run_r10_prepare cora 0 & P1=$!
run_r10_prepare citeseer 1 & P2=$!
wait "$P1"; wait "$P2"

run_r11_prepare cora 0 & P1=$!
run_r11_prepare citeseer 1 & P2=$!
wait "$P1"; wait "$P2"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" -u "$R11_RUNNER" \
  --phase seal --output "$R11_OUT" > "$OUT/logs/r11_seal.log" 2>&1

run_r11_evaluate cora 0 & P1=$!
run_r11_evaluate citeseer 1 & P2=$!
wait "$P1"; wait "$P2"

run_combo cora 0 & P1=$!
run_combo citeseer 1 & P2=$!
wait "$P1"; wait "$P2"

echo TGC_P_CONTENT_TWO_PROTO_INDEPENDENT_COMPLETE
