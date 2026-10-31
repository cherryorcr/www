#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation
SNAP=$ROOT/work_runs/tgr_cross_edge_transport_20260928
DATA_ROOT=/home/pc/research/GraphCLIP/processed_data
CACHE_ROOT=$ROOT/results/p5_rate_curve_small_20260919
CONSTRUCTION_ROOT=$ROOT/results/tgr_cross_edge_balanced_20260928
SEMANTIC_ROOT=$ROOT/results/tgc_fresh_semantic_recursive_20260926
OUT=$ROOT/results/tgr_cross_edge_balanced_eval_20260928
MINILM=/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip
PYTHONPATH_VALUE=$SNAP/code/v2_llm_fusion:$SNAP/code/frozen_v1_2026_09_09

for dataset in cora citeseer; do
  for order in order19611 order19612; do
    for stage in 1 2; do
      output="$OUT/${dataset}_${order}_s${stage}.json"
      if [[ -f "$output" ]]; then
        echo "SKIP $dataset $order s$stage"
        continue
      fi
      mkdir -p "$OUT/logs"
      PYTHONPATH="$PYTHONPATH_VALUE" CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
        "$SNAP/code/v2_llm_fusion/run_tgr_cross_edge_eval.py" \
        --dataset "$dataset" --order "$order" --stage "$stage" \
        --construction-root "$CONSTRUCTION_ROOT" \
        --semantic-root "$SEMANTIC_ROOT" \
        --data "$DATA_ROOT/$dataset.pt" \
        --text-cache "$CACHE_ROOT/${dataset}_r005/text_emb_pool_only.pt" \
        --minilm "$MINILM" --output "$output" --device cuda:0 \
        --modes c0 none message message_balanced \
        --seeds 150 151 152 153 154 155 \
        > "$OUT/logs/${dataset}_${order}_s${stage}.log" 2>&1
    done
  done
done
echo TGR_CROSS_EDGE_BALANCED_EVAL_20260928_COMPLETE
