#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation
SNAP=$ROOT/work_runs/tgr_cross_edge_transport_20260928
DATA_ROOT=/home/pc/research/GraphCLIP/processed_data
CACHE_ROOT=$ROOT/results/p5_rate_curve_small_20260919
C0_ROOT=$ROOT/results/tgc_fresh_r10_source_20260926
SEMANTIC_ROOT=$ROOT/results/tgc_fresh_semantic_recursive_20260926
OUT=$ROOT/results/tgr_cross_edge_transport_strict_20260928
PYTHONPATH_VALUE=$SNAP/code/v2_llm_fusion:$SNAP/code/frozen_v1_2026_09_09

for dataset in cora citeseer; do
  for order in order19611 order19612; do
    for stage in 1 2; do
      cell="$OUT/${dataset}_${order}_s${stage}"
      if [[ -f "$cell/protocol.json" ]]; then
        echo "SKIP $dataset $order s$stage"
        continue
      fi
      mkdir -p "$OUT/logs"
      PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" -u \
        "$SNAP/code/v2_llm_fusion/run_tgr_cross_edge_transport.py" \
        --dataset "$dataset" --order "$order" --stage "$stage" \
        --data "$DATA_ROOT/$dataset.pt" \
        --text-cache "$CACHE_ROOT/${dataset}_r005/text_emb_pool_only.pt" \
        --c0-root "$C0_ROOT" --semantic-root "$SEMANTIC_ROOT" \
        --output "$cell" --device cpu \
        --modes c0 none random semantic message \
        --cross-degree 1 --message-mix 1.0 --swap-fraction 1.0 \
        --random-seed 9821 \
        > "$OUT/logs/${dataset}_${order}_s${stage}.log" 2>&1
    done
  done
done
echo TGR_CROSS_EDGE_TRANSPORT_STRICT_20260928_COMPLETE
