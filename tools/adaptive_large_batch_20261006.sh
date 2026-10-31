#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?gpu id required}"
shift
ROOT="/home/pc/research/TAG_condensation"
SNAPSHOT="$ROOT/work_runs/adaptive_protocol_20261006_snapshot"
PYTHON="/home/pc/miniconda3/envs/tagsam/bin/python"
MINILM="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"
CACHE="$ROOT/results/public_baselines_text_cache_20261001"
ARMS=(
  cat_mmd_tag_cctp_top5
  cat_mmd_tag_x_conditioned_routing_coherent
  cat_mmd_tag_x_conditioned_routing_coherent_adaptive
  cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor
  k_center
  medoid
)
ORDERS=(9101 9102 9103)
CONSTRUCTION_SEEDS=(270927 314159 271828)

source_root_for() {
  case "$1" in
    wikics) echo "$ROOT/results/public_baselines_external_wikics_20261006_gpu0_retry" ;;
    bookhis) echo "$ROOT/results/public_baselines_external_bookhis_20261006_gpu1_retry" ;;
    *) echo "$ROOT/results/p5_large_base4_20260919" ;;
  esac
}

run_one() {
  local dataset="$1"
  local order_seed="$2"
  local construction_seed="$3"
  local source_root
  source_root="$(source_root_for "$dataset")"
  local name="adaptive_order_${dataset}_o${order_seed}_c${construction_seed}_20261006"
  local output="$ROOT/results/$name"
  local log="$ROOT/results/${name}.log"
  if [[ -f "$output/summary.json" ]]; then
    echo "SKIP $name"
    return
  fi
  echo "START $name"
  "$PYTHON" -u "$SNAPSHOT/code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py" \
    --datasets "$dataset" \
    --source-root "$source_root" \
    --xpa-root "$source_root" \
    --output-root "$output" \
    --minilm "$MINILM" \
    --text-cache-root "$CACHE" \
    --device "cuda:$GPU" \
    --arms "${ARMS[@]}" \
    --student-seeds 100 101 102 103 104 105 \
    --construction-seed "$construction_seed" \
    --order-seed "$order_seed" \
    > "$log" 2>&1
  echo "DONE $name"
}

for dataset in "$@"; do
  for construction_seed in "${CONSTRUCTION_SEEDS[@]}"; do
    for order_seed in "${ORDERS[@]}"; do
      run_one "$dataset" "$order_seed" "$construction_seed"
    done
  done
done

echo "ADAPTIVE_LARGE_BATCH_COMPLETE"
