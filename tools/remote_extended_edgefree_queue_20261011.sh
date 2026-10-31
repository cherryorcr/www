#!/usr/bin/env bash
set -u

ROOT=/home/pc/research/TAG_condensation
PY=/home/pc/miniconda3/envs/tagsam/bin/python
MINILM=/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip
TOOLS=$ROOT/tools
CODE=$ROOT/code/v2_llm_fusion
REFCODE=$ROOT/experiments/uq_q_seed_attribution_20261010/code
OUT=/dev/shm/extended_evidence_20261011
PROTO=$ROOT/results/extended_evidence_20261011/protocols
mkdir -p "$OUT" "$OUT/logs" "$OUT/cora_cache"

# Start only after the matched baseline queue has written its terminal status.
while [[ ! -s /dev/shm/baseline_seed_attribution_20261010/queue.status ]] || \
      ! grep -q '^done ' /dev/shm/baseline_seed_attribution_20261010/queue.status; do
  sleep 30
done

if [[ ! -s "$OUT/cora_cache/text_emb_pool_only.pt" ]]; then
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$CODE:$TOOLS:$ROOT" \
    "$PY" -u "$TOOLS/prepare_public_pool_text_cache.py" \
    --runner-dir "$CODE" --dataset cora \
    --data /home/pc/research/GraphCLIP/processed_data/cora.pt \
    --minilm "$MINILM" --output "$OUT/cora_cache/text_emb_pool_only.pt" \
    --device cuda:0 --batch-size 128 \
    > "$OUT/logs/cora_cache.log" 2>&1
fi

run_arxiv() {
  if [[ ! -s "$OUT/arxiv_o9001.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REFCODE:$CODE:$TOOLS:$ROOT" \
      "$PY" -u "$TOOLS/run_current_edgefree_dataset_20261010.py" \
      --dataset arxiv --reference "$PROTO/arxiv_o9001.json" \
      --text-cache "$ROOT/results/public_baselines_text_cache_20261001/arxiv" \
      --minilm "$MINILM" --output "$OUT/arxiv_o9001.json" \
      --device cuda:0 --construction-seed 271006 \
      > "$OUT/logs/arxiv_o9001.log" 2>&1
  fi
}

run_cora() {
  for order in 19611 19612 19613; do
    if [[ -s "$OUT/cora_o${order}.json" ]]; then
      continue
    fi
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REFCODE:$CODE:$TOOLS:$ROOT" \
      "$PY" -u "$TOOLS/run_current_edgefree_dataset_20261010.py" \
      --dataset cora --reference "$PROTO/cora_o${order}.json" \
      --text-cache "$OUT/cora_cache" --minilm "$MINILM" \
      --output "$OUT/cora_o${order}.json" --device cuda:0 \
      --construction-seed 271006 \
      > "$OUT/logs/cora_o${order}.log" 2>&1
  done
}

run_arxiv & pid_arxiv=$!
run_cora & pid_cora=$!
wait "$pid_arxiv"; rc_arxiv=$?
wait "$pid_cora"; rc_cora=$?
printf 'done %s arxiv=%s cora=%s\n' "$(date -Is)" "$rc_arxiv" "$rc_cora" > "$OUT/status"
exit $((rc_arxiv != 0 || rc_cora != 0))
