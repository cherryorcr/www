#!/usr/bin/env bash
set -euo pipefail
snapshot=/home/pc/research/TAG_condensation/experiments/v5_retention_20260919
output=/home/pc/research/TAG_condensation/results/v5_retention_20260919
python=/home/pc/miniconda3/envs/tagsam/bin/python
runner="$snapshot/code/v2_llm_fusion/run_v5_retention.py"
protocol="$snapshot/docs/current/V5_RETENTION_FROZEN_PROTOCOL_ZH_2026-09-19.md"
mkdir -p "$output"
cd "$snapshot"
"$python" "$runner" --dataset cora --output "$output/cora" --device cuda:0 --protocol "$protocol" > "$output/cora.log" 2>&1 &
cora_pid=$!
"$python" "$runner" --dataset citeseer --output "$output/citeseer" --device cuda:1 --protocol "$protocol" > "$output/citeseer.log" 2>&1 &
citeseer_pid=$!
status=0
wait "$cora_pid" || status=1
wait "$citeseer_pid" || status=1
exit "$status"
