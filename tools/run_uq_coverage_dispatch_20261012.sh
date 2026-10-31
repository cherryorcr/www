#!/usr/bin/env bash
set -euo pipefail

EXP=/home/pc/research/TAG_condensation/experiments/uq_q_coverage_20261012
OUT=/home/pc/research/TAG_condensation/results/uq_q_coverage_20261012
mkdir -p "$OUT"
exec 8>"$OUT/dispatch.lock"
flock -n 8 || exit 1
trap 'printf "failed %s\n" "$(date -Is)" > "$OUT/dispatch.status"' ERR

printf 'WikiCS on GPU0 %s\n' "$(date -Is)" > "$OUT/dispatch.status"
bash "$EXP/launch_first_batch_v2.sh" 0 wikics
printf 'SportsFit and BookHis queued on GPU1 %s\n' "$(date -Is)" > "$OUT/dispatch.status"
bash "$EXP/launch_first_batch_v2.sh" 1 sportsfit bookhis
printf 'complete %s\n' "$(date -Is)" > "$OUT/dispatch.status"
