#!/usr/bin/env bash
set -euo pipefail

EXP=/home/pc/research/TAG_condensation/experiments/uq_q_coverage_20261012
OUT=/home/pc/research/TAG_condensation/results/uq_q_coverage_20261012
exec 8>"$OUT/dispatch.lock"
flock -n 8 || exit 1
trap 'printf "failed %s\n" "$(date -Is)" > "$OUT/dispatch.status"' ERR

# WikiCS is complete; use available GPU0 without interrupting GPU1 confirmation.
printf 'BookHis then SportsFit on GPU0 %s\n' "$(date -Is)" > "$OUT/dispatch.status"
bash "$EXP/launch_first_batch_v2.sh" 0 bookhis sportsfit
printf 'complete %s\n' "$(date -Is)" > "$OUT/dispatch.status"
