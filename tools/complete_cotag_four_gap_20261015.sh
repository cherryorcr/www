#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/pc/research/TAG_condensation
OUT=$ROOT/results/cotag_q_four_gap_20261015_v1
LOGS="$OUT/logs/completion-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$LOGS"
status="$LOGS/status"
failures=0
record() { printf '%s %s\n' "$(date -Is)" "$*" >> "$status"; }
record 'waiting for registered full-history queue'
while pgrep -f '^bash /home/pc/research/TAG_condensation/tools/run_cotag_four_gap_queue_20261015.sh fullhistory 1$' > /dev/null; do
  sleep 10
done
record 'starting TAGSAM fixed-prefix budget diagnostic on GPU1'
bash "$ROOT/tools/run_cotag_tagsam_budget_queue_20261015.sh" 1 > "$LOGS/tagsam.log" 2>&1 || failures=$((failures+1))
record 'starting construction-seed 271109 text shard on GPU1'
bash "$ROOT/tools/run_cotag_four_gap_queue_20261015.sh" text 1 271109 > "$LOGS/text.log" 2>&1 || failures=$((failures+1))
record 'waiting for GPU0 text shard before isolated resource measurement'
while pgrep -f '^bash /home/pc/research/TAG_condensation/tools/run_cotag_four_gap_queue_20261015.sh text 0 271006 271108$' > /dev/null; do
  sleep 10
done
record 'starting isolated identity-operator benchmark on GPU1'
bash "$ROOT/tools/run_cotag_four_gap_queue_20261015.sh" cost 1 > "$LOGS/cost.log" 2>&1 || failures=$((failures+1))
record "complete failures=$failures"
test "$failures" -eq 0
