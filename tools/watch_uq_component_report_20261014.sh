#!/usr/bin/env bash
set -eu

UQROOT=/home/pc/research/TAG_condensation
UQCODE=$UQROOT/experiments/uq_component_ablations_20261014/code
UQOUT=$UQROOT/results/uq_component_ablations_20261014
UQPY=/home/pc/miniconda3/envs/tagsam/bin/python

while :; do
  primary=$(cat "$UQOUT/queue.status" 2>/dev/null || true)
  supplementary=$(cat "$UQOUT/supplementary.status" 2>/dev/null || true)
  case "$primary" in complete*) primary_done=1;; *) primary_done=0;; esac
  case "$supplementary" in complete*|blocked_*) supplementary_done=1;; *) supplementary_done=0;; esac
  if [ "$primary_done" -eq 1 ] && [ "$supplementary_done" -eq 1 ]; then
    break
  fi
  sleep 60
done

exec 9>"$UQOUT/report.lock"
flock -n 9 || exit 1
cd "$UQROOT"
OMP_NUM_THREADS=2 "$UQPY" "$UQCODE/report_uq_component_ablations_20261014.py" \
  > "$UQOUT/report_launcher.log" 2>&1
