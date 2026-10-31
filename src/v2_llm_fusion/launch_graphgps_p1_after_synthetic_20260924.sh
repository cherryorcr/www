#!/usr/bin/env bash
set -euo pipefail

while pgrep -f '[/]launch_graphgps_p1_synthetic_20260924.sh' >/dev/null; do
  sleep 30
done
exec /home/pc/research/TAG_condensation/code/v2_llm_fusion/launch_graphgps_p1_raw_20260924.sh
