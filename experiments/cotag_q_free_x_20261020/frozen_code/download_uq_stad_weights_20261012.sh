#!/usr/bin/env bash
set -eu
EXP=/home/pc/research/TAG_condensation/experiments/uq_final_evidence_20261012
PY=/home/pc/miniconda3/envs/tagsam/bin/python
MODEL=$EXP/models/qwen2.5-1.5b
URL=https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct/resolve/master/model.safetensors
TMP=$MODEL/model.safetensors.modelscope.tmp
curl --fail --location --silent --show-error --max-time 1200 --output "$TMP" "$URL"
"$PY" - "$TMP" "$MODEL/model.safetensors" <<'PY'
import hashlib
from pathlib import Path
import sys
source, target = map(Path, sys.argv[1:])
if source.stat().st_size != 3087467144:
    raise ValueError("unexpected model byte count")
digest = hashlib.sha256()
with source.open("rb") as stream:
    for part in iter(lambda: stream.read(8 * 1024**2), b""):
        digest.update(part)
if digest.hexdigest() != "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee":
    raise ValueError("official pinned model digest differs")
source.replace(target)
print("Pinned model weights verified", flush=True)
PY
"$PY" "$EXP/code/prepare_uq_stad_generator_20261012.py" --verify-local \
  --transfer modelscope --endpoint "$URL"
