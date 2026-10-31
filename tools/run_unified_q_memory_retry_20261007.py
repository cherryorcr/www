"""Retry the two SportsFit OOMs with mathematically unchanged checkpointing."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    root = Path("/home/pc/research/TAG_condensation")
    snapshot = Path(__file__).resolve().parent
    code = snapshot / "code"
    previous = root / "experiments/unified_q_diagnostics_20261007_retry/code"
    if not code.exists():
        shutil.copytree(previous, code)
        for name in ("unified_q_multimodal.py", "test_unified_q_diagnostics.py"):
            shutil.copy2(snapshot / name, code / name)
    if (code / "unified_q_multimodal.py").read_bytes() != (snapshot / "unified_q_multimodal.py").read_bytes():
        raise ValueError("immutable memory-retry snapshot differs")
    output = root / "results/unified_q_diagnostics_20261007"
    env = dict(os.environ, PYTHONPATH=str(code) + ":" + str(root / "code/v2_llm_fusion"),
               TOKENIZERS_PARALLELISM="false")
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q",
        str(code / "test_unified_q_multimodal.py"), str(code / "test_unified_q_diagnostics.py")], env=env)
    if tests.returncode:
        raise SystemExit(tests.returncode)
    arms = ("uq_30", "uq_v3")
    plan = {"reason": "both original SportsFit arms exhausted a 48-GiB GPU at stage zero",
        "change": "checkpoint large Sinkhorn blocks even with <=64 iterations; chunk 8",
        "unchanged": ["budget", "data", "loss", "outer steps", "transport iterations and tolerance", "seeds", "Student"],
        "core_sha256": hashlib.sha256((code / "unified_q_multimodal.py").read_bytes()).hexdigest(),
        "tasks": [{"tag": f"memretry_transfer_sportsfit_{a}", "arms": [a]} for a in arms]}
    path = output / "memory_retry_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("memory-retry plan changed")
    path.write_text(json.dumps(plan, indent=2))

    def run(item):
        gpu, arm = item
        tag = f"memretry_transfer_sportsfit_{arm}"
        destination = output / tag
        result = destination / "results.json"
        if result.exists() and json.loads(result.read_text()).get("status") == "complete":
            return {"tag": tag, "exit_code": 0, "reused": True}
        if shutil.disk_usage(root).free < 2 * 2**30:
            return {"tag": tag, "exit_code": -2, "reason": "disk reserve below 2 GiB"}
        command = [sys.executable, "-u", str(code / "run_unified_q_diagnostics_20261007.py"),
            "--dataset", "sportsfit", "--reference", str(root / "experiments/unified_q_diagnostics_20261007/sportsfit_reference.json"),
            "--arms", arm, "--construction-seed", "271006", "--sinkhorn-max-steps", "2048",
            "--output", str(destination), "--text-cache", str(root / "results/public_baselines_text_cache_20261001/sportsfit"),
            "--minilm", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", "--device", f"cuda:{gpu}"]
        print(json.dumps({"event": "start", "tag": tag, "gpu": gpu}), flush=True)
        start = time.time()
        with (output / f"{tag}.log").open("a") as log:
            status = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = {"tag": tag, "gpu": gpu, "exit_code": status.returncode, "seconds": time.time() - start}
        (output / f"memory_retry_lane{gpu}.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"event": "finish", **record}), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(run, enumerate(arms)))
    (output / "memory_retry_exit_codes.json").write_text(json.dumps(records, indent=2))
    if any(row["exit_code"] for row in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
