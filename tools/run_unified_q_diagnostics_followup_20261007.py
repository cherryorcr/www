"""Resume missing experiments in new folders after a solver ceiling failure."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("finish", "confirmation", "transfer"), required=True)
    args = p.parse_args()
    root = Path("/home/pc/research/TAG_condensation")
    output = root / "results/unified_q_diagnostics_20261007"
    snapshot = root / "experiments/unified_q_diagnostics_20261007_retry"
    minilm = "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"
    tasks = []
    if args.phase == "finish":
        plan = json.loads((output / "diagnostics_plan.json").read_text())
        for task in plan["tasks"]:
            if task["kind"] != "diag":
                continue
            result = output / task["tag"] / "results.json"
            if result.exists() and json.loads(result.read_text())["status"] == "complete":
                continue
            tasks.append(dict(task, tag="retry_" + task["tag"]))
        for dataset in ("wikics", "bookhis"):
            tasks.append({"kind": "operator", "dataset": dataset,
                          "tag": f"operator_{dataset}_fixed_rate"})
    elif args.phase == "confirmation":
        for order in (9101, 9102, 9103):
            for seed in (271006, 271007, 271008):
                if (order, seed) == (9101, 271006):
                    continue
                for dataset in ("wikics", "bookhis"):
                    tasks.append({"kind": "diag", "dataset": dataset, "arms": ["cat_cctp", "uq_v3"],
                        "tag": f"confirm_{dataset}_o{order}_c{seed}", "seed": seed, "order": order})
    else:
        for arm in ("cat_cctp", "uq_30", "uq_v3"):
            tasks.append({"kind": "diag", "dataset": "sportsfit", "arms": [arm],
                          "tag": f"transfer_sportsfit_{arm}", "seed": 271006})
    plan_path = output / f"followup_{args.phase}_plan.json"
    registered = {"reason": "512 iteration ceiling failed fixed 1e-4 tolerance; maximum 2048, no F1 tuning",
                  "tasks": tasks, "phase": args.phase, "solver_tolerance": 1e-4,
                  "solver_maximum": 2048}
    if plan_path.exists():
        registered = json.loads(plan_path.read_text())
        tasks = registered["tasks"]
    else:
        plan_path.write_text(json.dumps(registered, indent=2))

    def execute(task, gpu):
        tag, dataset = task["tag"], task["dataset"]
        cache = root / f"results/public_baselines_text_cache_20261001/{dataset}"
        env = dict(os.environ, TOKENIZERS_PARALLELISM="false")
        if task["kind"] == "operator":
            env["PYTHONPATH"] = str(root / "experiments/unified_q_diagnostics_20261007/ab") + ":" + str(root / "experiments/unified_q_revision_20261007/code") + ":" + str(root / "code/v2_llm_fusion")
            cmd = [sys.executable, "-u", str(root / "experiments/unified_q_diagnostics_20261007/ab/run_unified_q_operator_ab_20261007.py"),
                   "--source", str(root / f"results/unified_q_revision_verified_20261007/{dataset}_fixed_rate_order9101_seed271006")]
            finished = output / tag / "operator_ab.json"
        else:
            env["PYTHONPATH"] = str(snapshot / "code") + ":" + str(root / "code/v2_llm_fusion")
            reference = (root / f"results/unified_q_large_full_20261007/{dataset}_unified_q_smoke.json" if dataset != "sportsfit"
                         else root / "experiments/unified_q_diagnostics_20261007/sportsfit_reference.json")
            cmd = [sys.executable, "-u", str(snapshot / "code/run_unified_q_diagnostics_20261007.py"),
                   "--dataset", dataset, "--reference", str(reference), "--arms", *task["arms"],
                   "--construction-seed", str(task["seed"]), "--sinkhorn-max-steps", "2048"]
            if task.get("order", 9101) != 9101:
                cmd += ["--order-seed", str(task["order"])]
            finished = output / tag / "results.json"
        if finished.exists() and json.loads(finished.read_text()).get("status") == "complete":
            return {"tag": tag, "exit_code": 0, "reused": True}
        cmd += ["--output", str(output / tag), "--text-cache", str(cache), "--minilm", minilm, "--device", f"cuda:{gpu}"]
        if shutil.disk_usage(root).free < 2 * 2**30:
            return {"tag": tag, "exit_code": -2, "error": "disk safety threshold"}
        print(json.dumps({"event": "start", "tag": tag, "gpu": gpu}), flush=True)
        started = time.time()
        with (output / f"{tag}.log").open("a", encoding="utf-8") as log:
            child = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        result = {"tag": tag, "exit_code": child.returncode, "gpu": gpu, "seconds": time.time() - started}
        print(json.dumps({"event": "finish", **result}), flush=True)
        return result

    # Transfer uses one task per GPU because dense Q is substantially larger.
    count = 2 if args.phase == "transfer" else 4
    lanes = [[] for _ in range(count)]
    for i, task in enumerate(tasks):
        lanes[i % count].append(task)
    def lane(index):
        results = []
        for task in lanes[index]:
            results.append(execute(task, index % 2))
            (output / f"followup_{args.phase}_lane{index}.json").write_text(json.dumps(results, indent=2))
        return results
    with ThreadPoolExecutor(max_workers=count) as pool:
        results = [r for lane_results in pool.map(lane, range(count)) for r in lane_results]
    (output / f"followup_{args.phase}_exit_codes.json").write_text(json.dumps(results, indent=2))
    if any(r["exit_code"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
