"""Reproducible P1 budget and synthetic-row allocation sweep.

The sweep changes only memory budget policy, compression rate, or allocation
rule.  The construction remains synthetic-only append with frozen-old
topology, the text operator stays at the v3 concat control, and the student
protocol is inherited from run_unified_v3_continual.py.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_unified_v3_continual.py"

SMALL = ("cora", "citeseer")
LARGE = ("arxiv", "bookchild", "sportsfit")


def cases_for(dataset: str, phase: str) -> list[dict]:
    """Return a deterministic, resumable P1 matrix."""
    cases: list[dict] = []
    if phase in {"small", "all"} and dataset in SMALL:
        for rate in (0.01, 0.025, 0.05, 0.10):
            cases.append({"tag": f"rate_{rate:g}_sqrt",
                          "rate": rate, "allocation_mode": "sqrt",
                          "budget_mode": "fixed-rate"})
        for mode in ("legacy_15", "equal", "linear", "sqrt", "dispersion"):
            cases.append({"tag": f"alloc_{mode}_r005",
                          "rate": 0.05, "allocation_mode": mode,
                          "budget_mode": "fixed-rate"})
        cases.extend([
            {"tag": "policy_fixed_rate_r005_sqrt", "rate": 0.05,
             "allocation_mode": "sqrt", "budget_mode": "fixed-rate"},
            {"tag": "policy_fixed_total_sqrt", "rate": 0.05,
             "allocation_mode": "sqrt", "budget_mode": "fixed-total",
             "total_budget": 94 if dataset == "cora" else 127},
            {"tag": "policy_hybrid_r005_min5_sqrt", "rate": 0.05,
             "allocation_mode": "sqrt", "budget_mode": "hybrid",
             "min_increment": 5},
            {"tag": "policy_adaptive_r005_sqrt", "rate": 0.05,
             "allocation_mode": "sqrt", "budget_mode": "adaptive-rate"},
        ])
    if phase in {"large", "all"} and dataset in LARGE:
        for rate in (0.01, 0.05, 0.10):
            cases.append({"tag": f"rate_{rate:g}_sqrt",
                          "rate": rate, "allocation_mode": "sqrt",
                          "budget_mode": "fixed-rate"})
        for mode in ("equal", "sqrt", "dispersion"):
            cases.append({"tag": f"alloc_{mode}_r005",
                          "rate": 0.05, "allocation_mode": mode,
                          "budget_mode": "fixed-rate"})
        cases.extend([
            {"tag": "policy_hybrid_r005_min5_sqrt", "rate": 0.05,
             "allocation_mode": "sqrt", "budget_mode": "hybrid",
             "min_increment": 5},
            {"tag": "policy_adaptive_r005_sqrt", "rate": 0.05,
             "allocation_mode": "sqrt", "budget_mode": "adaptive-rate"},
        ])
    return cases


def run_case(dataset: str, case: dict, out_root: Path, args) -> dict:
    out = out_root / dataset / case["tag"]
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / "results.json"
    if result_file.exists():
        return {"dataset": dataset, "tag": case["tag"],
                "status": "existing", "output": str(out)}
    log = out / "run.log"
    cmd = [sys.executable, str(RUNNER),
           "--dataset", dataset,
           "--arms", "append_fixed",
           "--rate", str(case["rate"]),
           "--budget-mode", case["budget_mode"],
           "--allocation-mode", case["allocation_mode"],
           "--student-seeds", *[str(v) for v in args.student_seeds],
           "--output", str(out),
           "--device", args.device,
           "--minilm", args.minilm]
    if "total_budget" in case:
        cmd.extend(["--total-budget", str(case["total_budget"])])
    if "min_increment" in case:
        cmd.extend(["--min-increment", str(case["min_increment"])])
    if args.student_steps is not None:
        cmd.extend(["--student-steps", str(args.student_steps)])
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    status = "complete" if completed.returncode == 0 and result_file.exists() else "failed"
    return {"dataset": dataset, "tag": case["tag"], "status": status,
            "returncode": completed.returncode, "output": str(out)}


def summarize(root: Path) -> list[dict]:
    rows = []
    for result_file in sorted(root.glob("*/**/results.json")):
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            result_rows = data.get("results", [])
            final_stage = max((int(r["stage"]) for r in result_rows), default=-1)
            final = [r for r in result_rows if int(r["stage"]) == final_stage]
            if final:
                score = float(final[0].get("cum_dev_f1_mean", float("nan")))
                roots = int(final[0].get("roots", 0))
            else:
                score, roots = float("nan"), 0
            rows.append({"dataset": data.get("dataset"),
                         "tag": result_file.parent.name,
                         "budget_mode": data.get("budget_mode"),
                         "allocation_mode": data.get("allocation_mode"),
                         "rate": data.get("rate"),
                         "budgets": data.get("budgets"),
                         "effective_rates": data.get("effective_rates"),
                         "final_stage": final_stage,
                         "final_cum_dev_f1_mean": score,
                         "final_roots": roots,
                         "source_hash_sha256": data.get("source_hash_sha256"),
                         "result_file": str(result_file)})
        except Exception as exc:
            rows.append({"result_file": str(result_file), "status": "invalid",
                         "error": repr(exc)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["small", "large", "all"], default="small")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--student-seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", default="/home/pc/research/TAG_condensation/code/graphclip_unified_v3_2026_09_13/minilm_from_graphclip")
    args = parser.parse_args()
    datasets = args.datasets or (SMALL if args.phase == "small"
                                 else LARGE if args.phase == "large"
                                 else SMALL + LARGE)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    statuses = []
    for dataset in datasets:
        for case in cases_for(dataset, args.phase):
            print(json.dumps({"dataset": dataset, "case": case}), flush=True)
            statuses.append(run_case(dataset, case, root, args))
            if statuses[-1]["status"] == "failed":
                print(json.dumps(statuses[-1]), flush=True)
                raise SystemExit(1)
    summary = {"phase": args.phase, "datasets": list(datasets),
               "student_seeds": args.student_seeds,
               "student_steps": args.student_steps,
               "statuses": statuses,
               "results": summarize(root)}
    (root / f"summary_{args.phase}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "summary": str(root / f"summary_{args.phase}.json")}), flush=True)


if __name__ == "__main__":
    main()
