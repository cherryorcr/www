"""Summarize the frozen large-graph baseline comparison.

The semantic runner stores paired C0/semantic rows, while the coreset runner
stores a flat list of arm rows.  This script normalizes both layouts and
checks the protocol fields before producing a compact JSON/Markdown table.
It deliberately does not merge older base+2/base+3 results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("arxiv", "bookchild", "sportsfit")

# The first large-graph coreset run used this arm name, but its implementation
# was batched FPS after random projection rather than k-means medoid.  Keep the
# raw artifact name in the audit, while preventing the paper table from making
# a stronger algorithmic claim than the run supports.
ARM_LABELS = {
    "kmeans_medoid_subsampled": "real_node_fps_subsampled",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(row: dict) -> dict:
    return {
        "stage": int(row["stage"]),
        "budget": int(row.get("budget", row.get("effective_budget",
                                                   row.get("budget_reference", 0)))),
        "cum": float(row.get("cum_dev_f1_mean", 0.0)),
        "current": float(row.get("current_f1_mean", 0.0)),
        "old": (None if row.get("old_mean_f1_mean") is None
                else float(row["old_mean_f1_mean"])),
        "student_seconds": float(row.get("student_train_seconds", 0.0)),
        "build_seconds": float(row.get("stage_build_seconds", 0.0)),
        "wall_seconds": float(row.get("wall_seconds", 0.0)),
    }


def load_c0_and_semantic(path: Path) -> tuple[dict, dict, dict]:
    obj = load_json(path)
    protocol = obj["protocol"]
    rows = obj["summary"]["rows"]
    c0 = {int(row["stage"]): metric_row({"stage": row["stage"],
                                          "budget": row["budget"],
                                          **row["c0"]}) for row in rows}
    semantic = {int(row["stage"]): metric_row({"stage": row["stage"],
                                                "budget": row["budget"],
                                                **row["semantic_recursive"]})
                for row in rows}
    return c0, semantic, protocol


def load_flat_arms(path: Path) -> tuple[dict[str, dict[int, dict]], dict]:
    obj = load_json(path)
    arms: dict[str, dict[int, dict]] = {}
    for row in obj.get("results", []):
        raw_name = str(row["arm"])
        name = ARM_LABELS.get(raw_name, raw_name)
        arms.setdefault(name, {})[int(row["stage"])] = metric_row(row)
    return arms, obj


def resolve_results(root: Path, dataset: str) -> Path:
    """Resolve either a canonical directory or the temporary flat sync form."""
    candidates = (
        root / f"{dataset}_base4" / "results.json",
        root / f"{dataset}_base4_results.json",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no results.json for {dataset} below {root}; tried "
        + ", ".join(str(p) for p in candidates)
    )


def load_flat_arm_rows(path: Path) -> tuple[dict[str, dict[int, dict]], dict]:
    """Load all flat arms from a unified-v3 results file."""
    obj = load_json(path)
    arms: dict[str, dict[int, dict]] = {}
    for row in obj.get("results", []):
        arm = str(row.get("arm"))
        arms.setdefault(arm, {})[int(row["stage"])] = metric_row(row)
    return arms, obj


def load_single_arm(path: Path, arm: str) -> tuple[dict[int, dict], dict]:
    arms, obj = load_flat_arm_rows(path)
    if arm not in arms:
        raise ValueError(f"missing arm {arm!r} in {path}; found {sorted(arms)}")
    return arms[arm], obj


def protocol_signature(protocol: dict, manifest: dict | None = None) -> dict:
    source = protocol if "stage_protocol" in protocol else (manifest or protocol)
    student = source.get("student")
    student_batch = ((student or {}).get("batch")
                     if isinstance(student, dict) else None)
    return {
        "dataset": source.get("dataset"),
        "rate": source.get("memory_rate", source.get("rate")),
        "report_dev_mod": source.get("report_dev_mod"),
        "student_seeds": source.get("student_seeds"),
        "student_batch": source.get("student_batch", student_batch),
        "stage_classes": source.get("stage_classes"),
        "budgets": source.get("budgets"),
    }


def delta_summary(rows: dict[int, dict], c0: dict[int, dict]) -> dict:
    stages = sorted(rows)
    inc = [stage for stage in stages if stage > 0 and stage in c0]
    deltas = [rows[s]["cum"] - c0[s]["cum"] for s in inc]
    old = [rows[s]["old"] - c0[s]["old"] for s in inc
           if rows[s]["old"] is not None and c0[s]["old"] is not None]
    new = [rows[s]["current"] - c0[s]["current"] for s in inc]
    return {
        "stages": stages,
        "stage_cum": [rows[s]["cum"] for s in stages],
        "final_cum": rows[stages[-1]]["cum"],
        "incremental_mean_delta_pp": sum(deltas) / max(1, len(deltas)),
        "old_mean_delta_pp": sum(old) / max(1, len(old)),
        "new_mean_delta_pp": sum(new) / max(1, len(new)),
        "positive_incremental_cells": sum(d > 0 for d in deltas),
        "incremental_cells": len(deltas),
        "worst_incremental_delta_pp": min(deltas) if deltas else None,
        "delta_by_stage_pp": {str(s): rows[s]["cum"] - c0[s]["cum"]
                              for s in inc},
    }


def md_table(dataset: str, methods: dict[str, dict]) -> str:
    lines = [f"### {dataset}", "",
             "| method | stage 0 | stage 1 | stage 2 | stage 3 | stage 4 | "
             "incremental mean delta | old delta | new delta | positive | worst |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, summary in methods.items():
        values = summary["stage_cum"] + [None] * (5 - len(summary["stage_cum"]))
        fmt = lambda value: "n/a" if value is None else f"{value:.3f}"
        lines.append("| " + " | ".join([
            name, *(fmt(v) for v in values[:5]),
            fmt(summary["incremental_mean_delta_pp"]),
            fmt(summary["old_mean_delta_pp"]),
            fmt(summary["new_mean_delta_pp"]),
            f"{summary['positive_incremental_cells']}/{summary['incremental_cells']}",
            fmt(summary["worst_incremental_delta_pp"]),
        ]) + " |")
    return "\n".join(lines)


def md_cost_table(dataset: str, rows_by_name: dict[str, dict[int, dict]]) -> str:
    lines = [
        f"### {dataset}: reported compute and construction cost", "",
        "| method | build seconds (sum) | student seconds (sum) | raw wall seconds (sum) | reported total |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, rows in rows_by_name.items():
        build = sum(float(row.get("build_seconds", 0.0)) for row in rows.values())
        student = sum(float(row.get("student_seconds", 0.0)) for row in rows.values())
        wall = sum(float(row.get("wall_seconds", 0.0)) for row in rows.values())
        reported = wall if wall > 0.0 else build + student
        lines.append(f"| {name} | {build:.1f} | {student:.1f} | {wall:.1f} | {reported:.1f} |")
    return "\n".join(lines)


def cost_summary(rows_by_name: dict[str, dict[int, dict]]) -> dict[str, dict]:
    """Keep the same cost aggregation used by the Markdown table in JSON."""
    out = {}
    for name, rows in rows_by_name.items():
        build = sum(float(row.get("build_seconds", 0.0)) for row in rows.values())
        student = sum(float(row.get("student_seconds", 0.0))
                      for row in rows.values())
        wall = sum(float(row.get("wall_seconds", 0.0))
                   for row in rows.values())
        out[name] = {
            "build_seconds_sum": build,
            "student_seconds_sum": student,
            "wall_seconds_sum": wall,
            "reported_total_seconds": wall if wall > 0.0 else build + student,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c0-root", required=True,
                        help="root containing <dataset>_base4/results.json")
    parser.add_argument("--semantic-root", required=True)
    parser.add_argument("--coreset-root", required=True)
    parser.add_argument("--full-history-root", required=True,
                        help="root containing full-history results.json")
    parser.add_argument("--full-history-gpu-root", required=False,
                        help="root containing the GPU full-history results.json")
    parser.add_argument("--raw-root", required=False,
                        help="root containing raw_equal/raw_long results.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    c0_root = Path(args.c0_root)
    semantic_root = Path(args.semantic_root)
    coreset_root = Path(args.coreset_root)
    full_history_root = Path(args.full_history_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    report = {"datasets": {}, "protocol_warnings": []}
    markdown = ["# Frozen Large-Graph Baseline Comparison (2026-09-26)", "",
                "Only base+4, fixed-rate 5%, report-dev mod 3 runs are included.",
                "C0 is the paired C0 stored by the semantic recursive runner; "
                "coreset and full-history differences are interpreted as "
                "same-protocol absolute comparisons and are not treated as "
                "same-process confidence intervals. The historical arm named "
                "kmeans_medoid_subsampled is reported as real_node_fps_subsampled "
                "because that is what the implementation executed.", ""]

    for dataset in DATASETS:
        semantic_path = resolve_results(semantic_root, dataset)
        c0, semantic, semantic_protocol = load_c0_and_semantic(semantic_path)
        c0_manifest = load_json(resolve_results(c0_root, dataset))
        arms, coreset_manifest = load_flat_arms(resolve_results(coreset_root, dataset))
        full_history, full_history_manifest = load_single_arm(
            resolve_results(full_history_root, dataset),
            "full_history")
        all_methods = {"c0": c0, "semantic_recursive": semantic,
                       **arms, "full_history": full_history}
        method_manifests = {
            "c0": c0_manifest,
            "semantic_recursive": semantic_protocol,
            **{name: coreset_manifest for name in arms},
            "full_history": full_history_manifest,
        }
        if args.full_history_gpu_root:
            gpu_arms, gpu_manifest = load_flat_arm_rows(
                resolve_results(Path(args.full_history_gpu_root), dataset))
            for name, rows in gpu_arms.items():
                all_methods[name] = rows
                method_manifests[name] = gpu_manifest
        if args.raw_root:
            raw_arms, raw_manifest = load_flat_arm_rows(
                resolve_results(Path(args.raw_root), dataset))
            for name, rows in raw_arms.items():
                all_methods[name] = rows
                method_manifests[name] = raw_manifest
        summaries = {name: delta_summary(rows, c0)
                     for name, rows in all_methods.items()}
        report["datasets"][dataset] = {
            "protocol": {
                "c0": protocol_signature(c0_manifest, c0_manifest),
                "semantic": protocol_signature(semantic_protocol, semantic_protocol),
                "coreset": protocol_signature(coreset_manifest, coreset_manifest),
                "full_history": protocol_signature(
                    full_history_manifest, full_history_manifest),
                **{name: protocol_signature(manifest, manifest)
                   for name, manifest in method_manifests.items()
                   if name not in {"c0", "semantic_recursive", "full_history"}},
            },
            "methods": summaries,
            "costs": cost_summary(all_methods),
            "coreset_manifest": {
                key: coreset_manifest.get(key)
                for key in ("arms", "rate", "report_dev_mod", "student_seeds",
                            "student_batch", "stage_classes", "budgets",
                            "selection_candidate_cap", "selection_batch",
                "selection_projection_dim")
            },
            "method_notes": {
                "real_node_fps_subsampled": {
                    "raw_arm": "kmeans_medoid_subsampled",
                    "claim": "batched FPS with fixed random projection on a capped real-node candidate pool; not k-means medoid",
                },
                "full_history": {
                    "historical_raw_access": True,
                    "rewrites_previous_memory": True,
                },
                "full_history_gpu": {
                    "historical_raw_access": True,
                    "rewrites_previous_memory": True,
                    "note": "same construction family as full_history, GPU implementation and separately reported",
                },
                "raw_equal": {
                    "historical_raw_access": True,
                    "rewrites_previous_memory": True,
                    "note": "complete cumulative raw induced graph; equal optimizer steps; offline reference",
                },
                "raw_long": {
                    "historical_raw_access": True,
                    "rewrites_previous_memory": True,
                    "note": "complete cumulative raw induced graph; fixed 8192-step endpoint; not a convergence upper bound",
                },
            },
        }
        markdown.append(md_table(dataset, summaries))
        markdown.append("")
        markdown.append(md_cost_table(dataset, all_methods))
        markdown.append("")

    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    output.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"event": "complete",
                      "json": str(output.with_suffix('.json')),
                      "markdown": str(output.with_suffix('.md'))},
                     ensure_ascii=True))


if __name__ == "__main__":
    main()
