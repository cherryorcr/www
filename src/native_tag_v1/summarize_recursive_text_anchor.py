"""Aggregate strict stage-1 -> stage-2 Text-Anchored ASM validation."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

import torch


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return float(statistics.mean(values))


def canonical_edges(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index
    base = int(edge_index.max()) + 1
    order = torch.argsort(edge_index[0].long() * base + edge_index[1].long())
    return edge_index[:, order]


def field_audit(root: Path, seed: int) -> dict[str, Any]:
    graphs = {}
    for method in ("w0", "w001"):
        graphs[method] = torch.load(
            root / f"c{seed}_{method}_s1" / "selected_dev_artifact.pt",
            map_location="cpu", weights_only=True,
        )
    left, right = graphs["w0"], graphs["w001"]
    equality = {
        key: bool(torch.equal(left[key], right[key]))
        for key in ("x", "text_emb", "weights", "origin_ids", "stage")
    }
    equality["texts"] = left["texts"] == right["texts"]
    equality["edge_index_raw"] = bool(torch.equal(left["edge_index"], right["edge_index"]))
    equality["edge_index_canonical"] = bool(torch.equal(
        canonical_edges(left["edge_index"]), canonical_edges(right["edge_index"])
    ))
    return equality


def selection_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    pattern = re.compile(r"c(?P<seed>\d+)_(?P<method>w0|w001)_s(?P<stage>[12])(?:_recursive)?$")
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        match = pattern.fullmatch(directory.name)
        selection_path = directory / "selected_dev_artifact.selection.json"
        if not match or not selection_path.exists():
            continue
        value = load(selection_path)
        rows.append({
            "condensation_seed": int(match.group("seed")),
            "method": "text_anchor_0.01" if match.group("method") == "w001" else "asm_0",
            "stage": int(match.group("stage")),
            "selected_outer_step": int(value["selected_outer_step"]),
            "selected_dev_macro_f1": float(value["selected_metric_value"]),
            "selection_student_seeds": value["student_seeds"],
            "test_or_future_used_for_selection": value["test_or_future_used_for_selection"],
            "artifact_sha256": value["exported_artifact_sha256"],
        })
    return rows


def matrix_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in matrix["rows"]:
        match = re.search(r"confirm_c(?P<seed>\d+)_(?P<method>w0|w001)_s(?P<stage>[12])", source["run"])
        if not match:
            raise ValueError(f"cannot identify recursive run: {source['run']}")
        stage = int(source["model_stage"])
        cells = {(int(cell["eval_stage"]), cell["partition"]): cell["metric"]["macro_f1"]
                 for cell in source["cells"]}
        rows.append({
            "condensation_seed": int(match.group("seed")),
            "method": "text_anchor_0.01" if match.group("method") == "w001" else "asm_0",
            "stage": stage,
            "student_seed": int(source["student_seed"]),
            "cumulative_train_macro_f1": source["cumulative_seen"]["train"]["macro_f1"],
            "cumulative_dev_macro_f1": source["cumulative_seen"]["checkpoint"]["macro_f1"],
            "cumulative_test_macro_f1": source["cumulative_seen"]["test"]["macro_f1"],
            "old_dev_macro_f1": mean([cells[(index, "checkpoint")] for index in range(stage)]),
            "old_test_macro_f1": mean([cells[(index, "test")] for index in range(stage)]),
            "current_dev_macro_f1": cells[(stage, "checkpoint")],
            "current_test_macro_f1": cells[(stage, "test")],
            "future_test_macro_f1": ((source.get("cumulative_future_test") or {}).get("macro_f1")),
        })
    return rows


def aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "cumulative_train_macro_f1", "cumulative_dev_macro_f1", "cumulative_test_macro_f1",
        "old_dev_macro_f1", "old_test_macro_f1", "current_dev_macro_f1",
        "current_test_macro_f1", "future_test_macro_f1",
    ]
    result: dict[str, Any] = {"means": [], "paired_anchor_minus_asm": []}
    for stage in sorted({row["stage"] for row in rows}):
        stage_rows = [row for row in rows if row["stage"] == stage]
        for method in ("asm_0", "text_anchor_0.01"):
            group = [row for row in stage_rows if row["method"] == method]
            result["means"].append({
                "stage": stage,
                "method": method,
                "n": len(group),
                **{metric: mean([float(row[metric]) for row in group if row[metric] is not None])
                   for metric in metrics if any(row[metric] is not None for row in group)},
            })
        keyed = {(row["method"], row["condensation_seed"], row["student_seed"]): row
                 for row in stage_rows}
        pairs = sorted({(row["condensation_seed"], row["student_seed"]) for row in stage_rows})
        delta = {"stage": stage, "n_pairs": len(pairs)}
        for metric in metrics:
            values = [
                keyed[("text_anchor_0.01", *pair)][metric] - keyed[("asm_0", *pair)][metric]
                for pair in pairs if keyed[("text_anchor_0.01", *pair)][metric] is not None
            ]
            if not values:
                continue
            delta[metric] = {
                "mean_delta": mean(values),
                "positive_pairs": sum(value > 0 for value in values),
                "total_pairs": len(values),
                "by_condensation_seed": {
                    str(seed): mean([
                        keyed[("text_anchor_0.01", seed, student)][metric]
                        - keyed[("asm_0", seed, student)][metric]
                        for cond, student in pairs if cond == seed
                    ]) for seed in sorted({pair[0] for pair in pairs})
                },
            }
        result["paired_anchor_minus_asm"].append(delta)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix_path = args.root / "recursive_protocol_matrix.json"
    matrix = load(matrix_path)
    selections = selection_rows(args.root)
    if not selections or any(row["test_or_future_used_for_selection"] is not False for row in selections):
        raise ValueError("recursive artifact selection was not development-only")
    rows = matrix_rows(matrix)
    payload = {
        "protocol": "strict_recursive_text_anchor_validation_v1",
        "selection_rule": "maximize mean cumulative development Macro-F1 over fixed outer snapshots; ties use lower outer step",
        "selection_student_seeds": [100, 101],
        "confirmation_student_seeds": [102, 103, 104, 105, 106, 107],
        "test_or_future_used_for_selection": False,
        "confirmation_status": matrix["confirmation_status"],
        "stage1_candidate_baseline_field_audit": {
            str(seed): field_audit(args.root, seed) for seed in (44, 45)
        },
        "selections": selections,
        "confirmation_aggregates": aggregates(rows),
        "confirmation_rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["confirmation_aggregates"], indent=2))


if __name__ == "__main__":
    main()
