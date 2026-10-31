"""Audit the strict small-graph Bounded X+P artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def edge_set(edge: torch.Tensor, n: int) -> list[tuple[int, int]]:
    if edge.numel() == 0:
        return []
    return sorted((int(a), int(b)) for a, b in edge.t().tolist())


def audit_memory(previous: dict | None, current: dict, stage: int) -> list[str]:
    errors: list[str] = []
    if previous is None:
        return errors
    old = len(previous["x"])
    if len(current["x"]) <= old:
        errors.append(f"stage {stage}: memory did not grow beyond parent")
        return errors
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if not torch.equal(current[key][:old], previous[key]):
            errors.append(f"stage {stage}: parent prefix changed for {key}")
    if current["texts"][:old] != previous["texts"]:
        errors.append(f"stage {stage}: parent text prefix changed")
    current_edges = current["edge_index"].long()
    prefix = current_edges[:, (current_edges < old).all(dim=0)]
    if edge_set(prefix, old) != edge_set(previous["edge_index"].long(), old):
        errors.append(f"stage {stage}: old-old edge set changed")
    return errors


def main() -> None:
    errors: list[str] = []
    cells = []
    for dataset in ("cora", "citeseer"):
        root = ROOT / "results" / f"xp_joint_full_20261001_{dataset}"
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if summary.get("protocol") != "bounded-student-compatible-x-class+p-unified-v1":
            errors.append(f"{dataset}: summary protocol is not x-class+p")
        for block in summary["results"]:
            order = block["protocol"]["order"]
            if block["protocol"].get("method") != "x-class+p":
                errors.append(f"{dataset}/{order}: protocol method")
            previous = None
            stages = []
            for stage in range(3):
                cell = root / f"{dataset}_{order}"
                memory_path = cell / f"stage{stage}_memory.pt"
                memory = load(memory_path)
                stages.append(int(len(memory["x"])))
                errors.extend(f"{dataset}/{order}: {error}"
                              for error in audit_memory(previous, memory, stage))
                if stage > 0:
                    audit_path = cell / f"stage{stage}" / "audit.json"
                    audit = json.loads(audit_path.read_text(encoding="utf-8"))
                    variables = set(audit.get("outer_gradient_variables", []))
                    if "a_logits" in variables:
                        errors.append(f"{dataset}/{order}/s{stage}: A variable present")
                    if not {"x_class_relative", "p_logits"}.issubset(variables):
                        errors.append(f"{dataset}/{order}/s{stage}: missing X/P variable")
                    if audit.get("historical_raw_reads") != 0:
                        errors.append(f"{dataset}/{order}/s{stage}: historical raw read")
                    if audit.get("report_labels_used") is not False:
                        errors.append(f"{dataset}/{order}/s{stage}: report labels used")
                    if audit.get("mode") != "x-class+p":
                        errors.append(f"{dataset}/{order}/s{stage}: wrong audit mode")
                previous = memory
            cells.append({"dataset": dataset, "order": order,
                          "stage_rows": stages,
                          "parent_prefix_checked": True,
                          "x_p_only_checked": True})
    result = {
        "protocol": "small bounded X+P artifact audit 2026-10-01",
        "cells": cells,
        "error_count": len(errors),
        "errors": errors,
        "checks": [
            "parent synthetic X/text/mass/depth/stage prefix is unchanged",
            "parent old-old edge set is unchanged",
            "X/P are the only outer variables",
            "historical raw reads and report-label selection are disabled",
        ],
    }
    output = ROOT / "results" / "bounded_xp_final_evidence_20261001" / "small_xp_artifact_audit.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
