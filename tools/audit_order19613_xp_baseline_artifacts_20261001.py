"""Audit the additional order19613 X+P and adapter artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def audit_xp(dataset: str) -> dict:
    root = ROOT / "results" / f"xp_joint_independent_20261001_{dataset}" / f"{dataset}_order19613"
    errors: list[str] = []
    previous = None
    stages = []
    for audit_path in sorted(root.glob("stage*/audit.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("historical_raw_reads") != 0:
            errors.append(f"{audit_path}: historical_raw_reads != 0")
        if audit.get("report_labels_used") is not False:
            errors.append(f"{audit_path}: report_labels_used != false")
        memory = torch.load(audit_path.parent / "memory.pt", map_location="cpu", weights_only=False)
        if previous is not None:
            old_rows = len(previous["x"])
            for key in ("x", "text_emb", "mass", "depth", "stage"):
                if not torch.equal(memory[key][:old_rows], previous[key]):
                    errors.append(f"{audit_path}: parent prefix changed for {key}")
            if memory["texts"][:old_rows] != previous["texts"]:
                errors.append(f"{audit_path}: parent text prefix changed")
        previous = memory
        stages.append(int(audit.get("stage", -1)))
    return {"dataset": dataset, "stages": stages, "errors": errors, "passed": not errors}


def audit_external(dataset: str) -> dict:
    path = ROOT / "results" / f"external_continual_order19613_20261001_{dataset}" / dataset / "order19613" / "audit.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"dataset": dataset, "passed": payload.get("passed"),
            "fit_count": payload.get("fit_count"),
            "maximum_metric_error_pp": payload.get("maximum_metric_error_pp")}


def main() -> None:
    result = {
        "x_plus_p": [audit_xp(dataset) for dataset in ("cora", "citeseer")],
        "external_adapters": [audit_external(dataset) for dataset in ("cora", "citeseer")],
    }
    out = ROOT / "results" / "bounded_xp_final_evidence_20261001" / "order19613_artifact_audit.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if any(not row["passed"] for row in result["x_plus_p"]):
        raise SystemExit(1)
    if any(not row["passed"] for row in result["external_adapters"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
