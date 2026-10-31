"""Read-only collection of registered results; never read memories or sealed tables."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def collect(root):
    entries, issues = [], []
    for category in ("confirmation", "controls", "baselines", "stad"):
        for path in sorted((root / category).glob("*/results.json")):
            raw = path.read_bytes()
            try:
                result = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as error:
                issues.append(dict(path=str(path), error=str(error)))
                continue
            protocol = result["protocol"]
            stages = len(protocol["budgets"])
            arms = protocol.get("arms", [protocol.get("arm")])
            rows = result["rows"]
            complete, errors = [], []
            for arm in arms:
                selected = [row for row in rows if row["arm"] == arm]
                stage_ids = [row["stage"] for row in selected]
                if len(stage_ids) != len(set(stage_ids)):
                    errors.append(f"duplicate stages: {arm}")
                if sorted(stage_ids) == list(range(stages)):
                    complete.append(arm)
                for row in selected:
                    if row["memory_rows"] != protocol["budgets"][row["stage"]]:
                        errors.append(f"budget mismatch: {arm} s{row['stage']}")
                    for split in ("report_dev", "sealed_test"):
                        score = row["scores"][split]
                        if score["student_seeds"] != [150, 151, 152]:
                            errors.append(f"Student seeds differ: {arm} {split}")
                        if score["student_steps"] != 2048 or score["student_batch"] != 24:
                            errors.append(f"Student budget differs: {arm} {split}")
                        for key in ("final_f1", "accuracy", "old_f1", "new_f1"):
                            value = score[key]
                            if value is not None and not math.isfinite(value):
                                errors.append(f"nonfinite {key}: {arm} {split}")
            if result["status"] == "complete" and set(complete) != set(arms):
                errors.append("complete marker without all stage results")
            for row in rows:
                audit = row.get("audit", {})
                keep = ("constructor", "shared", "config", "assignment_parameter_count",
                        "residual_parameter_count", "legal_assignment_entries", "equal_parameter_budget",
                        "parameterization", "parameter_budget_caveat", "diagnostics", "assignments",
                        "initial_q_max_error", "historical_raw_reads", "old_update_count",
                        "old_x_relative_change_max", "text_hard_forward_max_error", "text_storage",
                        "forward_and_storage_match", "stored_vs_reference_string_max_error",
                        "q_row_sum_max_error", "q_column_relative_error", "sinkhorn_max_row_error",
                        "sinkhorn_max_column_relative_error", "distribution_matching", "trace",
                        "adaptations", "official_reproduction", "native_source", "native_loss",
                        "implementation", "source_sha256", "commit", "generator", "selection",
                        "winner", "label_rule", "generated_schema_retries",
                        "generated_schema_valid_fraction", "generation_candidates",
                        "report_dev_used_for_selection", "test_used_for_selection", "internal_split")
                row["audit"] = {key: audit[key] for key in keep if key in audit}
                row["audit_fields_omitted"] = sorted(set(audit) - set(keep))
            entries.append(dict(path=str(path.relative_to(root)),
                source_sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                complete_arms=complete, errors=errors, result=result))
    status = {}
    for name in ("confirmation_status.txt", "control_status.txt", "baseline_status.txt",
                 "native_retry_status.txt", "native_retry_queue.status"):
        path = root / name
        if path.exists():
            status[name] = path.read_text().splitlines()
    audits = {}
    for name in ("stad_sparse_audit.json", "control_numerical_audit.json"):
        path = root / name
        if path.exists():
            try:
                audits[name] = json.loads(path.read_text())
            except ValueError as error:
                issues.append(dict(path=str(path), error=str(error)))
    return dict(captured_at=datetime.now(timezone.utc).isoformat(), root=str(root),
                entries=entries, collection_issues=issues, status=status, audits=audits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = collect(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("preserve existing collection; use a new timestamp")
    args.output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(json.dumps(dict(captured_at=snapshot["captured_at"], entries=len(snapshot["entries"]),
        complete_trajectories=sum(len(e["complete_arms"]) for e in snapshot["entries"]),
        issues=snapshot["collection_issues"] + [e["errors"] for e in snapshot["entries"] if e["errors"]])))


if __name__ == "__main__":
    main()
