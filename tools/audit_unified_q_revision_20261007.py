"""Validate frozen UQ protocols and recorded metrics without modifying runs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def audit_results(root: Path, source_root: Path | None = None) -> dict:
    errors, warnings, diagnostics, cells = [], [], [], []
    shared_sources, shared_student = None, None
    splits, schedules, orders, identities = {}, {}, {}, set()
    complete_arms = stages = 0

    def check(ok, where, message):
        if not ok:
            errors.append({"location": where, "message": message})

    def near(actual, expected, where, message, tolerance=1e-6):
        check(actual is not None and math.isfinite(float(actual)) and
              abs(float(actual) - float(expected)) <= tolerance, where, message)

    for path in sorted(root.rglob("results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        protocol = result["protocol"]
        where = str(path.relative_to(root))
        key = tuple(protocol[k] for k in ("dataset", "budget_mode", "order_seed", "construction_seed"))
        check(key not in identities, where, "duplicate experiment identity")
        identities.add(key)
        separate = path.with_name("protocol.json")
        check(separate.exists() and json.loads(separate.read_text(encoding="utf-8")) == protocol,
              where, "protocol.json differs from embedded protocol")
        hashes = protocol["source_hashes"]
        shared_sources = hashes if shared_sources is None else shared_sources
        check(hashes == shared_sources, where, "source hashes changed between cells")
        if source_root is not None:
            for name, digest in hashes.items():
                source = source_root / name
                check(source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == digest,
                      where, f"local source does not match executed source: {name}")
        student = [protocol[k] for k in ("student", "student_steps", "student_batch", "student_seeds")]
        shared_student = student if shared_student is None else shared_student
        check(student == shared_student, where, "Student configuration changed")
        split = [protocol[k] for k in ("pool_mask_hash", "report_mask_hash")]
        check(split == splits.setdefault(key[0], split), where, "pool/report split changed")
        ordering = [protocol[k] for k in ("groups", "stage_id_hashes")]
        check(ordering == orders.setdefault((key[0], key[2]), ordering), where, "same order has different stages")
        if key[1] == "fixed_schedule":
            check(protocol["budgets"] == schedules.setdefault(key[0], protocol["budgets"]),
                  where, "fixed budget schedule changed")
        check(protocol["historical_raw_reads_by_constructor"] == 0, where, "old raw access declared")
        check(protocol["boundary_enabled"] is False, where, "boundary unexpectedly enabled")
        check(protocol["evaluation_split"] == "report-dev only, no test selection", where, "evaluation split changed")
        by_arm = defaultdict(list)
        raw_by_stage = {}
        for row in result["rows"]:
            by_arm[row["arm"]].append(row)
            stages += 1
            location = f"{where}:{row['arm']}:{row['stage']}"
            stage = row["stage"]
            check(0 <= stage < len(protocol["groups"]), location, "invalid stage")
            if not 0 <= stage < len(protocol["groups"]):
                continue
            budget = protocol["budgets"][stage]
            previous = 0 if stage == 0 else protocol["budgets"][stage - 1]
            check(row["memory_rows"] == budget and row["new_rows"] == budget - previous,
                  location, "recorded memory budget mismatch")
            check(row["raw_rows"] == raw_by_stage.setdefault(stage, row["raw_rows"]),
                  location, "methods used different delta sizes")
            for name in ("budget_ok", "prefix_ok", "finite_ok"):
                check(row["checks"].get(name) is True, location, f"runtime check failed or missing: {name}")
            score = row["score"]
            replicas = score["per_student"]
            check([s["seed"] for s in replicas] == protocol["student_seeds"], location, "Student seed mismatch")
            for name in ("student_steps", "student_batch", "student_seeds"):
                check(score[name] == protocol[name], location, f"score {name} mismatch")
            for name in ("final_f1", "accuracy", "new_f1", "old_f1"):
                values = [s[name] for s in replicas]
                if all(v is None for v in values):
                    check(score[name] is None, location, f"{name} should be null")
                else:
                    check(all(v is not None for v in values), location, f"missing {name} replicate")
                    if all(v is not None for v in values):
                        near(score[name], statistics.mean(values), location, f"{name} aggregation error")
            for name, sd in (("final_f1", "final_f1_std"), ("accuracy", "accuracy_std")):
                values = [s[name] for s in replicas]
                near(score[sd], statistics.stdev(values) if len(values) > 1 else 0,
                     location, f"{sd} aggregation error")
            old_count = sum(map(len, protocol["groups"][:stage]))
            new_count = len(protocol["groups"][stage])
            recomposed = ((score["old_f1"] or 0) * old_count + score["new_f1"] * new_count) / (old_count + new_count)
            near(score["final_f1"], recomposed, location, "old/new class F1 decomposition mismatch")
            if row["arm"] == "cat_cctp":
                continue
            record = row["audit"]
            near(row["checks"].get("propagation_max_error"), 0, location, "propagation mismatch")
            near(record.get("text_hard_forward_max_error"), 0, location, "text hard-forward mismatch", 1e-4)
            row_error = record.get("q_row_sum_max_error")
            check(row_error is not None and math.isfinite(row_error), location, "nonfinite Sinkhorn residual")
            # The frozen protocol fixes 30 iterations, not a convergence tolerance.
            # Report this numerical diagnostic separately from artifact corruption.
            if row_error is not None and math.isfinite(row_error) and row_error > 1e-3:
                warnings.append({"location": location, "message": "Sinkhorn row-mass residual exceeds 1e-3",
                                 "observed": row_error, "diagnostic_tolerance": 1e-3})
            check(record["historical_raw_reads"] == 0 and not record["uses_boundary_sketch"],
                  location, "constructor access/boundary violation")
            check(record["x_residual_ratio_max"] <= record["config"]["trust_radius"] + 1e-6,
                  location, "residual trust radius exceeded")
            if record["loss_weighting"] == "group_class":
                expected = 1 if stage == 0 else .5
                for name in ("real_new_loss_mass", "synthetic_new_loss_mass"):
                    near(record[name], expected, location, f"{name} mismatch")
            expected_steps = record["config"]["steps"]
            counts = (expected_steps, expected_steps)
            if row["arm"] == "uq_alternating":
                counts = ((expected_steps + 1) // 2, expected_steps // 2)
            check((record["q_update_count"], record["residual_update_count"]) == counts,
                  location, "optimizer update count mismatch")
            trace = record["trace"]
            for name in ("q_gradient_norm", "residual_gradient_norm"):
                check(all(math.isfinite(t[name]) for t in trace) and any(t[name] > 0 for t in trace),
                      location, f"nonfinite or inactive gradient path: {name}")
            check(all(math.isfinite(t["loss"]) for t in trace), location, "nonfinite objective")
            diagnostics.append({"dataset": key[0], "budget_mode": key[1], "order_seed": key[2],
                "construction_seed": key[3], "arm": row["arm"], "stage": stage,
                "initial_logged_loss": trace[0]["loss"], "last_logged_loss": trace[-1]["loss"],
                "q_row_error": record["q_row_sum_max_error"], "q_entropy": record["q_entropy"],
                "unique_text_ratio": record["unique_text_ratio"],
                "residual_ratio_max": record["x_residual_ratio_max"],
                "last_q_gradient_norm": trace[-1]["q_gradient_norm"],
                "last_residual_gradient_norm": trace[-1]["residual_gradient_norm"]})
        for arm, rows in by_arm.items():
            observed = sorted(r["stage"] for r in rows)
            check(observed == list(range(len(rows))), where, f"missing/duplicate stage for {arm}")
            complete_arms += int(observed == list(range(len(protocol["groups"]))))
            if result["status"] == "complete":
                check(len(rows) == len(protocol["groups"]), where, f"complete cell missing stages: {arm}")
        expected_arms = {"cat_cctp", *protocol["arms"]}
        if result["status"] == "complete":
            check(set(by_arm) == expected_arms, where, "complete cell missing planned methods")
        if key[1] == "fixed_rate":
            cumulative = previous = 0
            for stage, raw in sorted(raw_by_stage.items()):
                cumulative += raw
                budget = max(round(protocol["memory_rate_target"] * cumulative), previous + len(protocol["groups"][stage]))
                check(protocol["budgets"][stage] == budget, where, "fixed-rate budget mismatch")
                previous = budget
        cells.append({"cell": str(path.parent.relative_to(root)), "status": result["status"],
                      "observed_stage_rows": len(result["rows"])})
    queue = {"planned_cells": None, "finished_cells": None, "complete": None}
    plan_path, exits_path = root / "confirmation_plan.json", root / "confirmation_exit_codes.json"
    if plan_path.exists():
        tasks = json.loads(plan_path.read_text(encoding="utf-8"))["tasks"]
        queue["planned_cells"] = len(tasks)
        queue["complete"] = False
        if exits_path.exists():
            exits = json.loads(exits_path.read_text(encoding="utf-8"))
            queue["finished_cells"] = len(exits)
            expected_names = {f"{d}_{mode}_order{o}_seed{s}" for d, o, s, mode, _ in tasks}
            check(len(exits) == len(tasks) and {e["cell"] for e in exits} == expected_names,
                  "confirmation queue", "missing or duplicate queue completion records")
            check(all(e["exit_code"] == 0 for e in exits), "confirmation queue", "nonzero experiment exit code")
            complete_names = {c["cell"] for c in cells if c["status"] == "complete"}
            queue["complete"] = (expected_names <= complete_names and
                                 all(e["exit_code"] == 0 for e in exits) and len(exits) == len(tasks))
            check(queue["complete"], "confirmation queue", "queue finished without complete expected artifacts")
    return {"integrity_passed": not errors, "strict_transport_tolerance_met": not warnings,
            "errors": errors, "numerical_warnings": warnings, "complete_method_cells": complete_arms,
            "observed_stage_rows": stages, "cells": cells, "diagnostics": diagnostics, "confirmation_queue": queue,
            "limitations": ["Metric aggregation is checked; original predictions are not re-evaluated.",
                "Access checks combine recorded declarations and constructor source review, not an OS sandbox proof.",
                "Report-dev results do not establish held-out test or unseen-class generalization."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    report = audit_results(args.root, args.source_root)
    (args.root / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    brief = {k: v for k, v in report.items() if k not in ("diagnostics", "cells", "numerical_warnings")}
    brief["numerical_warning_count"] = len(report["numerical_warnings"])
    brief["maximum_row_mass_residual"] = max((d["q_row_error"] for d in report["diagnostics"]), default=0)
    print(json.dumps(brief, indent=2))
    if not report["integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
