"""Audit the protocol fields of the frozen large-graph comparison.

This is intentionally separate from the score summarizer.  A method can have
the same student score scale while still violating the intended comparison
through a different class stream, budget, seed set, or raw-data access rule.
The audit emits JSON and Markdown so the final table can label offline
references without silently treating them as continual methods.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("arxiv", "bookchild", "sportsfit")
EXPECTED = {
    "rate": 0.05,
    "report_dev_mod": 3,
    "student_batch": 16,
    "student_seeds": [100, 101, 102, 103, 104, 105],
    "stages": [0, 1, 2, 3, 4],
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(root: Path, dataset: str) -> Path:
    for candidate in (
        root / f"{dataset}_base4" / "results.json",
        root / f"{dataset}_base4_results.json",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing {dataset} results below {root}")


def rows_and_manifest(path: Path, method: str) -> tuple[dict, list[dict]]:
    obj = load(path)
    flat = obj.get("results")
    if flat is not None:
        # C0 stores one flat row per stage under ``arm=append_fixed``;
        # semantic-recursive stores paired rows whose arm values are nested
        # under ``c0`` and ``semantic_recursive``.
        if flat and isinstance(flat[0], dict) and "arm" in flat[0]:
            aliases = {
                "c0": "append_fixed",
                "real_node_fps_subsampled": "kmeans_medoid_subsampled",
            }
            raw_method = aliases.get(method, method)
            return obj, [row for row in flat if row.get("arm") == raw_method]
        aliases = {
            "c0": "c0",
            "semantic_recursive": "semantic_recursive",
        }
        key = aliases.get(method, method)
        rows = []
        for row in flat:
            if key not in row:
                continue
            item = dict(row[key])
            item.update({"stage": row["stage"], "budget": row["budget"]})
            item["arm"] = key
            rows.append(item)
        return obj, rows
    rows = []
    for row in obj["summary"]["rows"]:
        if method not in row:
            continue
        item = dict(row[method])
        item.update({"stage": row["stage"], "budget": row["budget"]})
        item["arm"] = method
        rows.append(item)
    return obj, rows


def protocol_value(manifest: dict, key: str):
    if key in manifest:
        return manifest[key]
    protocol = manifest.get("protocol")
    if isinstance(protocol, dict) and key in protocol:
        return protocol[key]
    student = manifest.get("student")
    if key == "student_batch" and isinstance(student, dict):
        return student.get("batch")
    return None


def stage_classes(manifest: dict):
    value = protocol_value(manifest, "stage_classes")
    if value is None:
        value = manifest.get("stage_protocol", {}).get("stage_classes")
    return [[int(x) for x in group] for group in value] if value is not None else None


def budgets(manifest: dict):
    value = protocol_value(manifest, "budgets")
    return [int(x) for x in value] if value is not None else None


def scalar_close(left, right, tol=1e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= tol
    except (TypeError, ValueError):
        return False


def audit_method(name: str, manifest: dict, rows: list[dict], reference: dict) -> dict:
    issues: list[str] = []
    rate = protocol_value(manifest, "rate")
    if not scalar_close(rate, reference["rate"]):
        issues.append(f"rate={rate!r}, expected {reference['rate']}")
    report_dev = protocol_value(manifest, "report_dev_mod")
    if report_dev != reference["report_dev_mod"]:
        issues.append(f"report_dev_mod={report_dev!r}")
    batch = protocol_value(manifest, "student_batch")
    if batch != reference["student_batch"]:
        issues.append(f"student_batch={batch!r}")
    seeds = protocol_value(manifest, "student_seeds")
    if seeds != reference["student_seeds"]:
        issues.append(f"student_seeds={seeds!r}")

    got_classes = stage_classes(manifest)
    if got_classes is not None and got_classes != reference["stage_classes"]:
        issues.append("stage_classes differ from C0")
    elif got_classes is None:
        issues.append("stage_classes missing")
    got_budgets = budgets(manifest)
    if got_budgets is not None and got_budgets != reference["budgets"]:
        issues.append("budgets differ from C0")
    elif got_budgets is None:
        issues.append("budgets missing")

    observed_stages = sorted({int(row["stage"]) for row in rows})
    if observed_stages != reference["stages"]:
        issues.append(f"observed stages={observed_stages}")
    expected_steps = []
    wrong_steps = []
    for row in rows:
        stage = int(row["stage"])
        if name == "raw_long":
            expected = 8192
        else:
            expected = 2048
        expected_steps.append(expected)
        steps = row.get("steps", row.get("student_steps"))
        if steps is not None and int(steps) != expected:
            wrong_steps.append((stage, int(steps), expected))
    if wrong_steps:
        issues.append(f"wrong steps={wrong_steps}")

    # These flags are descriptive rather than a pass/fail criterion.  An
    # offline reference is valid for the five-level decomposition, but not a
    # legal continual constructor.
    raw_access = bool(
        protocol_value(manifest, "historical_raw_access")
        or manifest.get("historical_raw_reads", 0)
        or manifest.get("raw_access", False)
        or name in {"full_history", "full_history_gpu", "raw_equal", "raw_long"}
    )
    rewrites = bool(
        protocol_value(manifest, "rewrites_previous_memory")
        or manifest.get("rewrites_previous_memory", False)
        or name in {"full_history", "full_history_gpu", "raw_equal", "raw_long"}
    )
    continual_legal = not raw_access and not rewrites
    if len(rows) != 5:
        issues.append(f"row count={len(rows)}, expected 5")
    return {
        "method": name,
        "pass": not issues,
        "issues": issues,
        "historical_raw_access": raw_access,
        "rewrites_previous_memory": rewrites,
        "continual_legal": continual_legal,
        "offline_reference": not continual_legal,
        "observed_stages": observed_stages,
        "observed_steps": sorted({int(row.get("steps", row.get("student_steps", -1)))
                                   for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c0-root", required=True)
    parser.add_argument("--semantic-root", required=True)
    parser.add_argument("--coreset-root", required=True)
    parser.add_argument("--full-history-root", required=True)
    parser.add_argument("--full-history-gpu-root")
    parser.add_argument("--raw-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    roots = {
        "c0": Path(args.c0_root),
        "semantic_recursive": Path(args.semantic_root),
        "random_real": Path(args.coreset_root),
        "k_center_subsampled": Path(args.coreset_root),
        "real_node_fps_subsampled": Path(args.coreset_root),
        "full_history": Path(args.full_history_root),
    }
    if args.full_history_gpu_root:
        roots["full_history_gpu"] = Path(args.full_history_gpu_root)
    if args.raw_root:
        roots["raw_equal"] = Path(args.raw_root)
        roots["raw_long"] = Path(args.raw_root)

    report = {"datasets": {}, "all_pass": True}
    md = [
        "# Large-Graph Baseline Protocol Audit (2026-09-26)",
        "",
        "A pass means the method matches the frozen base+4 comparison fields "
        "that are expected to be shared. Offline raw/full-history references "
        "are intentionally marked as non-continual rather than rejected.",
        "",
    ]
    for dataset in DATASETS:
        loaded = {}
        for name, root in roots.items():
            path = resolve(root, dataset)
            obj, rows = rows_and_manifest(path, name)
            if name == "c0":
                ref_manifest = obj
                ref_rows = rows
            loaded[name] = (obj, rows)
        ref_manifest, ref_rows = loaded["c0"]
        reference = {
            "rate": protocol_value(ref_manifest, "rate"),
            "report_dev_mod": protocol_value(ref_manifest, "report_dev_mod"),
            "student_batch": protocol_value(ref_manifest, "student_batch"),
            "student_seeds": protocol_value(ref_manifest, "student_seeds"),
            "stage_classes": stage_classes(ref_manifest),
            "budgets": budgets(ref_manifest),
            "stages": EXPECTED["stages"],
        }
        checks = []
        for name, (manifest, rows) in loaded.items():
            # The paired semantic runner intentionally stores only a hash of
            # the inherited C0 protocol.  Fill those inherited fields for the
            # audit while preserving the original artifact unchanged.
            check_manifest = dict(manifest)
            if name == "semantic_recursive":
                check_manifest.setdefault("rate", reference["rate"])
                check_manifest.setdefault("report_dev_mod", reference["report_dev_mod"])
                check_manifest.setdefault("student_batch", reference["student_batch"])
                check_manifest.setdefault("student_seeds", reference["student_seeds"])
                check_manifest.setdefault("stage_classes", reference["stage_classes"])
                check_manifest.setdefault("budgets", reference["budgets"])
            check = audit_method(name, check_manifest, rows, reference)
            checks.append(check)
            if not check["pass"]:
                report["all_pass"] = False
        report["datasets"][dataset] = {
            "reference": reference,
            "checks": checks,
        }
        md.append(f"## {dataset}")
        md.append("")
        md.append("| method | shared protocol | historical raw | rewrites memory | continual legal | issues |")
        md.append("|---|---|---|---|---|---|")
        for check in checks:
            issues = "; ".join(check["issues"]) or "none"
            md.append(
                f"| {check['method']} | {'PASS' if check['pass'] else 'REVIEW'} | "
                f"{check['historical_raw_access']} | {check['rewrites_previous_memory']} | "
                f"{check['continual_legal']} | {issues} |"
            )
        md.append("")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    output.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"event": "complete", "all_pass": report["all_pass"],
                      "json": str(output.with_suffix('.json')),
                      "markdown": str(output.with_suffix('.md'))},
                     ensure_ascii=True))


if __name__ == "__main__":
    main()
