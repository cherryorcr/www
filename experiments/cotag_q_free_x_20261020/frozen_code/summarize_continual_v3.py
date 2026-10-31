"""Summarize unified v3 continual result directories into a Markdown table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(root: Path) -> list[dict]:
    rows = []
    for result_file in sorted(root.glob("**/results.json")):
        data = json.loads(result_file.read_text(encoding="utf-8"))
        dataset = data.get("dataset", result_file.parent.name)
        protocol = data.get("protocol", data.get("protocol_name", "unknown"))
        grouped = {}
        for item in data.get("results", []):
            arm = item.get("arm", item.get("arm_stage", "unknown"))
            stage = item.get("stage")
            if stage is None and "_s" in arm:
                try:
                    stage = int(arm.rsplit("_s", 1)[1])
                except ValueError:
                    stage = None
            if stage is None:
                continue
            grouped.setdefault(arm, {})[int(stage)] = item
        for arm, stages in sorted(grouped.items()):
            final = stages[max(stages)]
            rows.append({
                "dataset": dataset,
                "arm": arm,
                "protocol": protocol,
                "s0": stages.get(0, {}).get("cum_dev_f1_mean"),
                "s1": stages.get(1, {}).get("cum_dev_f1_mean"),
                "s2": stages.get(2, {}).get("cum_dev_f1_mean"),
                "old2": stages.get(2, {}).get("old_mean_f1_mean"),
                "cur2": stages.get(2, {}).get("current_f1_mean"),
                "rows": final.get("roots"),
                "seeds": final.get("student_seeds"),
                "source": str(result_file),
            })
    return rows


def fmt(value):
    return "-" if value is None else f"{float(value):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = read_rows(Path(args.root))
    lines = [
        "# Unified v3 Continual Main Table",
        "",
        "Values are dev Macro-F1 (%), evaluated with the manifest's student seeds.",
        "",
        "| dataset | arm | s0 cum | s1 cum | s2 cum | s2 old | s2 current | final rows |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {dataset} | {arm} | {s0} | {s1} | {s2} | {old2} | {cur2} | {rows} |".format(
            dataset=row["dataset"], arm=row["arm"], s0=fmt(row["s0"]),
            s1=fmt(row["s1"]), s2=fmt(row["s2"]), old2=fmt(row["old2"]),
            cur2=fmt(row["cur2"]), rows=row["rows"] or "-"))
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
