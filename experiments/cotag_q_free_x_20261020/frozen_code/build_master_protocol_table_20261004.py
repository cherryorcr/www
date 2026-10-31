from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(row: dict, *names: str):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def add_row(rows: list[dict], **values):
    rows.append({
        "protocol_id": values.get("protocol_id", ""),
        "protocol_scope": values.get("protocol_scope", ""),
        "classification": values.get("classification", ""),
        "main_table_status": values.get("main_table_status", ""),
        "directly_comparable_with_current_xsp": values.get(
            "directly_comparable_with_current_xsp", "no"),
        "method": values.get("method", ""),
        "dataset": values.get("dataset", ""),
        "order": values.get("order", ""),
        "stage": values.get("stage", ""),
        "macro_f1": metric(values, "macro_f1", "final_f1"),
        "accuracy": metric(values, "accuracy", "final_accuracy"),
        "old_f1": values.get("old_f1"),
        "new_f1": values.get("new_f1"),
        "worst_stage_f1": values.get("worst_stage_f1"),
        "memory_rows": values.get("memory_rows"),
        "student_seconds": values.get("student_seconds"),
        "construction_seconds": values.get("construction_seconds"),
        "student_seeds": values.get("student_seeds", ""),
        "construction_seed": values.get("construction_seed", ""),
        "historical_raw_reads": values.get("historical_raw_reads", ""),
        "source": values.get("source", ""),
        "warning": values.get("warning", ""),
    })


def build_rows(external_root: Path, gpu0: Path, gpu1: Path,
               variant_summary: Path) -> list[dict]:
    rows: list[dict] = []
    external = load_json(external_root / "docs" /
                         "external_baseline_summary_20261002.json")

    strict_protocol = (
        "Cora/Citeseer; 3-stage; 5% cumulative; current delta + parent; "
        "Student seeds 150-155")
    for item in external["comparison_table"]:
        method = item["method"]
        status = ("analysis-only (heuristic; excluded from primary ranking)"
                  if method == "semantic-recursive" else "strict external")
        add_row(
            rows,
            protocol_id="strict_small_3stage_seed150_155",
            protocol_scope=strict_protocol,
            classification=item.get("classification", "mechanism-adapted"),
            main_table_status=status,
            method=method,
            dataset=item["dataset"],
            order=item["order"],
            stage=item["stage"],
            macro_f1=item.get("macro_f1"),
            accuracy=item.get("accuracy"),
            old_f1=item.get("old_f1"),
            new_f1=item.get("new_f1"),
            worst_stage_f1=item.get("worst_stage_f1"),
            memory_rows=item.get("memory_rows"),
            student_seconds=item.get("student_seconds"),
            construction_seconds=item.get("construction_seconds"),
            student_seeds=",".join(map(str, item.get("student_seeds", []))),
            construction_seed=(44 if method == "PUMA-inspired" else 270927),
            historical_raw_reads=0,
            source=item.get("source", "external summary"),
            warning=("semantic-recursive retained for historical context only"
                     if method == "semantic-recursive" else ""),
        )

    for item in external["adapted_reference_table"]:
        add_row(
            rows,
            protocol_id="small_adapted_reference_seed100_105",
            protocol_scope=("Cora/Citeseer; order unspecified; adapted/reference; "
                            "Student seeds 100-105"),
            classification="adapted/reference",
            main_table_status="supplementary only",
            method=item["method"],
            dataset=item["dataset"],
            order=item["order"],
            stage=item["stage"],
            macro_f1=item.get("macro_f1"),
            accuracy=item.get("accuracy"),
            old_f1=item.get("old_f1"),
            new_f1=item.get("new_f1"),
            worst_stage_f1=item.get("worst_stage_f1"),
            memory_rows=item.get("memory_rows"),
            student_seconds=item.get("student_seconds"),
            construction_seconds=item.get("construction_seconds"),
            student_seeds=item.get("student_seeds", ""),
            historical_raw_reads="recorded, not identical to strict table",
            source=item.get("source", "external adapted summary"),
            warning=item.get("metadata_warning", "seed/order mismatch"),
        )

    method_names = {
        "random_real": "Random real-node coreset",
        "k_center": "K-center",
        "medoid": "Medoid",
        "cat_mmd_tag_cctp_top5": "CaT + CCTP top-5",
        "gecc_hard_tag": "GECC-hard-TAG",
        "cat_mmd_tag_class_assigned_prompt_margin":
            "Class-shared prompt-margin (diagnostic)",
        "cat_mmd_tag_x_slot_partition": "XSP",
        "cat_mmd_tag_x_slot_partition_st": "XSP + ST-Sinkhorn",
    }
    large_files = [gpu0 / "arxiv_base4_results.json",
                   gpu0 / "bookchild_base4_results.json",
                   gpu1 / "sportsfit_base4_results.json"]
    for path in large_files:
        result = load_json(path)
        protocol = result["protocol"]
        for item in result["rows"]:
            if int(item["stage"]) != 4:
                continue
            arm = item["arm"]
            if arm == "cat_mmd_tag_class_assigned_prompt_margin":
                status = "diagnostic excluded from fair primary ranking"
                warning = "class-shared one payload per class plus prompt margin"
            elif arm in {"cat_mmd_tag_x_slot_partition",
                         "cat_mmd_tag_x_slot_partition_st"}:
                status = "candidate main method" if arm.endswith("partition") \
                    else "candidate variant"
                warning = ""
            else:
                status = "large adapted baseline"
                warning = "mechanism-adapted, not official native reproduction"
            score = item["score"]
            audit = item.get("construction_audit", {})
            add_row(
                rows,
                protocol_id="large_base4_seed100_105_314159",
                protocol_scope=("Arxiv/BookChild/SportsFit; base+4; 5% cumulative; "
                                "current delta only; Student seeds 100-105"),
                classification=("candidate" if arm.startswith("cat_mmd_tag_x_slot_partition")
                                else "mechanism-adapted"),
                main_table_status=status,
                directly_comparable_with_current_xsp=("yes" if arm !=
                    "cat_mmd_tag_class_assigned_prompt_margin" else "no"),
                method=method_names.get(arm, arm),
                dataset=result["dataset"],
                order="frozen base+4 order",
                stage=item["stage"],
                final_f1=score.get("final_f1"),
                final_accuracy=score.get("accuracy"),
                old_f1=score.get("old_f1"),
                new_f1=score.get("new_f1"),
                worst_stage_f1=score.get("worst_stage_f1"),
                memory_rows=item.get("memory_rows"),
                student_seconds=score.get("student_seconds"),
                construction_seconds=item.get("construction_seconds"),
                student_seeds=",".join(map(str, protocol["student_seeds"])),
                construction_seed=protocol.get("construction_seed", ""),
                historical_raw_reads=audit.get("historical_raw_reads", 0),
                source=str(path),
                warning=warning,
            )

    variants = load_json(variant_summary)
    for item in variants.get("variant_pooled", []):
        add_row(
            rows,
            protocol_id="large_variant_sweep_pooled_seed100_105",
            protocol_scope=("three-large pooled variant sweep; base+4; 5%; "
                            "construction seed 270927; Student seeds 100-105"),
            classification="variant diagnostic",
            main_table_status="supplementary variant sweep; not primary winner",
            directly_comparable_with_current_xsp="same family, different frozen run",
            method=item.get("method", item.get("arm", "")),
            dataset="Arxiv+BookChild+SportsFit pooled",
            order="pooled",
            stage="final pooled",
            final_f1=item.get("final_f1"),
            final_accuracy=item.get("final_accuracy"),
            old_f1=item.get("old_f1"),
            new_f1=item.get("new_f1"),
            worst_stage_f1=item.get("worst_stage_f1"),
            student_seconds=item.get("student_seconds"),
            construction_seconds=item.get("construction_seconds"),
            student_seeds="100,101,102,103,104,105",
            construction_seed=270927,
            historical_raw_reads=0,
            source=str(variant_summary),
            warning="earlier variant sweep; do not override latest frozen baseline",
        )
    return rows


def fmt(value):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def aggregate(rows: list[dict], protocol_id: str, include_status=None):
    selected = [r for r in rows if r["protocol_id"] == protocol_id]
    if include_status is not None:
        selected = [r for r in selected if include_status(r)]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        groups[row["method"]].append(row)
    out = []
    for method, values in sorted(groups.items()):
        record = {"method": method, "n": len(values)}
        for key in ("macro_f1", "accuracy", "old_f1", "new_f1",
                    "worst_stage_f1", "student_seconds",
                    "construction_seconds"):
            nums = [float(v[key]) for v in values
                    if v.get(key) not in (None, "")]
            record[key] = sum(nums) / len(nums) if nums else None
        out.append(record)
    return out


def write_outputs(rows: list[dict], outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with (outdir / "master_protocol_table_20261004.csv").open(
            "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    strict = aggregate(rows, "strict_small_3stage_seed150_155",
                       lambda r: "analysis-only" not in r["main_table_status"])
    large = aggregate(rows, "large_base4_seed100_105_314159",
                      lambda r: "diagnostic" not in r["main_table_status"])
    variants = aggregate(rows, "large_variant_sweep_pooled_seed100_105")
    lines = [
        "# Master protocol-aware comparison table (2026-10-04)", "",
        "This table keeps protocol-mismatched results in one inventory while "
        "preventing invalid cross-protocol ranking.", "",
        "## Protocol consistency", "",
        "| Dimension | External strict table | Current XSP large table | Same? |",
        "|---|---|---|---|",
        "| Dataset | Cora/Citeseer | Arxiv/BookChild/SportsFit | No |",
        "| Stage protocol | 3-stage, final stage 2 | base+4, final stage 4 | No |",
        "| Student seeds | 150-155 | 100-105 | No |",
        "| Memory rate | 5% cumulative | 5% cumulative | Yes |",
        "| Access | current delta + parent, no historical raw | current delta only, no historical raw | Yes |",
        "| Student family | canonical 2-layer GCN + frozen MiniLM + InfoNCE | same family | Yes |",
        "| Construction seed | 270927 (PUMA 44) | 314159 | No |",
        "",
        "Therefore the strict small-graph rows and large-graph XSP rows are "
        "reported together for auditability, but are not one direct leaderboard.",
        "",
        "## Aggregated strict external table", "",
        "| Method | Cells | Final F1 | Accuracy | Old F1 | New F1 | Worst-stage F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in strict:
        lines.append("|{method}|{n}|{macro_f1}|{accuracy}|{old_f1}|{new_f1}|{worst_stage_f1}|".format(
            method=row["method"], n=row["n"], macro_f1=fmt(row["macro_f1"]),
            accuracy=fmt(row["accuracy"]), old_f1=fmt(row["old_f1"]),
            new_f1=fmt(row["new_f1"]), worst_stage_f1=fmt(row["worst_stage_f1"])))
    lines += ["", "## Aggregated current large frozen table", "",
              "| Method | Cells | Final F1 | Accuracy | Old F1 | New F1 | Worst-stage F1 | Construction s |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in large:
        lines.append("|{method}|{n}|{macro_f1}|{accuracy}|{old_f1}|{new_f1}|{worst_stage_f1}|{construction_seconds}|".format(
            method=row["method"], n=row["n"], macro_f1=fmt(row["macro_f1"]),
            accuracy=fmt(row["accuracy"]), old_f1=fmt(row["old_f1"]),
            new_f1=fmt(row["new_f1"]), worst_stage_f1=fmt(row["worst_stage_f1"]),
            construction_seconds=fmt(row["construction_seconds"])))
    lines += ["", "## Variant sweep pooled table", "",
              "| Variant | Final F1 | Accuracy | Old F1 | New F1 | Construction s |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in variants:
        lines.append("|{method}|{macro_f1}|{accuracy}|{old_f1}|{new_f1}|{construction_seconds}|".format(
            method=row["method"], macro_f1=fmt(row["macro_f1"]),
            accuracy=fmt(row["accuracy"]), old_f1=fmt(row["old_f1"]),
            new_f1=fmt(row["new_f1"]), construction_seconds=fmt(row["construction_seconds"])))
    lines += ["", "## Full row-level inventory", "",
              "| Protocol | Status | Method | Dataset | Order | Stage | F1 | Acc | Old | New | Worst | Seeds | Warning |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in rows:
        lines.append("|{protocol_id}|{status}|{method}|{dataset}|{order}|{stage}|{f1}|{acc}|{old}|{new}|{worst}|{seeds}|{warning}|".format(
            protocol_id=row["protocol_id"], status=row["main_table_status"],
            method=row["method"], dataset=row["dataset"], order=row["order"],
            stage=row["stage"], f1=fmt(row["macro_f1"]), acc=fmt(row["accuracy"]),
            old=fmt(row["old_f1"]), new=fmt(row["new_f1"]),
            worst=fmt(row["worst_stage_f1"]), seeds=row["student_seeds"],
            warning=row["warning"]))
    (outdir / "master_protocol_table_20261004.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    metadata = {
        "row_count": len(rows),
        "strict_aggregate": strict,
        "large_aggregate": large,
        "variant_aggregate": variants,
        "recommendation": {
            "primary": "XSP",
            "optional_variant": "XSP + ST-Sinkhorn as ablation/candidate, not promoted",
            "reason": "latest frozen large comparison has XSP mean Final F1 21.60 versus XSP-ST 20.64; XSP wins Arxiv and BookChild but loses SportsFit to medoid",
            "claim_boundary": "not universal SOTA; moderate average gain under a large-graph base+4 protocol",
        },
    }
    (outdir / "master_protocol_table_20261004.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path(r"D:\Aprojects\tag_baseline\EXTERNAL_BASELINE_HANDOFF_PACKAGE_20261004\EXTERNAL_BASELINE_HANDOFF_PACKAGE_20261004"),
    )
    parser.add_argument(
        "--gpu0",
        type=Path,
        default=Path("results/public_baselines_frozen_20261004b_gpu0"),
    )
    parser.add_argument(
        "--gpu1",
        type=Path,
        default=Path("results/public_baselines_frozen_20261004b_gpu1"),
    )
    parser.add_argument(
        "--variant-summary",
        type=Path,
        default=Path("results/x_slot_partition_variants_summary_20261004/summary.json"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/master_protocol_table_20261004"),
    )
    args = parser.parse_args()
    rows = build_rows(args.external_root, args.gpu0, args.gpu1,
                      args.variant_summary)
    write_outputs(rows, args.outdir)
    print(json.dumps({"rows": len(rows), "outdir": str(args.outdir)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
