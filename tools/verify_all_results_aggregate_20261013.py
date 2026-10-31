"""Independently check the result package against source hashes and XLSX cells."""

import csv
import hashlib
import json
import math
import posixpath
import statistics
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/all_results_aggregate_20261013"
bundle = json.loads((OUT / "bundle.json").read_text(encoding="utf-8"))
manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
errors = []
checks = 0


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        errors.append(message)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)


rows = bundle["all_method_results"]
check(len(rows) == len({r["record_id"] for r in rows}), "Duplicate method record IDs")
check(Counter(r["dataset"] for r in rows) == manifest["dataset_counts"], "Dataset counts")
check(len(rows) == manifest["counts"]["baseline_rows"] + manifest["counts"]["internal_rows"], "Method counts")
panels = {r["panel_id"] for r in bundle["protocol_panels"]}
for row in rows:
    check(row["panel_id"] in panels, f"Missing panel: {row['record_id']}")
    check(bool(row.get("source")), f"Missing source: {row['record_id']}")
    for field in ["final_f1", "accuracy", "old_f1", "new_f1", "stage_average_f1"]:
        if row.get(field) not in (None, ""):
            check(0 <= float(row[field]) <= 100, f"Out-of-range {field}: {row['record_id']}")

pairs = bundle["paired_statistics"]
pair_key = lambda r: tuple(r.get(k) for k in ["category", "dataset", "split", "left", "right", "metric"])
check(len(pairs) == len({pair_key(r) for r in pairs}), "Duplicate paired statistic identities")
missing_ci = 0
for row in pairs:
    check(close(row["positive_ratio"], row["positive_cells"] / row["cells"]), f"Positive ratio: {pair_key(row)}")
    if row.get("ci_low") is None or row.get("ci_high") is None:
        missing_ci += 1
        check(row.get("ci_low") is None and row.get("ci_high") is None, "Partial CI bounds")
    else:
        check(row["ci_low"] <= row["mean_delta"] <= row["ci_high"], f"CI bounds: {pair_key(row)}")

links = bundle["topology_link_prediction"]
link_key = lambda r: tuple(r[k] for k in ["dataset", "order", "split_seed", "stage", "method"])
check(len(links) == len({link_key(r) for r in links}), "Duplicate link-prediction identities")
check({r["split_seed"] for r in links} == {29001, 29002, 29003}, "Link split seeds")
check(any(r["stage"] == 0 for r in links), "Lost stage zero")
for row in bundle["link_prediction_summary"]:
    group = [r for r in links if r["dataset"] == row["dataset"] and r["method"] == row["method"]]
    final = [r for r in group if r["stage"] == max(x["stage"] for x in group)]
    check(len(final) == row["n_splits"] == 3, f"LP summary splits: {row['dataset']}/{row['method']}")
    for metric in ["auc", "ap"]:
        check(close(statistics.mean(r[metric] for r in final), row[metric]), f"LP mean: {metric}")
        check(close(statistics.stdev(r[metric] for r in final), row[f"{metric}_sd"]), f"LP SD: {metric}")

for row in bundle["source_inventory"]:
    source = ROOT / row["path"]
    check(source.is_file(), f"Missing source file: {row['path']}")
    if source.is_file():
        check(source.stat().st_size == int(row["bytes"]), f"Source byte size changed: {row['path']}")
        check(hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"], f"Source hash changed: {row['path']}")

with (ROOT / "results/uq_final_evidence_report_20261013_132047/summary.csv").open(encoding="utf-8-sig", newline="") as source:
    official_summary = list(csv.DictReader(source))
primary = [r for r in rows if r["panel_id"] == "01_sealed"]
for row in primary:
    matches = [s for s in official_summary if s["category"] == "confirmation" and s["dataset"] == row["dataset"] and s["split"] == "sealed_test" and s["label"] == row["method"]]
    check(len(matches) == 1, f"Primary source identity: {row['record_id']}")
    if matches:
        for key, source_key in {"final_f1": "final_f1_mean", "accuracy": "accuracy_mean", "f1_sd": "final_f1_cell_std", "accuracy_sd": "accuracy_cell_std", "memory_rows": "memory_rows", "actual_rate": "actual_rate"}.items():
            check(close(row[key], matches[0][source_key]), f"Primary source value: {row['record_id']}/{key}")

snapshot = ROOT / "results/uq_baseline_expansion_20261013/report_expansion_utc_20261014_021125_595059"
with (snapshot / "matched.csv").open(encoding="utf-8-sig", newline="") as source:
    matched = list(csv.DictReader(source))
with (snapshot / "sources.csv").open(encoding="utf-8-sig", newline="") as source:
    registered = {r["path"]: r for r in csv.DictReader(source)}
receipt = json.loads((ROOT / "results/uq_baseline_expansion_20261013/latest_report_receipt.json").read_text(encoding="utf-8"))
independent = json.loads((snapshot / "independent_verification.json").read_text(encoding="utf-8"))
check(receipt["report"] == snapshot.relative_to(ROOT).as_posix(), "Snapshot receipt path")
check((receipt["completed"], receipt["registered"]) == (57, 59), "Snapshot 57/59 coverage")
check(receipt["independent_verification"] == independent["status"] == "passed", "Independent snapshot verification")
check(len(matched) == independent["matched_rows_verified"] == 114, "Snapshot matched rows")
expansion = [r for r in rows if r["panel_id"] == "00_expansion_sealed"]
qualified = [r for r in rows if r["panel_id"] == "00_qualified_sealed"]
check(len(expansion) == 50 and len(qualified) == 7, "Five-dataset expansion and qualified reference rows")
check(Counter(r["dataset"] for r in expansion) == dict.fromkeys(("arxiv", "sportsfit", "bookhis", "cora", "citeseer"), 10), "Ten configurations per dataset")
check(Counter(r["dataset"] for r in qualified) == {"arxiv": 2, "sportsfit": 2, "bookhis": 3}, "Qualified reference coverage")
sealed_source = {(r["dataset"], r["arm"]): r for r in matched if r["split"] == "sealed_test"}
check(len(sealed_source) == 57, "Unique sealed snapshot identities")
for row in expansion + qualified:
    key = (row["dataset"], row["arm"])
    source_row = sealed_source.get(key)
    check(source_row is not None, f"Expansion source identity: {key}")
    if source_row is None:
        continue
    check(row["underlying_sources"] == source_row["source"], f"Expansion source path: {key}")
    check(row["underlying_sha256"] == source_row["source_sha256"] == registered[source_row["source"]]["source_sha256"], f"Registered source hash: {key}")
    check(registered[source_row["source"]]["status"] == "complete", f"Registered source status: {key}")
    check(source_row["student_source_hash"] == receipt["checks"]["student_source_hash"], f"Student hash: {key}")
    for field, original in {"final_f1": "final_f1", "accuracy": "accuracy", "f1_sd": "f1_student_std", "accuracy_sd": "acc_student_std", "memory_rows": "memory_rows", "actual_rate": "actual_rate"}.items():
        check(close(row[field], source_row[original]), f"Expansion snapshot value: {key}/{field}")

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
sheet_cells = {}
table_counts = {}
with zipfile.ZipFile(OUT / "all_results_aggregate_20261013.xlsx") as xlsx:
    shared = []
    if "xl/sharedStrings.xml" in xlsx.namelist():
        shared = ["".join(t.itertext()) for t in ET.fromstring(xlsx.read("xl/sharedStrings.xml")).findall("s:si", NS)]
    wb = ET.fromstring(xlsx.read("xl/workbook.xml"))
    rels = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(xlsx.read("xl/_rels/workbook.xml.rels")).findall(f"{{{REL_NS}}}Relationship")}
    for sheet in wb.findall("s:sheets/s:sheet", NS):
        target = rels[sheet.attrib[f"{{{NS['r']}}}id"]]
        path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
        doc = ET.fromstring(xlsx.read(path))
        cells = {}
        for cell in doc.findall("s:sheetData/s:row/s:c", NS):
            kind = cell.get("t", "n")
            value = cell.findtext("s:v", default=None, namespaces=NS)
            if kind == "s" and value is not None:
                value = shared[int(value)]
            elif kind == "inlineStr":
                value = "".join(t.text or "" for t in cell.findall("s:is//s:t", NS))
            elif kind == "n" and value is not None:
                value = float(value)
            check(kind != "e", f"Excel error cell: {sheet.get('name')}/{cell.get('r')}")
            cells[cell.get("r")] = (value, kind)
        sheet_cells[sheet.get("name")] = cells
        if sheet.get("name") != "Audit":
            check(doc.find("s:sheetViews/s:sheetView/s:pane", NS) is not None, f"Missing freeze pane: {sheet.get('name')}")
    check(len(sheet_cells) == 23, "Workbook sheet count")
    for name in xlsx.namelist():
        if name.startswith("xl/tables/") and name.endswith(".xml"):
            table = ET.fromstring(xlsx.read(name))
            bounds = table.get("ref").split(":")
            first = int("".join(c for c in bounds[0] if c.isdigit()))
            last = int("".join(c for c in bounds[-1] if c.isdigit()))
            table_counts[table.get("name")] = last - first
            filt = table.find("s:autoFilter", NS)
            check(filt is not None and filt.get("ref") == table.get("ref"), f"Missing table filter: {table.get('name')}")


def column_name(number):
    result = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        result = chr(65 + remainder) + result
    return result


def compare_export(sheet_name, data, fields):
    cells = sheet_cells[sheet_name]
    columns = {value: address[:-1] for address, (value, _) in cells.items() if address.endswith("4") and not address[:-1][-1:].isdigit()}
    for offset, row in enumerate(data, 5):
        for field, label in fields.items():
            expected = row.get(field)
            actual, kind = cells.get(f"{columns[label]}{offset}", (None, "n"))
            if expected in (None, ""):
                check(actual is None, f"Missing value not blank: {sheet_name}/{field}/{offset}")
            elif isinstance(expected, (int, float)):
                check(kind == "n" and actual is not None and close(expected, actual), f"Numeric export: {sheet_name}/{field}/{offset}")
            else:
                check(expected == actual, f"Text export: {sheet_name}/{field}/{offset}")


method_fields = {"record_id": "ID", "panel_id": "Panel", "dataset_display": "Dataset", "method": "Method", "final_f1": "F1 (%)", "accuracy": "Acc (%)", "f1_sd": "F1 SD", "accuracy_sd": "Acc SD", "source": "Source", "sd_unit": "SD unit", "fidelity": "Identity"}
compare_export("All Methods", rows, method_fields)
compare_export("Paired Statistics", pairs, {"dataset_display": "Dataset", "left_method": "Left method", "right_method": "Right method", "mean_delta": "Mean delta (pp)", "ci_low": "95% CI lower", "ci_high": "95% CI upper", "positive_ratio": "Positive ratio"})
for dataset in manifest["dataset_counts"]:
    display = next(r["dataset_display"] for r in rows if r["dataset"] == dataset)
    data = [r for r in rows if r["dataset"] == dataset]
    check(table_counts[f"Results{display}"] == len(data), f"Dataset table rows: {display}")
    ids = {sheet_cells[display].get(f"A{i}", (None,))[0] for i in range(5, len(data) + 5)}
    check(ids == {r["record_id"] for r in data}, f"Dataset row identities: {display}")
check(table_counts["AllMethodResults"] == len(rows), "All Methods table count")
check(table_counts["PairedStatistics"] == len(pairs), "Paired table count")
check(table_counts["SourceInventory"] == len(bundle["source_inventory"]), "Source table count")
check(table_counts["QDiagnostics"] == len(bundle["q_diagnostics"]), "Q diagnostic table count")
check(table_counts["TopologyLinkPrediction"] == len(links), "LP table count")

summary_cells = sheet_cells["Summary"]
check(summary_cells.get("A6", (None,))[0] == "Dataset", "Summary expansion header")
check(table_counts["ExpansionSealed"] == 50 and table_counts["QualifiedReferences"] == 7 and table_counts["PrimarySealed"] == 9, "Summary table row counts")
for start, group, label in ((7, expansion, "Expansion"), (61, qualified, "Qualified"), (72, primary, "Nine-cell")):
    for offset, row in enumerate(group):
        index = start + offset
        check(summary_cells.get(f"A{index}", (None,))[0] == row["dataset_display"], f"{label} Summary dataset: {index}")
        check(summary_cells.get(f"B{index}", (None,))[0] == row["method"], f"{label} Summary method: {index}")
        check(summary_cells.get(f"J{index}", (None,))[0] == row["record_id"], f"{label} Summary ID: {index}")
        for column, field in (("C", "final_f1"), ("D", "accuracy"), ("E", "f1_sd"), ("F", "accuracy_sd")):
            actual = summary_cells.get(f"{column}{index}", (None,))[0]
            expected = row[field]
            check(actual is not None and close(actual, expected), f"{label} Summary {field}: {index}")
check("Five datasets x ten configurations" in str(summary_cells.get("A4", (None,))[0]), "Summary evidence label")
check("Nine-cell sealed confirmation" in str(summary_cells.get("A69", (None,))[0]), "Summary stability label")
check("Full-history reference" in str(summary_cells.get("A82", (None,))[0]), "Summary full-history label")

verification = {"status": "passed" if not errors else "failed", "checks": checks, "method_rows": len(rows), "expansion_rows": len(expansion), "qualified_reference_rows": len(qualified), "registered_source_hashes_crosschecked": len(expansion) + len(qualified), "source_files_hash_verified": len(bundle["source_inventory"]), "paired_rows": len(pairs), "paired_rows_without_ci": missing_ci, "link_stage_rows": len(links), "sheets": list(sheet_cells), "table_row_counts": table_counts, "errors": errors}
(OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(verification, ensure_ascii=True))
raise SystemExit(bool(errors))
