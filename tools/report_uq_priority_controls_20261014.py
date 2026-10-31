"""Summarize the priority controls without changing frozen primary tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path("/home/pc/research/TAG_condensation")
OUT = ROOT / "results/uq_priority_controls_20261014"
DATASETS = ("arxiv", "sportsfit", "bookhis")
SEEDS = (271006, 271108, 271109)


def primary(dataset: str, seed: int) -> dict:
    path = ROOT / "results/uq_final_evidence_20261012/confirmation" \
        / f"{dataset}_o9101_c{seed}" / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in payload["rows"] if r["arm"] == "uq_dm_edgefree"]
    return sorted(rows, key=lambda r: r["stage"])[-1]["scores"]["sealed_test"]


def ci(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    # t_{.975,2}=4.303; use the exact small-sample critical value for n=3.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(values), 1.96)
    m = mean(values)
    half = critical * stdev(values) / (len(values) ** 0.5)
    return m - half, m + half


def fmt(v, digits=2):
    return "--" if v is None else f"{v:.{digits}f}"


def rows_from_payload():
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = OUT / "payload" / f"{dataset}_o9101_c{seed}.json"
            if not path.exists():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))["payload"]
            controls = {x["mode"]: x for x in value["controls"]}
            primary_scores = controls["primary"]["scores"]["sealed_test"]
            for mode, item in controls.items():
                scores = item["scores"]["sealed_test"]
                rows.append(dict(dataset=dataset, seed=seed, family="payload_storage",
                                 arm=mode, f1=scores["final_f1"], accuracy=scores["accuracy"],
                                 delta_f1=scores["final_f1"] - primary_scores["final_f1"],
                                 delta_accuracy=scores["accuracy"] - primary_scores["accuracy"],
                                 primary_hash_match=value.get("primary_hash_match"),
                                 x_unchanged=item.get("x_unchanged", True),
                                 text_unique_ratio=item.get("text_unique_ratio")))
    return rows


def rows_from_graph():
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            path = OUT / "no_graph" / f"{dataset}_o9101_c{seed}.json"
            if not path.exists():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))["no_graph"]
            scores = value["rows"][-1]["scores"]["sealed_test"]
            base = primary(dataset, seed)
            rows.append(dict(dataset=dataset, seed=seed, family="constructor_graph",
                             arm="no_graph_message", f1=scores["final_f1"],
                             accuracy=scores["accuracy"], delta_f1=scores["final_f1"] - base["final_f1"],
                             delta_accuracy=scores["accuracy"] - base["accuracy"],
                             primary_hash_match=False, x_unchanged=False,
                             text_unique_ratio=None))
    return rows


def write_csv(path: Path, rows: list[dict]):
    fields = ["dataset", "seed", "family", "arm", "f1", "accuracy", "delta_f1",
              "delta_accuracy", "primary_hash_match", "x_unchanged", "text_unique_ratio"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in fields} for row in rows)


def main():
    rows = rows_from_payload() + rows_from_graph()
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "priority_controls_cells.csv", rows)
    lines = ["# Unified-Q-EF priority controls", "",
             "All results use the current-delta-only, frozen-prefix, cumulative exact-5% protocol and the common Prompt Student. The frozen primary tables are not overwritten.", "",
             "## Payload storage-only controls", "",
             "These controls keep the reconstructed/archived primary X and labels fixed and replace only stored text. ArXiv uses archived memory; SportsFit/BookHis use a source replay when an archive was not retained, and the hash match is reported explicitly.", ""]
    for dataset in DATASETS:
        selected = [r for r in rows if r["dataset"] == dataset and r["family"] == "payload_storage"]
        lines += [f"### {dataset}", "", "| Control | n | F1 mean +/- SD | Acc mean +/- SD | Delta F1 vs same-run primary | Positive cells | X unchanged |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for arm in ("primary", "uq_class_shared_payload", "uq_random5_payload"):
            values = [r for r in selected if r["arm"] == arm]
            if not values:
                continue
            fs = [r["f1"] for r in values]
            ac = [r["accuracy"] for r in values]
            ds = [r["delta_f1"] for r in values] if arm != "primary" else []
            lines.append(f"| {arm} | {len(values)} | {fmt(mean(fs))} +/- {fmt(stdev(fs) if len(fs)>1 else None)} | {fmt(mean(ac))} +/- {fmt(stdev(ac) if len(ac)>1 else None)} | {fmt(mean(ds) if ds else None)} | {(sum(v>0 for v in ds) if ds else '--')}/{len(ds) if ds else '--'} | {all(r['x_unchanged'] for r in values)} |")
    lines += ["", "## No-graph-message constructor control", "", "This control removes current-delta edges only during construction. Downstream evaluation still uses the common full raw transductive graph.", ""]
    for dataset in DATASETS:
        selected = [r for r in rows if r["dataset"] == dataset and r["family"] == "constructor_graph"]
        if not selected:
            lines += [f"### {dataset}", "", "pending", ""]
            continue
        ds = [r["delta_f1"] for r in selected]
        da = [r["delta_accuracy"] for r in selected]
        low, high = ci(ds)
        lines += [f"### {dataset}", "", f"- completed cells: {len(selected)}/{len(SEEDS)}", f"- no-graph F1: {fmt(mean([r['f1'] for r in selected]))} +/- {fmt(stdev([r['f1'] for r in selected]) if len(selected)>1 else None)}", f"- paired F1 delta vs frozen primary: {fmt(mean(ds))} pp, CI {('['+fmt(low)+', '+fmt(high)+']') if low is not None else '--'}, positive cells {sum(v>0 for v in ds)}/{len(ds)}, worst {fmt(min(ds))} pp", f"- paired Accuracy delta: {fmt(mean(da))} pp", ""]
    lines += ["## Interpretation limits", "", "The payload controls test stored-text materialization, not a new construction-time text feedback path. No-graph-message is a mechanism ablation, not a new main method. A positive or negative single cell is not sufficient; use the completed paired cells and retain all replay/hash disclosures.", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(rows=len(rows), output=str(OUT / "summary.md"))))


if __name__ == "__main__":
    main()
