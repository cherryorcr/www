"""Summarize paired constructions, with students nested inside each memory."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

from scipy.stats import t

SEEDS = [271006, 271108, 271109]
ARMS = ["uq_dm_edgefree", "gecc_hard", "cat_cctp"]
NAMES = {"uq_dm_edgefree": "CoTAG-Q", "gecc_hard": "GECC-hard adaptation",
         "cat_cctp": "CaT adaptation"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired(left, right):
    assert sorted(left) == sorted(right) == SEEDS
    differences = [left[s] - right[s] for s in SEEDS]
    mean = stats.mean(differences)
    half = float(t.ppf(.975, 2)) * stats.stdev(differences) / math.sqrt(3)
    return dict(n_pairs=3, difference_pp=mean, ci95_low=mean-half, ci95_high=mean+half,
                positive_pairs=sum(d > 0 for d in differences),
                positive_pair_ratio=sum(d > 0 for d in differences)/3,
                worst_difference_pp=min(differences),
                differences=json.dumps(dict(zip(SEEDS, differences))))


def csv_new(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def aggregate(rows, keys, measures):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        assert sorted(r["construction_seed"] for r in items) == SEEDS
        result = dict(zip(keys, key), constructions=3)
        for metric in measures:
            values = [r[metric] for r in items]
            result[metric] = stats.mean(values)
            result[metric + "_construction_sd"] = stats.stdev(values)
        out.append(result)
    return out


def make_pairs(rows):
    grouped = defaultdict(dict)
    keys = ("dataset", "stage", "mode", "split", "cohort")
    for row in rows:
        grouped[tuple(row[k] for k in keys)].setdefault(row["arm"], {})[row["construction_seed"]] = row
    output = []
    for key, arms in sorted(grouped.items()):
        assert sorted(arms) == sorted(ARMS)
        for baseline in ("gecc_hard", "cat_cctp"):
            for metric in ("f1", "accuracy"):
                left = {s: r[metric] for s, r in arms["uq_dm_edgefree"].items()}
                right = {s: r[metric] for s, r in arms[baseline].items()}
                output.append(dict(zip(keys, key), main="uq_dm_edgefree", baseline=baseline,
                                   metric=metric, **paired(left, right)))
    return output


def main(root, output):
    verification = read(root / "analysis/verification_v1.json")
    assert verification["passed"] and len(verification["runs"]) == 18
    rows, students, resources, source_receipts = [], [], [], []
    protocols = {}
    for path in sorted((root / "runs").rglob("results.json")):
        result = read(path)
        p = result["protocol"]
        assert result["status"] == "complete" and len(result["rows"]) == 5
        identity = dict(dataset=p["dataset"], arm=p["arm"], construction_seed=p["construction_seed"])
        normalized = dict(p)
        for field in ("arm", "construction_seed"):
            normalized.pop(field)
        normalized["q_config"] = {k: v for k, v in p["q_config"].items() if k != "seed"}
        if p["dataset"] in protocols:
            assert protocols[p["dataset"]] == normalized, "Cross-arm protocol mismatch"
        protocols[p["dataset"]] = normalized
        source_receipts.append(dict(path=path.relative_to(root).as_posix(),
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for stage in result["rows"]:
            for mode, splits in stage["scores"]["prediction"].items():
                for split, scores in splits.items():
                    for cohort in ("cumulative", "current", "historical"):
                        if mode == "current_snapshot" and cohort != "current":
                            continue
                        if scores[cohort] is None:
                            continue
                        basic = dict(identity, stage=str(stage["stage"]), mode=mode, split=split, cohort=cohort)
                        per_student = [r[cohort] for r in scores["per_student"]]
                        row = dict(basic, nodes=per_student[0]["nodes"])
                        for key in ("f1", "accuracy"):
                            row[key] = stats.mean(r[key] for r in per_student)
                        rows.append(row)
                        for individual in scores["per_student"]:
                            students.append(dict(basic, student_seed=individual["seed"], **individual[cohort]))
        stages = result["rows"]
        training = [r for stage in stages for r in stage["scores"]["training_resources"]]
        resource = dict(identity,
            construction_seconds=sum(s["construction"]["seconds"] for s in stages),
            construction_peak_gib=max(s["construction"]["peak_mb"] for s in stages)/1024,
            construction_reserved_peak_gib=max(s["construction"]["reserved_peak_mb"] for s in stages)/1024,
            student_seconds_per_five_stage_sequence=sum(r["train_seconds"] for r in training)/3,
            student_training_peak_gib=max(r["train_peak_mb"] for r in training)/1024,
            student_training_reserved_peak_gib=max(r["train_reserved_peak_mb"] for r in training)/1024,
            memory_rows=stages[-1]["construction"]["memory_rows"],
            final_memory_archive_mib=(path.parent / "s4/memory.pt.gz").stat().st_size/2**20)
        for mode in ("arrived_snapshot", "current_snapshot"):
            values = [r["inference"][mode] for r in training]
            resource[mode + "_inference_seconds_per_sequence"] = sum(v["seconds"] for v in values)/3
            resource[mode + "_inference_peak_gib"] = max(v["peak_mb"] for v in values)/1024
        resources.append(resource)
    stage_mean = []
    groups = defaultdict(list)
    keys = ("dataset", "arm", "construction_seed", "mode", "split", "cohort")
    for row in rows:
        if row["cohort"] != "historical":
            groups[tuple(row[k] for k in keys)].append(row)
    for key, items in sorted(groups.items()):
        assert sorted(r["stage"] for r in items) == list("01234")
        stage_mean.append(dict(zip(keys, key), stage="five_stage_mean",
                               f1=stats.mean(r["f1"] for r in items),
                               accuracy=stats.mean(r["accuracy"] for r in items)))
    summary = aggregate(rows + stage_mean, ("dataset", "arm", "stage", "mode", "split", "cohort"), ("f1", "accuracy"))
    pairs = make_pairs(rows + stage_mean)
    same_query = []
    lookups = {(r["dataset"], r["arm"], r["construction_seed"], r["stage"], r["split"], r["mode"]): r
               for r in rows + stage_mean if r["cohort"] == "current"}
    for key, current in sorted(lookups.items()):
        if key[-1] != "current_snapshot":
            continue
        arrived = lookups[key[:-1] + ("arrived_snapshot",)]
        same_query.append(dict(zip(("dataset", "arm", "construction_seed", "stage", "split"), key[:-1]),
                               f1=arrived["f1"]-current["f1"], accuracy=arrived["accuracy"]-current["accuracy"]))
    same_query_summary = aggregate(same_query, ("dataset", "arm", "stage", "split"), ("f1", "accuracy"))
    resource_summary = aggregate(resources, ("dataset", "arm"),
                                 [k for k in resources[0] if k not in ("dataset", "arm", "construction_seed")])
    output.mkdir(parents=True, exist_ok=False)
    for name, table in (("per_student", students), ("per_construction", rows + stage_mean),
                        ("summary", summary), ("paired_comparisons", pairs), ("resources", resources),
                        ("resource_summary", resource_summary), ("same_current_queries", same_query_summary)):
        csv_new(output / (name + ".csv"), table)
    json_new(output / "summary.json", dict(summary=summary, pairs=pairs, resources=resource_summary,
                                            same_current_queries=same_query_summary))
    json_new(output / "sources.json", dict(files=source_receipts,
        summary_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        uncertainty="Student average within memory, sample SD and unadjusted paired t95% CI over 3 construction seeds. Same fixed chronology and split; not 9 independent constructions.",
        stage_mean="Equal mean of five stage metrics within each construction, then average constructions; not pooled node-level F1.",
        resource_scope="PyTorch allocated/reserved peaks include the resident frozen encoder, exclude CPU archives and untracked CUDA allocations. No matched full-history/static resource control."))
    text = ["# Chronological citation subsets", "", "Three constructions; three students averaged within each memory.",
            "Sample SD across constructions. Test and report-dev remain separate.", ""]
    for split in ("test", "report_dev"):
        text += [f"## {split}", "", "| Dataset | Method | Final cumulative F1 / Acc | Five-stage cumulative F1 | Final current-only F1 / Acc |", "|---|---|---:|---:|---:|"]
        for dataset in ("acm", "dblp"):
            for arm in ARMS:
                def get(stage, mode, cohort):
                    return next(r for r in summary if (r["dataset"], r["arm"], r["split"], r["stage"], r["mode"], r["cohort"]) == (dataset, arm, split, stage, mode, cohort))
                final, average, current = get("4", "arrived_snapshot", "cumulative"), get("five_stage_mean", "arrived_snapshot", "cumulative"), get("4", "current_snapshot", "current")
                def fmt(r, metric):
                    return f"{r[metric]:.2f} +/- {r[metric + '_construction_sd']:.2f}"
                text.append(f"| {dataset.upper()} | {NAMES[arm]} | {fmt(final, 'f1')} / {fmt(final, 'accuracy')} | {fmt(average, 'f1')} | {fmt(current, 'f1')} / {fmt(current, 'accuracy')} |")
        text += ["", "| Dataset | CoTAG-Q minus | Evaluation | F1 difference (pp) | Paired 95% CI | Positive pairs | Worst pair (pp) |", "|---|---|---|---:|---:|---:|---:|"]
        for r in pairs:
            if r["split"] != split or r["stage"] not in ("4", "five_stage_mean") or r["metric"] != "f1":
                continue
            if (r["mode"], r["cohort"]) not in (("arrived_snapshot", "cumulative"), ("current_snapshot", "current")):
                continue
            text.append(f"| {r['dataset'].upper()} | {NAMES[r['baseline']]} | {r['stage']}, {r['mode']} | {r['difference_pp']:+.2f} | [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}] | {r['positive_pairs']}/3 | {r['worst_difference_pp']:+.2f} |")
        text.append("")
    (output / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(output), constructions=len(resources), stage_metric_rows=len(rows),
                          all_runs_complete=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
