"""Independently verify saved predictions and summarize fixed-feature pairs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cotag_temporal_bibliographic_20261022"))
import verify_results as V
import summarize as S

sys.path.insert(0, str(HERE))
from controls import assert_numeric_equal, assert_frozen_prefix, assert_single_source, text_statistics


def load_memory(path):
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def inputs(project, dataset, protocol):
    if dataset == "arxiv":
        source = Path(protocol["source"]["path"])
        assert V.sha(source) == protocol["source"]["sha256"]
        raw = torch.load(source, map_location="cpu", weights_only=False)
        names = list(map(str, raw["all_labels"]))
        labels = torch.tensor([names.index(str(v)) for v in raw["label_list"]])
        ids = raw["original_node_ids"].long()
        arrival = torch.bucketize(raw["node_year"].flatten().long(), torch.tensor([2015, 2016, 2017, 2018, 2020]), right=False)
        texts = list(raw["text_list"])
    else:
        folder = project / "results/cotag_temporal_bibliographic_20261022/data" / dataset
        encoding = V.read(folder / "encoding.json")
        assert V.sha(folder / "encoded.pt") == encoding["encoded_data_sha256"]
        raw = torch.load(folder / "encoded.pt", map_location="cpu", weights_only=False)
        labels, ids, arrival, texts, names = raw["y"], raw["original_ids"], raw["arrival"], raw["texts"], raw["names"]
    masks = dict(test=ids % 10 == 2, report_dev=ids % 10 == 3)
    pool = ~torch.isin(ids % 10, torch.tensor([1, 2, 3]))
    stages = [torch.where(pool & (arrival == s))[0] for s in range(5)]
    assert [V.tensor_hash(v) for v in stages] == protocol["stage_id_hashes"]
    assert V.tensor_hash(pool) == protocol["pool_mask_hash"]
    assert V.tensor_hash(masks["test"]) == protocol["test_mask_hash"]
    assert V.tensor_hash(masks["report_dev"]) == protocol["report_mask_hash"]
    return dict(labels=labels, arrival=arrival, texts=texts, classes=len(names), masks=masks, stages=stages)


def check_metrics(scores, predictions, data, stage):
    checks = 0
    assert sorted(predictions) == [150, 151, 152]
    assert set(scores["prediction"]) == {"arrived_snapshot", "current_snapshot"}
    assert all(set(value) == set(scores["prediction"]) for value in predictions.values())
    arrival = data["arrival"]
    for mode, splits in scores["prediction"].items():
        assert set(splits) == {"test", "report_dev"}
        ids = torch.where(arrival <= stage if mode == "arrived_snapshot" else arrival == stage)[0]
        for seed in predictions:
            assert torch.equal(predictions[seed][mode]["ids"], ids)
            prediction = predictions[seed][mode]["pred"]
            assert len(prediction) == len(ids)
            assert bool(((prediction >= 0) & (prediction < data["classes"])).all())
        for split, aggregate in splits.items():
            for cohort, cohort_mask in (("cumulative", arrival <= stage), ("current", arrival == stage), ("historical", arrival < stage)):
                local = (cohort_mask & data["masks"][split])[ids]
                values = []
                assert [r["seed"] for r in aggregate["per_student"]] == [150, 151, 152]
                for row in aggregate["per_student"]:
                    pred = predictions[row["seed"]][mode]["pred"]
                    metric = V.metrics(data["labels"][ids][local].numpy(), pred[local].numpy(), data["classes"])
                    if metric is None:
                        assert row[cohort] is None
                    else:
                        assert row[cohort]["nodes"] == metric["nodes"]
                        for key in ("f1", "accuracy"):
                            V.near(row[cohort][key], metric[key])
                            checks += 1
                        values.append(metric)
                if not values:
                    assert aggregate[cohort] is None
                else:
                    assert aggregate[cohort]["nodes"] == int(local.sum())
                    for key in ("f1", "accuracy"):
                        V.near(aggregate[cohort][key], statistics.mean(v[key] for v in values))
                        V.near(aggregate[cohort][key + "_sd"], statistics.stdev(v[key] for v in values))
            if mode == "current_snapshot":
                assert aggregate["current"] == aggregate["cumulative"] and aggregate["historical"] is None
    return checks


def verify_run(path, data):
    result = V.read(path)
    p = result["protocol"]
    assert result["status"] == "complete" and [r["stage"] for r in result["rows"]] == list(range(5))
    assert V.read(path.with_name("protocol.json")) == p
    assert p["posthoc_sensitivity"] and not p["clustering_rerun"] and not p["method_selection_from_test"]
    assert p["source_count"] == 1 and p["text_chars"] == 320
    assert p["student_seeds"] == [150, 151, 152] and p["student_steps"] == 2048 and p["student_batch"] == 24
    for field, name in (("runner_sha256", "run_single.py"), ("controls_sha256", "controls.py"), ("registration_sha256", "REGISTRATION.md")):
        assert p[field] == V.sha(HERE / name)
    for field, source_path in (
        ("evaluator_sha256", HERE.parent / "cotag_temporal_bibliographic_20261022/run_temporal.py"),
        ("runtime_sha256", HERE.parent / "cotag_temporal_bibliographic_20261022/runtime.py"),
        ("arxiv_loader_sha256", HERE.parent / "cotag_fidelity_temporal_20261021/run_temporal.py"),
    ):
        assert p[field] == V.sha(source_path)
    source = Path(p["source_result_path"])
    assert V.sha(source) == p["source_result_sha256"]
    original = V.read(source)
    assert original["protocol"] == p["source_protocol"]
    for key in ("budgets", "stage_id_hashes", "pool_mask_hash", "report_mask_hash", "test_mask_hash"):
        assert original["protocol"][key] == p[key]
    parent, metric_checks = None, 0
    diagnostics = []
    for row in result["rows"]:
        stage = row["stage"]
        folder = path.parent / f"s{stage}"
        construction, scores = row["construction"], row["scores"]
        assert V.read(folder / "construction.json") == construction
        assert V.read(folder / "scores.json") == scores
        memory = load_memory(folder / "memory.pt.gz")
        old = load_memory(source.parent / f"s{stage}/memory.pt.gz")
        assert V.memory_hash(old) == construction["source_memory_hash"] == original["rows"][stage]["construction"]["memory_hash"]
        assert V.memory_hash(memory) == construction["memory_hash"] == scores["memory_hash"]
        assert_numeric_equal(memory, old)
        assert_frozen_prefix(memory, parent)
        assert len(memory["x"]) == p["budgets"][stage]
        assert memory["edge_index"].shape == (2, 0)
        for key in ("x", "text_emb", "mass"):
            assert bool(torch.isfinite(memory[key]).all())
        keep = 0 if parent is None else len(parent["x"])
        audit = construction["audit"]
        assert audit["original_nearest5_text_exact"] and audit["numeric_memory_exact"]
        assert len(audit["source_ids"]) == len(memory["x"])-keep
        assert_single_source(audit["source_ids"], audit["reference_source_ids"], data["stages"][stage].tolist())
        allowed = set(data["stages"][stage].tolist())
        for j, (single, five) in enumerate(zip(audit["source_ids"], audit["reference_source_ids"])):
            assert set(five) <= allowed and len(set(five)) == len(five) <= 5
            assert all(int(data["labels"][index]) == int(memory["labels"][keep+j]) for index in five)
            assert memory["texts"][keep+j] == str(data["texts"][single[0]])[:320]
            chars = max(1, (320-(len(five)-1))//len(five))
            assert old["texts"][keep+j] == " ".join(str(data["texts"][i])[:chars] for i in five)[:320]
        assert text_statistics(memory["texts"][keep:]) == audit["new_text_statistics"]
        assert text_statistics(old["texts"][keep:]) == audit["reference_text_statistics"]
        assert text_statistics(memory["texts"]) == audit["all_memory_text_statistics"]
        assert sum(a != b for a, b in zip(memory["texts"], old["texts"])) == audit["changed_cumulative_rows"]
        assert V.sha(folder / "predictions.pt.gz") == scores["prediction_archive_sha256"]
        metric_checks += check_metrics(scores, load_memory(folder / "predictions.pt.gz"), data, stage)
        diagnostics.append(dict(dataset=p["dataset"], construction_seed=p["construction_seed"], stage=stage,
            **audit["new_text_statistics"], reference_mean_chars=audit["reference_text_statistics"]["mean_chars"],
            reference_unique_ratio=audit["reference_text_statistics"]["unique_text_ratio"],
            changed_new_rows=audit["changed_new_rows"], attachment_seconds=construction["attachment_seconds"],
            propagation_seconds=construction["propagation_seconds"],
            validation_seconds=construction["validation_seconds"]))
        parent = memory
    return dict(dataset=p["dataset"], seed=p["construction_seed"], passed=True, stages=5,
                independently_recomputed_metric_scalars=metric_checks,
                numeric_tensors_identical=True, own_prefix_frozen=True, text_sources_legal=True), diagnostics


def references(project, dataset, seed):
    if dataset == "arxiv":
        root = project / "results/cotag_fidelity_temporal_20261021"
        return {"gecc_nearest5": root / f"temporal/gecc_hard_c{seed}/results.json",
                "cat_cctp": root / f"temporal/cat_cctp_c{seed}/results.json",
                "uq_dm_edgefree": root / f"precision_repair_v2/temporal/uq_dm_edgefree_c{seed}/results.json"}
    root = project / "results/cotag_temporal_bibliographic_20261022/runs" / dataset
    return {variant: root / f"{arm}_c{seed}/results.json" for variant, arm in
            (("gecc_nearest5", "gecc_hard"), ("cat_cctp", "cat_cctp"), ("uq_dm_edgefree", "uq_dm_edgefree"))}


def collect(value, dataset, variant):
    assert value["status"] == "complete"
    output = []
    for stage in value["rows"]:
        for mode, splits in stage["scores"]["prediction"].items():
            for raw_split, score in splits.items():
                split = "test" if raw_split == "historical_test" else raw_split
                assert [r["seed"] for r in score["per_student"]] == [150, 151, 152]
                for cohort in ("cumulative", "current", "historical"):
                    if score[cohort] is None or (mode == "current_snapshot" and cohort != "current"):
                        continue
                    output.append(dict(dataset=dataset, variant=variant, stage=str(stage["stage"]), mode=mode,
                        split=split, source_split=raw_split, cohort=cohort, construction_seed=value["protocol"]["construction_seed"],
                        f1=statistics.mean(r[cohort]["f1"] for r in score["per_student"]),
                        accuracy=statistics.mean(r[cohort]["accuracy"] for r in score["per_student"])))
    return output


def stage_averages(rows):
    groups = defaultdict(list)
    keys = ("dataset", "variant", "construction_seed", "mode", "split", "cohort")
    for row in rows:
        if row["cohort"] != "historical":
            groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, values in groups.items():
        assert sorted(r["stage"] for r in values) == list("01234")
        output.append(dict(zip(keys, key), stage="five_stage_mean",
                           f1=statistics.mean(r["f1"] for r in values),
                           accuracy=statistics.mean(r["accuracy"] for r in values)))
    return output


def main(root, output):
    project = root.parent.parent
    expected = {root / "runs" / d / f"gecc_nearest1_c{s}/results.json"
                for d in ("acm", "dblp", "arxiv") for s in S.SEEDS}
    assert set((root / "runs").rglob("results.json")) == expected
    output.mkdir(parents=True, exist_ok=False)
    verification, diagnostics, rows, receipts, resources = [], [], [], [], []
    data_cache = {}
    for path in sorted(expected):
        value = V.read(path)
        p = value["protocol"]
        dataset, seed = p["dataset"], p["construction_seed"]
        if dataset not in data_cache:
            data_cache[dataset] = inputs(project, dataset, p)
        verified, diag = verify_run(path, data_cache[dataset])
        verification.append(verified)
        diagnostics.extend(diag)
        rows.extend(collect(value, dataset, "gecc_nearest1"))
        receipts.append(dict(path=str(path), sha256=V.sha(path), variant="gecc_nearest1"))
        for row in value["rows"]:
            cost = row["construction"]
            for student in row["scores"]["training_resources"]:
                resource = dict(dataset=dataset, construction_seed=seed, stage=row["stage"],
                    student_seed=student["seed"], train_seconds=student["train_seconds"],
                    train_peak_gib=student["train_peak_mb"]/1024,
                    train_reserved_peak_gib=student["train_reserved_peak_mb"]/1024,
                    propagation_seconds=cost["propagation_seconds"],
                    validation_seconds=cost["validation_seconds"],
                    attachment_seconds=cost["attachment_seconds"],
                    adaptation_validation_peak_gib=cost["peak_mb"]/1024)
                for mode, inference in student["inference"].items():
                    resource[mode+"_seconds"] = inference["seconds"]
                    resource[mode+"_peak_gib"] = inference["peak_mb"]/1024
                resources.append(resource)
        for variant, source_path in references(project, dataset, seed).items():
            source = V.read(source_path)
            for key in ("budgets", "stage_id_hashes", "pool_mask_hash", "report_mask_hash", "test_mask_hash",
                        "student_seeds", "student_steps", "student_batch", "memory_rate", "source", "source_hashes"):
                assert source["protocol"][key] == p[key]
            if "student" in source["protocol"]:
                assert source["protocol"]["student"] == p["student"]
            rows.extend(collect(source, dataset, variant))
            receipts.append(dict(path=str(source_path), sha256=V.sha(source_path), variant=variant))
        print(json.dumps(dict(event="verified", **verified)), flush=True)
    rows += stage_averages(rows)
    summary = S.aggregate(rows, ("dataset", "variant", "stage", "mode", "split", "cohort"), ("f1", "accuracy"))
    paired = []
    grouped = defaultdict(dict)
    pair_keys = ("dataset", "stage", "mode", "split", "cohort")
    for row in rows:
        grouped[tuple(row[k] for k in pair_keys)].setdefault(row["variant"], {})[row["construction_seed"]] = row
    for key, variants in sorted(grouped.items()):
        for left, right in (("gecc_nearest1", "gecc_nearest5"), ("uq_dm_edgefree", "gecc_nearest1"), ("uq_dm_edgefree", "gecc_nearest5")):
            for metric in ("f1", "accuracy"):
                paired.append(dict(zip(pair_keys, key), left=left, right=right, metric=metric,
                    **S.paired({s: r[metric] for s, r in variants[left].items()}, {s: r[metric] for s, r in variants[right].items()})))
    for filename, table in (("per_construction", rows), ("summary", summary), ("paired_comparisons", paired), ("text_diagnostics", diagnostics), ("resource_measurements", resources)):
        S.csv_new(output / f"{filename}.csv", table)
    S.json_new(output / "verification.json", dict(passed=True, results=verification,
        verifier_sha256=V.sha(Path(__file__)), scope="All new memories, exact frozen numeric tensors/prefixes, source texts/classes/masks and raw prediction metrics; original reference scores reused without rerunning students."))
    S.json_new(output / "summary.json", dict(summary=summary, pairs=paired))
    S.json_new(output / "sources.json", dict(files=receipts,
        scope="Post-hoc text-policy sensitivity; 3 matched construction IDs with 3 students averaged within memory, not 9 independent constructions. Unadjusted paired t95% intervals, df2. No original result replaced."))
    lines = ["# GECC single-source text sensitivity", "", "Three constructions, three students averaged per memory; sample SD over constructions.",
             "Post-hoc sensitivity on previously inspected masks. Original nearest-5 remains unchanged.", ""]
    for split in ("test", "report_dev"):
        lines += [f"## {split}", "", "| Dataset | Method | Final cumulative F1 / Acc | Five-stage cumulative F1 | Final current-only F1 | Five-stage current-only F1 |", "|---|---|---:|---:|---:|---:|"]
        for dataset in ("arxiv", "acm", "dblp"):
            for variant in ("cat_cctp", "gecc_nearest5", "gecc_nearest1", "uq_dm_edgefree"):
                def get(stage, mode, cohort):
                    return next(r for r in summary if (r["dataset"],r["variant"],r["split"],r["stage"],r["mode"],r["cohort"]) == (dataset,variant,split,stage,mode,cohort))
                def fmt(row, key="f1"):
                    return f"{row[key]:.2f} +/- {row[key+'_construction_sd']:.2f}"
                final = get("4", "arrived_snapshot", "cumulative")
                lines.append(f"| {dataset} | {variant} | {fmt(final)} / {fmt(final,'accuracy')} | {fmt(get('five_stage_mean','arrived_snapshot','cumulative'))} | {fmt(get('4','current_snapshot','current'))} | {fmt(get('five_stage_mean','current_snapshot','current'))} |")
        lines += ["", "| Dataset | Stage | Snapshot | nearest-1 minus nearest-5 F1 (pp) | Paired 95% CI | Positive pairs |", "|---|---|---|---:|---:|---:|"]
        for p in paired:
            if p["split"] != split or p["left"] != "gecc_nearest1" or p["metric"] != "f1" or p["stage"] not in ("4","five_stage_mean"):
                continue
            if (p["mode"],p["cohort"]) not in (("arrived_snapshot","cumulative"),("current_snapshot","current")):
                continue
            lines.append(f"| {p['dataset']} | {p['stage']} | {p['mode']} | {p['difference_pp']:+.2f} | [{p['ci95_low']:+.2f}, {p['ci95_high']:+.2f}] | {p['positive_pairs']}/3 |")
        lines.append("")
    with (output / "summary.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")
    print(json.dumps(dict(output=str(output), passed=True, runs=len(verification))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    main(args.root, args.output)
