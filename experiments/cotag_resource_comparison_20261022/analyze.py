"""Audit raw memories and predictions; summarize matched construction replicates."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "previous_resource_analysis", HERE.parent / "cotag_memory_optimization_20261022/analyze.py")
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)
from protocol_utils import payload_bytes

SEEDS = (271006, 271108, 271109)
ARMS = ("incremental_q", "full_history", "cat_cctp", "gecc_hard")
METRICS = ("f1", "accuracy", "peak_gib", "reserved_gib", "seconds", "cpu_rss_gib",
           "payload_mib", "student_peak_gib", "student_seconds", "inference_peak_gib")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_rows(dataset, left, right, runs):
    a = {r["construction_seed"]: r for r in runs if r["dataset"] == dataset and r["arm"] == left}
    b = {r["construction_seed"]: r for r in runs if r["dataset"] == dataset and r["arm"] == right}
    matched = sorted(a.keys() & b.keys())
    if not matched:
        return []
    rows = []
    for metric in METRICS:
        values = [a[s][metric] - b[s][metric] for s in matched]
        rows.append(dict(dataset=dataset, left=left, right=right, metric=metric,
                         **V.stats(values)))
    for metric in ("peak_gib", "seconds", "student_peak_gib"):
        values = [100 * (1 - a[s][metric] / b[s][metric]) for s in matched]
        rows.append(dict(dataset=dataset, left=left, right=right,
                         metric=metric + "_reduction_pct", **V.stats(values)))
    return rows


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for dataset in ("arxiv", "sportsfit"):
        for seed in SEEDS:
            path = args.reused / f"{dataset}_combined_c{seed}/results.json"
            result = read(path)
            assert result["status"] == "complete"
            records[(dataset, "incremental_q", seed)] = (path, result)
    for path in sorted(args.input.glob("*/results.json")):
        result = read(path)
        p = result["protocol"]
        b = p["base"]
        key = (b["dataset"], p["arm"], b["construction_seed"])
        assert key not in records
        assert key[1] in ARMS[1:]
        records[key] = (path, result)

    receipt = dict(memories=0, prefixes=0, predictions=0, independent_metric_checks=0,
                   protocol_matches=0, cohort_matches=0, statuses=[], source_results=[])
    stages, evaluations, runs, failures = [], [], [], []
    cohorts = {}
    for (dataset, arm, seed), (path, result) in sorted(records.items()):
        reused = arm == "incremental_q"
        base = result["protocol"] if reused else result["protocol"]["base"]
        source_path, source = records[(dataset, "incremental_q", seed)]
        assert base == source["protocol"]
        assert base["dataset"] == dataset and base["construction_seed"] == seed
        assert base["student_steps"] == 2048 and base["student_batch"] == 24
        assert base["student_seeds"] == [150, 151, 152] and base["order_seed"] == 9101
        assert base["config"]["steps"] == 30 and base["config"]["lambda_gradient"] > 0
        assert base["config"]["assignment_storage"] == "class_blocks"
        assert len(base["groups"]) == (3 if dataset == "arxiv" else 5)
        if not reused:
            p = result["protocol"]
            assert p["source_result_sha256"] == V.digest(source_path)
            assert p["report_dev_only"] and not p["same_total_compute"]
            assert p["full_history"] == (arm == "full_history")
            assert p["gpu_name"] == base["device_name"]
            assert p["torch_version"] == base["torch_version"]
            assert p["cpu_threads"] == base["cpu_threads"] == 4
            for key, filename in (("runner_hash", "run.py"), ("registration_hash", "README.md"),
                                  ("protocol_utils_hash", "protocol_utils.py")):
                assert p[key] == V.digest(HERE / filename)
            receipt["protocol_matches"] += 1
        receipt["source_results"].append(dict(run=path.parent.name, sha256=V.digest(path)))
        receipt["statuses"].append(dict(dataset=dataset, arm=arm, seed=seed, status=result["status"]))
        info = dict(dataset=dataset, arm=arm, construction_seed=seed)
        ref_memory = V.load_memory(source_path.parent / source["rows"][-1]["memory_path"])
        previous = None
        last_payload = None
        for row in result["rows"]:
            stage = row["stage"]
            memory_path = path.parent / row["memory_path"]
            assert V.digest(memory_path) == row["memory_file_sha256"]
            memory = V.load_memory(memory_path)
            assert V.memory_hash(memory) == row["memory_hash"]
            assert len(memory["x"]) == base["budgets"][stage]
            assert torch.equal(memory["labels"], ref_memory["labels"][:len(memory["x"])])
            assert not memory["edge_index"].numel()
            for field in ("x", "text_emb", "mass"):
                assert torch.isfinite(memory[field]).all()
            if previous is not None:
                count = len(previous["x"])
                for field in ("x", "text_emb", "labels", "mass"):
                    assert torch.equal(memory[field][:count], previous[field])
                assert memory["texts"][:count] == previous["texts"]
                if "text_source_ids" in memory and "text_source_ids" in previous:
                    assert memory["text_source_ids"][:count] == previous["text_source_ids"]
                receipt["prefixes"] += 1
            previous = memory
            if arm == "full_history":
                assert row["raw_rows"] == sum(base["stage_raw_rows"])
                assert row["raw_id_hash"] == result["protocol"]["full_pool_hash"]
                assert row["new_rows"] == base["budgets"][-1]
                assert row["rng_stage"] == 0
            else:
                assert row["raw_rows"] == base["stage_raw_rows"][stage]
                assert row["new_rows"] == base["budgets"][stage] - (base["budgets"][stage-1] if stage else 0)
                if not reused:
                    assert row["raw_id_hash"] == base["stage_hashes"][stage]
            last_payload = payload_bytes(memory)
            if "payload_bytes" in row:
                assert last_payload == row["payload_bytes"]
            receipt["memories"] += 1
            audit = row["audit"]
            if arm in ("incremental_q", "full_history"):
                assert row["checks"]["sinkhorn_feasible"]
                costs = audit["q_cost"]["class_costs"]
                assert audit["assignment_parameter_entries"] == sum(c["raw_rows"] * c["slots"] for c in costs)
            stages.append(dict(**info, stage=stage, raw_rows=row["raw_rows"], new_rows=row["new_rows"],
                memory_rows=len(memory["x"]), raw_edges=row.get("raw_edges"),
                peak_gib=row["peak_bytes"]/2**30, reserved_gib=row["reserved_bytes"]/2**30,
                entry_gib=row["entry_bytes"]/2**30, cpu_rss_gib=row["cpu_peak_rss_bytes"]/2**30,
                seconds=row["construction_seconds"], cpu_process_seconds=row.get("cpu_process_seconds"),
                payload_mib=last_payload["total"]/2**20,
                assignment_entries=audit.get("assignment_parameter_entries"),
                final_loss=audit.get("trace", [{}])[-1].get("loss") if audit.get("trace") else None))
            scores = row["scores"]
            if not scores:
                continue
            assert scores["split"] == "report_dev"
            assert [s["seed"] for s in scores["per_student"]] == base["student_seeds"]
            f1s, accs = [], []
            for score in scores["per_student"]:
                pred_path = path.parent / score["prediction_path"]
                assert V.digest(pred_path) == score["prediction_sha256"]
                with np.load(pred_path) as a:
                    f1, acc = V.independent_scores(a["truth"], a["pred"], a["classes"])
                    key = (dataset, stage)
                    cohort = {k: a[k].copy() for k in ("ids", "truth", "classes")}
                    if key in cohorts:
                        for k in cohort:
                            assert np.array_equal(cohort[k], cohorts[key][k])
                        receipt["cohort_matches"] += 1
                    else:
                        cohorts[key] = cohort
                assert abs(f1 - score["f1"]) < 1e-9
                assert abs(acc - score["accuracy"]) < 1e-9
                receipt["predictions"] += 1
                receipt["independent_metric_checks"] += 2
                f1s.append(f1)
                accs.append(acc)
                evaluations.append(dict(**info, stage=stage, student_seed=score["seed"], f1=f1, accuracy=acc,
                    student_seconds=score["train_seconds"], student_peak_gib=score["train_peak_bytes"]/2**30,
                    inference_seconds=score["inference_seconds"], inference_peak_gib=score["inference_peak_bytes"]/2**30,
                    cache_rebuild_seconds=score["cache_rebuild_seconds"]))
            assert abs(np.mean(f1s) - scores["f1"]) < 1e-9
            assert abs(np.mean(accs) - scores["accuracy"]) < 1e-9
        if result["status"] != "complete":
            failures.append(dict(**info, status=result["status"], saved_stages=len(result["rows"]),
                failure=result.get("failure"), failed_stage=result.get("failed_stage"),
                failure_peak_gib=result.get("failure_peak_bytes", 0)/2**30))
            continue
        expected_stages = [len(base["groups"])-1] if arm == "full_history" else list(range(len(base["groups"])))
        assert [r["stage"] for r in result["rows"]] == expected_stages
        final = result["rows"][-1]
        assert final["scores"]
        final_scores = final["scores"]["per_student"]
        runs.append(dict(**info, reused=reused, stages=len(result["rows"]),
            f1=final["scores"]["f1"], accuracy=final["scores"]["accuracy"],
            peak_gib=max(r["peak_bytes"] for r in result["rows"])/2**30,
            reserved_gib=max(r["reserved_bytes"] for r in result["rows"])/2**30,
            seconds=sum(r["construction_seconds"] for r in result["rows"]),
            cpu_rss_gib=max(r["cpu_peak_rss_bytes"] for r in result["rows"])/2**30,
            payload_mib=last_payload["total"]/2**20,
            student_peak_gib=float(np.mean([s["train_peak_bytes"] for s in final_scores]))/2**30,
            student_seconds=float(np.mean([s["train_seconds"] for s in final_scores])),
            inference_peak_gib=float(np.mean([s["inference_peak_bytes"] for s in final_scores]))/2**30))

    summaries, paired = [], []
    for dataset in ("arxiv", "sportsfit"):
        for arm in ARMS:
            selected = [r for r in runs if r["dataset"] == dataset and r["arm"] == arm]
            if not selected:
                continue
            row = dict(dataset=dataset, arm=arm, constructions=len(selected))
            for metric in METRICS:
                stats = V.stats([r[metric] for r in selected])
                row[metric + "_mean"], row[metric + "_sd"] = stats["mean"], stats["sd"]
            summaries.append(row)
        for other in ARMS[1:]:
            paired += paired_rows(dataset, "incremental_q", other, runs)
    for name, values in (("stages", stages), ("evaluation", evaluations), ("runs", runs),
                         ("summary", summaries), ("paired", paired), ("failures", failures)):
        V.write_csv(args.output / (name + ".csv"), values)
    receipt["complete_runs"] = len(runs)
    receipt["new_complete_runs"] = sum(not r["reused"] for r in runs)
    receipt["failed_runs"] = len([f for f in failures if f["status"] == "failed"])
    if args.require_finished:
        expected = {(d, a, s) for d in ("arxiv", "sportsfit") for a in ARMS for s in SEEDS}
        assert set(records) == expected
        assert all(r["status"] in ("complete", "failed") for _, r in records.values())
        assert receipt["protocol_matches"] == 18
        if not failures:
            assert receipt["memories"] == 78 and receipt["prefixes"] == 54
            assert receipt["predictions"] == 126 and receipt["independent_metric_checks"] == 252
            assert receipt["complete_runs"] == 24
    (args.output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    lines = ["# Common construction resource comparison", "",
        "Report-development only. ArXiv uses the archived three-stage class stream; SportsFit uses five stages.",
        "Three constructions, each averaged over three student seeds. SD is across construction means.", "",
        "| Dataset | Method | n | F1 mean +/- SD | Accuracy | Allocated GiB | Reserved GiB | Construction s | CPU RSS GiB | Student GiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        sd = "NA" if r["f1_sd"] is None else f"{r['f1_sd']:.3f}"
        lines.append(f"| {r['dataset']} | {r['arm']} | {r['constructions']} | {r['f1_mean']:.3f} +/- {sd} | {r['accuracy_mean']:.3f} | {r['peak_gib_mean']:.3f} | {r['reserved_gib_mean']:.3f} | {r['seconds_mean']:.2f} | {r['cpu_rss_gib_mean']:.3f} | {r['student_peak_gib_mean']:.3f} |")
    lines += ["", "## Interpretation", "",
        "- Full-history uses all final legal raw training nodes and edges. It is an access-relaxed reference.",
        "- Full-history has one 30-update call; incremental Q has three/five 30-update calls. Total compute is not matched.",
        "- Incremental Q reuses the six verified combined runs. Baselines retain CaT 3000 updates and GECC 20 restarts.",
        "- Construction includes the resident text encoder, excludes original-graph GPU evaluation caches, extraction and serialization.",
        "- CPU RSS includes the CPU source loader. This experiment does not establish lower total host memory for streaming.",
        "- Student and original-graph inference are measured separately. Their peaks must not be presented as construction peaks.",
        "- Times are descriptive across batches and GPU lanes; no isolated timing speedup claim.",
        "- Both baselines are mechanism adaptations, not unchanged official reproductions.",
        "- Failure is missing evidence, not zero performance. Paired intervals use construction seeds, not nine independent students.",
        "", "See paired.csv for paired 95% t intervals, sign counts and worst paired outcome.",
        f"Verified {receipt['memories']} memories, {receipt['prefixes']} prefixes, {receipt['predictions']} predictions and {receipt['independent_metric_checks']} independently recomputed scores."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("source_results", "statuses")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reused", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-finished", action="store_true")
    main(parser.parse_args())
