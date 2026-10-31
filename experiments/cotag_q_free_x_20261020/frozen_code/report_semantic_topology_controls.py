"""Fail-closed paired reporting for semantic topology controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import summarize_topology_causal_ablation as S

BASE_ARMS = ("c0", "internal_only", "t_degree")
NEW_ARMS = ("knn_nearest", "knn_capacity", "cycles_nearest", "cycles_capacity")
CONTRASTS = dict(S.CONTRASTS)
CONTRASTS.update({f"{a}-{b}": {a: 1, b: -1} for a in NEW_ARMS for b in BASE_ARMS})
CONTRASTS.update({f"{a}-{b}": {a: 1, b: -1} for a, b in (
    ("knn_capacity", "knn_nearest"), ("cycles_capacity", "cycles_nearest"),
    ("cycles_nearest", "knn_nearest"), ("cycles_capacity", "knn_capacity"))})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_dataset(root, baseline):
    manifest, base_manifest = S.read_json(root / "manifest.json"), S.read_json(baseline / "manifest.json")
    if tuple(manifest["arms"]) != NEW_ARMS:
        raise ValueError("unexpected new arms")
    for name in ("dataset", "orders", "seeds", "splits", "rate", "nc_steps", "nc_batch",
                 "lp_steps", "data_sha256", "text_sha256"):
        if manifest[name] != base_manifest[name]:
            raise ValueError("mismatched comparison protocol: " + name)
    if sha(baseline / "manifest.json") != manifest["baseline_manifest_sha256"]:
        raise ValueError("baseline manifest hash changed")
    records, base_records = S.read_json(root / "results.json"), S.read_json(baseline / "results.json")
    S.validate(records, manifest)
    S.validate(base_records, base_manifest)
    separate = S.read_records(root)
    S.validate(separate, manifest)
    if {S.record_key(r): r for r in records} != {S.record_key(r): r for r in separate}:
        raise ValueError("per-fit and combined records disagree")
    audit = S.read_json(root / "independent_audit.json")
    counts = {task: sum(r["task"] == task for r in records) for task in ("nc", "lp")}
    if not audit["passed"] or audit["fits"] != len(records):
        raise ValueError("checkpoint audit incomplete")
    if any(audit[task + "_fits"] != count for task, count in counts.items()):
        raise ValueError("checkpoint audit counts disagree")
    completion = S.read_json(root / "complete.json")
    expected_completion = {"nc_fits": counts["nc"], "lp_fits": counts["lp"],
                           "total_fits": len(records), "expected": len(records)}
    receipt = dict(dataset=manifest["dataset"], matrix_valid=True,
                   counts=expected_completion, runner_complete=completion,
                   runner_count_metadata_valid=completion == expected_completion,
                   original_files_preserved=True, strict_metric_reload_passed=audit["strict_metric_reload_passed"],
                   reload_warnings=audit["reload_warnings"],
                   source_hashes={str(p): sha(p) for p in (
                       root / "manifest.json", root / "results.json", root / "independent_audit.json",
                       baseline / "manifest.json", baseline / "results.json")})
    return manifest, records, base_manifest, base_records, receipt


def paired_contrasts(trajectories, contrasts=CONTRASTS):
    matched, grouped = defaultdict(dict), defaultdict(list)
    for r in trajectories:
        key = tuple(r[k] for k in ("task", "order", "split", "seed", "horizon", "metric"))
        if r["arm"] in matched[key]:
            raise ValueError("duplicate trajectory arm")
        matched[key][r["arm"]] = r["value"]
    for (task, order, split, seed, horizon, metric), values in matched.items():
        for name, weights in contrasts.items():
            if not weights.keys() <= values.keys():
                raise ValueError("unpaired contrast: " + name)
            delta = sum(values[a] * w for a, w in weights.items())
            for scope in (order, "all"):
                grouped[(task, scope, horizon, metric, name)].append(delta)
    return [dict(task=t, order=o, horizon=h, metric=m, contrast=c, unit="pp",
                 mean_pp=S.avg(v), sd_pp=S.sd(v), n=len(v),
                 positive=sum(x > 1e-12 for x in v), negative=sum(x < -1e-12 for x in v),
                 ties=sum(abs(x) <= 1e-12 for x in v))
            for (t, o, h, m, c), v in grouped.items()]


def cost_rows(records, dataset):
    grouped = defaultdict(list)
    for r in records:
        for order in (r["order"], "all"):
            grouped[(r["task"], order, r["arm"])].append(r["cost"])
    rows = []
    for (task, order, arm), costs in grouped.items():
        values = [c["train_seconds"] for c in costs]
        if not all(v >= 0 for v in values):
            raise ValueError("invalid training cost")
        rows.append(dict(dataset=dataset, task=task, order=order, arm=arm, n=len(costs),
                         mean_train_seconds=mean(values), sd_train_seconds=S.sd(values),
                         total_train_seconds=sum(values),
                         max_peak_allocated_bytes=max(c["peak_allocated_bytes"] for c in costs),
                         source_batch="new" if arm in NEW_ARMS else "reused"))
    return rows


def supplement_rows(data, manifest):
    dataset = manifest["dataset"]
    if data["passed"] is not True or data["dataset"] != dataset:
        raise ValueError("supplement failed or belongs to another dataset")
    if data["frozen_sources_checked"] != manifest["sources"]:
        raise ValueError("supplement frozen source hashes disagree")
    stages = {(int(order), stage) for order, groups in manifest["orders"].items()
              for stage in range(len(groups))}
    operators = ("c0", "single_ring", "knn", "cycles")

    def indexed(rows, field, names):
        result = {(r["order"], r["stage"], r[field]): r for r in rows}
        expected = {(order, stage, name) for order, stage in stages for name in names}
        if (len(rows) != len(result) or set(result) != expected
                or any(r["dataset"] != dataset for r in rows)):
            raise ValueError("supplement matrix mismatch: " + field)
        return result

    structural = indexed(data["structural"], "arm", NEW_ARMS)
    timings = indexed(data["internal_microbenchmark"], "operator", operators)
    if (data["memory_checks"] != len(structural)
            or data["cycle_degree_checks"] != 2 * len(stages)):
        raise ValueError("supplement check counts disagree")
    for row in timings.values():
        samples = row["seconds"]
        if (not samples or row["repeats"] != len(samples)
                or not all(math.isfinite(s) and s >= 0 for s in samples)
                or not math.isfinite(row["median_seconds"])
                or not math.isclose(row["median_seconds"], median(samples), rel_tol=0, abs_tol=1e-12)
                or not isinstance(row["new_rows"], int) or row["new_rows"] < 1):
            raise ValueError("invalid internal timing samples")
    for order, stage in stages:
        sizes = {timings[order, stage, op]["new_rows"] for op in operators}
        if len(sizes) != 1:
            raise ValueError("timed operators use different blocks")
        n = sizes.pop()
        payloads = set()
        for arm in NEW_ARMS:
            row = structural[order, stage, arm]
            match = "t_degree" if arm.endswith("capacity") else "internal_only"
            if row["matched_baseline"] != match or row["same_current_cross"] is not True:
                raise ValueError("supplement cross policy mismatch")
            if arm.startswith("cycles") and (row["same_full_degree"] is not True
                    or row["max_new_internal_degree"] != min(2, n - 1)):
                raise ValueError("supplement cycle degree mismatch")
            if any(not isinstance(row[k], int) or row[k] < 0 for k in
                   ("memory_file_bytes", "tensor_bytes", "text_utf8_bytes")):
                raise ValueError("invalid memory storage size")
            if row["tensor_bytes"] <= 0 or row["memory_file_bytes"] < row["tensor_bytes"]:
                raise ValueError("invalid memory storage size")
            payloads.add((row["tensor_bytes"], row["text_utf8_bytes"]))
        if len(payloads) != 1:
            raise ValueError("matched memory payload sizes disagree")
    benchmarks, storage = [], []
    for op in operators:
        rows = [r for r in timings.values() if r["operator"] == op]
        values = [r["median_seconds"] for r in rows]
        benchmarks.append(dict(dataset=dataset, operator=op, blocks=len(rows),
                               measured_runs=sum(r["repeats"] for r in rows),
                               mean_block_median_seconds=mean(values),
                               min_block_median_seconds=min(values),
                               max_block_median_seconds=max(values),
                               min_new_rows=min(r["new_rows"] for r in rows),
                               max_new_rows=max(r["new_rows"] for r in rows)))
    for arm in NEW_ARMS:
        rows = [r for r in structural.values() if r["arm"] == arm]
        final = [r for r in rows if r["stage"] == len(manifest["orders"][str(r["order"])]) - 1]
        storage.append(dict(dataset=dataset, arm=arm, memories=len(rows), final_memories=len(final),
                            cross_policy_checks=sum(r["same_current_cross"] for r in rows),
                            full_degree_matches=sum(r["same_full_degree"] for r in rows),
                            max_new_internal_degree=max(r["max_new_internal_degree"] for r in rows),
                            final_file_mib_min=min(r["memory_file_bytes"] for r in final) / 2**20,
                            final_file_mib_max=max(r["memory_file_bytes"] for r in final) / 2**20,
                            final_tensor_mib_min=min(r["tensor_bytes"] for r in final) / 2**20,
                            final_tensor_mib_max=max(r["tensor_bytes"] for r in final) / 2**20))
    return benchmarks, storage


def write_markdown(path, endpoints, contrasts, benchmarks, storage):
    lines = ["# Semantic Topology Controls: Audited Tables", "",
             "New arms: 1440 fits. Prior arms are reused, not rerun. LP units: percent; differences: pp.",
             "Stage means are computed within trajectories. Old/cross groups omit undefined stage 0.",
             "SD and positive counts are descriptive repeats, not independent datasets or significance tests.", ""]
    metrics = ("all_cumulative_f1", "seen_cumulative_f1", "seen_old_f1", "seen_new_f1", "all_auc", "all_ap")
    for dataset in ("arxiv", "bookchild"):
        lines += [f"## {dataset}", "", "| horizon | arm | metric | mean | SD | n |",
                  "|---|---|---|---:|---:|---:|"]
        for r in endpoints:
            if r["dataset"] != dataset or r["order"] != "all" or r["metric"] not in metrics:
                continue
            lines.append(f"| {r['horizon']} | {r['arm']} | {r['metric']} | {r['mean']:.4f} | {r['sd']:.4f} | {r['n']} |")
        lines += ["", "| horizon | contrast | metric | mean pp | paired SD | positive / n |",
                  "|---|---|---|---:|---:|---:|"]
        for r in contrasts:
            if r["dataset"] != dataset or r["order"] != "all" or r["metric"] not in metrics:
                continue
            lines.append(f"| {r['horizon']} | {r['contrast']} | {r['metric']} | {r['mean_pp']:+.4f} | {r['sd_pp']:.4f} | {r['positive']}/{r['n']} |")
    lines += ["", "## Post-result Internal-Topology Microbenchmark", "",
              "Same blocks; one warmup followed by rotated measured runs. Mean of within-block medians.",
              "Includes the existing implementation's kernel/edge selection and device transfers.",
              "Excludes cross edges, raw-data loading, X/text construction, I/O and student training.",
              "Not end-to-end speedup evidence, not an additional student fit.", "",
              "| dataset | operator | blocks | mean median (s) | min (s) | max (s) |",
              "|---|---|---:|---:|---:|---:|"]
    for r in benchmarks:
        lines.append(f"| {r['dataset']} | {r['operator']} | {r['blocks']} | "
                     f"{r['mean_block_median_seconds']:.6f} | {r['min_block_median_seconds']:.6f} | "
                     f"{r['max_block_median_seconds']:.6f} |")
    lines += ["", "## Saved Memory Storage", "",
              "MiB = 2^20 bytes. Ranges cover final-stage memories across class orders.",
              "Serialized size includes metadata; it is not process RAM or construction scratch space.", "",
              "| dataset | arm | checked | degree matches | final file MiB min | max |",
              "|---|---|---:|---:|---:|---:|"]
    for r in storage:
        lines.append(f"| {r['dataset']} | {r['arm']} | {r['memories']} | {r['full_degree_matches']} | "
                     f"{r['final_file_mib_min']:.4f} | {r['final_file_mib_max']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args):
    endpoints, contrasts, costs, receipts, trajectories = [], [], [], [], []
    benchmarks, storage, fingerprints = [], [], []
    root, baseline = Path(args.root), Path(args.baseline_root)
    for dataset in ("arxiv", "bookchild"):
        manifest, records, base_manifest, base_records, receipt = validate_dataset(root / dataset, baseline / dataset)
        supplement_path = root / dataset / "supplemental_topology_checks.json"
        supplement = S.read_json(supplement_path)
        if supplement["source_sha256"] != sha(Path(__file__).with_name("check_semantic_topology_supplement.py")):
            raise ValueError("supplement script hash changed")
        timing_rows, storage_rows = supplement_rows(supplement, manifest)
        benchmarks.extend(timing_rows)
        storage.extend(storage_rows)
        fingerprints.append(supplement["minilm_files_sha256_after_run"])
        receipt["supplement"] = dict(passed=True, memory_checks=supplement["memory_checks"],
                                     cycle_degree_checks=supplement["cycle_degree_checks"],
                                     file_sha256=sha(supplement_path),
                                     hardware=supplement["hardware"], limits=supplement["limits"])
        traj = S.trajectories(base_records, base_manifest) + S.trajectories(records, manifest)
        endpoints.extend(dict(dataset=dataset, **r) for r in S.summarize_trajectories(traj))
        contrasts.extend(dict(dataset=dataset, **r) for r in paired_contrasts(traj))
        costs.extend(cost_rows(base_records + records, dataset))
        trajectories.extend(dict(dataset=dataset, **r) for r in traj)
        receipts.append(receipt)
    if not fingerprints[0] or any(f != fingerprints[0] for f in fingerprints[1:]):
        raise ValueError("post-run text encoder fingerprints disagree")
    for name, rows in (("comparison_endpoints_percent.csv", endpoints),
                       ("paired_semantic_contrasts_pp.csv", contrasts),
                       ("comparison_trajectories_percent.csv", trajectories), ("training_costs.csv", costs),
                       ("internal_topology_cost_summary.csv", benchmarks), ("memory_storage_summary.csv", storage)):
        S.write_csv(root / name, rows)
    status = dict(new_fits=sum(r["counts"]["total_fits"] for r in receipts),
                  reused_fits=2160, datasets=receipts, report_script_sha256=sha(__file__),
                  independent_confirmation=False,
                  post_run_encoder_fingerprints_match=True, end_to_end_cost_complete=False,
                  warning="Frozen completion counts are wrong; use validated counts in this receipt. Original runner/fits are retained. This is not independent confirmation.")
    (root / "comparison_integrity.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    write_markdown(root / "SEMANTIC_TOPOLOGY_CONTROLS_RESULTS_ZH_2026-09-21.md", endpoints, contrasts,
                   benchmarks, storage)
    print(json.dumps(dict(new_fits=status["new_fits"], reused_fits=2160,
                          paired_rows=len(contrasts), endpoint_rows=len(endpoints))))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--baseline-root", required=True)
    main(p.parse_args())
