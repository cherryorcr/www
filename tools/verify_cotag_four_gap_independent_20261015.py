"""Verify the four-gap export directly from the snapshotted Student records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


DATASETS = ("arxiv", "sportsfit", "bookhis")
SEEDS = (271006, 271108, 271109)
SPLITS = ("report_dev", "sealed_test")
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1", "stage_f1")
MATCH_KEYS = (
    "dataset", "order_seed", "construction_seed", "groups", "budgets",
    "stage_raw_rows", "student", "student_steps", "student_batch",
    "student_seeds", "pool_mask_hash", "report_mask_hash", "test_mask_hash",
    "stage_id_hashes",
)
T95_DF2 = 4.302652729911275


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, context):
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), context
        for index, (value, target) in enumerate(zip(actual, expected)):
            close(value, target, (context, index))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert math.isfinite(float(actual)) and math.isclose(
            float(actual), expected, rel_tol=1e-10, abs_tol=1e-8
        ), (context, actual, expected)
    else:
        assert actual == expected, (context, actual, expected)


def describe(values):
    avg, sd = mean(values), stdev(values)
    half = T95_DF2 * sd / math.sqrt(3)
    return dict(n=3, mean=avg, sd=sd, ci_low=avg-half, ci_high=avg+half,
                positive=sum(value > 0 for value in values), worst=min(values),
                values=values)


def verify(root, report, output):
    manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["pending"] == []
    assert len(manifest["sources"]) == 48
    raw = {}
    for source in manifest["sources"]:
        path = root / source["path"]
        close(digest(path), source["sha256"], source["path"])
        raw[source["path"]] = json.loads(path.read_text(encoding="utf-8"))
        assert raw[source["path"]]["status"] == "complete"
    assert len(raw) == 48
    close(digest(root / "tools/summarize_cotag_four_gap_20261015.py"),
          manifest["summary_script_sha256"], "snapshotted summarizer")

    export = json.loads((report / "data.json").read_text(encoding="utf-8"))
    expected = {key: [] for key in export}
    batch = "results/cotag_q_four_gap_20261015_v1"
    cells, primaries = {}, {}
    warnings = []

    def protocol(result, primary):
        for key in MATCH_KEYS:
            close(result["protocol"][key], primary["protocol"][key], key)

    def rows_for(result, arm=None, final=False):
        p = result["protocol"]
        rows = sorted((r for r in result["rows"] if arm is None or r["arm"] == arm),
                      key=lambda r: r["stage"])
        stages = [len(p["budgets"])-1] if final else list(range(len(p["budgets"])))
        assert [r["stage"] for r in rows] == stages
        for row in rows:
            assert row["memory_rows"] == (p["reference_rows"] if final else p["budgets"][row["stage"]])
            if not final:
                assert all(row["checks"][k] for k in ("budget_ok", "prefix_ok", "finite_ok"))
                assert row["checks"].get("sinkhorn_feasible") is not False
                if row["checks"].get("numerical_warning"):
                    warnings.append(dict(dataset=p["dataset"], stage=row["stage"],
                                         task=p.get("task", p["name"])))
            for split in SPLITS:
                score = row["scores"][split]
                assert score["split"] == split
                assert sorted(s["seed"] for s in score["per_student"]) == sorted(p["student_seeds"])
                for key in ("student_steps", "student_batch", "student_seeds"):
                    close(score[key], p[key], key)
                for metric in METRICS[:-1]:
                    values = [s[metric] for s in score["per_student"]]
                    if metric == "old_f1" and row["stage"] == 0 and score[metric] is None:
                        assert all(v is None for v in values)
                        continue
                    assert all(v is not None and 0 <= v <= 100 for v in values)
                    close(score[metric], mean(values), (split, metric))
        return rows

    for ds in DATASETS:
        for seed in SEEDS:
            primary = raw[f"results/uq_final_evidence_20261012/confirmation/{ds}_o9101_c{seed}/results.json"]
            primaries[ds, seed] = primary
            arms = dict(
                full=primary,
                no_text_init=raw[f"{batch}/text/{ds}_o9101_c{seed}_no_text_init/results.json"],
                no_pair_feedback=raw[f"{batch}/text/{ds}_o9101_c{seed}_no_pair_feedback/results.json"],
                both_off=raw[f"results/uq_exact5_no_node_text_20261015/{ds}_o9101_c{seed}/results.json"],
            )
            for arm, result in arms.items():
                protocol(result, primary)
                config, base = result["protocol"]["q_config"], primary["protocol"]["q_config"]
                changes = {k: (base[k], config[k]) for k in base if base[k] != config[k]}
                wanted = {}
                if arm in ("no_text_init", "both_off"):
                    wanted["text_weight"] = (0.35, 0.0)
                if arm in ("no_pair_feedback", "both_off"):
                    wanted["pair_weight"] = (0.5, 0.0)
                close(changes, wanted, (ds, seed, arm, "intervention"))
                rows = rows_for(result, "uq_dm_edgefree" if arm == "full" else None)
                for split in SPLITS:
                    score = rows[-1]["scores"][split]
                    cell = dict(dataset=ds, construction_seed=seed, arm=arm, split=split)
                    cell.update({m: mean(s[m] for s in score["per_student"]) for m in METRICS[:-1]})
                    cell["stage_f1"] = mean(mean(s["final_f1"] for s in r["scores"][split]["per_student"]) for r in rows)
                    cell["construction_seconds"] = sum(r["construction_seconds"] for r in rows)
                    cell["construction_peak_mb"] = max(r["construction_peak_mb"] for r in rows)
                    cells[ds, seed, arm, split] = cell
                    expected["text_cells"].append(cell)
                    expected["text_students"].extend(dict(dataset=ds, construction_seed=seed,
                        arm=arm, split=split, **s) for s in score["per_student"])
        for split in SPLITS:
            for metric in METRICS:
                for arm in arms:
                    values = [cells[ds, seed, arm, split][metric] for seed in SEEDS]
                    expected["text_summary"].append(dict(dataset=ds, split=split, arm=arm,
                                                        metric=metric, **describe(values)))
                    if arm != "full":
                        differences = [cells[ds, seed, "full", split][metric]-cells[ds, seed, arm, split][metric] for seed in SEEDS]
                        expected["text_contrasts"].append(dict(dataset=ds, split=split, control=arm,
                            metric=metric, orientation="full minus control", **describe(differences)))
                interaction = [cells[ds,s,"full",split][metric]-cells[ds,s,"no_text_init",split][metric]
                    -cells[ds,s,"no_pair_feedback",split][metric]+cells[ds,s,"both_off",split][metric] for s in SEEDS]
                expected["text_interactions"].append(dict(dataset=ds, split=split, metric=metric,
                    orientation="full - no-init - no-pair + both-off", **describe(interaction)))

    for ds in DATASETS:
        for arm in ("full_ef", "full_topology"):
            result = raw[f"{batch}/fullhistory/{ds}_o9101_c271006_{arm}/results.json"]
            protocol(result, primaries[ds, SEEDS[0]])
            p = result["protocol"]
            row = rows_for(result, final=True)[0]
            assert p["reference_rows"] == sum(p["stage_raw_rows"])
            close(p["actual_final_rate"], p["budgets"][-1]/p["reference_rows"], "rate")
            for coverage in result["sampling_coverage"]:
                assert coverage["draws"] == p["student_steps"]*p["student_batch"]
                assert abs(coverage["fraction"]-coverage["unique_rows"]/p["reference_rows"]) < 5e-8
            for split in SPLITS:
                ref = {s["seed"]: s for s in row["scores"][split]["per_student"]}
                for metric in METRICS[:-1]:
                    pairs = []
                    for s in p["student_seeds"]:
                        ours = mean(next(r[metric] for r in rows_for(primaries[ds,c], "uq_dm_edgefree")[-1]["scores"][split]["per_student"] if r["seed"] == s) for c in SEEDS)
                        b = ref[s][metric]
                        pairs.append(dict(student_seed=s, ours_mean_over_constructions=ours,
                                          reference_value=b, retention=100*ours/b, difference=ours-b))
                    expected["fullhistory_student_pairs"].extend(dict(dataset=ds, split=split,
                        reference=arm, metric=metric, **r) for r in pairs)
                    ours_avg = mean(r["ours_mean_over_constructions"] for r in pairs)
                    ref_avg = mean(r["reference_value"] for r in pairs)
                    expected["fullhistory_retention"].append(dict(dataset=ds, split=split, reference=arm,
                        metric=metric, ours=ours_avg, reference_value=ref_avg,
                        ratio_of_means=100*ours_avg/ref_avg,
                        mean_paired_student_ratio=mean(r["retention"] for r in pairs),
                        paired_student_ratio_sd=stdev(r["retention"] for r in pairs),
                        paired_student_difference_mean=mean(r["difference"] for r in pairs),
                        full_rows=p["reference_rows"], synthetic_rows=p["budgets"][-1],
                        full_edges=p["reference_edges"], actual_rate=p["actual_final_rate"],
                        sampled_loss_row_coverage=mean(r["fraction"] for r in result["sampling_coverage"]),
                        note="3 Student pairs after averaging 3 construction seeds; full-history is not triplicated"))

        result = raw[f"{batch}/tagsam_budget/{ds}_o9101_c271006/results.json"]
        protocol(result, primaries[ds, SEEDS[0]])
        expected["tagsam_replay"].append(dict(dataset=ds, **result["archived_100_replay"]))
        assert sorted(r["outer_steps"] for r in result["rows"]) == [100, 500]
        for split in SPLITS:
            for r in result["rows"]:
                score = r["scores"][split]
                expected["tagsam_budget"].append(dict(dataset=ds, split=split, outer_steps=r["outer_steps"],
                    **{m: mean(s[m] for s in score["per_student"]) for m in METRICS[:-1]},
                    student_f1_sd=stdev(s["final_f1"] for s in score["per_student"]),
                    response_seconds=r["response_seconds"], response_peak_mb=r["response_peak_mb"]))

        result = raw[f"{batch}/cost/{ds}_o9101_c271006_dense_vs_sparse/results.json"]
        protocol(result, primaries[ds, SEEDS[0]])
        op = result["operator_equivalence"]
        assert op["passed"] and all(op[k] == 0 for k in ("forward_max_error", "first_gradient_max_error", "second_gradient_max_error"))
        a = result["allocation_accounting"]
        assert a["dense_entries"] == a["raw_rows"]*a["slots"]
        close(a["illegal_fraction"], 1-a["class_legal_entries"]/a["dense_entries"], "illegal mass")
        close(a["dense_logits_fp32_mib"], 4*a["dense_entries"]/2**20, "logits bytes")
        close(a["logits_plus_gradient_and_two_adam_states_mib"], 16*a["dense_entries"]/2**20, "Adam bytes")
        close(a["synthetic_dense_identity_mib"], 4*a["slots"]**2/2**20, "identity bytes")
        for r in result["rows"]:
            expected["cost"].append(dict(dataset=ds, arm=r["arm"], stage=0,
                **{k:r[k] for k in ("construction_seconds", "construction_peak_mb", "construction_reserved_mb")},
                **op, **result["construction_equivalence"], **a))

    key_fields = {
        "text_cells": ("dataset", "construction_seed", "arm", "split"),
        "text_students": ("dataset", "construction_seed", "arm", "split", "seed"),
        "text_summary": ("dataset", "split", "arm", "metric"),
        "text_contrasts": ("dataset", "split", "control", "metric"),
        "text_interactions": ("dataset", "split", "metric"),
        "fullhistory_retention": ("dataset", "split", "reference", "metric"),
        "fullhistory_student_pairs": ("dataset", "split", "reference", "metric", "student_seed"),
        "tagsam_budget": ("dataset", "split", "outer_steps"),
        "tagsam_replay": ("dataset",), "cost": ("dataset", "arm"),
    }
    checked_scalars = 0
    for table, records in expected.items():
        keys = key_fields[table]
        def record_key(record):
            return tuple(str(record[k]) for k in keys)
        json_records = {record_key(r): r for r in export[table]}
        with (report / f"{table}.csv").open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        csv_records = {record_key(r): r for r in csv_rows}
        assert len(records) == len(json_records) == len(csv_rows) == len(csv_records)
        for record in records:
            key = record_key(record)
            assert set(json_records[key]) == set(record)
            for field, value in record.items():
                close(json_records[key][field], value, (table, key, field, "json"))
                encoded = csv_records[key][field]
                if isinstance(value, list):
                    decoded = json.loads(encoded)
                    assert len(decoded) == len(value)
                    for actual, wanted in zip(decoded, value):
                        close(actual, wanted, (table, key, field, "csv array"))
                elif isinstance(value, bool):
                    close(encoded, str(value), (table, key, field, "csv bool"))
                else:
                    close(encoded, value, (table, key, field, "csv"))
                checked_scalars += 1
    for figure in ("text_factorial", "fullhistory_matched", "identity_cost"):
        provenance = json.loads((report / f"{figure}.source.json").read_text(encoding="utf-8"))
        close(provenance["data_sha256"], digest(report / "data.json"), figure)
    warnings = list({json.dumps(r, sort_keys=True): r for r in warnings}.values())
    audit = dict(status="pass", source_hashes_verified=len(raw),
        summarizer_snapshot_verified=True, export_rows={k:len(v) for k,v in expected.items()},
        compared_fields=checked_scalars, numerical_warnings=warnings,
        data_sha256=digest(report / "data.json"), verifier_sha256=digest(Path(__file__)),
        scope="Independent stdlib recomputation from raw Student records; no aggregation helpers imported",
        limits="Checks saved records and exports, not the truth of unobserved training behavior; masks previously inspected")
    assert not output.exists(), "Keep audit outputs immutable; choose a new path"
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.snapshot, args.report, args.output)
