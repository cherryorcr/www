"""Summarize archived student measurements and audited assignment dimensions."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


STUDENT_HASHES = {
    "arxiv": "836da7459598e299bc259bb32e2998824d9dbc878cdd30421ac99dfbdf94959f",
    "sportsfit": "f5204e0fd7a6e14bb86a99cb4891bbba19990565976707dbd650108254d46d0e",
    "bookhis": "392c007dae4a5c74b62cdf93a9bf1f18b3eea22600017d7647b08d9d76487b33",
}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    summary, pairs, sources = [], [], []
    for dataset, expected_hash in STUDENT_HASHES.items():
        path = args.student / (dataset + ".json")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["dataset"] == dataset and result["status"] == "complete"
        p = result["protocol"]
        assert p["student_steps"] == 2048 and p["student_batch"] == 24
        assert p["student_seeds"] == [150, 151, 152]
        assert not p["evaluation_performed"] and not p["evaluation_cache_resident"]
        assert not p["construction_in_timing"] and not p["text_encoder_resident"]
        assert p["warmup_updates"] == 2
        sources.append(dict(dataset=dataset, file=path.name, sha256=expected_hash))
        groups = {}
        for condition in ("memory_5pct", "full_raw_graph", "full_raw_edgefree"):
            rows = [r for r in result["rows"] if r["condition"] == condition]
            assert len(rows) == 3
            assert sorted(r["student_seed"] for r in rows) == p["student_seeds"]
            groups[condition] = {r["student_seed"]: r for r in rows}
            row = dict(dataset=dataset, condition=condition, student_seeds=3,
                       nodes=rows[0]["nodes"], edges=rows[0]["edges"])
            for metric in ("peak_allocated_mb", "seconds"):
                values = [r[metric] for r in rows]
                row[metric + "_mean"] = statistics.mean(values)
                row[metric + "_sd"] = statistics.stdev(values)
            summary.append(row)
        for reference in ("full_raw_graph", "full_raw_edgefree"):
            for metric in ("peak_allocated_mb", "seconds"):
                values = [100 * (1 - groups["memory_5pct"][s][metric] /
                                 groups[reference][s][metric]) for s in p["student_seeds"]]
                pairs.append(dict(dataset=dataset, reference=reference, metric=metric,
                                  reduction_pct=statistics.mean(values),
                                  reduction_sd=statistics.stdev(values),
                                  minimum_reduction_pct=min(values)))
    write_csv(args.output / "student_resources.csv", summary)
    write_csv(args.output / "student_reductions.csv", pairs)

    stages, geometry = [], []
    for dataset in ("arxiv", "sportsfit"):
        all_dimensions = []
        for seed in (271006, 271108, 271109):
            path = args.reused / f"{dataset}_combined_c{seed}/results.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            assert result["status"] == "complete"
            p = result["protocol"]
            assert p["config"]["assignment_storage"] == "class_blocks"
            total_rows = sum(p["stage_raw_rows"])
            total_entries = sum(r["audit"]["assignment_parameter_entries"] for r in result["rows"])
            dimensions = []
            seen_classes = set()
            for r in result["rows"]:
                costs = r["audit"]["q_cost"]["class_costs"]
                classes = {c["class"] for c in costs}
                assert not classes.intersection(seen_classes)
                seen_classes.update(classes)
                entries = sum(c["raw_rows"] * c["slots"] for c in costs)
                assert entries == r["audit"]["assignment_parameter_entries"]
                largest = max(costs, key=lambda c: c["raw_rows"] * c["slots"])
                dense = r["raw_rows"] * r["new_rows"]
                dimensions.append((r["stage"], r["raw_rows"], r["new_rows"], entries))
                stages.append(dict(dataset=dataset, construction_seed=seed, stage=r["stage"],
                    raw_rows=r["raw_rows"], new_slots=r["new_rows"], dense_entries=dense,
                    legal_entries=entries, stored_fraction=entries/dense,
                    raw_fraction_of_full=r["raw_rows"]/total_rows,
                    legal_fraction_of_full=entries/total_entries,
                    largest_class=largest["class"], largest_class_rows=largest["raw_rows"],
                    largest_class_slots=largest["slots"],
                    largest_class_entries=largest["raw_rows"]*largest["slots"],
                    peak_gib=r["peak_bytes"]/2**30))
            all_dimensions.append(dimensions)
        assert all(d == all_dimensions[0] for d in all_dimensions)
        d = all_dimensions[0]
        geometry.append(dict(dataset=dataset, stages=len(d), final_raw_rows=total_rows,
            final_slots=p["budgets"][-1], full_dense_entries=total_rows*p["budgets"][-1],
            full_legal_entries=total_entries, max_stage_legal_entries=max(r[3] for r in d),
            max_stage_legal_fraction=max(r[3] for r in d)/total_entries))
    write_csv(args.output / "assignment_stages.csv", stages)
    write_csv(args.output / "assignment_geometry.csv", geometry)
    receipt = dict(student_sources=sources, student_measurements=27,
                   assignment_runs=6, assignment_stages=len(stages),
                   note="MiB values use historical field names ending in _mb; student encoder is not resident.")
    (args.output / "support_verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(student_summary=summary, student_reductions=pairs, assignment_geometry=geometry), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--reused", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
