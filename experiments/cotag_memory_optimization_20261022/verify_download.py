"""Verify a newly extracted bundle, its analysis tables and the repeatability check."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze import independent_scores


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, analysis):
    remote_analysis = root / "results/cotag_memory_optimization_20261022_final"
    manifest_path = remote_analysis / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest:
        path = (root / entry["path"]).resolve()
        assert path.is_relative_to(root.resolve()), path
        assert path.stat().st_size == entry["bytes"], path
        assert digest(path) == entry["sha256"], path

    tables = list(remote_analysis.glob("*.csv"))
    assert len(tables) == 7
    exact_tables, norm_rounding = [], []
    for path in tables:
        local = analysis / path.name
        if path.read_bytes() == local.read_bytes():
            exact_tables.append(path.name)
            continue
        assert path.name == "memory_comparison.csv", path.name
        with path.open(newline="") as a, local.open(newline="") as b:
            remote_rows, local_rows = list(csv.DictReader(a)), list(csv.DictReader(b))
        assert len(remote_rows) == len(local_rows)
        for remote, local_row in zip(remote_rows, local_rows):
            assert remote.keys() == local_row.keys()
            for field in remote:
                if remote[field] == local_row[field]:
                    continue
                # Float32 CPU norm reductions can differ by a few ulps across hosts.
                assert field == "x_relative_l2", (field, remote, local_row)
                av, bv = float(remote[field]), float(local_row[field])
                assert np.isclose(av, bv, rtol=3e-7, atol=1e-12)
                norm_rounding.append(dict(dataset=remote["dataset"], variant=remote["variant"],
                    construction_seed=remote["construction_seed"], stage=remote["stage"],
                    remote=av, local=bv, absolute_difference=abs(av-bv)))

    runs = root / "results/cotag_memory_optimization_20261022_v2"
    replay_root = root / "results/cotag_memory_optimization_20261022_replay"
    replay = json.loads((replay_root / "results.json").read_text())
    assert replay["status"] == "complete"
    assert {r["variant"] for r in replay["rows"]} == {"original", "combined"}
    comparisons = []
    for record in replay["rows"]:
        variant = record["variant"]
        original_root = runs / f"sportsfit_{variant}_c271108"
        original = json.loads((original_root / "results.json").read_text())
        final = original["rows"][-1]
        assert record["memory_hash"] == final["memory_hash"]
        assert record["original_scores"] == final["scores"]
        scores = record["repeated_scores"]
        assert len(scores["per_student"]) == 3
        for new, old in zip(scores["per_student"], final["scores"]["per_student"]):
            assert old["seed"] == new["seed"]
            path = replay_root / variant / new["prediction_path"]
            assert digest(path) == new["prediction_sha256"]
            with np.load(path) as b, np.load(original_root / old["prediction_path"]) as a:
                for field in ("ids", "truth", "classes"):
                    assert np.array_equal(a[field], b[field]), field
                f1, acc = independent_scores(b["truth"], b["pred"], b["classes"])
                assert abs(f1 - new["f1"]) < 1e-9
                assert abs(acc - new["accuracy"]) < 1e-9
                agreement = float(np.mean(a["pred"] == b["pred"]))
            comparisons.append(dict(variant=variant, student_seed=new["seed"],
                f1_change=new["f1"] - old["f1"], accuracy_change=new["accuracy"] - old["accuracy"],
                prediction_agreement=agreement))
        for metric in ("f1", "accuracy"):
            assert abs(np.mean([r[metric] for r in scores["per_student"]]) - scores[metric]) < 1e-9

    receipt = dict(manifest_files=len(manifest), matching_analysis_tables=len(tables),
                   byte_identical_tables=sorted(exact_tables),
                   float32_norm_rounding=norm_rounding,
                   replay_predictions=len(comparisons), replay_metric_checks=2 * len(comparisons),
                   replay_comparisons=comparisons)
    output = analysis / "download_verification.json"
    if output.exists():
        assert json.loads(output.read_text()) == receipt
    else:
        output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    args = p.parse_args()
    verify(args.root, args.analysis)
