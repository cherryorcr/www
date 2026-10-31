"""Verify saved artifacts and predictions without importing experiment code."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata
import zipfile

import numpy as np
import torch

SEEDS = [271006, 271108, 271109]
STUDENTS = [150, 151, 152]
ARMS = ["cat_cctp", "gecc_hard", "uq_dm_edgefree"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def memory_hash(memory):
    digest = hashlib.sha256(json.dumps(memory["texts"], ensure_ascii=True).encode())
    for key in ("x", "text_emb", "labels", "mass", "edge_index"):
        digest.update(tensor_hash(memory[key]).encode())
    return digest.hexdigest()


def near(actual, expected):
    assert math.isfinite(float(actual))
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), (actual, expected)


def metrics(labels, predictions, classes):
    """Fixed-taxonomy macro F1, computed from a confusion matrix, not sklearn."""
    if not len(labels):
        return None
    counts = np.bincount(classes * labels + predictions,
                         minlength=classes * classes).reshape(classes, classes)
    true_positive = np.diag(counts).astype(float)
    denominator = counts.sum(0) + counts.sum(1)
    per_class = np.divide(2 * true_positive, denominator,
                          out=np.zeros(classes), where=denominator != 0)
    return dict(nodes=len(labels), f1=float(100 * per_class.mean()),
                accuracy=float(100 * true_positive.sum() / len(labels)))


def canonical(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize(
        "NFKC", html.unescape(str(text))).casefold()))


def raw_records(archive_path, dataset):
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            with archive.open(member) as source:
                stream = io.TextIOWrapper(source, encoding="utf-8")
                if dataset == "dblp":
                    for line in stream:
                        if line.strip():
                            yield json.loads(line)
                else:
                    record = {}
                    for line in stream:
                        line = line.rstrip("\r\n")
                        if line.startswith("#*"):
                            if record:
                                yield record
                            record = dict(title=line[2:], references=[])
                        elif record:
                            for tag, key in (("#index", "id"), ("#t", "year"),
                                             ("#c", "venue"), ("#!", "abstract")):
                                if line.startswith(tag):
                                    record[key] = line[len(tag):]
                                    break
                            if line.startswith("#%"):
                                record["references"].append(line[2:])
                    if record:
                        yield record


def verify_raw(dataset, folder, records, manifest, edges):
    archive = folder.parent / "raw" / Path(manifest["source_url"]).name
    assert archive.stat().st_size == manifest["archive_bytes"]
    assert sha(archive) == manifest["archive_sha256"]
    title_index = {canonical(r["title"]): i for i, r in enumerate(records)}
    assert len(title_index) == len(records)
    found, aliases, references = set(), {}, [set() for _ in records]
    total = 0
    for raw in raw_records(archive, dataset):
        total += 1
        index = title_index.get(canonical(raw.get("title", "")))
        if index is None:
            continue
        saved = records[index]
        try:
            year = int(raw.get("year", ""))
        except (ValueError, TypeError):
            continue
        venue = " " + canonical(raw.get("venue", "")) + " "
        labels = [label for label, rules in manifest["venue_rules"].items()
                  if any(" " + canonical(rule) + " " in venue for rule in rules)]
        if year != saved["year"] or labels != [saved["label"]]:
            continue
        source_id = str(raw.get("id", ""))
        assert source_id
        aliases[source_id] = index
        references[index].update(map(str, raw.get("references", [])))
        if source_id == saved["id"]:
            assert html.unescape(str(raw["title"])).strip() == saved["title"]
            assert html.unescape(str(raw.get("abstract", ""))).strip() == saved["abstract"]
            assert str(raw.get("venue", "")) == saved["venue"]
            found.add(index)
    assert len(found) == len(records)
    counts, directed = Counter(), set()
    for origin, refs in enumerate(references):
        for reference in refs:
            target = aliases.get(reference)
            if target is None:
                counts["references_outside_retained_nodes"] += 1
            elif origin == target:
                counts["self_references_removed"] += 1
            elif records[target]["year"] > records[origin]["year"]:
                counts["forward_year_references_removed"] += 1
            else:
                directed.add((origin, target))
    assert counts == Counter(manifest["edge_filter_counts"])
    assert len(directed) == manifest["directed_citations"]
    assert sorted(directed | {(b, a) for a, b in directed}) == edges
    return dict(source_records_scanned=total, selected_primary_records_matched=len(found),
                selected_alias_records=len(aliases), source_fields_and_citations_match=True,
                archive_sha256=manifest["archive_sha256"],
                scope="Independent parsing of selected source records and their citations; not an independent census of all eligibility exclusions.")


def verify_data(root, dataset, project):
    folder = root / "data" / dataset
    manifest, encoding = read(folder / "manifest.json"), read(folder / "encoding.json")
    with (folder / "records.jsonl").open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    edges = [tuple(e) for e in read(folder / "edges.json")]
    assert sha(folder / "records.jsonl") == manifest["records_sha256"]
    assert sha(folder / "edges.json") == manifest["edges_sha256"]
    assert sha(folder / "manifest.json") == encoding["dataset_manifest_sha256"]
    assert sha(folder / "encoded.pt") == encoding["encoded_data_sha256"]
    code = project / "experiments/cotag_temporal_bibliographic_20261022"
    assert sha(code / "prepare_data.py") == manifest["preparation_code_sha256"]
    assert sha(code / "encode_data.py") == encoding["encoding_code_sha256"]
    for filename, expected in encoding["encoder_hashes"].items():
        assert sha(Path(encoding["encoder_path"]) / filename) == expected
    raw = torch.load(folder / "encoded.pt", map_location="cpu", weights_only=False)
    hashes = torch.tensor([int(hashlib.sha256(f"{dataset}:{r['id']}".encode()).hexdigest()[:15], 16)
                           for r in records], dtype=torch.long)
    assert len(set(hashes.tolist())) == len(records) == manifest["nodes"]
    assert hashes.tolist() == [r["split_hash_id"] for r in records]
    pool = ~torch.isin(hashes % 10, torch.tensor([1, 2, 3]))
    years = torch.tensor([r["year"] for r in records])
    training_years = sorted(years[pool].tolist())
    bounds = [training_years[math.ceil(len(training_years) * q) - 1]
              for q in (.4, .55, .7, .85, 1.)]
    assert bounds == manifest["boundary_years"] and len(set(bounds)) == 5
    arrival = torch.tensor([next(s for s, end in enumerate(bounds) if year <= end)
                            for year in years.tolist()])
    assert arrival.tolist() == [r["stage"] for r in records]
    expected = dict(original_ids=hashes, node_year=years, arrival=arrival,
                    y=torch.tensor([manifest["classes"].index(r["label"]) for r in records]),
                    edge_index=torch.tensor(edges).T.contiguous())
    for key, value in expected.items():
        assert torch.equal(raw[key], value), key
    assert raw["names"] == manifest["classes"]
    assert raw["source_ids"] == [r["id"] for r in records]
    assert raw["texts"] == [r["text"] for r in records]
    assert raw["prompts"] == [f"a research paper about {name}" for name in manifest["classes"]]
    for row in records:
        assert row["text"] == "Title: " + row["title"] + (
            ". Abstract: " + row["abstract"] if row["abstract"] else "")
    assert tuple(raw["x"].shape) == (len(records), 384)
    assert bool(torch.isfinite(raw["x"]).all())
    stage_ids = [torch.where(pool & (arrival == stage))[0] for stage in range(5)]
    counts = [len(ids) for ids in stage_ids]
    budgets = [round(.05 * sum(counts[:s + 1])) for s in range(5)]
    masks = dict(report_dev=hashes % 10 == 3, test=hashes % 10 == 2)
    source_check = verify_raw(dataset, folder, records, manifest, edges)
    report = dict(nodes=len(records), classes=manifest["classes"], boundaries=bounds,
                  train_counts=counts, budgets=budgets, actual_final_rate=budgets[-1]/sum(counts),
                  source_check=source_check, source_hashes_match=True)
    print(json.dumps(dict(event="dataset_verified", dataset=dataset, **report)), flush=True)
    return dict(data=raw, records=records, pool=pool, ids=stage_ids, budgets=budgets,
                masks=masks, manifest=manifest, encoding=encoding), report


def verify_run(path, expected, project):
    result = read(path)
    protocol = result["protocol"]
    assert result["status"] == "complete"
    assert read(path.with_name("protocol.json")) == protocol
    assert protocol["student_seeds"] == STUDENTS
    assert protocol["student_steps"] == 2048 and protocol["student_batch"] == 24
    assert protocol["memory_rate"] == .05
    assert protocol["method_selection_from_test"] is False
    assert protocol["source"] == expected["manifest"] and protocol["encoding"] == expected["encoding"]
    assert protocol["budgets"] == expected["budgets"] and protocol["class_floor_adjustments"] == []
    assert protocol["stage_id_hashes"] == [tensor_hash(ids) for ids in expected["ids"]]
    assert protocol["pool_mask_hash"] == tensor_hash(expected["pool"])
    for key, mask in (("report_mask_hash", "report_dev"), ("test_mask_hash", "test")):
        assert protocol[key] == tensor_hash(expected["masks"][mask])
    code = project / "experiments/cotag_temporal_bibliographic_20261022"
    prior = project / "experiments/cotag_fidelity_temporal_20261021"
    for field, source in (("runner_hash", code / "run_temporal.py"),
                          ("runtime_hash", code / "runtime.py"),
                          ("registration_sha256", code / "REGISTRATION.md"),
                          ("precision_code_sha256", prior / "solver_precision.py")):
        assert protocol[field] == sha(source), field
    source_paths = {}
    for name, recorded_hash in protocol["source_hashes"].items():
        candidates = [project / "experiments/uq_final_evidence_20261012/code" / name,
                      project / "code/v2_llm_fusion" / name, prior / name]
        matches = [p for p in candidates if p.is_file() and sha(p) == recorded_hash]
        assert matches, name
        source_paths[name] = str(matches[0])
    assert protocol["q_config"]["steps"] == 30
    assert protocol["q_config"]["sinkhorn_max_steps"] == 16384
    assert protocol["q_config"]["sinkhorn_tolerance"] == 1e-4
    assert [row["stage"] for row in result["rows"]] == list(range(5))
    parent, metric_count = None, 0
    data, pools = expected["data"], expected["ids"]
    arrival, classes = data["arrival"], len(data["names"])
    for row in result["rows"]:
        stage = row["stage"]
        folder = path.parent / f"s{stage}"
        construction, scores = read(folder / "construction.json"), read(folder / "scores.json")
        assert construction == row["construction"] and scores == row["scores"]
        with gzip.open(folder / "memory.pt.gz", "rb") as stream:
            memory = torch.load(stream, map_location="cpu", weights_only=False)
        assert memory_hash(memory) == construction["memory_hash"] == scores["memory_hash"]
        budget = expected["budgets"][stage]
        for key in ("x", "text_emb", "labels", "mass", "texts", "stage", "depth"):
            assert len(memory[key]) == budget
        assert memory["edge_index"].shape == (2, 0)
        for key in ("x", "text_emb", "mass"):
            assert bool(torch.isfinite(memory[key]).all())
        keep = 0 if parent is None else len(parent["x"])
        if parent is not None:
            for key in ("x", "text_emb", "labels", "mass", "stage", "depth"):
                assert torch.equal(parent[key], memory[key][:keep]), (stage, key)
            assert parent["texts"] == memory["texts"][:keep]
        assert bool((memory["stage"][keep:] == stage).all())
        assert bool(torch.isin(memory["labels"][keep:], torch.unique(data["y"][pools[stage]])).all())
        assert construction["access"] == dict(current_stage_only=True, source_rows=len(pools[stage]),
                                              historical_raw_rows=0, future_raw_rows=0, holdout_rows=0)
        if protocol["arm"] == "uq_dm_edgefree":
            audit = construction["audit"]
            assert audit["q_update_count"] == audit["residual_update_count"] == 30
            assert construction["checks"]["sinkhorn_feasible"]
            assert set(audit["q_cost"]["prototype_source_ids"]) <= set(pools[stage].tolist())
        archive = folder / "predictions.pt.gz"
        assert sha(archive) == scores["prediction_archive_sha256"]
        with gzip.open(archive, "rb") as stream:
            predictions = torch.load(stream, map_location="cpu", weights_only=False)
        assert sorted(predictions) == STUDENTS
        for mode, splits in scores["prediction"].items():
            indices = torch.where(arrival <= stage if mode == "arrived_snapshot" else arrival == stage)[0]
            for seed in STUDENTS:
                saved = predictions[seed][mode]
                assert torch.equal(saved["ids"], indices)
                assert len(saved["pred"]) == len(indices)
                assert bool(((saved["pred"] >= 0) & (saved["pred"] < classes)).all())
            for split, aggregate in splits.items():
                assert [r["seed"] for r in aggregate["per_student"]] == STUDENTS
                for cohort in ("cumulative", "current", "historical"):
                    cohort_mask = {"cumulative": arrival <= stage, "current": arrival == stage,
                                   "historical": arrival < stage}[cohort]
                    local = (cohort_mask & expected["masks"][split])[indices]
                    recomputed = []
                    for per_student in aggregate["per_student"]:
                        prediction = predictions[per_student["seed"]][mode]["pred"]
                        value = metrics(data["y"][indices][local].numpy(), prediction[local].numpy(), classes)
                        if value is None:
                            assert per_student[cohort] is None
                        else:
                            assert per_student[cohort]["nodes"] == value["nodes"]
                            for key in ("f1", "accuracy"):
                                near(per_student[cohort][key], value[key])
                                metric_count += 1
                            recomputed.append(value)
                    if not recomputed:
                        assert aggregate[cohort] is None
                    else:
                        assert aggregate[cohort]["nodes"] == int(local.sum())
                        for key in ("f1", "accuracy"):
                            values = [v[key] for v in recomputed]
                            near(aggregate[cohort][key], statistics.mean(values))
                            near(aggregate[cohort][key + "_sd"], statistics.stdev(values))
                    if mode == "current_snapshot":
                        assert aggregate["historical"] is None
                        assert aggregate["current"] == aggregate["cumulative"]
        parent = memory
    return dict(path=str(path.relative_to(project)), passed=True, stages=5,
                independently_recomputed_metric_scalars=metric_count, source_paths=source_paths,
                prefix_and_hashes_match=True, fixed_budget_match=True,
                future_inference_nodes=0, checkpoints_finite=True)


def main(root, output):
    project = root.parent.parent
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  verifier_sha256=sha(Path(__file__)), datasets={}, runs=[])
    datasets = {}
    for name in ("acm", "dblp"):
        datasets[name], report["datasets"][name] = verify_data(root, name, project)
    title_sets = [{canonical(r["title"]) for r in datasets[name]["records"]} for name in ("acm", "dblp")]
    assert not title_sets[0] & title_sets[1]
    report["cross_dataset_exact_normalized_title_overlap"] = 0
    expected_paths = {root / "runs" / dataset / f"{arm}_c{seed}" / "results.json"
                      for dataset in datasets for arm in ARMS for seed in SEEDS}
    assert set((root / "runs").rglob("results.json")) == expected_paths
    for path in sorted(expected_paths):
        dataset = path.parents[1].name
        try:
            row = verify_run(path, datasets[dataset], project)
        except Exception as error:
            import traceback
            row = dict(path=str(path), passed=False, error=repr(error), traceback=traceback.format_exc())
        report["runs"].append(row)
        print(json.dumps(dict(event="run_verified", path=str(path.parent), passed=row["passed"])), flush=True)
    report["passed"] = all(row["passed"] for row in report["runs"])
    report["scope"] = ("All saved memories, prefixes, prediction metrics, protocol/source hashes and selected raw source records. "
                       "Does not instrument historical process access or independently re-encode text embeddings.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=report["passed"], output=str(output), runs=len(report["runs"]))))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    main(args.root, args.output)
