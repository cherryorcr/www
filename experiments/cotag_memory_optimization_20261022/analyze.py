"""Verify saved artifacts and produce construction-level paired resource tables."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
import torch


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_memory(path):
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def memory_hash(memory):
    h = hashlib.sha256(json.dumps(memory["texts"], ensure_ascii=True).encode())
    for key in ("x", "text_emb", "labels", "mass", "edge_index"):
        data = memory[key].detach().cpu().contiguous().numpy().tobytes()
        h.update(hashlib.sha256(data).hexdigest().encode())
    return h.hexdigest()


def independent_scores(truth, pred, classes):
    f1 = []
    for cls in classes:
        tp = int(((truth == cls) & (pred == cls)).sum())
        fp = int(((truth != cls) & (pred == cls)).sum())
        fn = int(((truth == cls) & (pred != cls)).sum())
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return 100 * np.mean(f1), 100 * np.mean(truth == pred)


def stats(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=mean, sd=sd, ci_low=None if half is None else mean-half,
                ci_high=None if half is None else mean+half,
                positive=int((values > 0).sum()), worst=float(values.min()))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        w = csv.DictWriter(stream, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    stages, runs, pilot, evaluations, comparisons = [], [], [], [], []
    receipts = dict(memories=0, predictions=0, independent_metric_checks=0, prefixes=0)
    records = {}
    for path in sorted(args.input.glob("*/results.json")):
        result = json.loads(path.read_text())
        p = result["protocol"]
        key = (p["dataset"], p["variant"], p["construction_seed"])
        records[key] = (path, result)
        base = dict(dataset=key[0], variant=key[1], construction_seed=key[2])
        previous = None
        for row in result["rows"]:
            file = path.parent / row["memory_path"]
            assert digest(file) == row["memory_file_sha256"], file
            memory = load_memory(file)
            assert memory_hash(memory) == row["memory_hash"], file
            assert len(memory["x"]) == p["budgets"][row["stage"]]
            assert memory["edge_index"].numel() == 0
            for field in ("x", "text_emb", "mass"):
                assert torch.isfinite(memory[field]).all()
            if previous is not None:
                n = len(previous["x"])
                for field in ("x", "text_emb", "mass", "labels"):
                    assert torch.equal(memory[field][:n], previous[field])
                assert memory["texts"][:n] == previous["texts"]
                if "text_source_ids" in memory and "text_source_ids" in previous:
                    assert memory["text_source_ids"][:n] == previous["text_source_ids"]
                receipts["prefixes"] += 1
            previous = memory
            receipts["memories"] += 1
            entry = dict(**base, stage=row["stage"], raw_rows=row["raw_rows"], new_rows=row["new_rows"],
                seconds=row["construction_seconds"], peak_gib=row["peak_bytes"]/2**30,
                reserved_gib=row["reserved_bytes"]/2**30, entry_gib=row["entry_bytes"]/2**30,
                cpu_rss_gib=row["cpu_peak_rss_bytes"]/2**30,
                parameter_entries=row["audit"]["assignment_parameter_entries"],
                max_sinkhorn_steps=row["audit"]["sinkhorn_max_iterations"],
                final_loss=row["audit"]["trace"][-1]["loss"],
                text_cosine=row["audit"]["text_projection"]["hard_soft_cosine"],
                f1=None if row["scores"] is None else row["scores"]["f1"],
                accuracy=None if row["scores"] is None else row["scores"]["accuracy"])
            costs = row["audit"]["q_cost"]["class_costs"]
            expected_entries = (sum(c["raw_rows"] * c["slots"] for c in costs)
                                if key[1] in ("class_blocks", "combined") else row["raw_rows"] * row["new_rows"])
            assert entry["parameter_entries"] == expected_entries
            stages.append(entry)
            if row["stage"] == 0 and key[2] == 271006:
                pilot.append(entry)
            if row["scores"]:
                f1s, accuracies = [], []
                for score in row["scores"]["per_student"]:
                    pred_file = path.parent / score["prediction_path"]
                    assert digest(pred_file) == score["prediction_sha256"]
                    with np.load(pred_file) as a:
                        f1, acc = independent_scores(a["truth"], a["pred"], a["classes"])
                    assert abs(f1 - score["f1"]) < 1e-9
                    assert abs(acc - score["accuracy"]) < 1e-9
                    receipts["predictions"] += 1
                    receipts["independent_metric_checks"] += 2
                    f1s.append(f1)
                    accuracies.append(acc)
                    evaluations.append(dict(**base, stage=row["stage"], student_seed=score["seed"],
                        f1=f1, accuracy=acc, train_seconds=score["train_seconds"],
                        train_peak_gib=score["train_peak_bytes"]/2**30,
                        inference_seconds=score["inference_seconds"],
                        inference_peak_gib=score["inference_peak_bytes"]/2**30,
                        cache_rebuild_seconds=score["cache_rebuild_seconds"]))
                assert abs(np.mean(f1s) - row["scores"]["f1"]) < 1e-9
                assert abs(np.mean(accuracies) - row["scores"]["accuracy"]) < 1e-9
        if result["status"] == "complete":
            assert len(result["rows"]) == len(p["groups"])
            final = result["rows"][-1]
            runs.append(dict(**base, stages=len(p["groups"]), f1=final["scores"]["f1"],
                accuracy=final["scores"]["accuracy"],
                peak_gib=max(r["peak_bytes"] for r in result["rows"])/2**30,
                reserved_gib=max(r["reserved_bytes"] for r in result["rows"])/2**30,
                seconds=sum(r["construction_seconds"] for r in result["rows"])))

    for (dataset, variant, seed), (path, result) in records.items():
        if variant == "original" or (dataset, "original", seed) not in records:
            continue
        old_path, old_result = records[(dataset, "original", seed)]
        for field in ("groups", "budgets", "stage_raw_rows", "pool_hash", "report_hash",
                      "stage_hashes", "student_steps", "student_batch", "student_seeds"):
            assert result["protocol"][field] == old_result["protocol"][field], field
        a_cfg, b_cfg = dict(result["protocol"]["config"]), dict(old_result["protocol"]["config"])
        a_cfg.pop("assignment_storage")
        b_cfg.pop("assignment_storage")
        assert a_cfg == b_cfg
        for row, old in zip(result["rows"], old_result["rows"]):
            a, b = load_memory(old_path.parent / old["memory_path"]), load_memory(path.parent / row["memory_path"])
            assert torch.equal(a["labels"], b["labels"])
            n = len(a["x"])
            def sources(record, stage):
                return [(r["stage"], local) for r in record["rows"] if r["stage"] <= stage
                        for local in r["audit"]["text_projection"]["selected_rows"]]
            a_sources = sources(old_result, old["stage"])
            b_sources = sources(result, row["stage"])
            assert len(a_sources) == len(b_sources) == n
            comparisons.append(dict(dataset=dataset, variant=variant, construction_seed=seed, stage=row["stage"],
                x_max_abs=float((a["x"] - b["x"]).abs().max()),
                x_relative_l2=float((a["x"] - b["x"]).norm() / a["x"].norm().clamp_min(1e-12)),
                text_source_agreement=sum(x == y for x, y in zip(a_sources, b_sources))/n,
                text_string_agreement=sum(x == y for x, y in zip(a["texts"], b["texts"]))/n,
                exact_memory=memory_hash(a) == memory_hash(b)))

    aggregates, paired = [], []
    for dataset in sorted(set(r["dataset"] for r in runs)):
        for variant in ("original", "combined"):
            chosen = [r for r in runs if r["dataset"] == dataset and r["variant"] == variant]
            if not chosen:
                continue
            item = dict(dataset=dataset, variant=variant, constructions=len(chosen))
            for metric in ("f1", "accuracy", "peak_gib", "seconds", "reserved_gib"):
                s = stats([r[metric] for r in chosen])
                item[metric + "_mean"], item[metric + "_sd"] = s["mean"], s["sd"]
            aggregates.append(item)
        originals = {r["construction_seed"]: r for r in runs if r["dataset"] == dataset and r["variant"] == "original"}
        combined = {r["construction_seed"]: r for r in runs if r["dataset"] == dataset and r["variant"] == "combined"}
        matched = sorted(originals.keys() & combined.keys())
        if matched:
            for metric in ("f1", "accuracy", "peak_gib", "seconds"):
                values = [combined[s][metric] - originals[s][metric] for s in matched]
                paired.append(dict(dataset=dataset, metric=metric, **stats(values)))
            for metric in ("peak_gib", "seconds"):
                values = [100 * (1-combined[s][metric]/originals[s][metric]) for s in matched]
                paired.append(dict(dataset=dataset, metric=metric + "_reduction_pct", **stats(values)))
    for name, rows in (("stages", stages), ("pilot", pilot), ("runs", runs), ("summary", aggregates),
                       ("paired", paired), ("memory_comparison", comparisons), ("evaluation", evaluations)):
        write_csv(args.output / f"{name}.csv", rows)
    receipts["complete_streams"] = len(runs)
    receipts["statuses"] = [{"path": str(path), "status": r["status"]} for path, r in records.values()]
    if args.require_complete:
        expected = {(d, v, s) for d in ("arxiv", "sportsfit")
                    for v in ("original", "combined") for s in (271006, 271108, 271109)}
        expected |= {(d, v, 271006) for d in ("arxiv", "sportsfit")
                     for v in ("cache_offload", "class_blocks")}
        assert set(records) == expected
        assert receipts["memories"] == 52 and receipts["predictions"] == 84
        assert receipts["complete_streams"] == 12 and receipts["prefixes"] == 36
        for (_, variant, _), (_, result) in records.items():
            assert result["status"] == ("complete" if variant in ("original", "combined") else "pilot_complete")
    (args.output / "verification.json").write_text(json.dumps(receipts, indent=2))
    lines = ["# Construction memory optimization", "", "Report-development diagnostic. Three constructions; three students averaged per memory.",
             "ArXiv uses the archived three-stage stream; SportsFit uses five stages.", "",
             "## Four-way stage-zero pilot", "", "| Dataset | Implementation | Peak GiB | Time s | Entry GiB | Parameters | F1 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in pilot:
        lines.append(f"| {r['dataset']} | {r['variant']} | {r['peak_gib']:.3f} | {r['seconds']:.2f} | {r['entry_gib']:.3f} | {r['parameter_entries']:,} | {r['f1']:.3f} |")
    lines += ["", "## Complete streams", "", "| Dataset | Implementation | Constructions | F1 mean +/- SD | Accuracy mean +/- SD | Peak GiB | Construction s |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in aggregates:
        sd = lambda x: "NA" if x is None else f"{x:.3f}"
        lines.append(f"| {r['dataset']} | {r['variant']} | {r['constructions']} | {r['f1_mean']:.3f} +/- {sd(r['f1_sd'])} | {r['accuracy_mean']:.3f} +/- {sd(r['accuracy_sd'])} | {r['peak_gib_mean']:.3f} | {r['seconds_mean']:.2f} |")
    lines += ["", "## Paired contrasts", "", "Combined minus original; reductions use original as denominator. CI: two-sided 95% t, construction as unit.", "", "| Dataset | Metric | n | Mean | CI |", "|---|---|---:|---:|---|"]
    for r in paired:
        ci = "unavailable" if r["ci_low"] is None else f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['n']} | {r['mean']:.3f} | {ci} |")
    lines += ["", "## Interpretation boundaries", "", "- Absolute allocated peak includes the resident encoder; reserved peak is separately reported.",
              "- Construction time excludes evaluation-cache rebuild and student training; those are in evaluation.csv.",
              "- Times from two concurrent GPU lanes are descriptive. CPU RSS is sampled every 20 ms.",
              "- Losses and GM remain unchanged. Small GPU rounding differences can change hard text choices.",
              "- Existing CaT/GECC results were not rerun; these tables do not establish lowest construction cost among methods.",
              "- No sealed-test outcomes were used; no significance/equivalence claim from a single pilot.",
              "", f"Verified: {receipts['memories']} memories, {receipts['prefixes']} prefixes, {receipts['predictions']} predictions, {receipts['independent_metric_checks']} independently recomputed scores."]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-complete", action="store_true")
    main(p.parse_args())
