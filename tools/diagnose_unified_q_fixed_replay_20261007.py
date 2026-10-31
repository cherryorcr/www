"""Inspect replay drift without changing, deleting or selecting any runs."""
import argparse
import json
from pathlib import Path
import torch


def compare(root, dataset):
    pilot = root / "results/unified_q_inloop_cat_20261007" / dataset
    diagnostic = root / "results/unified_q_inloop_validation_20261007" / f"diagnostics_{dataset}"
    original = json.loads((pilot / "results.json").read_text())
    replay = json.loads((diagnostic / "results.json").read_text())
    reference_rows = {r["stage"]: r for r in original["rows"] if r["arm"] == "uq_dm_edgefree"}
    stage_comparison = []
    for row in replay["rows"]:
        if row["arm"] != "unchanged":
            continue
        stage_comparison.append(dict(stage=row["stage"],
            exact_memory_hash=row["memory_hash"] == reference_rows[row["stage"]]["memory_hash"],
            f1_delta=row["score"]["final_f1"]-reference_rows[row["stage"]]["score"]["final_f1"]))
    report = dict(dataset=dataset, stages=stage_comparison,
        source_hashes={key:[original["protocol"]["source_hashes"].get(key), value]
            for key,value in replay["protocol"]["source_hashes"].items()}, tensor_comparison=[])
    saved = diagnostic / "diagnostic_memory.pt"
    if not saved.exists():
        report["status"] = "diagnostic checkpoint pending"
        return report
    expected = torch.load(pilot / "latest_uq_dm_edgefree.pt", weights_only=False, map_location="cpu")["memory"]
    bundle = torch.load(saved, weights_only=False, map_location="cpu")["memories"]
    actual = bundle["unchanged"]
    for stage in sorted(set(actual["stage"].tolist())):
        mask = actual["stage"] == stage
        item = dict(stage=stage, rows=int(mask.sum()), tensors={})
        for key in ("x", "text_emb", "labels", "mass", "depth", "stage"):
            left,right = actual[key][mask],expected[key][mask]
            difference = left.double()-right.double()
            item["tensors"][key] = dict(exact=torch.equal(left,right),
                max_abs=float(difference.abs().max()),
                relative_norm=float(difference.norm()/right.double().norm().clamp_min(1e-12)))
        ids = torch.where(mask)[0].tolist()
        for key in ("texts", "text_source_ids", "support_ids"):
            item[key+"_mismatches"] = sum(actual[key][i] != expected[key][i] for i in ids)
        report["tensor_comparison"].append(item)
    report["within_run_content_fixed"] = {
        arm:all(torch.equal(memory[k],actual[k]) for k in ("x","labels","mass"))
        and (not arm.startswith("topology") or (
            torch.equal(memory["text_emb"],actual["text_emb"]) and memory["texts"] == actual["texts"]))
        for arm,memory in bundle.items()}
    report["status"] = "inspected"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    parser.add_argument("--datasets", nargs="+", default=["wikics", "bookhis", "sportsfit"])
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    results=[compare(args.root,d) for d in args.datasets]
    content=json.dumps(results,indent=2)
    if args.output:
        args.output.write_text(content)
    print(content)


if __name__ == "__main__":
    main()
