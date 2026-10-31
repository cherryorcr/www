"""Construction-seed confirmation for matched external adapters.

This runner fills the one remaining fairness gap in the pre-submission bundle:
CaT-CCTP and GECC-hard are reconstructed with the same three construction
seeds used by Unified-Q, under the current-delta-only, edge-free protocol.
Only compact JSON metrics are written; synthetic tensors stay in RAM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

import run_student_method_factorial_20261010 as F
import run_unified_q_attribution_20261008 as A


ARMS = ("cat_cctp", "gecc_hard")
SEEDS = (271006, 271108, 271109)
STUDENT_SEEDS = (150, 151, 152)
STEPS = 2048
BATCH = 24


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")
    temp.replace(path)


def load_protocol(root: Path, dataset: str, order: int) -> tuple[dict, Path]:
    source = root / "results" / "unified_q_attribution_20261008" / f"{dataset}_o{order}"
    payload = json.loads((source / "results.json").read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete source: {source}")
    return payload, source


def construct(data: dict, ids: list[torch.Tensor], budgets: list[int],
              arm: str, seed: int, device: str) -> tuple[list[dict], list[dict]]:
    parent = None
    prefixes = []
    audits = []
    for stage, raw_ids in enumerate(ids):
        delta = A.L.delta_view(data, raw_ids, stage)
        old_rows = 0 if parent is None else len(parent["x"])
        new_rows = int(budgets[stage]) - old_rows
        if new_rows <= 0 or new_rows > len(raw_ids):
            raise AssertionError(f"invalid append budget at stage {stage}: {new_rows}")
        started = time.perf_counter()
        if arm == "cat_cctp":
            block, audit = A.R.make_baseline(
                delta, new_rows, arm, data["encoder"], seed, device)
        elif arm == "gecc_hard":
            # GECC construction is deliberately run on CPU so the hard
            # clustering path is deterministic across GPU lanes.
            block, audit = A.M.gecc_block(
                delta, new_rows, data["encoder"],
                A.M.Config(seed=seed, cat_steps=3000,
                           text_sources=5, text_mode="nearest"), "cpu")
            block["labels"] = A.M.row_labels(audit["allocation"])
        else:
            raise ValueError(arm)
        if "text_source_ids" not in block:
            block["text_source_ids"] = [list(v) for v in audit["text_source_ids"]]
        parent = A.append_block(parent, block)
        if parent["edge_index"].numel():
            raise AssertionError("baseline acquired non-empty topology")
        audits.append({
            "stage": stage,
            "new_rows": int(new_rows),
            "memory_rows": int(len(parent["x"])),
            "construction_seconds": float(time.perf_counter() - started),
            "implementation": audit.get("implementation"),
        })
        prefixes.append(parent)
    return prefixes, audits


def run_cell(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    source_json, source = load_protocol(root, args.dataset, args.order)
    protocol = source_json["protocol"]
    cache = root / "results" / "public_baselines_text_cache_20261001" / args.dataset
    manifest = {"groups": protocol["groups"],
                "report_dev_mod": protocol.get("report_dev_mod", 3)}
    data = A.L.load_large(args.dataset, args.minilm, cache, args.device, manifest)
    loaded = F.validate_loaded_protocol(data, protocol)
    ids = A.L.stage_ids(data)
    budgets = [int(v) for v in protocol["budgets"]]
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            prefixes, audits = construct(data, ids, budgets, arm, seed, args.device)
            stage_scores = []
            for stage, prefix in enumerate(prefixes):
                score = A.L.score(prefix, data, stage, list(STUDENT_SEEDS),
                                  args.device, STEPS, BATCH)
                stage_scores.append(score)
            final = stage_scores[-1]
            rows.append({
                "dataset": args.dataset,
                "order_seed": int(args.order),
                "construction_seed": int(seed),
                "arm": arm,
                "method_class": "current-delta-only external adapter",
                "memory_rows": int(len(prefixes[-1]["x"])),
                "memory_hash": A.V.memory_hash(prefixes[-1]),
                "final": final,
                "stage_scores": stage_scores,
                "construction_audits": audits,
                "historical_raw_reads_by_constructor": 0,
                "inference_operator": protocol.get("inference_operator"),
                "loaded_data_hashes": loaded,
            })
            del prefixes
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    return {
        "status": "complete",
        "protocol": {
            "name": "matched-external-construction-seed-confirmation-20261010",
            "dataset": args.dataset,
            "order_seed": int(args.order),
            "construction_seeds": list(SEEDS),
            "student_seeds": list(STUDENT_SEEDS),
            "student_steps": STEPS,
            "student_batch": BATCH,
            "evaluation_split": "report-dev; not sealed test",
            "access": protocol.get("access"),
            "topology": "edge-free/self-loop operator",
            "source_results": str(source / "results.json"),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("wikics", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--order", type=int, choices=(9101, 9102, 9103), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = run_cell(args)
    write_json(args.output, result)
    print(json.dumps({"event": "complete", "dataset": args.dataset,
                      "order": args.order, "output": str(args.output),
                      "rows": len(result["rows"])}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
