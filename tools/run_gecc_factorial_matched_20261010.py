"""Rebuild only the GECC adapter with deterministic CPU propagation.

The factorial replay keeps the three frozen arms unchanged.  This helper
replaces only GECC in a copied cell so KMeans does not consume slightly
different GPU sparse-matmul values across runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_student_method_factorial_20261010 as F


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    source = root / "results" / "unified_q_attribution_20261008" / (
        f"{args.dataset}_o{args.order_seed}")
    base_path = Path(args.base_dir) / f"{args.dataset}_o{args.order_seed}.json"
    source_json = json.loads((source / "results.json").read_text(encoding="utf-8"))
    protocol = source_json["protocol"]
    cache = root / "results" / "public_baselines_text_cache_20261001" / args.dataset
    manifest = {"groups": protocol["groups"],
                "report_dev_mod": protocol.get("report_dev_mod", 3)}
    data = F.A.L.load_large(args.dataset, args.minilm, cache, args.device, manifest)
    loaded_hashes = F.validate_loaded_protocol(data, protocol)
    ids = F.A.L.stage_ids(data)
    budgets = [int(value) for value in protocol["budgets"]]
    gecc_memory, gecc_hashes, gecc_audits = F.construct_gecc(
        data, ids, budgets, int(protocol["construction_seed"]), "cpu")
    gecc_prompt = F.prompt_score(gecc_memory, data, len(budgets) - 1,
                                 args.device)
    gecc_legacy = F.legacy_score(gecc_memory, data, len(budgets) - 1,
                                 args.device)
    gecc_legacy["inference_operator"] = "shared data[real_adj]"

    base = json.loads(base_path.read_text(encoding="utf-8"))
    rows = [row for row in base["rows"] if row["arm"] != "gecc_hard"]
    rows.append({
        "dataset": args.dataset,
        "order_seed": args.order_seed,
        "arm": "gecc_hard",
        "method_class": "current-delta-only GECC adapter; deterministic CPU reconstruction",
        "stage": len(budgets) - 1,
        "memory_rows": len(gecc_memory["x"]),
        "memory_hash": F.A.V.memory_hash(gecc_memory),
        "student": gecc_prompt,
        "legacy": gecc_legacy,
    })
    output = Path(args.output_dir) / f"{args.dataset}_o{args.order_seed}.json"
    result = dict(base)
    result["rows"] = rows
    result["method_hashes"] = dict(base["method_hashes"])
    result["method_hashes"]["gecc_hard"] = {
        "final_memory_hash": F.A.V.memory_hash(gecc_memory),
        "stage_memory_hashes": gecc_hashes,
    }
    result["construction_audits"] = dict(base.get("construction_audits", {}))
    result["construction_audits"]["gecc_hard"] = gecc_audits
    result["protocol"] = dict(base["protocol"])
    result["protocol"]["loaded_data_hashes"] = loaded_hashes
    result["protocol"]["gecc_adapter"] = (
        "current delta only; sklearn hard Lloyd with matched class budget; "
        "CPU propagated features for deterministic reconstruction; self-loop memory")
    result["protocol"]["gecc_construction_device"] = "cpu"
    F.write_json(output, result)
    print(json.dumps({"event": "complete", "dataset": args.dataset,
                      "order": args.order_seed,
                      "memory_hash": result["method_hashes"]["gecc_hard"]["final_memory_hash"],
                      "output": str(output)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("wikics", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
