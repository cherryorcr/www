"""Read only legal stage pools before applying a sparse-class STAD adaptation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

import run_uq_final_confirmation_20261012 as E


def audit(args):
    if args.output.exists():
        raise FileExistsError("preserve prior audit; choose a new output")
    records = []
    for dataset in args.datasets:
        reference = E.R.L.read(args.experiment / f"{dataset}_reference.json")
        groups = E.R.V.choose_groups(reference, args.order_seed)
        data = E.R.L.load_large(dataset, args.minilm, args.cache_root / dataset, args.device,
            dict(groups=groups, report_dev_mod=reference.get("report_dev_mod", 3)))
        for stage, ids in enumerate(E.R.L.stage_ids(data)):
            y = data["y"][ids].long().cpu()
            classes, counts = torch.unique(y, return_counts=True)
            rare = [{"class": int(c), "nodes": int(n)} for c, n in zip(classes, counts) if int(n) < 3]
            records.append(dict(dataset=dataset, order_seed=args.order_seed, stage=stage,
                legal_rows=len(ids), minimum_class_size=int(counts.min()), classes=len(classes),
                class_counts={str(int(c)): int(n) for c, n in zip(classes, counts)},
                rare_classes=rare, rare_rows=sum(r["nodes"] for r in rare),
                pool_mask_hash=E.R.V.hash_tensor(data["pool_mask"]),
                stage_ids_hash=E.R.V.hash_tensor(ids),
                evaluation_nodes_used=False))
        del data
    result = dict(captured_at=datetime.now(timezone.utc).isoformat(), records=records,
        access="current legal stage pool only; no report-dev/test nodes", selection_from_scores=False)
    E.R.V.write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "records": len(records),
        "stages_with_rare_classes": [r for r in records if r["rare_classes"]]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["bookhis", "sportsfit", "arxiv"])
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, default=9101)
    parser.add_argument("--device", default="cpu")
    options = parser.parse_args()
    torch.set_num_threads(2)
    audit(options)
