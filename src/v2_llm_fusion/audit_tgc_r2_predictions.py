"""Independently reload downstream checkpoints and recompute report-dev metrics."""
import argparse
import json
from pathlib import Path

import torch

import run_tgc_revision_gate as R
from run_tgc_r2_capacity import CONFIGS


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--text-cache-root", default="/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919")
    args = ap.parse_args()
    torch.set_num_threads(2)
    memo = {}
    errors = []
    count = 0
    max_difference = 0.0
    for dataset in ("cora", "citeseer"):
        cache = Path(args.text_cache_root) / f"{dataset}_r005" / "text_emb_pool_only.pt"
        data = R.R.load_data(dataset, args.minilm, args.device, cache)
        for config in CONFIGS:
            for order in ("canonical", "order19101"):
                cell = args.root / f"{dataset}_{config}" / dataset / order
                manifest = read(cell / "manifest.json")
                R.A.set_stream(data, manifest["stage_classes"], .05)
                if R.C.tensor_hash({"report": data["report_dev_mask"]}) != manifest["report_mask_sha256"]:
                    raise ValueError("report mask changed")
                for stage in range(3):
                    for arm in ("c0", "tgc_r"):
                        for seed in manifest["student_seeds"]:
                            name = f"{arm}_s{stage}_seed{seed}"
                            row = read(cell / (name + ".json"))
                            checkpoint = cell / (name + ".pt")
                            checkpoint_hash = R.C.sha(checkpoint)
                            if row["checkpoint_sha256"] != checkpoint_hash:
                                raise ValueError(f"checkpoint damaged: {checkpoint}")
                            key = (dataset, order, stage, checkpoint_hash)
                            if key not in memo:
                                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                                model = R.G.CLIPStudent(data["x"].shape[1]).to(args.device)
                                model.load_state_dict(saved["state"])
                                memo[key] = R.C.evaluate(model, data, stage)
                                del model
                            scores = memo[key]
                            count += 1
                            for scope in ("all", "seen"):
                                for metric in ("cumulative_f1", "old_f1", "new_f1", "accuracy"):
                                    value, expected = scores[scope][metric], row["metrics"][scope][metric]
                                    if value is None and expected is None:
                                        continue
                                    diff = abs(value - expected) if value is not None and expected is not None else float("inf")
                                    max_difference = max(diff, max_difference)
                                    if diff > 1e-7:
                                        errors.append({"cell": str(cell), "fit": name, "scope": scope,
                                                       "metric": metric, "stored": expected,
                                                       "reloaded": value, "difference_pp": diff})
        print(json.dumps({"dataset": dataset, "records_checked": count,
                          "unique_checkpoint_evaluations": len(memo), "errors": len(errors)}), flush=True)
    result = {"passed": count == 1008 and not errors, "records_checked": count,
              "unique_checkpoint_evaluations": len(memo), "errors": errors,
              "max_metric_difference_pp": max_difference,
              "tolerance_pp": 1e-7, "test_labels_evaluated": False}
    R.save_json(args.output, result)
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
