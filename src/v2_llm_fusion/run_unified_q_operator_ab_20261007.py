"""Factor training objective and raw evaluation operator without rebuilding memory."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

import graphtext_prompt_student as P
import run_unified_q_large_smoke_20261007 as L
import run_unified_q_revision_20261007 as V
from run_unified_q_student_ab_20261007 import score_prediction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    source = json.loads((args.source / "results.json").read_text())
    protocol = source["protocol"]
    data = L.load_large(protocol["dataset"], args.minilm, args.text_cache, args.device, protocol)
    if V.hash_tensor(data["report_dev_mask"]) != protocol["report_mask_hash"]:
        raise AssertionError("report mask changed")
    adj = {"legacy": L.U.G.norm_adj(data["edge_index"], len(data["x"]), args.device),
           "common": data["real_adj"]}
    result = {"dataset": protocol["dataset"], "budget_mode": protocol["budget_mode"],
        "steps": 2048, "batch": 24, "seeds": [150, 151, 152], "source": str(args.source),
        "runner_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": [],
        "status": "running", "repeated_seed_is_not_an_independent_seed": True}
    for arm in ("cat_cctp", "uq_semantic"):
        saved = torch.load(args.source / f"latest_{arm}.pt", map_location="cpu", weights_only=False)
        memory, stage = saved["memory"], saved["stage"]
        archive = next(r for r in source["rows"] if r["arm"] == arm and r["stage"] == stage)
        if V.memory_hash(memory) != archive["memory_hash"]:
            raise AssertionError("memory hash differs")
        for student in ("legacy", "prompt"):
            for repeat, seed in ((0, 150), (0, 151), (0, 152), (1, 150)):
                model = (L.U.G.train_student(memory, seed, args.device, 2048, lr=0.001, batch=24)
                         if student == "legacy" else P.train_prompt_student(
                             memory, data["prompt_emb"], seed, args.device, 2048, batch=24))
                for operator, matrix in adj.items():
                    with torch.no_grad():
                        prediction = model.logits(data["x_device"], matrix,
                            data["prompt_emb"].to(args.device)).argmax(1).cpu()
                    metrics = score_prediction(prediction, data, stage)
                    result["rows"].append({"arm": arm, "student": student, "seed": seed,
                        "repeat": repeat, "operator": operator, "stage": stage,
                        "memory_hash": archive["memory_hash"], **metrics})
                V.write_json(args.output / "operator_ab.json", result)
                print(json.dumps({"event": "operator_ab", "arm": arm, "student": student,
                                  "seed": seed, "repeat": repeat}), flush=True)
                del model
    result["status"] = "complete"
    V.write_json(args.output / "operator_ab.json", result)


if __name__ == "__main__":
    main()
