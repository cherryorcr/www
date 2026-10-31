"""Same archived memory, common metrics, legacy versus Prompt Student."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import graphtext_prompt_student as P
import run_unified_q_revision_20261007 as V
import run_unified_q_large_smoke_20261007 as L


def prefix(memory: dict, rows: int) -> dict:
    result = {}
    total = len(memory["x"])
    for key, value in memory.items():
        if key == "edge_index":
            result[key] = value[:, (value < rows).all(0)].clone()
        elif isinstance(value, torch.Tensor) and value.ndim and len(value) == total:
            result[key] = value[:rows].clone()
        elif isinstance(value, list) and len(value) == total:
            result[key] = value[:rows]
        else:
            result[key] = value
    return result


def score_prediction(pred: torch.Tensor, data: dict, stage: int) -> dict:
    groups = data["stage_classes"]
    seen = [int(c) for g in groups[:stage + 1] for c in g]
    old = [int(c) for g in groups[:stage] for c in g]
    new = [int(c) for c in groups[stage]]
    y = data["y"].long().cpu()
    mask = data["report_dev_mask"].cpu() & torch.isin(y, torch.tensor(seen))
    yy, pp = y[mask].numpy(), pred.cpu()[mask].numpy()
    cls = f1_score(yy, pp, labels=seen, average=None, zero_division=0) * 100
    values = dict(zip(seen, cls))
    return {"final_f1": float(cls.mean()),
            "accuracy": float(100 * accuracy_score(yy, pp)),
            "old_f1": float(np.mean([values[c] for c in old])) if old else None,
            "new_f1": float(np.mean([values[c] for c in new])),
            "per_class_f1": {str(c): float(v) for c, v in values.items()},
            "report_nodes": int(mask.sum()),
            "prediction_hash": V.hash_tensor(pred.cpu()[mask])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    archived = json.loads((args.source / "results.json").read_text())
    protocol = archived["protocol"]
    data = L.load_large(protocol["dataset"], args.minilm, args.text_cache,
                        args.device, protocol)
    for key in ("report_dev_mask", "pool_mask"):
        expected = protocol["report_mask_hash" if key == "report_dev_mask" else "pool_mask_hash"]
        if V.hash_tensor(data[key]) != expected:
            raise AssertionError(f"changed {key}")
    if [V.hash_tensor(v) for v in L.stage_ids(data)] != protocol["stage_id_hashes"]:
        raise AssertionError("stage identity changed")
    legacy = L.U.G
    adjs = {"legacy": legacy.norm_adj(data["edge_index"], len(data["x"]), args.device),
            "prompt": data["real_adj"]}
    config = {"source_protocol": protocol, "seeds": [150, 151, 152], "steps": 2048,
              "batch": 24, "evaluation": "report-dev; all-prompt inference",
              "legacy_source_hash": hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest(),
              "prompt_source_hash": hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest(),
              "runner_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = args.output / "student_ab.json"
    result = json.loads(path.read_text()) if path.exists() else {
        "protocol": config, "rows": [], "status": "running"}
    if result["protocol"] != config:
        raise ValueError("A/B output protocol changed; choose a new directory")
    for arm in ("cat_cctp", "uq_semantic"):
        checkpoint = torch.load(args.source / f"latest_{arm}.pt", map_location="cpu", weights_only=False)
        final = checkpoint["memory"]
        for stage, budget in enumerate(protocol["budgets"]):
            memory = prefix(final, budget)
            source_row = next(r for r in archived["rows"] if r["arm"] == arm and r["stage"] == stage)
            if V.memory_hash(memory) != source_row["memory_hash"]:
                raise AssertionError("reconstructed prefix differs from archived memory")
            for student in ("legacy", "prompt"):
                for seed in config["seeds"]:
                    identity = (arm, stage, student, seed)
                    if any((r["arm"], r["stage"], r["student"], r["seed"]) == identity for r in result["rows"]):
                        continue
                    V.sync(args.device)
                    start = time.perf_counter()
                    model = (legacy.train_student(memory, seed, args.device, 2048, lr=0.001, batch=24)
                             if student == "legacy" else P.train_prompt_student(
                                 memory, data["prompt_emb"], seed, args.device, 2048, batch=24))
                    with torch.no_grad():
                        pred = model.logits(data["x_device"], adjs[student],
                                            data["prompt_emb"].to(args.device)).argmax(1).cpu()
                    V.sync(args.device)
                    metrics = score_prediction(pred, data, stage)
                    if student == "prompt":
                        old_score = next(v for v in source_row["score"]["per_student"] if v["seed"] == seed)
                        metrics["archived_f1_difference"] = metrics["final_f1"] - old_score["final_f1"]
                    result["rows"].append({"arm": arm, "stage": stage, "student": student,
                        "seed": seed, "memory_rows": budget, "memory_hash": V.memory_hash(memory),
                        "x_hash": V.hash_tensor(memory["x"]),
                        "text_hash": V.hash_tensor(memory["text_emb"]),
                        "edge_hash": V.hash_tensor(memory["edge_index"]),
                        "seconds": time.perf_counter() - start, **metrics})
                    V.write_json(path, result)
                    print(json.dumps({"event": "ab", "arm": arm, "stage": stage,
                                      "student": student, "seed": seed, **metrics}), flush=True)
                    del model
    result["status"] = "complete"
    V.write_json(path, result)


if __name__ == "__main__":
    main()
