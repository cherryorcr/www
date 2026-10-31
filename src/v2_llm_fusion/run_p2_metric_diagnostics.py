"""Evaluate prompt-scope and cross-task false-positive accounting explicitly."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

import run_unified_v3_continual as U


def class_metrics(y, predictions, seen, old, new, classes):
    matrix = confusion_matrix(y, predictions, labels=list(range(classes)))
    tp = np.diag(matrix)
    denom = matrix.sum(0) + matrix.sum(1)
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(classes, dtype=float), where=denom > 0) * 100
    return {"confusion_matrix": matrix.tolist(), "per_class_f1": f1.tolist(),
            "cumulative_f1": float(f1[seen].mean()),
            "old_class_f1_global_support": float(f1[old].mean()) if old else None,
            "new_class_f1_global_support": float(f1[new].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--minilm", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    matrix, output = Path(args.matrix), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    first = json.loads((matrix / "concat_t64_g5_r0.5_a0.5/results.json").read_text())
    ref = json.loads((Path(first["reference"]) / "results.json").read_text())
    D = U.load_unified(ref["dataset"], ref["data"], args.minilm, args.device,
        n_stages=ref["n_stages"], split_mode=ref["split_mode"],
        base_fraction=ref["base_fraction"], report_dev_mod=ref["report_dev_mod"])
    prompt = D["encoder"].encode(D["prompts"]).to(args.device)
    adj = U.G.norm_adj(D["edge_index"], len(D["x"]), args.device)
    x = D["x"].to(args.device)
    for operator in ("concat", "mean_pool", "set_pool"):
        source = matrix / f"{operator}_t64_g5_r0.5_a0.5"
        result_file = output / f"{operator}.json"
        if result_file.exists():
            continue
        original = json.loads((source / "results.json").read_text())
        rows = []
        for row in original["results"]:
            stage = row["stage"]
            memory_file = source / f"append_fixed_s{stage}_memory.pt"
            memory = torch.load(memory_file, map_location="cpu", weights_only=False)
            seen = sum(D["stage_classes"][:stage + 1], [])
            old, new = sum(D["stage_classes"][:stage], []), D["stage_classes"][stage]
            mask = D["report_dev_mask"] & torch.isin(D["y"], torch.tensor(seen))
            y = D["y"][mask].numpy()
            if set(np.unique(y)) != set(seen):
                raise AssertionError("diagnostics require support for every seen class")
            scores = []
            for i, seed in enumerate(original["student_seeds"]):
                model = U.G.train_student(U.G.memory_to_trainable(memory), seed, args.device,
                    U.protocol_v3.student_steps(len(memory["x"])), batch=16)
                with torch.no_grad():
                    logits = model.logits(x, adj, prompt).cpu()[mask]
                predictions = logits.argmax(1).numpy()
                metrics = class_metrics(y, predictions, seen, old, new, len(D["prompts"]))
                if abs(metrics["cumulative_f1"] - row["per_student"][i]["cumulative_macro_f1"]) > 1e-8:
                    raise AssertionError("canonical identity replay differed")
                seen_prediction = np.array(seen)[logits[:, seen].argmax(1).numpy()]
                metrics["seen_prompt_only_f1"] = 100 * f1_score(y, seen_prediction,
                    labels=seen, average="macro", zero_division=0)
                metrics["seed"] = seed
                scores.append(metrics)
                del model
            summary = {"stage": stage, "seen_classes": seen, "old_classes": old,
                "new_classes": new, "per_student": scores,
                "source_memory_sha256": hashlib.sha256(memory_file.read_bytes()).hexdigest(),
                "canonical_replay_match": True,
                "cumulative_f1": float(np.mean([s["cumulative_f1"] for s in scores])),
                "seen_prompt_only_f1": float(np.mean([s["seen_prompt_only_f1"] for s in scores])),
                "old_class_f1_global_support": (float(np.mean([s["old_class_f1_global_support"] for s in scores]))
                                                 if old else None),
                "new_class_f1_global_support": float(np.mean([s["new_class_f1_global_support"] for s in scores]))}
            rows.append(summary)
            print(json.dumps({k: v for k, v in summary.items() if k != "per_student"}), flush=True)
        result_file.write_text(json.dumps({"dataset": ref["dataset"], "operator": operator,
            "test_accessed": False, "student_seeds": original["student_seeds"],
            "rows": rows, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
    print(f"P2_METRIC_DIAGNOSTICS_DONE {ref['dataset']}", flush=True)


if __name__ == "__main__":
    main()
