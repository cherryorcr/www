"""Predefined student-seed, batch, and text-order stress tests for P2 artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import run_unified_v3_continual as U
import p2_text_operators as T


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reverse_targets(memory, parts_memory, encoder, operator):
    result = dict(memory)
    texts, embeddings = [], []
    for audit in parts_memory["text_input_audit"]:
        parts = list(reversed(audit["inputs"]))
        text = " ".join(parts)
        if T.token_count(encoder.tokenizer, text) > 64:
            raise AssertionError("order perturbation changed the content-token allowance")
        if operator == "concat":
            emb = encoder.encode([text])[0]
        else:
            encoded = encoder.encode(parts)
            if operator == "mean_pool":
                emb = F.normalize(encoded.mean(0), dim=0)
            elif operator == "set_pool":
                query = F.normalize(encoded.mean(0), dim=0)
                weights = torch.softmax((F.normalize(encoded, dim=1) @ query) / 0.2, dim=0)
                emb = F.normalize((encoded * weights[:, None]).sum(0), dim=0)
            else:
                raise ValueError("reversal is defined for concat/mean/set attention only")
        texts.append(text)
        embeddings.append(emb)
    result.update(texts=texts, text_emb=torch.stack(embeddings))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--minilm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--cohorts", nargs="+", default=["independent", "batch8", "batch32", "reverse"])
    args = ap.parse_args()
    matrix, out = Path(args.matrix), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    first = json.loads((matrix / args.names[0] / "results.json").read_text())
    ref = json.loads((Path(first["reference"]) / "results.json").read_text())
    D = U.load_unified(ref["dataset"], ref["data"], args.minilm, args.device,
        n_stages=ref["n_stages"], split_mode=ref["split_mode"],
        base_fraction=ref["base_fraction"], report_dev_mod=ref["report_dev_mod"])
    if ref["test_accessed"] or sha(ref["data"]) != ref["source_hash_sha256"]:
        raise ValueError("source protocol changed")
    D["prompt_emb"] = D["encoder"].encode(D["prompts"])
    D["real_adj"] = U.G.norm_adj(D["edge_index"], len(D["x"]), args.device)
    D["x_device"] = D["x"].to(args.device)
    for cohort in args.cohorts:
        if cohort not in {"independent", "batch8", "batch32", "reverse"}:
            raise ValueError("unknown confirmation cohort")
        seeds = list(range(400, 406)) if cohort == "independent" else list(U.protocol_v3.STUDENT_SEEDS)
        batch = {"batch8": 8, "batch32": 32}.get(cohort, 16)
        for name in args.names:
            source = matrix / name
            original = json.loads((source / "results.json").read_text())
            operator = original["spec"]["operator"]
            if cohort == "reverse" and operator not in {"concat", "mean_pool", "set_pool"}:
                continue
            target = out / cohort / name
            target.mkdir(parents=True, exist_ok=True)
            result_path = target / "results.json"
            if result_path.exists():
                done = json.loads(result_path.read_text())
                if done["source_results_sha256"] != sha(source / "results.json"):
                    raise ValueError("confirmation source hash changed")
                print(f"SKIP {cohort} {name}", flush=True)
                continue
            results = []
            print(f"RUN {ref['dataset']} {cohort} {name}", flush=True)
            for row in original["results"]:
                stage = row["stage"]
                memory_path = source / f"append_fixed_s{stage}_memory.pt"
                memory = torch.load(memory_path, map_location="cpu", weights_only=False)
                max_diff = 0.0
                if cohort == "reverse":
                    parts = torch.load(matrix / "mean_pool_t64_g5_r0.5_a0.5" /
                        f"append_fixed_s{stage}_memory.pt", map_location="cpu", weights_only=False)
                    altered = reverse_targets(memory, parts, D["encoder"], operator)
                    max_diff = float((altered["text_emb"] - memory["text_emb"]).abs().max())
                    memory = altered
                start = time.perf_counter()
                report_scores, disjoint_scores = [], []
                steps = U.protocol_v3.student_steps(len(memory["x"]))
                for seed in seeds:
                    model = U.G.train_student(U.G.memory_to_trainable(memory), seed,
                                              args.device, steps, batch=batch)
                    for mask, dest in ((D["report_dev_mask"], report_scores),
                                       (D["selector_dev_mask"], disjoint_scores)):
                        dest.append(U.G.eval_stagewise(model, D["x_device"], D["real_adj"],
                            D["prompt_emb"], D["y"], mask, stage))
                    del model
                scores = [s["cumulative_macro_f1"] for s in report_scores]
                result = {"stage": stage, "roots": len(memory["x"]), "student_steps": steps,
                    "student_batch": batch, "student_seeds": seeds,
                    "source_memory_sha256": sha(memory_path),
                    "text_order_max_embedding_change": max_diff,
                    "cum_dev_f1_mean": float(np.mean(scores)),
                    "cum_dev_f1_std": float(np.std(scores)),
                    "current_f1_mean": float(np.mean([s["current_f1"] for s in report_scores])),
                    "old_mean_f1_mean": (float(np.mean([s["old_mean_f1"] for s in report_scores]))
                                         if stage > 0 else None),
                    "per_student": report_scores,
                    "disjoint_dev_per_student": disjoint_scores,
                    "disjoint_dev_f1_mean": float(np.mean([s["cumulative_macro_f1"] for s in disjoint_scores])),
                    "student_training_runs": len(seeds),
                    "student_train_seconds": time.perf_counter() - start}
                results.append(result)
                print(json.dumps({"cohort": cohort, "name": name, "stage": stage,
                                  "f1": result["cum_dev_f1_mean"]}), flush=True)
            result_path.write_text(json.dumps({"protocol": "P2-artifact-confirmation-v1",
                "dataset": ref["dataset"], "cohort": cohort, "name": name,
                "source_results_sha256": sha(source / "results.json"),
                "source": str(source), "student_seeds": seeds, "student_batch": batch,
                "test_accessed": False, "fitting_used_dev_labels": False,
                "report_dev_mod": ref["report_dev_mod"], "disjoint_dev_mod": ref["dev_mod"],
                "code_sha256": sha(__file__), "results": results}, indent=2))
    print(f"P2_CONFIRMATION_DONE {ref['dataset']}", flush=True)


if __name__ == "__main__":
    main()
