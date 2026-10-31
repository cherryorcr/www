"""Separate true text order effects from floating-point/retraining effects."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import p2_text_operators as T
import run_unified_v3_continual as U


def pool_parts(parts, encoder, operator):
    encoded = encoder.encode(parts)
    if operator == "mean_pool":
        weights = torch.full((len(encoded),), 1.0 / len(encoded))
    elif operator == "set_pool":
        query = F.normalize(encoded.mean(0), dim=0)
        weights = torch.softmax((F.normalize(encoded, dim=1) @ query) / 0.2, dim=0)
    else:
        raise ValueError("only mean and set attention are supported")
    return F.normalize((encoded * weights[:, None]).sum(0), dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--minilm", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    matrix, out = Path(args.matrix), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    first = json.loads((matrix / "mean_pool_t64_g5_r0.5_a0.5/results.json").read_text())
    ref = json.loads((Path(first["reference"]) / "results.json").read_text())
    D = U.load_unified(ref["dataset"], ref["data"], args.minilm, args.device,
        n_stages=ref["n_stages"], split_mode=ref["split_mode"],
        base_fraction=ref["base_fraction"], report_dev_mod=ref["report_dev_mod"])
    D["prompt_emb"] = D["encoder"].encode(D["prompts"])
    D["real_adj"] = U.G.norm_adj(D["edge_index"], len(D["x"]), args.device)
    D["x_device"] = D["x"].to(args.device)
    for operator in ("mean_pool", "set_pool"):
        source = matrix / f"{operator}_t64_g5_r0.5_a0.5"
        original = json.loads((source / "results.json").read_text())
        modes = ["identity", "same_order", "reverse_exact", "canonical_order", "canonical_independent"]
        for mode in modes:
            target = out / f"{operator}_{mode}"
            target.mkdir(exist_ok=True)
            if (target / "results.json").exists():
                print(f"SKIP {operator} {mode}", flush=True)
                continue
            seeds = list(range(400, 406)) if mode == "canonical_independent" else list(U.protocol_v3.STUDENT_SEEDS)
            results, old_parts, old_embeddings = [], [], []
            print(f"RUN {ref['dataset']} {operator} {mode}", flush=True)
            for row in original["results"]:
                stage = row["stage"]
                memory_file = source / f"append_fixed_s{stage}_memory.pt"
                saved = torch.load(memory_file, map_location="cpu", weights_only=False)
                memory = dict(saved)
                invariance_error = 0.0
                if mode != "identity":
                    # Transform only the new block, preserving the tested old prefix.
                    keep = row["keep_old_rows"]
                    if len(old_parts) != keep:
                        raise AssertionError("numerical stress test rewrote old rows")
                    for audit in saved["text_input_audit"][keep:]:
                        parts = list(audit["inputs"])
                        if mode == "reverse_exact":
                            parts = list(reversed(parts))
                        elif mode.startswith("canonical"):
                            parts = sorted(parts)
                            reverse_sorted = sorted(reversed(audit["inputs"]))
                            if parts != reverse_sorted:
                                raise AssertionError("canonicalization is not permutation invariant")
                        emb = pool_parts(parts, D["encoder"], operator)
                        if mode.startswith("canonical"):
                            second = pool_parts(sorted(reversed(audit["inputs"])), D["encoder"], operator)
                            invariance_error = max(invariance_error, float((second - emb).abs().max()))
                            if not torch.equal(second, emb):
                                raise AssertionError("canonicalized set representation is not bitwise invariant")
                        if sum(T.token_count(D["encoder"].tokenizer, p) for p in parts) > 64:
                            raise AssertionError("token budget changed")
                        old_parts.append(parts)
                        old_embeddings.append(emb)
                    memory.update(text_emb=torch.stack(old_embeddings),
                                  texts=[" ".join(p) for p in old_parts])
                    audits = []
                    for previous, parts, text in zip(saved["text_input_audit"], old_parts, memory["texts"]):
                        audit = dict(previous)
                        audit.update(inputs=parts,
                            input_sha256=hashlib.sha256("\0".join(parts).encode()).hexdigest(),
                            retained_tokens=T.token_count(D["encoder"].tokenizer, text))
                        audits.append(audit)
                    memory["text_input_audit"] = audits
                torch.save(memory, target / f"append_fixed_s{stage}_memory.pt")
                delta = float((memory["text_emb"] - saved["text_emb"]).abs().max())
                metrics = U.run_students(memory, stage, D,
                    SimpleNamespace(student_seeds=seeds, student_steps=None, student_batch=16), args.device)
                original_scores = [s["cumulative_macro_f1"] for s in row["per_student"]]
                current_scores = [s["cumulative_macro_f1"] for s in metrics["per_student"]]
                results.append({"stage": stage, "source_memory_sha256": hashlib.sha256(memory_file.read_bytes()).hexdigest(),
                    "max_embedding_change": delta, "canonical_order_invariance_error": invariance_error,
                    "matches_original_scores": current_scores == original_scores,
                    "original_f1": row["cum_dev_f1_mean"], **metrics})
                print(json.dumps({"operator": operator, "mode": mode, "stage": stage,
                    "embedding_change": delta, "f1": metrics["cum_dev_f1_mean"]}), flush=True)
            (target / "results.json").write_text(json.dumps({
                "protocol": "P2-numerical-guard-v1", "dataset": ref["dataset"],
                "operator": operator, "mode": mode, "student_seeds": seeds,
                "source": str(source), "test_accessed": False,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "results": results}, indent=2))
    print(f"P2_NUMERICAL_GUARD_DONE {ref['dataset']}", flush=True)


if __name__ == "__main__":
    main()
