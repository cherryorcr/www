"""Evaluate genuine learned set compression without rewriting old memory."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import run_unified_v3_continual as U
import p2_text_operators as T
from p2_learned_set import fit_source_only


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CaptureEncoder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.tokenizer = encoder.tokenizer
        self.last_embeddings = None

    def encode(self, texts):
        self.last_embeddings = self.encoder.encode(texts)
        return self.last_embeddings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--minilm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--construction-seeds", nargs="+", type=int, default=[700, 701, 702])
    ap.add_argument("--student-seeds", nargs="+", type=int, default=list(U.protocol_v3.STUDENT_SEEDS))
    args = ap.parse_args()
    root, out = Path(args.reference), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ref = json.loads((root / "results.json").read_text())
    if ref["test_accessed"] or ref["response_marginal"] or ref["allocation_mode"] != "dispersion":
        raise ValueError("invalid isolation reference")
    if digest(ref["data"]) != ref["source_hash_sha256"]:
        raise ValueError("source file changed")
    D = U.load_unified(ref["dataset"], ref["data"], args.minilm, args.device,
        n_stages=ref["n_stages"], split_mode=ref["split_mode"],
        base_fraction=ref["base_fraction"], report_dev_mod=ref["report_dev_mod"])
    D["text_emb"] = torch.load(root / "text_emb_pool_only.pt", map_location="cpu", weights_only=False)
    if D["text_emb"][~D["pool_mask"]].abs().max() != 0:
        raise ValueError("unsealed source cache")
    D["prompt_emb"] = D["encoder"].encode(D["prompts"])
    D["real_adj"] = U.G.norm_adj(D["edge_index"], len(D["x"]), args.device)
    D["x_device"] = D["x"].to(args.device)
    capture = CaptureEncoder(D["encoder"])
    joint = torch.cat((F.normalize(D["x"].float(), dim=1), D["text_emb"]), dim=1)
    stages = []
    for row in ref["results"]:
        if row["arm"] != "append_fixed":
            continue
        start = time.perf_counter()
        memory = torch.load(root / f"append_fixed_s{row['stage']}_memory.pt", map_location="cpu", weights_only=False)
        clusters = []
        for cls in D["stage_classes"][row["stage"]]:
            ids = torch.where(D["pool_mask"] & (D["y"] == cls))[0]
            count = int(row["allocated_new_rows_by_class"][str(cls)])
            if count:
                assign = U.SCA.cluster_assign(joint[ids], torch.ones(len(ids)), count, device=args.device)
                clusters.extend(ids[assign == j] for j in range(int(assign.max()) + 1))
        reconstruction = torch.stack([D["x"][m].float().mean(0) for m in clusters])
        if not torch.allclose(reconstruction, memory["x"][row["keep_old_rows"]:], atol=1e-6, rtol=0):
            raise AssertionError("cluster reconstruction mismatch")
        bags = torch.zeros(len(clusters), 5, 384)
        masks = torch.zeros(len(clusters), 5)
        texts, audits, teacher_targets = [], [], []
        for i, members in enumerate(clusters):
            text, _, audit = T.construct(joint, D["text_emb"], members, D["texts"],
                capture, operator="mean_pool", budget=64, g=5)
            encoded = capture.last_embeddings
            bags[i, :len(encoded)] = encoded
            masks[i, :len(encoded)] = 1
            teacher_targets.append(D["encoder"].encode([text])[0])
            audit["operator"] = "learned_deepsets"
            texts.append(text)
            audits.append(audit)
        stages.append((row, memory, bags, masks, torch.stack(teacher_targets), texts,
                       audits, time.perf_counter() - start))
    for seed in args.construction_seeds:
        target = out / f"learned_deepsets_t64_g5_seed{seed}"
        target.mkdir(exist_ok=True)
        if (target / "results.json").exists():
            old = json.loads((target / "results.json").read_text())
            if (old["student_seeds"] != args.student_seeds
                    or old["construction_seed"] != seed
                    or old["code_sha256"] != digest(Path(__file__).with_name("p2_learned_set.py"))
                    or old["reference_results_sha256"] != digest(root / "results.json")
                    or len(old["results"]) != len(stages)):
                raise ValueError("existing result protocol mismatch")
            print(f"SKIP construction seed {seed}", flush=True)
            continue
        old_texts, old_emb, old_audits, results = [], [], [], []
        for row, refmem, bags, masks, teachers, texts, audits, preparation in stages:
            start = time.perf_counter()
            embeddings, state, fit_audit = fit_source_only(
                bags, masks, teachers, seed=seed, steps=300, device=args.device)
            if len(old_texts) != row["keep_old_rows"]:
                raise AssertionError("old rows must all be retained")
            old_texts.extend(texts)
            old_emb.extend(embeddings)
            old_audits.extend(audits)
            memory = dict(refmem)
            memory.update(texts=list(old_texts), text_emb=torch.stack(old_emb),
                          text_input_audit=list(old_audits))
            torch.save(state, target / f"set_model_s{row['stage']}.pt")
            torch.save(memory, target / f"append_fixed_s{row['stage']}_memory.pt")
            fit_seconds = time.perf_counter() - start
            metrics = U.run_students(memory, row["stage"], D,
                SimpleNamespace(student_seeds=args.student_seeds, student_steps=None,
                                student_batch=ref["student_batch"]), args.device)
            results.append({"arm": "append_fixed", "stage": row["stage"],
                "keep_old_rows": row["keep_old_rows"], "new_rows": len(texts),
                "fit": fit_audit, "text_audit": T.aggregate_audit(old_audits),
                "build_seconds": preparation + fit_seconds,
                "teacher_encoding_extra_tokens": sum(T.token_count(D["encoder"].tokenizer, t) for t in texts),
                "reference_memory_sha256": digest(root / f"append_fixed_s{row['stage']}_memory.pt"),
                "topology_and_features_fixed": True, **metrics})
            print(json.dumps({"seed": seed, "stage": row["stage"],
                              "f1": metrics["cum_dev_f1_mean"],
                              "fit_step": fit_audit["selected_step"]}), flush=True)
        result = {"protocol": "P2-learned-set-source-distill-v1", "dataset": ref["dataset"],
            "reference": str(root), "reference_results_sha256": digest(root / "results.json"),
            "construction_seed": seed, "student_seeds": args.student_seeds,
            "spec": {"operator": "learned_deepsets", "budget": 64, "g": 5},
            "student_batch": ref["student_batch"], "test_accessed": False, "selector_used": False,
            "stage_classes": D["stage_classes"], "source_scope": "current_delta_only",
            "extra_cost": "source_teacher_encoding_and_300_set_optimizer_steps_per_stage",
            "code_sha256": digest(Path(__file__).with_name("p2_learned_set.py")),
            "results": results}
        (target / "results.json").write_text(json.dumps(result, indent=2))
    print(f"P2_LEARNED_SET_DONE {ref['dataset']}", flush=True)


if __name__ == "__main__":
    main()
