"""Static full-history TAGSAM baseline on the canonical GraphCLIP task.

This is an explicitly named adaptation, not an official TAGSAM reproduction.
At every stage the constructor can read the complete legal raw pool seen so
far, rebuilds the whole synthetic graph, and optimizes all synthetic features
from scratch with the pinned official TAGSAM response loss.  The downstream
student and report-dev evaluation are exactly the audited small-graph ones.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

import run_small_comparison_audited as C
import run_tagsam_graphclip_comparison as T
import run_unified_v3_continual as U


P = C.P
G = C.G
SCA = C.SCA
STATIC_ARMS = ["tagsam_static_selfloop", "tagsam_static_semantic"]


def load_data(dataset: str, minilm: str, device: str) -> dict:
    data = U.load_unified(dataset, U.DATA_CFG[dataset]["path"], minilm, device,
                           n_stages=3, split_mode="canonical")
    cache = C.artifact_dir(dataset, "v4") / "text_emb_pool_only.pt"
    data["text_emb"] = torch.load(cache, map_location="cpu", weights_only=False)
    excluded = ~data["pool_mask"]
    if excluded.any() and float(data["text_emb"][excluded].abs().max()) != 0.0:
        raise ValueError("source embedding cache contains excluded rows")
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = G.norm_adj(data["edge_index"], len(data["x"]), device)
    data["x_device"] = data["x"].to(device)
    return data


def stage_inputs(data: dict):
    stages = [torch.where(data["pool_mask"] & torch.isin(
        data["y"], torch.tensor(classes)))[0]
              for classes in data["stage_classes"]]
    budgets = [math.floor(0.05 * sum(len(v) for v in stages[:s + 1]))
               for s in range(len(stages))]
    return stages, budgets


def build_initial(data: dict, source_ids: torch.Tensor, budget: int,
                  stage: int, device: str) -> dict:
    seen_classes = [int(c) for group in data["stage_classes"][:stage + 1]
                    for c in group]
    block = U.build_stage_block(
        data, source_ids, budget, data["encoder"], stage,
        "joint_g5_c320", "concat", device, "dispersion", seen_classes,
        0.5, 0.5)
    block["construction"] = "full-history joint dispersion initialization"
    return block


def construct_static(data: dict, arm: str, stages: list[torch.Tensor],
                     budgets: list[int], out: Path, device: str,
                     loss_fn, outer_steps: int, teacher_steps: int):
    memories, audits = [], []
    for stage, _ in enumerate(stages):
        source_ids = torch.cat(stages[:stage + 1])
        memory_path = out / f"{arm}_s{stage}_memory.pt"
        audit_path = out / f"{arm}_s{stage}_construction.json"
        if memory_path.exists() and audit_path.exists():
            memory = torch.load(memory_path, map_location="cpu", weights_only=False)
            audit = json.loads(audit_path.read_text())
            if C.tensor_hash(memory) != audit["tensor_sha256"]:
                raise ValueError(f"cached static memory hash mismatch: {memory_path}")
        else:
            started = time.perf_counter()
            raw = C.raw_support(data, source_ids, stage)
            teacher_path = out / f"teacher_s{stage}.pt"
            teacher = G.CLIPStudent(data["x"].shape[1]).to(device)
            if teacher_path.exists():
                saved = torch.load(teacher_path, map_location="cpu", weights_only=False)
                if (saved["support_sha256"] != C.tensor_hash(raw)
                        or saved["steps"] != teacher_steps
                        or saved["seed"] != 9000 + stage):
                    raise ValueError("stale static teacher support")
                teacher.load_state_dict(saved["state"])
            else:
                teacher = G.train_student(raw, 9000 + stage, device, teacher_steps)
                torch.save({
                    "state": {k: v.detach().cpu() for k, v in teacher.state_dict().items()},
                    "seed": 9000 + stage, "steps": teacher_steps,
                    "support_sha256": C.tensor_hash(raw),
                }, teacher_path)
            teacher.eval()
            initial = build_initial(data, source_ids, budgets[stage], stage, device)
            memory, opt_audit = T.optimize(
                initial, None, raw, teacher, stage, arm, device, loss_fn,
                outer_steps)
            if not torch.equal(initial["text_emb"], memory["text_emb"]):
                raise AssertionError("static optimizer rewrote frozen text")
            if not torch.isfinite(memory["x"]).all():
                raise FloatingPointError("static synthetic features are not finite")
            audit = {
                "arm": arm, "stage": stage,
                "source_rows": len(source_ids), "budget": budgets[stage],
                "source_ids_sha256": C.tensor_hash({"source_ids": source_ids}),
                "teacher_steps": teacher_steps, "outer_steps": outer_steps,
                "official_response_loss": "model.py:wBCELoss",
                "initial_tensor_sha256": C.tensor_hash(initial),
                "teacher_state_sha256": C.tensor_hash(teacher.state_dict()),
                "teacher_checkpoint_sha256": C.sha(teacher_path),
                "memory_file_sha256": None,
                "optimization": opt_audit,
                "build_seconds": time.perf_counter() - started,
                "tensor_sha256": C.tensor_hash(memory),
            }
            torch.save(memory, memory_path)
            audit["memory_file_sha256"] = C.sha(memory_path)
            C.save_json(audit_path, audit)
        if len(memory["x"]) != budgets[stage]:
            raise AssertionError(f"static budget mismatch at stage {stage}")
        memories.append(memory)
        audits.append(audit)
    return memories, audits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAG_condensation/code/graphclip_unified_v3_2026_09_13/minilm_from_graphclip")
    ap.add_argument("--official", default="/home/pc/research/TAGSAM_official")
    ap.add_argument("--outer-steps", type=int, default=500)
    ap.add_argument("--teacher-steps", type=int, default=P.TEACHER_STEPS)
    ap.add_argument("--student-seeds", nargs="+", type=int, default=list(P.STUDENT_SEEDS))
    ap.add_argument("--arms", nargs="+", choices=STATIC_ARMS, default=STATIC_ARMS)
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    loss_fn, official = T.official_loss(Path(args.official))
    data = load_data(args.dataset, args.minilm, args.device)
    stages, budgets = stage_inputs(data)
    manifest = {
        "version": "tagsam-graphclip-static-global-v1",
        "dataset": args.dataset,
        "source_sha256": C.sha(U.DATA_CFG[args.dataset]["path"]),
        "runner_sha256": C.sha(__file__),
        "student_runner_sha256": C.sha(C.__file__),
        "dependencies": {Path(m.__file__).name: C.sha(m.__file__) for m in (T, U, G, SCA, P)},
        "text_cache_sha256": C.sha(C.artifact_dir(args.dataset, "v4") / "text_emb_pool_only.pt"),
        "official": official,
        "arms": args.arms, "student_seeds": args.student_seeds,
        "budgets": budgets,
        "stage_classes": data["stage_classes"],
        "stage_pool_counts": [len(v) for v in stages],
        "rate": 0.05, "outer_steps": args.outer_steps,
        "inner_steps": 10, "inner_batch": 16,
        "reference_cap": 256, "reference_seed": "777 + stage",
        "teacher_seeds": [9000, 9001, 9002],
        "synthetic_optimizer": "SGD X lr=100 momentum=0.5; X gradient clip=5",
        "distilled_rates": "initial=0.005; SGD lr=2e-6 momentum=0.5; used only in construction",
        "teacher_steps": args.teacher_steps,
        "data_access": "full legal raw pool through current stage",
        "historical_memory": "not reused; all synthetic rows rebuilt at every stage",
        "topology": "self-loop faithful arm or semantic text-kNN control",
        "initialization": "joint_g5_c320 dispersion per seen class",
        "optimization": "official pinned TAGSAM wBCELoss response matching; synthetic X only",
        "student": "canonical two-layer GCN/MiniLM/InfoNCE/Adam protocol",
        "evaluation": "report-dev Macro-F1 on transductive full real graph",
        "test_labels_evaluated": False,
        "report_dev_used_for_selection": False,
        "outer_checkpoint_selection": "fixed terminal step; no dev-driven tuning",
        "adaptations": ["GCN/MiniLM instead of official encoders",
            "joint dispersion and concat initialization instead of official NPMI selector",
            "10 inner minibatch updates instead of 15 inner epochs",
            "fixed reference cap256 instead of official up-to-2000 sampling",
            "canonical downstream Adam instead of distilled learning rates",
            "500 outer default instead of official 5000; not evidence of convergence"],
        "adaptation_notice": "not an official end-to-end TAGSAM reproduction",
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("static output already has a different manifest")
    C.save_json(manifest_path, manifest)
    all_results, all_audits = [], {}
    for arm in args.arms:
        memories, audits = construct_static(
            data, arm, stages, budgets, out, args.device, loss_fn,
            args.outer_steps, args.teacher_steps)
        all_audits[arm] = audits
        for stage, memory in enumerate(memories):
            rows = []
            steps = P.student_steps(budgets[stage])
            for seed in args.student_seeds:
                key = f"{arm}_s{stage}_seed{seed}"
                row_path, checkpoint = out / f"{key}.json", out / f"{key}.pt"
                if row_path.exists() and checkpoint.exists():
                    row = json.loads(row_path.read_text())
                    if (row["support_sha256"] != C.tensor_hash(memory)
                            or row["steps"] != steps):
                        raise ValueError(f"stale static fit: {key}")
                else:
                    row = C.fit(memory, seed, steps, args.device, data, stage,
                                checkpoint=checkpoint)
                    C.save_json(row_path, row)
                rows.append(row)
            result = {"arm": arm, "stage": stage, "rows": len(memory["x"]),
                      "steps": steps, "per_seed": rows,
                      "metrics": C.aggregate(rows)}
            all_results.append(result)
            C.save_json(out / "progress.json", {
                "manifest": manifest, "results": all_results,
                "audits": all_audits})
            print(json.dumps({"arm": arm, "stage": stage,
                              "all": result["metrics"]["all"]}), flush=True)
    C.save_json(out / "results.json", {
        "manifest": manifest, "results": all_results,
        "audits": all_audits})
    print("TAGSAM_STATIC_GLOBAL_DONE", flush=True)


if __name__ == "__main__":
    main()
