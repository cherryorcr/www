"""Gate 2'': replicate the continual gates on Citeseer (second TAG dataset).

Class-incremental Citeseer (6 classes, GraphCLIP processed_data/citeseer.pt):
  stage 0 [0,1] (agents, machine learning)
  stage 1 [2,3] (information retrieval, databases)
  stage 2 [4,5] (human computer interaction, artificial intelligence)

Class-index -> prompt mapping audited on 2026-09-08 by mean-text-embedding vs
prompt similarity (clean one-to-one diagonal; see AUDIT block in results).

Arms: append_fixed / recluster / full_history (same semantics as Cora Gate
1''), plus raw_upper_bound and the budget rule check (keep vs c=15 floor).
Pipeline, dev rule (node_id%10==1), student protocol (steps=min(2048,16k))
identical to the Cora runs. Test never read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2_llm_fusion"))
sys.path.insert(1, str(Path(__file__).parent))
import protocol_v3
import run_gate1_continual as G

# ---- dataset-specific configuration (overrides Cora constants) ----
G.CLASS_NAMES = ["agents", "machine learning", "information retrieval",
                 "databases", "human computer interaction", "artificial intelligence"]
G.PROMPTS = [f"a research paper about {n}" for n in G.CLASS_NAMES]
G.STAGE_CLASSES = [[0, 1], [2, 3], [4, 5]]
CLASS_AUDIT = "mean-text-emb vs prompt cosine gives clean diagonal (2026-09-08)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/pc/research/GraphCLIP/processed_data/citeseer.pt")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--arms", nargs="+", default=["append_fixed", "recluster", "full_history"])
    parser.add_argument("--include-raw-upper", action="store_true")
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=list(protocol_v3.STUDENT_SEEDS))
    parser.add_argument("--student-batch", type=int,
                        default=protocol_v3.STUDENT.batch)
    parser.add_argument("--student-steps", type=int, default=None,
                        help="fixed steps; otherwise protocol-scaled")
    parser.add_argument("--raw-steps-cap", type=int, default=protocol_v3.RAW_STEPS_CAP)
    parser.add_argument("--topology-mode", choices=["global", "frozen_old"],
                        default="global")
    parser.add_argument("--cross-degree", type=int, default=1)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing runs are immutable")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    x, y, texts = data.x, data.y, list(data.raw_texts)
    n = len(x)
    dev_mask, _, pool_mask = protocol_v3.split_masks(torch.arange(n))
    device = args.device
    encoder = G.FrozenMiniLM(args.minilm, device)
    text_emb = encoder.encode(texts)
    prompt_emb = encoder.encode(G.PROMPTS)
    real_adj = G.norm_adj(data.edge_index, n, device)
    xg = x.float().to(device)

    stage_train = []
    for cls in G.STAGE_CLASSES:
        m = torch.isin(y, torch.tensor(cls)) & pool_mask
        stage_train.append(torch.where(m)[0])
    cum = np.cumsum([len(s) for s in stage_train])
    budgets = [math.floor(args.rate * c) for c in cum]
    print(json.dumps({"stage_train_counts": [len(s) for s in stage_train],
                      "budgets": budgets}), flush=True)

    def run_students(memory_or_syn, seen, tag, is_memory=True, steps_cap=2048):
        syn = G.memory_to_trainable(memory_or_syn) if is_memory else memory_or_syn
        if is_memory:
            steps = protocol_v3.student_steps(len(syn["x"]), override=args.student_steps)
        else:
            steps = protocol_v3.raw_steps(override=args.raw_steps_cap)
        rows = []
        for seed in args.student_seeds:
            model = G.train_student(syn, seed, device, steps,
                                    batch=args.student_batch)
            rows.append(G.eval_stagewise(model, xg, real_adj, prompt_emb, y, dev_mask, seen))
            del model
        agg = {"arm_stage": tag, "roots": len(syn["x"]), "steps": steps,
               "student_steps": steps,
               "student_batch": args.student_batch,
               "student_seeds": list(args.student_seeds),
               "cum_mean": float(np.mean([r["cumulative_macro_f1"] for r in rows])),
               "cum_std": float(np.std([r["cumulative_macro_f1"] for r in rows])),
               "cur_mean": float(np.mean([r["current_f1"] for r in rows])),
               "old_mean": (float(np.mean([r["old_mean_f1"] for r in rows])) if seen else None),
               "per_stage_mean": [float(np.mean([r["per_stage"][s]["macro_f1"] for r in rows]))
                                  for s in range(seen + 1)],
               "per_student": rows}
        print(json.dumps({k: agg[k] for k in ("arm_stage", "roots", "cum_mean",
                                              "old_mean", "cur_mean")}), flush=True)
        return agg

    results, audits = [], {}
    for arm in args.arms:
        memory = None
        for stage in range(3):
            ids = stage_train[stage]
            dx, de = x[ids], text_emb[ids]
            dt = [texts[int(i)] for i in ids]
            k_t = budgets[stage]
            if arm == "full_history":
                pool = torch.cat(stage_train[: stage + 1])
                memory = G.condense_static(x[pool], text_emb[pool],
                                           [texts[int(i)] for i in pool], k_t, encoder)
            elif stage == 0:
                memory = G.condense_static(dx, de, dt, k_t, encoder)
            elif arm == "append_fixed":
                keep, k_new = protocol_v3.append_quota(
                    k_t, len(memory["x"]), len(G.STAGE_CLASSES[stage]))
                block = G.condense_static(dx, de, dt, k_new, encoder)
                block["stage"] = torch.full((len(block["x"]),), stage, dtype=torch.long)
                memory = G.merge_memory(memory, block, keep,
                                        topology_mode=args.topology_mode,
                                        cross_degree=args.cross_degree)
            else:
                memory, audit = G.continual_recluster(memory, dx, de, dt, k_t, stage, encoder)
                audits[f"{arm}_s{stage}"] = audit
            torch.save(memory, out / f"{arm}_s{stage}_memory.pt")
            results.append(run_students(memory, stage, f"{arm}_s{stage}"))

    if args.include_raw_upper:
        pool = torch.cat(stage_train)
        remap = torch.full((n,), -1, dtype=torch.long)
        remap[pool] = torch.arange(len(pool))
        e = data.edge_index
        km = (remap[e[0]] >= 0) & (remap[e[1]] >= 0)
        syn = {"x": x[pool].float(), "text_emb": text_emb[pool],
               "texts": [texts[int(i)] for i in pool],
               "edge_index": torch.stack((remap[e[0, km]], remap[e[1, km]])),
               "weights": torch.ones(len(pool))}
        results.append(run_students(syn, 2, "raw_upper_bound", is_memory=False,
                                    steps_cap=args.raw_steps_cap))

    json.dump({"results": results, "recluster_audits": audits, "budgets": budgets,
               "rate": args.rate, "dev_rule": "node_id % 10 == 1",
               **protocol_v3.config_dict(student_seeds=args.student_seeds),
               "protocol": protocol_v3.PROTOCOL_NAME,
               "student_batch": args.student_batch,
               "student_steps_override": args.student_steps,
               "student_seeds": args.student_seeds,
               "topology_mode": args.topology_mode,
               "cross_degree": args.cross_degree,
               "stage_classes": G.STAGE_CLASSES, "class_names": G.CLASS_NAMES,
               "class_mapping_audit": CLASS_AUDIT,
               "test_accessed": False},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
