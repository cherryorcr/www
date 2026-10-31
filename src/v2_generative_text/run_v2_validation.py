"""v2 quick validation: LLM summarization vs v1 string concatenation on Cora stage-2.

Compare generative text condensation (Qwen3 greedy summaries) against v1's
truncation+concatenation baseline. Single stage-2 run, dev evaluation only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add frozen_v1 to path for reusing infra
sys.path.insert(0, str(Path(__file__).parent.parent / "frozen_v1_2026_09_09"))

import torch
import torch.nn.functional as F
from run_gate1_continual import (
    FrozenMiniLM, CLIPStudent, weighted_cluster,
    memory_to_trainable, train_student, eval_stagewise, norm_adj,
    CLASS_NAMES, PROMPTS, STAGE_CLASSES, FUSE_G, FUSE_CHARS
)

# v2 imports
from llm_summarizer import ClusterSummarizer


def condense_v1(x, text_emb, texts, k, encoder):
    """v1 baseline: truncation+concatenation."""
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)
    assign = weighted_cluster(joint, torch.ones(len(x)), k)
    syn_x, fused, mass = [], [], []
    for j in range(int(assign.max()) + 1):
        members = torch.where(assign == j)[0]
        syn_x.append(x[members].float().mean(0))
        mass.append(float(len(members)))
        # v1 fusion: top-g geometric similarity, truncate+concat
        center = joint[members].mean(0, keepdim=True)
        d = torch.cdist(joint[members], center).flatten()
        order = members[torch.argsort(d)][:FUSE_G]
        per = max(48, FUSE_CHARS // max(1, len(order)))
        fused.append(" ".join(texts[int(i)][:per] for i in order))
    emb = encoder.encode(fused)
    return {"x": torch.stack(syn_x), "text_emb": emb, "texts": fused,
            "mass": torch.tensor(mass), "depth": torch.zeros(len(syn_x), dtype=torch.long),
            "stage": torch.zeros(len(syn_x), dtype=torch.long)}


def condense_v2(x, text_emb, texts, k, encoder, llm_summarizer):
    """v2: LLM-generated semantic summaries (temperature=0 greedy decode)."""
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)
    assign = weighted_cluster(joint, torch.ones(len(x)), k)
    syn_x, fused, mass = [], [], []
    for j in range(int(assign.max()) + 1):
        members = torch.where(assign == j)[0]
        syn_x.append(x[members].float().mean(0))
        mass.append(float(len(members)))
        # v2 fusion: LLM summarization
        center = joint[members].mean(0, keepdim=True)
        d = torch.cdist(joint[members], center).flatten()
        order = members[torch.argsort(d)][:FUSE_G]
        member_texts = [texts[int(i)] for i in order]
        summary = llm_summarizer.summarize_cluster(
            member_texts, cluster_size=len(members),
            budget_chars=FUSE_CHARS, max_input_texts=FUSE_G
        )
        fused.append(summary)
    emb = encoder.encode(fused)
    return {"x": torch.stack(syn_x), "text_emb": emb, "texts": fused,
            "mass": torch.tensor(mass), "depth": torch.zeros(len(syn_x), dtype=torch.long),
            "stage": torch.zeros(len(syn_x), dtype=torch.long)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/pc/research/GraphCLIP/processed_data/cora.pt")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--qwen", default="/home/pc/models/Qwen3-32B-AWQ")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--student-steps", type=int, default=2048)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError(f"{out}/results.json already exists")

    # Load data
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    x, y, texts = data.x, data.y, list(data.raw_texts)
    n = len(x)
    dev_mask = (torch.arange(n) % 10) == 1
    device = args.device

    # Frozen text encoder
    print("Loading frozen MiniLM...", flush=True)
    encoder = FrozenMiniLM(args.minilm, device)
    text_emb = encoder.encode(texts)
    prompt_emb = encoder.encode(PROMPTS)

    # LLM summarizer (v2 only)
    print(f"Loading Qwen3 from {args.qwen}...", flush=True)
    llm_summarizer = ClusterSummarizer(args.qwen, device=device)

    # Real graph for eval
    real_adj = norm_adj(data.edge_index, n, device)
    xg = x.float().to(device)

    # Stage-2 train pool (classes [4,5,6], dev excluded)
    stage2_cls = torch.tensor(STAGE_CLASSES[2])
    stage2_train = torch.where(torch.isin(y, stage2_cls) & ~dev_mask)[0]
    k = 121  # frozen_v1 budget for stage-2
    print(json.dumps({"stage2_train_nodes": len(stage2_train), "budget": k}), flush=True)

    dx = x[stage2_train]
    de = text_emb[stage2_train]
    dt = [texts[int(i)] for i in stage2_train]

    results = []
    for variant, condense_fn in [("v1_concat", condense_v1),
                                  ("v2_llm_summary", lambda *a: condense_v2(*a, llm_summarizer))]:
        print(f"\n=== {variant} ===", flush=True)
        memory = condense_fn(dx, de, dt, k, encoder)
        torch.save(memory, out / f"{variant}_memory.pt")

        # Determinism audit
        memory2 = condense_fn(dx, de, dt, k, encoder)
        for key in ("x", "text_emb", "mass"):
            if not torch.allclose(memory[key], memory2[key], atol=1e-6):
                raise RuntimeError(f"{variant} non-deterministic on {key}")
        print(f"  Determinism: PASS", flush=True)

        # Train students
        syn = memory_to_trainable(memory)
        per_student = []
        for seed in args.student_seeds:
            model = train_student(syn, seed, device, args.student_steps)
            metrics = eval_stagewise(model, xg, real_adj, prompt_emb, y, dev_mask, seen_stages=2)
            per_student.append(metrics)
            print(f"  seed={seed} cum_f1={metrics['cumulative_macro_f1']:.2f}", flush=True)
            del model

        import numpy as np
        agg = {
            "variant": variant,
            "roots": len(memory["x"]),
            "cum_dev_f1_mean": float(np.mean([r["cumulative_macro_f1"] for r in per_student])),
            "cum_dev_f1_std": float(np.std([r["cumulative_macro_f1"] for r in per_student])),
            "per_student": per_student
        }
        results.append(agg)
        print(json.dumps({k: agg[k] for k in ("variant", "cum_dev_f1_mean", "cum_dev_f1_std")}), flush=True)

    json.dump({"results": results, "budget": k, "student_steps": args.student_steps,
               "stage": 2, "dev_rule": "node_id % 10 == 1"},
              open(out / "results.json", "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
