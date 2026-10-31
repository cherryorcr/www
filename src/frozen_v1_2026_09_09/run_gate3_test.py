"""Gate 3'': one-time test read under the frozen v1 method.

Test partition: node_id % 10 == 2 (labels never previously read; features
participated label-free in condensation sources -- transductive protocol).
Every artifact x student is evaluated exactly once. Dev numbers are NOT
recomputed here; this file produces the final confirmation table.

Arms per dataset (all artifacts frozen before this run):
  method_append_c15   Cora: rebuilt s2 memory (keep76+new45, deterministic
                      from saved s1); Citeseer: saved append s2 (rule not
                      binding). Also evaluates s0/s1 memories of the method.
  append_default      saved keep-all append s2 (Cora only, context)
  recluster / full_history   saved s2 memories (context)
  raw_upper           all raw nodes, steps capped 8192 (context)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
import run_gate1_continual as G

CONFIGS = {
    "cora": {
        "data": "/home/pc/research/GraphCLIP/processed_data/cora.pt",
        "class_names": ["case based", "genetic algorithms", "neural networks",
                        "probabilistic methods", "reinforcement learning",
                        "rule learning", "theory"],
        "stage_classes": [[0, 1], [2, 3], [4, 5, 6]],
        "runs": "/home/pc/research/TAGSAM/problem_validation_results/gate1_scaled_2026_09_08",
        "append_gpu": "_gpu0", "fullhist_gpu": "_gpu1",
        "c_rule_new": 45,
    },
    "citeseer": {
        "data": "/home/pc/research/GraphCLIP/processed_data/citeseer.pt",
        "class_names": ["agents", "machine learning", "information retrieval",
                        "databases", "human computer interaction",
                        "artificial intelligence"],
        "stage_classes": [[0, 1], [2, 3], [4, 5]],
        "runs": "/home/pc/research/TAGSAM/problem_validation_results/gate2_citeseer_2026_09_08",
        "append_gpu": "_gpu0", "fullhist_gpu": "_gpu1",
        "c_rule_new": None,  # rule not binding; use saved append s2
    },
}


def shrink_by_mass(memory: dict, keep: int) -> dict:
    order = torch.argsort(memory["mass"], descending=True, stable=True)[:keep]
    order = torch.sort(order).values
    out = {k: memory[k][order] for k in ("x", "text_emb", "mass", "depth", "stage")}
    out["texts"] = [memory["texts"][int(i)] for i in order]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(CONFIGS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104, 105])
    args = parser.parse_args()
    cfg = CONFIGS[args.dataset]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("Gate 3'' is one-time; existing results are immutable")

    G.CLASS_NAMES = cfg["class_names"]
    G.PROMPTS = [f"a research paper about {n}" for n in cfg["class_names"]]
    G.STAGE_CLASSES = cfg["stage_classes"]

    data = torch.load(cfg["data"], map_location="cpu", weights_only=False)
    x, y, texts = data.x, data.y, list(data.raw_texts)
    n = len(x)
    test_mask = (torch.arange(n) % 10) == 2          # ONE-TIME READ
    device = args.device
    encoder = G.FrozenMiniLM("/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", device)
    text_emb = encoder.encode(texts)
    prompt_emb = encoder.encode(G.PROMPTS)
    real_adj = G.norm_adj(data.edge_index, n, device)
    xg = x.float().to(device)

    dev_mask = (torch.arange(n) % 10) == 1
    stage_train = []
    for cls in cfg["stage_classes"]:
        m = torch.isin(y, torch.tensor(cls)) & ~dev_mask
        stage_train.append(torch.where(m)[0])
    budgets = [math.floor(0.05 * c) for c in np.cumsum([len(s) for s in stage_train])]

    def evaluate(memory_or_syn, seen, tag, is_memory=True, steps_cap=2048):
        syn = G.memory_to_trainable(memory_or_syn) if is_memory else memory_or_syn
        steps = min(steps_cap, 16 * len(syn["x"]))
        rows = []
        for seed in args.student_seeds:
            model = G.train_student(syn, seed, device, steps)
            rows.append(G.eval_stagewise(model, xg, real_adj, prompt_emb, y, test_mask, seen))
            del model
        agg = {"arm": tag, "roots": len(syn["x"]), "steps": steps,
               "test_cum_mean": float(np.mean([r["cumulative_macro_f1"] for r in rows])),
               "test_cum_std": float(np.std([r["cumulative_macro_f1"] for r in rows])),
               "test_acc_mean": float(np.mean([r["cumulative_accuracy"] for r in rows])),
               "test_cur_mean": float(np.mean([r["current_f1"] for r in rows])),
               "test_old_mean": (float(np.mean([r["old_mean_f1"] for r in rows])) if seen else None),
               "per_student": rows}
        print(json.dumps({k: agg[k] for k in ("arm", "roots", "test_cum_mean",
                                              "test_old_mean", "test_cur_mean")}), flush=True)
        return agg

    def load_mem(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    runs = cfg["runs"]
    results = []
    # method stages 0/1 (identical to saved append memories)
    m0 = load_mem(f"{runs}{cfg['append_gpu']}/append_fixed_s0_memory.pt")
    m1 = load_mem(f"{runs}{cfg['append_gpu']}/append_fixed_s1_memory.pt")
    results.append(evaluate(m0, 0, "method_s0"))
    results.append(evaluate(m1, 1, "method_s1"))
    # method stage 2 under the frozen c rule
    if cfg["c_rule_new"] is not None:
        new_k = cfg["c_rule_new"]
        keep = budgets[2] - new_k
        ids2 = stage_train[2]
        block = G.condense_static(x[ids2], text_emb[ids2], [texts[int(i)] for i in ids2],
                                  new_k, encoder)
        block["stage"] = torch.full((len(block["x"]),), 2, dtype=torch.long)
        old_part = shrink_by_mass(m1, keep)
        m2 = {k: (old_part[k] + block[k] if k == "texts"
                   else torch.cat((old_part[k], block[k])))
              for k in ("x", "text_emb", "texts", "mass", "depth", "stage")}
        torch.save(m2, out / "method_s2_memory.pt")
        results.append(evaluate(m2, 2, f"method_s2_c15_keep{keep}_new{new_k}"))
        results.append(evaluate(load_mem(f"{runs}{cfg['append_gpu']}/append_fixed_s2_memory.pt"),
                                2, "append_default_s2"))
    else:
        results.append(evaluate(load_mem(f"{runs}{cfg['append_gpu']}/append_fixed_s2_memory.pt"),
                                2, "method_s2_rule_not_binding"))
    # context arms
    results.append(evaluate(load_mem(f"{runs}{cfg['append_gpu']}/recluster_s2_memory.pt"),
                            2, "recluster_s2"))
    results.append(evaluate(load_mem(f"{runs}{cfg['fullhist_gpu']}/full_history_s2_memory.pt"),
                            2, "full_history_s2"))
    pool = torch.cat(stage_train)
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[pool] = torch.arange(len(pool))
    e = data.edge_index
    km = (remap[e[0]] >= 0) & (remap[e[1]] >= 0)
    raw = {"x": x[pool].float(), "text_emb": text_emb[pool],
           "texts": [texts[int(i)] for i in pool],
           "edge_index": torch.stack((remap[e[0, km]], remap[e[1, km]])),
           "weights": torch.ones(len(pool))}
    results.append(evaluate(raw, 2, "raw_upper", is_memory=False, steps_cap=8192))

    json.dump({"results": results, "budgets": budgets,
               "test_rule": "node_id % 10 == 2, first and only read",
               "confirmation_accessed": True,
               "frozen_version": "frozen_v1_2026_09_09"},
              open(out / "results.json", "w"), indent=1)
    print("GATE3 DONE", flush=True)


if __name__ == "__main__":
    main()
