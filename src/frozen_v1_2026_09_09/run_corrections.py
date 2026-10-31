"""Correction experiments after the Gate-2'' review.

1. EQUIVALENCE AUDIT (answers "is append>full_history just a weak-condenser
   artifact?"): with a deterministic constructor, append is byte-identical to
   "full-history with per-stage stratified quotas" (same inputs, same budgets,
   same constructor => same blocks). We verify this by reconstructing the
   stratified memory from raw and comparing it field-by-field to the saved
   append memory. If equal, the append-vs-joint gap is EXACTLY the effect of
   joint clustering on a mixed pool vs stratified clustering -- the continual
   constraint itself costs nothing under this constructor.

2. CITESEER BUDGET SWEEP: transfer check for the c rule selected on Cora.
   s2 total budget 143; old memory 110 rows; sweep old_keep in
   {110, 100, 90, 80} (=> new block 33/43/53/63; c per new class ~16/21/26/31).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2_llm_fusion"))
import protocol_v3

sys.path.insert(0, str(Path(__file__).parent))
import run_gate1_continual as G

# citeseer configuration
G.CLASS_NAMES = ["agents", "machine learning", "information retrieval",
                 "databases", "human computer interaction", "artificial intelligence"]
G.PROMPTS = [f"a research paper about {n}" for n in G.CLASS_NAMES]
G.STAGE_CLASSES = [[0, 1], [2, 3], [4, 5]]


def shrink_by_mass(memory: dict, keep: int) -> dict:
    order = torch.argsort(memory["mass"], descending=True, stable=True)[:keep]
    order = torch.sort(order).values
    out = {k: memory[k][order] for k in ("x", "text_emb", "mass", "depth", "stage")}
    out["texts"] = [memory["texts"][int(i)] for i in order]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--append-memory", default="/home/pc/research/TAGSAM/problem_validation_results/gate2_citeseer_2026_09_08_gpu0/append_fixed_s2_memory.pt")
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=list(protocol_v3.STUDENT_SEEDS))
    parser.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    parser.add_argument("--student-steps", type=int, default=None)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing runs are immutable")

    data = torch.load("/home/pc/research/GraphCLIP/processed_data/citeseer.pt",
                      map_location="cpu", weights_only=False)
    x, y, texts = data.x, data.y, list(data.raw_texts)
    n = len(x)
    dev_mask = (torch.arange(n) % 10) == 1
    device = args.device
    encoder = G.FrozenMiniLM("/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", device)
    text_emb = encoder.encode(texts)
    prompt_emb = encoder.encode(G.PROMPTS)
    real_adj = G.norm_adj(data.edge_index, n, device)
    xg = x.float().to(device)

    stage_train = []
    for cls in G.STAGE_CLASSES:
        m = torch.isin(y, torch.tensor(cls)) & ~dev_mask
        stage_train.append(torch.where(m)[0])
    cum = np.cumsum([len(s) for s in stage_train])
    budgets = [math.floor(0.05 * c) for c in cum]

    # ---------- 1. equivalence audit ----------
    saved = torch.load(args.append_memory, map_location="cpu", weights_only=False)
    blocks = []
    prev = 0
    for s, ids in enumerate(stage_train):
        quota = budgets[s] - prev
        prev = budgets[s]
        b = G.condense_static(x[ids], text_emb[ids], [texts[int(i)] for i in ids],
                              quota, encoder)
        b["stage"] = torch.full((len(b["x"]),), s, dtype=torch.long)
        blocks.append(b)
    stratified = {k: (sum([b[k] for b in blocks], []) if k == "texts"
                       else torch.cat([b[k] for b in blocks]))
                  for k in ("x", "text_emb", "texts", "mass", "depth", "stage")}
    audit = {}
    for k in ("x", "text_emb", "mass", "stage"):
        audit[k] = bool(torch.allclose(stratified[k].float(), saved[k].float(), atol=1e-5))
    audit["texts"] = stratified["texts"] == saved["texts"]
    audit["equivalent"] = all(audit.values())
    print(json.dumps({"equivalence_audit": audit}), flush=True)

    # ---------- 2. citeseer budget sweep ----------
    m1 = torch.load(str(Path(args.append_memory).parent / "append_fixed_s1_memory.pt"),
                    map_location="cpu", weights_only=False)
    ids2 = stage_train[2]

    def run_students(memory, tag):
        syn = G.memory_to_trainable(memory)
        steps = protocol_v3.student_steps(len(memory["x"]), override=args.student_steps)
        rows = []
        for seed in args.student_seeds:
            model = G.train_student(syn, seed, device, steps, batch=args.student_batch)
            rows.append(G.eval_stagewise(model, xg, real_adj, prompt_emb, y, dev_mask, 2))
            del model
        agg = {"arm": tag, "roots": len(memory["x"]),
               "cum_mean": float(np.mean([r["cumulative_macro_f1"] for r in rows])),
               "cum_std": float(np.std([r["cumulative_macro_f1"] for r in rows])),
               "old_mean": float(np.mean([r["old_mean_f1"] for r in rows])),
               "cur_mean": float(np.mean([r["current_f1"] for r in rows])),
               "per_student": rows}
        print(json.dumps({k: agg[k] for k in ("arm", "roots", "cum_mean", "old_mean", "cur_mean")}), flush=True)
        return agg

    results = []
    for keep in [110, 100, 90, 80]:
        keep = min(keep, len(m1["x"]))
        new_k = budgets[2] - keep
        old_part = shrink_by_mass(m1, keep)
        block = G.condense_static(x[ids2], text_emb[ids2], [texts[int(i)] for i in ids2],
                                  new_k, encoder)
        block["stage"] = torch.full((len(block["x"]),), 2, dtype=torch.long)
        mem = {k: (old_part[k] + block[k] if k == "texts"
                    else torch.cat((old_part[k], block[k])))
               for k in ("x", "text_emb", "texts", "mass", "depth", "stage")}
        results.append(run_students(mem, f"keep{keep}_new{new_k}"))

    json.dump({**protocol_v3.config_dict(dataset="citeseer",
               student_batch=args.student_batch,
               student_steps=args.student_steps,
               student_seeds=args.student_seeds),
               "equivalence_audit": audit, "budget_sweep": results,
               "budgets": budgets, "test_accessed": False},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
