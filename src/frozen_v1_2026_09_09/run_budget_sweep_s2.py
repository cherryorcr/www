"""Budget allocation sweep at stage 2 + the missing raw upper bound.

Question: is new-class plasticity (48 vs full-raw ~66 current F1) limited by
the 34-row new block, or by the constructor/text tower?

Arms (total budget fixed at 121 = floor(0.05 * 2437)):
  keep87_new34   status quo (append-constructive v1.1)
  keep77_new44 / keep67_new54 / keep57_new64
     old memory SHRUNK by deleting lowest-mass rows (deterministic,
     label-free, no re-clustering / no text edits of surviving rows),
     freed rows go to a larger stage-2 constructive block
  raw_upper_bound   ALL raw training nodes seen so far, real X/text/edges,
     trained directly (no condensation). Steps scaled by size with a higher
     cap so it is not under-trained (the earlier full-real=43 artifact).

Pipeline identical to run_gate1_continual.py; stages 0 and 1 reuse the
append-constructive path; dev-only; test never read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2_llm_fusion"))
import protocol_v3
sys.path.insert(0, str(Path(__file__).parent))
from run_gate1_continual import (FrozenMiniLM, CLIPStudent, norm_adj,
                                 condense_static, memory_to_trainable,
                                 train_student, eval_stagewise,
                                 STAGE_CLASSES, PROMPTS)


def shrink_by_mass(memory: dict, keep: int) -> dict:
    """Delete lowest-mass rows; surviving rows untouched. Deterministic."""
    order = torch.argsort(memory["mass"], descending=True, stable=True)[:keep]
    order = torch.sort(order).values
    out = {}
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        out[key] = memory[key][order]
    out["texts"] = [memory["texts"][int(i)] for i in order]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/pc/research/GraphCLIP/processed_data/cora.pt")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--old-keep", type=int, nargs="+", default=[87, 77, 67, 57])
    parser.add_argument("--include-raw-upper", action="store_true")
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=list(protocol_v3.STUDENT_SEEDS))
    parser.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument("--raw-steps-cap", type=int, default=protocol_v3.RAW_STEPS_CAP)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing runs are immutable")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    x, y, texts = data.x, data.y, list(data.raw_texts)
    n = len(x)
    dev_mask = (torch.arange(n) % 10) == 1
    device = args.device
    encoder = FrozenMiniLM(args.minilm, device)
    text_emb = encoder.encode(texts)
    prompt_emb = encoder.encode(PROMPTS)
    real_adj = norm_adj(data.edge_index, n, device)
    xg = x.float().to(device)

    stage_train = []
    for cls in STAGE_CLASSES:
        m = torch.isin(y, torch.tensor(cls)) & ~dev_mask
        stage_train.append(torch.where(m)[0])
    cum = np.cumsum([len(s) for s in stage_train])
    budgets = [math.floor(args.rate * c) for c in cum]

    def run_students(syn: dict, steps: int, seen: int, tag: str) -> dict:
        rows = []
        for seed in args.student_seeds:
            model = train_student(syn, seed, device, steps, batch=args.student_batch)
            rows.append(eval_stagewise(model, xg, real_adj, prompt_emb, y, dev_mask, seen))
            del model
        agg = {"arm": tag, "roots": len(syn["x"]), "steps": steps,
               "cum_mean": float(np.mean([r["cumulative_macro_f1"] for r in rows])),
               "cum_std": float(np.std([r["cumulative_macro_f1"] for r in rows])),
               "old_mean": float(np.mean([r["old_mean_f1"] for r in rows])),
               "cur_mean": float(np.mean([r["current_f1"] for r in rows])),
               "per_stage_mean": [float(np.mean([r["per_stage"][s]["macro_f1"] for r in rows]))
                                  for s in range(seen + 1)],
               "per_student": rows}
        print(json.dumps({k: agg[k] for k in ("arm", "roots", "cum_mean", "old_mean", "cur_mean")}),
              flush=True)
        return agg

    # rebuild the append-constructive memory for stages 0 and 1 (deterministic)
    ids0, ids1, ids2 = stage_train
    m0 = condense_static(x[ids0], text_emb[ids0], [texts[int(i)] for i in ids0],
                         budgets[0], encoder)
    k1_new = budgets[1] - len(m0["x"])
    b1 = condense_static(x[ids1], text_emb[ids1], [texts[int(i)] for i in ids1],
                         k1_new, encoder)
    b1["stage"] = torch.full((len(b1["x"]),), 1, dtype=torch.long)
    m1 = {key: (m0[key] + b1[key] if key == "texts" else torch.cat((m0[key], b1[key])))
          for key in ("x", "text_emb", "texts", "mass", "depth", "stage")}

    results = []
    for keep in args.old_keep:
        keep = min(keep, len(m1["x"]))
        new_k = budgets[2] - keep
        old_part = shrink_by_mass(m1, keep)
        block = condense_static(x[ids2], text_emb[ids2], [texts[int(i)] for i in ids2],
                                new_k, encoder)
        block["stage"] = torch.full((len(block["x"]),), 2, dtype=torch.long)
        mem = {key: (old_part[key] + block[key] if key == "texts"
                      else torch.cat((old_part[key], block[key])))
               for key in ("x", "text_emb", "texts", "mass", "depth", "stage")}
        torch.save(mem, out / f"mem_keep{keep}_new{new_k}.pt")
        syn = memory_to_trainable(mem)
        steps = protocol_v3.student_steps(len(mem["x"]), override=args.student_steps)
        results.append(run_students(syn, steps, 2, f"keep{keep}_new{new_k}"))

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
        steps = protocol_v3.raw_steps(override=args.raw_steps_cap)
        results.append(run_students(syn, steps, 2, "raw_upper_bound"))

    json.dump({**protocol_v3.config_dict(dataset="cora", rate=args.rate,
               student_batch=args.student_batch,
               student_steps=args.student_steps,
               student_seeds=args.student_seeds,
               raw_steps_cap=args.raw_steps_cap),
               "results": results, "budgets": budgets, "rate": args.rate,
               "dev_rule": "node_id % 10 == 1", "test_accessed": False},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
