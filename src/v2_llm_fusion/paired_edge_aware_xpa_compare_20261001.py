"""Paired same-process comparison for append, X+P and X+P+A memories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_tgc_semantic_recursive_large_20260926 as L
import run_unified_v3_continual as U


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate_memory(memory: dict, stage: int, data: dict,
                    seeds: list[int], device: str) -> dict:
    syn = {"x": memory["x"], "text_emb": memory["text_emb"],
           "texts": memory["texts"], "edge_index": memory["edge_index"],
           "weights": memory.get("mass")}
    rows = []
    for seed in seeds:
        steps = U.protocol_v3.student_steps(len(syn["x"]))
        model = U.G.train_student(syn, int(seed), device, steps, batch=16)
        rows.append(U.G.eval_stagewise(
            model, data["x_device"], data["real_adj"], data["prompt_emb"],
            data["y"], data["report_dev_mask"], stage))
        del model
    return {
        "final_f1": float(sum(r["cumulative_macro_f1"] for r in rows)
                             / len(rows)),
        "accuracy": float(sum(r["cumulative_accuracy"] for r in rows)
                           / len(rows)),
        "old_f1": (float(sum(r["old_mean_f1"] for r in rows) / len(rows))
                   if stage > 0 else None),
        "new_f1": float(sum(r["current_f1"] for r in rows) / len(rows)),
        "per_student": rows,
    }


def prepare_data(dataset: str, source: Path, output: Path, minilm: str,
                 device: str) -> dict:
    manifest = json.loads((source / "results.json").read_text())
    nested = manifest.get("protocol")
    protocol = nested if isinstance(nested, dict) else manifest
    groups = [[int(v) for v in group]
              for group in protocol.get("stage_classes", protocol.get("groups", []))]
    data = U.load_unified(
        dataset, U.DATA_CFG[dataset]["path"], minilm, device,
        n_stages=5, split_mode="base_incremental", base_fraction=0.4,
        report_dev_mod=int(protocol.get("report_dev_mod", 3)))
    cumulative, seen = [], []
    for group in groups:
        seen.extend(group)
        cumulative.append(list(seen))
    data["stage_classes"] = groups
    U.G.STAGE_CLASSES = cumulative
    try:
        import run_gate1_continual as legacy
        legacy.STAGE_CLASSES = cumulative
    except ImportError:
        pass
    data = L.set_pool_text_embeddings(data, source, output)
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = U.G.norm_adj(data["edge_index"], len(data["x"]), device)
    data["x_device"] = data["x"].float().to(device)
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--xp-root", type=Path, required=True)
    ap.add_argument("--xpa-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--minilm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--student-seeds", nargs="+", type=int,
                    default=[100, 101, 102, 103, 104, 105])
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    source = args.source_root / f"{args.dataset}_base4"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = prepare_data(args.dataset, source, args.output.parent, args.minilm,
                        args.device)
    stage = int(args.stage)
    memories = {
        "append-edge": load(source / f"append_fixed_s{stage}_memory.pt"),
        "X+P": load(args.xp_root / f"{args.dataset}_base4/xpa_s{stage}_memory.pt"),
        "X+P+A": load(args.xpa_root / f"{args.dataset}_base4/xpa_s{stage}_memory.pt"),
    }
    result = {"dataset": args.dataset, "stage": stage,
              "student_seeds": args.student_seeds,
              "same_process": True, "edge_aware": True, "arms": {}}
    for name, memory in memories.items():
        result["arms"][name] = evaluate_memory(
            memory, stage, data, args.student_seeds, args.device)
        print(json.dumps({"event": "arm_complete", "arm": name,
                          **{k: result["arms"][name][k]
                             for k in ("final_f1", "accuracy", "old_f1", "new_f1")}}),
              flush=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True),
                           encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(args.output)}),
          flush=True)


if __name__ == "__main__":
    main()
