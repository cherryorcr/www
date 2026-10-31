"""Evaluate append-fixed memories with the exact edge-aware Student path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import run_tgc_semantic_recursive_large_20260926 as L
import run_unified_v3_continual as U


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                    encoding="utf-8")


def run_students(memory: dict, stage: int, data: dict, seeds: list[int],
                 device: str) -> dict:
    syn = {
        "x": memory["x"],
        "text_emb": memory["text_emb"],
        "texts": memory["texts"],
        "edge_index": memory["edge_index"],
        "weights": memory.get("mass"),
    }
    rows = []
    for seed in seeds:
        steps = U.protocol_v3.student_steps(len(syn["x"]))
        model = U.G.train_student(syn, int(seed), device, steps, batch=16)
        rows.append(U.G.eval_stagewise(
            model, data["x_device"], data["real_adj"], data["prompt_emb"],
            data["y"], data["report_dev_mask"], stage))
        del model
    return {
        "roots": int(len(syn["x"])),
        "student_steps": int(steps),
        "student_seeds": seeds,
        "student_training_runs": len(seeds),
        "cum_dev_f1_mean": float(sum(r["cumulative_macro_f1"] for r in rows)
                                  / len(rows)),
        "current_f1_mean": float(sum(r["current_f1"] for r in rows)
                                  / len(rows)),
        "old_mean_f1_mean": (
            float(sum(r["old_mean_f1"] for r in rows) / len(rows))
            if stage > 0 else None),
        "per_stage_mean": [
            float(sum(r["per_stage"][s]["macro_f1"] for r in rows)
                  / len(rows)) for s in range(stage + 1)],
        "per_student": rows,
    }


def evaluate_dataset(dataset: str, source_root: Path, output_root: Path,
                     minilm: str, device: str, seeds: list[int],
                     max_stage: int) -> None:
    source = source_root / f"{dataset}_base4"
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
    data = L.set_pool_text_embeddings(data, source, output_root)
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = U.G.norm_adj(data["edge_index"], len(data["x"]), device)
    data["x_device"] = data["x"].float().to(device)
    out = output_root / f"{dataset}_base4"
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for stage in range(max_stage + 1):
        memory = load(source / f"append_fixed_s{stage}_memory.pt")
        summary = run_students(memory, stage, data, seeds, device)
        rows.append({"dataset": dataset, "stage": stage,
                     "aggregate": {
                         "mean_f1": summary["cum_dev_f1_mean"],
                         "mean_accuracy": float(sum(
                             r["cumulative_accuracy"]
                             for r in summary["per_student"])
                             / len(summary["per_student"])),
                         "old_f1": summary["old_mean_f1_mean"],
                         "new_f1": summary["current_f1_mean"],
                     },
                     "summary": summary})
        print(json.dumps({"event": "stage_complete", "dataset": dataset,
                          "stage": stage,
                          "final_f1": summary["cum_dev_f1_mean"]}),
              flush=True)
    save(out / "protocol.json", {
        "name": "edge-aware-append-fixed-reference-20261001",
        "dataset": dataset, "stage_protocol": "base+4",
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "student_graph_mode": "edge-aware", "student_seeds": seeds,
        "historical_raw_reads": 0, "memory_source": "append_fixed",
    })
    save(out / "results.json", {"rows": rows, "complete": True})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--minilm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-stage", type=int, default=4)
    ap.add_argument("--student-seeds", nargs="+", type=int,
                    default=[100, 101, 102, 103, 104, 105])
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    for dataset in args.datasets:
        evaluate_dataset(dataset, args.source_root, args.output_root,
                         args.minilm, args.device, args.student_seeds,
                         args.max_stage)
    print(json.dumps({"event": "complete", "output": str(args.output_root)}),
          flush=True)


if __name__ == "__main__":
    main()
