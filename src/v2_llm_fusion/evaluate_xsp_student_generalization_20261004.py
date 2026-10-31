"""Evaluate frozen XSP memories with multiple downstream Students.

This script deliberately does not reconstruct memory.  It loads the frozen
Current-XSP and ST-Sinkhorn artifacts from one construction seed and changes
only the downstream graph-text Student architecture.  All variants use the
same InfoNCE objective, batch size, optimization steps, and report-dev
protocol.  The result is a Student-generalization check, not a new method
search.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_gate1_continual as G
import run_public_continual_baselines_large_20261001 as BASE
import run_unified_v3_continual as U


class ShallowCLIPStudent(torch.nn.Module):
    """One graph propagation layer followed by the shared text projection."""

    def __init__(self, in_dim: int, out: int = 384, text_dim: int = 384):
        super().__init__()
        self.graph = torch.nn.Linear(in_dim, out)
        self.text_proj = torch.nn.Linear(text_dim, out)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_graph(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return F.normalize(adj @ self.graph(x), dim=1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_proj(t), dim=1)

    def logits(self, x: torch.Tensor, adj: torch.Tensor,
               t: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100) * (
            self.encode_graph(x, adj) @ self.encode_text(t).T)


class LinearCLIPStudent(torch.nn.Module):
    """Graph-free linear control with the same text and InfoNCE heads."""

    def __init__(self, in_dim: int, out: int = 384, text_dim: int = 384):
        super().__init__()
        self.graph = torch.nn.Linear(in_dim, out)
        self.text_proj = torch.nn.Linear(text_dim, out)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_graph(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        del adj
        return F.normalize(self.graph(x), dim=1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_proj(t), dim=1)

    def logits(self, x: torch.Tensor, adj: torch.Tensor,
               t: torch.Tensor) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100) * (
            self.encode_graph(x, adj) @ self.encode_text(t).T)


STUDENT_BUILDERS = {
    "canonical_2layer": lambda dim: G.CLIPStudent(dim),
    "shallow_1layer": lambda dim: ShallowCLIPStudent(dim),
    "linear": lambda dim: LinearCLIPStudent(dim),
}


def load_memory(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def train_variant(memory: dict, student_name: str, seed: int, device: str,
                  steps: int, batch: int):
    torch.manual_seed(int(seed))
    model = STUDENT_BUILDERS[student_name](int(memory["x"].shape[1])).to(device)
    x = memory["x"].float().to(device)
    text = memory["text_emb"].float().to(device)
    edge_index = memory.get("edge_index", torch.empty((2, 0), dtype=torch.long))
    adj = G.norm_adj(edge_index.long(), len(x), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(int(seed) * 7 + 1)
    n = len(x)
    for _ in range(int(steps)):
        rows = torch.randperm(n, generator=generator)[:min(int(batch), n)].to(device)
        graph = model.encode_graph(x, adj)[rows]
        target_text = model.encode_text(text[rows])
        logits = model.logit_scale.exp().clamp(max=100) * (graph @ target_text.T)
        target = torch.arange(len(rows), device=device)
        loss = (F.cross_entropy(logits, target)
                + F.cross_entropy(logits.T, target)) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def score_stage(memory: dict, stage: int, data: dict, student_name: str,
                seeds: list[int], device: str, batch: int,
                steps_override: int | None) -> dict:
    steps = U.protocol_v3.student_steps(len(memory["x"]),
                                        override=steps_override)
    scores = []
    started = time.perf_counter()
    for seed in seeds:
        model = train_variant(memory, student_name, int(seed), device, steps, batch)
        scores.append(BASE.eval_stagewise_exact(
            model, data, data["stage_classes"], stage))
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    def mean(key: str) -> float:
        values = [float(item[key]) for item in scores]
        return float(np.mean(values))

    def std(key: str) -> float:
        values = [float(item[key]) for item in scores]
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    final = scores[-1]
    return {
        "student": student_name,
        "stage": int(stage),
        "student_steps": int(steps),
        "student_batch": int(batch),
        "student_seeds": [int(v) for v in seeds],
        "student_seconds": float(time.perf_counter() - started),
        "per_student": scores,
        "final_f1": mean("cumulative_macro_f1"),
        "final_f1_std": std("cumulative_macro_f1"),
        "final_accuracy": mean("cumulative_accuracy"),
        "final_accuracy_std": std("cumulative_accuracy"),
        "old_f1": (mean("old_mean_f1") if stage > 0 else None),
        "new_f1": mean("current_f1"),
        "stage_avg_f1": mean("stage_avg_f1"),
        "stage_avg_accuracy": mean("stage_avg_accuracy"),
        "worst_stage_f1": min(
            float(item["cumulative_macro_f1"]) for item in scores),
        "terminal": {
            "f1": float(final["cumulative_macro_f1"]),
            "accuracy": float(final["cumulative_accuracy"]),
        },
    }


def protocol_from_result(root: Path, dataset: str) -> dict:
    payload = json.loads((root / f"{dataset}_base4" / "results.json").read_text(
        encoding="utf-8"))
    return payload["protocol"]


def run_dataset(dataset: str, source_root: Path, text_cache_root: Path | None,
                minilm: str, output_root: Path,
                arms: list[str], students: list[str], seeds: list[int],
                device: str, batch: int, steps_override: int | None) -> dict:
    protocol = protocol_from_result(source_root, dataset)
    groups = BASE.stage_groups(protocol)
    data = U.load_unified(
        dataset, U.DATA_CFG[dataset]["path"], minilm, device,
        n_stages=5, split_mode="base_incremental", base_fraction=0.4,
        report_dev_mod=int(protocol.get("report_dev_mod", 3)))
    if text_cache_root is not None:
        cache_root = text_cache_root / f"{dataset}_base4"
        if not (cache_root / "text_emb_pool_only.pt").exists():
            cache_root = text_cache_root / dataset
    else:
        cache_root = source_root / dataset
    data = BASE.set_pool_text_embeddings(data, cache_root)
    BASE.configure_eval(data, groups, device)
    out_dir = output_root / f"{dataset}_base4"
    out_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for arm in arms:
        for student_name in students:
            arm_rows = []
            for stage in range(len(groups)):
                path = (source_root / f"{dataset}_base4"
                        / f"{arm}_s{stage}_memory.pt")
                if not path.exists():
                    raise FileNotFoundError(path)
                memory = load_memory(path)
                score = score_stage(memory, stage, data, student_name,
                                    seeds, device, batch, steps_override)
                record = {
                    "dataset": dataset,
                    "stage": int(stage),
                    "arm": arm,
                    "memory_rows": int(len(memory["x"])),
                    "memory_file": path.name,
                    "score": score,
                }
                rows.append(record)
                arm_rows.append(record)
                print(json.dumps({
                    "event": "student_generalization_stage_complete",
                    "dataset": dataset, "arm": arm,
                    "student": student_name, "stage": stage,
                    "f1": score["final_f1"],
                    "accuracy": score["final_accuracy"],
                }), flush=True)
            terminal = arm_rows[-1]["score"]
            rows_for_arm = [item["score"] for item in arm_rows]
            rows.append({
                "dataset": dataset,
                "arm": arm,
                "student": student_name,
                "overall": {
                    "final_f1": terminal["final_f1"],
                    "final_f1_std": terminal["final_f1_std"],
                    "final_accuracy": terminal["final_accuracy"],
                    "final_accuracy_std": terminal["final_accuracy_std"],
                    "old_f1": terminal["old_f1"],
                    "new_f1": terminal["new_f1"],
                    "stage_avg_f1": terminal["stage_avg_f1"],
                    "stage_avg_accuracy": terminal["stage_avg_accuracy"],
                    "worst_stage_f1": min(
                        float(item["final_f1"]) for item in rows_for_arm),
                    "student_seconds": float(sum(
                        item["student_seconds"] for item in rows_for_arm)),
                },
            })
    result = {
        "protocol": {
            "name": "xsp-student-generalization-20261004",
            "dataset": dataset,
            "groups": groups,
            "budgets": protocol["budgets"],
            "source_root": str(source_root),
            "text_cache_root": (None if text_cache_root is None
                                 else str(text_cache_root)),
            "arms": arms,
            "students": students,
            "student_seeds": seeds,
            "student_batch": int(batch),
            "student_steps": "max(50,min(2048,16*memory_rows))",
            "memory_reconstructed": False,
            "historical_raw_reads": 0,
            "report_test_labels_used_for_construction": False,
        },
        "dataset": dataset,
        "rows": rows,
    }
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    return result


def summarize(results: list[dict], output_root: Path) -> dict:
    overall = []
    for result in results:
        for item in result["rows"]:
            if "overall" in item:
                overall.append({"dataset": result["dataset"], **item})
    keys = ["final_f1", "final_accuracy", "old_f1", "new_f1",
            "stage_avg_f1", "stage_avg_accuracy"]
    table = []
    methods = sorted({(item["arm"], item["student"]) for item in overall})
    for arm, student in methods:
        subset = [item["overall"] for item in overall
                  if item["arm"] == arm and item["student"] == student]
        row = {"arm": arm, "student": student}
        for key in keys:
            values = [item[key] for item in subset if item[key] is not None]
            row[key] = float(np.mean(values)) if values else None
        row["datasets"] = len(subset)
        table.append(row)

    paired = []
    by_key = {(item["dataset"], item["arm"], item["student"]): item["overall"]
              for item in overall}
    for dataset in sorted({item["dataset"] for item in overall}):
        for student in sorted({item["student"] for item in overall}):
            current = by_key.get((dataset, "cat_mmd_tag_x_slot_partition", student))
            st = by_key.get((dataset, "cat_mmd_tag_x_slot_partition_st", student))
            if current is None or st is None:
                continue
            paired.append({
                "dataset": dataset,
                "student": student,
                "f1_delta_st_vs_current": float(st["final_f1"] - current["final_f1"]),
                "accuracy_delta_st_vs_current": float(st["final_accuracy"] - current["final_accuracy"]),
                "old_f1_delta_st_vs_current": float(st["old_f1"] - current["old_f1"]),
                "new_f1_delta_st_vs_current": float(st["new_f1"] - current["new_f1"]),
            })
    summary = {
        "protocol": results[0]["protocol"],
        "datasets": [result["dataset"] for result in results],
        "overall": table,
        "paired": paired,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    lines = [
        "# XSP Student generalization",
        "",
        "Frozen Current-XSP and ST-Sinkhorn memories; only the downstream Student changes.",
        "",
        "| Arm | Student | Final F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Stage avg Accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"|{row['arm']}|{row['student']}|{row['final_f1']:.2f}|"
            f"{row['final_accuracy']:.2f}|{row['old_f1']:.2f}|"
            f"{row['new_f1']:.2f}|{row['stage_avg_f1']:.2f}|"
            f"{row['stage_avg_accuracy']:.2f}|")
    lines += ["", "| Dataset | Student | ST - Current F1 | ST - Current Accuracy | ST - Current Old F1 | ST - Current New F1 |",
              "|---|---|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(
            f"|{row['dataset']}|{row['student']}|"
            f"{row['f1_delta_st_vs_current']:+.2f}|"
            f"{row['accuracy_delta_st_vs_current']:+.2f}|"
            f"{row['old_f1_delta_st_vs_current']:+.2f}|"
            f"{row['new_f1_delta_st_vs_current']:+.2f}|")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--text-cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", default=[
        "cat_mmd_tag_x_slot_partition", "cat_mmd_tag_x_slot_partition_st"])
    parser.add_argument("--students", nargs="+", default=list(STUDENT_BUILDERS))
    parser.add_argument("--student-seeds", nargs="+", type=int,
                        default=[100, 101, 102, 103, 104, 105])
    parser.add_argument("--student-batch", type=int, default=16)
    parser.add_argument("--student-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    torch.set_num_threads(2)
    results = []
    for dataset in args.datasets:
        results.append(run_dataset(
            dataset, args.source_root, args.text_cache_root, args.minilm,
            args.output_root,
            args.arms, args.students, args.student_seeds, args.device,
            args.student_batch, args.student_steps))
    summary = summarize(results, args.output_root)
    print(json.dumps({"event": "complete", "output": str(args.output_root),
                      "overall": summary["overall"]}, ensure_ascii=True),
          flush=True)


if __name__ == "__main__":
    main()
