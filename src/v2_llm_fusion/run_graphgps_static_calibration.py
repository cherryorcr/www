"""GraphGPS scratch calibration on the frozen continual-memory protocol.

This runner is intentionally a calibration tool, not a new condensation
method.  It trains the released GraphCLIP GPS graph tower from scratch on one
of the existing memory artifacts (or the legal raw-pool reference), while the
MiniLM text embeddings remain frozen.  Training examples are materialized as
2-hop ego graphs and receive fresh 32-step random-walk positional encodings on
the graph being consumed.  Report-dev inference always rebuilds ego graphs on
the complete real TAG, matching the released GraphCLIP input convention.

The output contains an input/PE audit, source hashes, stagewise macro-F1, and
per-seed timing.  No report/test labels are used for construction, selection,
or checkpoint choice.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.utils import k_hop_subgraph

import run_unified_v3_continual as U
from small_comparison_metrics import score_logits


DEFAULT_MEMORY_ROOT = Path(
    "/home/pc/research/TAG_condensation/results/tgc_r11_c0_anchor_20260923b"
)
DEFAULT_KCENTER_ROOT = Path(
    "/home/pc/research/TAG_condensation/results/content_gap_attribution_20260922"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value) -> str:
    digest = hashlib.sha256()
    if isinstance(value, dict):
        items = sorted(value.items())
    else:
        items = [("value", value)]
    for key, item in items:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((key, str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
            digest.update(json.dumps([key, item], ensure_ascii=True).encode())
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def induced_edges(edges: torch.Tensor, ids: torch.Tensor, total: int) -> torch.Tensor:
    ids = ids.long().cpu()
    remap = torch.full((int(total),), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids), dtype=torch.long)
    local = remap[edges.long().cpu()]
    return local[:, (local >= 0).all(0)]


def make_raw_pool_memory(data: dict, stage: int) -> dict:
    seen = {int(c) for group in data["stage_classes"][:stage + 1] for c in group}
    ids = torch.tensor(
        [int(i) for i in torch.where(data["pool_mask"])[0].tolist()
         if int(data["y"][i]) in seen],
        dtype=torch.long,
    )
    if len(ids) == 0:
        raise ValueError("raw pool has no nodes for the requested stage")
    texts = [data["texts"][int(i)] for i in ids]
    text_emb = data["encoder"].encode(texts)
    return {
        "x": data["x"][ids].float().cpu(),
        "text_emb": text_emb.float().cpu(),
        "texts": texts,
        "edge_index": induced_edges(data["edge_index"], ids, len(data["x"])),
        "mass": torch.ones(len(ids)),
        "depth": torch.zeros(len(ids), dtype=torch.long),
        "stage": torch.zeros(len(ids), dtype=torch.long),
        "source_ids": ids,
        "raw_pool": True,
    }


def load_memory(args, data: dict, arm: str, stage: int) -> tuple[dict, dict]:
    if arm == "raw_pool":
        memory = make_raw_pool_memory(data, stage)
        return memory, {
            "arm": arm,
            "source": "legal raw pool induced graph",
            "source_sha256": None,
            "historical_raw_reads": 0,
            "node_count": int(len(memory["x"])),
        }
    if arm == "c0":
        name = "c0"
        source = args.memory_root / args.dataset / args.order / f"{name}_s{stage}_memory.pt"
    elif arm == "r11":
        name = args.r11_arm
        source = args.memory_root / args.dataset / args.order / f"{name}_s{stage}_memory.pt"
    elif arm == "kcenter":
        source = args.kcenter_root / args.dataset / f"kcenter_fixed_topology_s{stage}_memory.pt"
    else:
        raise ValueError(f"unknown arm: {arm}")
    if not source.exists():
        raise FileNotFoundError(source)
    memory = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(memory, dict) or "x" not in memory or "edge_index" not in memory:
        raise ValueError(f"invalid memory artifact: {source}")
    if memory["x"].ndim != 2 or memory["x"].shape[1] != 384:
        raise ValueError(f"expected 384-dimensional x in {source}")
    if memory["text_emb"].shape != memory["x"].shape:
        raise ValueError(f"text embedding shape mismatch in {source}")
    return memory, {
        "arm": arm,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "historical_raw_reads": 0,
        "node_count": int(len(memory["x"])),
        "memory_tensor_sha256": tensor_hash({
            "x": memory["x"], "text_emb": memory["text_emb"],
            "edge_index": memory["edge_index"],
        }),
    }


def ego_graph(x: torch.Tensor, edges: torch.Tensor, node: int,
              walk_length: int = 32) -> tuple[Data, int]:
    subset, local_edges, center, _ = k_hop_subgraph(
        int(node), 2, edges.long(), relabel_nodes=True, num_nodes=len(x)
    )
    graph = Data(x=x[subset].float(), edge_index=local_edges.long(),
                 num_nodes=int(len(subset)))
    graph = AddRandomWalkPE(walk_length=walk_length, attr_name="pe")(graph)
    return graph, int(center[0])


def materialize_examples(x: torch.Tensor, edges: torch.Tensor,
                         nodes: torch.Tensor, walk_length: int = 32) -> list[tuple[Data, int]]:
    return [ego_graph(x, edges, int(node), walk_length) for node in nodes.tolist()]


def pack_examples(examples: list[tuple[Data, int]], indices: list[int],
                  device: torch.device) -> tuple[Batch, torch.Tensor]:
    graphs = [examples[int(i)][0] for i in indices]
    centers_local = [examples[int(i)][1] for i in indices]
    batch = Batch.from_data_list(graphs).to(device)
    centers = batch.ptr[:-1] + torch.tensor(centers_local, dtype=torch.long,
                                            device=device)
    return batch, centers


class ScratchGPSStudent(nn.Module):
    def __init__(self, official_module):
        super().__init__()
        self.graph = official_module.GPS(
            in_dim=384,
            channels=1024,
            out_dim=384,
            pe_dim=8,
            num_layers=12,
            attn_type="multihead",
            attn_kwargs={"dropout": 0.0},
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_graph(self, batch: Batch, centers: torch.Tensor) -> torch.Tensor:
        embedding, _ = self.graph(batch.x, batch.pe, batch.edge_index,
                                  batch.batch, centers)
        return F.normalize(embedding.float(), dim=1)


def contrastive_loss(graph_emb: torch.Tensor, text_emb: torch.Tensor,
                     logit_scale: torch.Tensor) -> torch.Tensor:
    text_emb = F.normalize(text_emb.float(), dim=1)
    logits = logit_scale.exp().clamp(max=100) * (graph_emb @ text_emb.T)
    target = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, target) +
            F.cross_entropy(logits.T, target)) * 0.5


@torch.no_grad()
def eval_report(model: ScratchGPSStudent, data: dict, report_examples,
                report_nodes: torch.Tensor, prompts: torch.Tensor,
                device: torch.device, batch_size: int) -> tuple[list[dict], dict]:
    model.eval()
    logits = []
    for start in range(0, len(report_nodes), batch_size):
        index = list(range(start, min(start + batch_size, len(report_nodes))))
        batch, centers = pack_examples(report_examples, index, device)
        emb = model.encode_graph(batch, centers)
        logits.append((emb @ F.normalize(prompts.float(), dim=1).T).cpu())
    logits = torch.cat(logits, dim=0)
    target = data["y"][report_nodes].numpy()
    stage_metrics = [score_logits(logits, target, data["stage_classes"], stage)
                     for stage in range(len(data["stage_classes"]))]
    summary = {
        "final_all_cumulative_f1": float(stage_metrics[-1]["all"]["cumulative_f1"]),
        "final_seen_cumulative_f1": float(stage_metrics[-1]["seen"]["cumulative_f1"]),
        "stage_average_all_cumulative_f1": float(np.mean(
            [item["all"]["cumulative_f1"] for item in stage_metrics])),
        "stage_average_seen_cumulative_f1": float(np.mean(
            [item["seen"]["cumulative_f1"] for item in stage_metrics])),
    }
    return stage_metrics, summary


def train_one(memory: dict, data: dict, official_module, seed: int,
              steps: int, batch_size: int, eval_batch_size: int,
              device: torch.device, walk_length: int) -> dict:
    seed_all(seed)
    model = ScratchGPSStudent(official_module).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    nodes = torch.arange(len(memory["x"]), dtype=torch.long)
    train_examples = materialize_examples(memory["x"].cpu(),
                                           memory["edge_index"].cpu(),
                                           nodes, walk_length)
    report_nodes = data["report_dev_mask"].nonzero(as_tuple=True)[0].cpu()
    report_examples = materialize_examples(data["x"].cpu(),
                                           data["edge_index"].cpu(),
                                           report_nodes, walk_length)
    prompts = data["encoder"].encode(data["prompts"]).to(device)
    text = memory["text_emb"].float().to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed * 1009 + 17)
    losses = []
    started = time.perf_counter()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for step in range(1, int(steps) + 1):
        count = min(int(batch_size), len(train_examples))
        selected = torch.randperm(len(train_examples), generator=generator)[:count]
        batch, centers = pack_examples(train_examples, selected.tolist(), device)
        graph_emb = model.encode_graph(batch, centers)
        loss = contrastive_loss(graph_emb, text[selected.to(device)], model.logit_scale)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    elapsed = time.perf_counter() - started
    stage_metrics, summary = eval_report(
        model, data, report_examples, report_nodes, prompts, device, eval_batch_size
    )
    peak = (float(torch.cuda.max_memory_allocated(device) / 2**20)
            if torch.cuda.is_available() and device.type == "cuda" else 0.0)
    result = {
        "seed": int(seed),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "seconds": float(elapsed),
        "peak_allocated_mb": peak,
        "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]),
        "loss_min": float(min(losses)),
        "stage_metrics": stage_metrics,
        "summary": summary,
        "train_nodes": int(len(memory["x"])),
        "train_graph_count": int(len(train_examples)),
        "train_graph_nodes_total": int(sum(int(g.num_nodes) for g, _ in train_examples)),
        "report_nodes": int(len(report_nodes)),
    }
    del model, optimizer, train_examples, report_examples
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    parser.add_argument("--order", default="order19601")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=["c0"])
    parser.add_argument("--student-seeds", nargs="+", type=int, default=[150])
    parser.add_argument(
        "--student-steps", type=int, default=0,
        help="override steps; 0 uses max(50,min(2048,16*rows)) per arm",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--walk-length", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--graphclip-root", type=Path,
                        default=Path("/home/pc/research/GraphCLIP"))
    parser.add_argument("--minilm", type=Path, default=Path(
        "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"))
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--memory-root", type=Path, default=DEFAULT_MEMORY_ROOT)
    parser.add_argument("--kcenter-root", type=Path, default=DEFAULT_KCENTER_ROOT)
    parser.add_argument("--r11-arm", default="anchor025")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true",
                        help="mark this as a protocol smoke test; does not change the input")
    args = parser.parse_args()
    if args.stage < 0 or args.stage > 2:
        parser.error("stage must be 0, 1, or 2")
    if args.student_steps < 0 or args.batch_size < 1:
        parser.error("student-steps must be non-negative and batch-size positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    seed_all(0)
    data_path = args.data_path or Path(
        "/home/pc/research/GraphCLIP/processed_data" / Path(f"{args.dataset}.pt")
    )
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    data = U.load_unified(args.dataset, str(data_path), str(args.minilm), str(device))
    sys.path.insert(0, str(args.graphclip_root))
    official = importlib.import_module("models.gt")
    report_nodes = data["report_dev_mask"].nonzero(as_tuple=True)[0]
    result = {
        "status": "running",
        "dataset": args.dataset,
        "order": args.order,
        "stage": int(args.stage),
        "arms": list(args.arms),
        "student_seeds": list(args.student_seeds),
        "student_steps_override": int(args.student_steps),
        "student_steps_rule": "override if >0 else max(50,min(2048,16*memory_rows))",
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "model": "released GraphCLIP GPS-12L-1024 scratch graph tower + frozen MiniLM embeddings",
        "training_objective": "symmetric graph-text InfoNCE; no labels in training",
        "split": "GraphCLIP fixed split; report-dev only for metrics",
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "stage_classes": data["stage_classes"],
        "split_counts": {key: int(data[key].sum()) for key in
                         ("pool_mask", "selector_dev_mask", "report_dev_mask", "test_mask")},
        "report_nodes": int(len(report_nodes)),
        "pe": {
            "type": "random-walk positional encoding",
            "walk_length": int(args.walk_length),
            "recomputed_on_consumed_graph": True,
            "training_graph": "2-hop ego per memory node",
            "report_graph": "2-hop ego per report-dev node on complete real TAG",
        },
        "label_access": {"construction": False, "candidate_selection": False,
                          "checkpoint_selection": False, "report_only": True},
        "source_files": {
            "runner": str(Path(__file__)),
            "runner_sha256": sha256_file(Path(__file__)),
            "unified_loader": str(Path(U.__file__)),
            "unified_loader_sha256": sha256_file(Path(U.__file__)),
            "official_model": str(Path(official.__file__)),
            "official_model_sha256": sha256_file(Path(official.__file__)),
        },
        "arm_audits": {},
        "runs": [],
        "smoke": bool(args.smoke),
        "started_at": time.time(),
    }
    for arm in args.arms:
        memory, audit = load_memory(args, data, arm, args.stage)
        result["arm_audits"][arm] = audit
        print(json.dumps({"event": "arm_loaded", "arm": arm,
                          "nodes": int(len(memory["x"])),
                          "edges": int(memory["edge_index"].shape[1])}), flush=True)
        steps = (int(args.student_steps) if args.student_steps > 0 else
                 max(50, min(2048, 16 * len(memory["x"]))))
        for seed in args.student_seeds:
            row = train_one(memory, data, official, int(seed), steps,
                            args.batch_size, args.eval_batch_size, device,
                            args.walk_length)
            row.update({"arm": arm, "stage": int(args.stage)})
            result["runs"].append(row)
            print(json.dumps({"event": "run_complete", "arm": arm,
                              "seed": seed, **row["summary"]}), flush=True)
    result["status"] = "complete"
    result["finished_at"] = time.time()
    result["seconds"] = float(result["finished_at"] - result["started_at"])
    save_json(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
