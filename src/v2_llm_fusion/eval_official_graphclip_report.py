"""Frozen official GraphCLIP checkpoint inference on current report-dev nodes.

The local adapter uses complete two-hop ego graphs from the original TAG.
It does not retrain, select a checkpoint, or claim an official benchmark
reproduction. Stagewise scores are slices of one frozen model, not continual
learning runs. Test-node features may supply transductive graph context;
test labels never enter metrics or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.utils import k_hop_subgraph

import run_unified_v3_continual as U
from small_comparison_metrics import score_logits


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_state(state):
    selected = {k.removeprefix("graph_model."): v for k, v in state.items()
                if k.startswith("graph_model.")}
    if not selected:
        raise ValueError("checkpoint has no graph_model weights")
    return selected


def state_hash(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(str((key, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def report_nodes(data):
    mask = data["report_dev_mask"]
    if not mask.any():
        raise ValueError("empty report-dev split")
    for split in ("pool_mask", "selector_dev_mask", "test_mask"):
        if (mask & data[split]).any():
            raise ValueError(f"report-dev overlaps {split}")
    return mask.nonzero(as_tuple=True)[0]


def ego_graph(x, edges, node, subgraph_cap=None):
    subset, local_edges, center, _ = k_hop_subgraph(
        int(node), 2, edges, relabel_nodes=True, num_nodes=len(x))
    if subgraph_cap is not None and len(subset) > subgraph_cap:
        if subgraph_cap < 2:
            raise ValueError("subgraph_cap must be at least 2")
        center_local = int(center[0])
        rest = torch.arange(len(subset), device=subset.device)
        rest = rest[rest != center_local]
        keep_local = torch.cat((torch.tensor([center_local], device=subset.device),
                                rest[:subgraph_cap - 1]))
        remap = torch.full((len(subset),), -1, dtype=torch.long,
                           device=subset.device)
        remap[keep_local] = torch.arange(len(keep_local), device=subset.device)
        edge_keep = (remap[local_edges[0]] >= 0) & (remap[local_edges[1]] >= 0)
        local_edges = remap[local_edges[:, edge_keep]]
        subset = subset[keep_local]
        center = torch.tensor([0], device=subset.device)
    graph = Data(x=x[subset], edge_index=local_edges, num_nodes=len(subset))
    graph = AddRandomWalkPE(walk_length=32, attr_name="pe")(graph)
    return graph, int(center[0])


def pack_graphs(items, device):
    graphs, local_centers = zip(*items)
    batch = Batch.from_data_list(list(graphs)).to(device)
    centers = batch.ptr[:-1] + torch.tensor(local_centers, device=device)
    return batch, centers


@torch.inference_mode()
def predict(model, x, edges, nodes, prompt_emb, batch_size, device,
            subgraph_cap=None):
    logits, subgraphs = [], []
    for start in range(0, len(nodes), batch_size):
        chunk = nodes[start:start + batch_size]
        items = [ego_graph(x, edges, int(node), subgraph_cap=subgraph_cap)
                 for node in chunk]
        for node, (graph, _) in zip(chunk, items):
            subgraphs.append({"node_id": int(node), "nodes": graph.num_nodes,
                              "edges": graph.num_edges})
        batch, centers = pack_graphs(items, device)
        embedding, _ = model(batch.x, batch.pe, batch.edge_index,
                             batch.batch, centers)
        value = F.normalize(embedding.float(), dim=1) @ prompt_emb.T
        if not torch.isfinite(value).all():
            raise ValueError("non-finite checkpoint logits")
        logits.append(value.cpu())
        if start == 0 or (start // batch_size + 1) % 20 == 0:
            print(json.dumps({"completed": start + len(chunk), "total": len(nodes),
                              "batch_nodes": batch.num_nodes}), flush=True)
    return torch.cat(logits), subgraphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cora", "citeseer", "arxiv",
                                                "sportsfit", "bookhis", "wikics"],
                        required=True)
    parser.add_argument("--graphclip-root", type=Path,
                        default=Path("/home/pc/research/GraphCLIP"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--minilm", type=Path, default=Path(
        "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-graphs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data", type=Path,
                        help="optional TAG data file; otherwise use the local registry")
    parser.add_argument("--max-report-nodes", type=int, default=0,
                        help="deterministic prefix cap for large-graph reference runs")
    parser.add_argument("--record-subgraphs", action="store_true",
                        help="store per-node ego sizes; disabled by default for large graphs")
    parser.add_argument("--subgraph-cap", type=int, default=None,
                        help="deterministic cap for large-graph ego references")
    args = parser.parse_args()
    if args.batch_graphs < 1 or args.threads < 1:
        parser.error("batch-graphs and threads must be positive")
    if args.output.exists():
        parser.error("output already exists; use a new path")
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.graphclip_root))
    official = importlib.import_module("models.gt")
    checkpoint = args.checkpoint or args.graphclip_root / "checkpoints/pretrained_graphclip.pt"
    if args.data is None:
        if args.dataset in {"cora", "citeseer"}:
            data_path = args.graphclip_root / "processed_data" / f"{args.dataset}.pt"
        else:
            data_path = Path(U.DATA_CFG[args.dataset]["path"])
    else:
        data_path = args.data
    data = U.load_unified(args.dataset, str(data_path), str(args.minilm), args.device)
    nodes = report_nodes(data)
    if args.max_report_nodes:
        if args.max_report_nodes < 1:
            parser.error("--max-report-nodes must be positive when supplied")
        nodes = nodes[:args.max_report_nodes]
    if data["x"].shape[1] != 384:
        raise ValueError(
            f"official GraphCLIP checkpoint expects 384-dim node features; "
            f"{args.dataset} provides {data['x'].shape[1]}"
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    graph = official.GPS(in_dim=384, channels=1024, out_dim=384, pe_dim=8,
                         num_layers=12, attn_type="multihead",
                         attn_kwargs={"dropout": 0.0})
    graph.load_state_dict(graph_state(state), strict=True)
    text_state = {k.removeprefix("text_model."): v for k, v in state.items()
                  if k.startswith("text_model.")}
    if not text_state:
        raise ValueError("checkpoint has no text_model weights")
    text_model = data["encoder"].model
    prior = text_model.state_dict()
    mismatches = [k for k, v in text_state.items()
                  if k not in prior or not torch.equal(prior[k].cpu(), v)]
    text_model.load_state_dict(text_state, strict=True)
    text_model.requires_grad_(False).eval()
    graph.requires_grad_(False).to(args.device).eval()
    graph_before, text_before = state_hash(graph), state_hash(text_model)
    parameters = {"graph": sum(p.numel() for p in graph.parameters()),
                  "text": sum(p.numel() for p in text_model.parameters())}
    del state, text_state, prior
    prompts = F.normalize(data["encoder"].encode(data["prompts"]), dim=1).to(args.device)
    print(json.dumps({"dataset": args.dataset, "report_nodes": len(nodes),
                      "checkpoint_loaded_strict": True, "parameters": parameters,
                      "text_cache_mismatches_before_load": len(mismatches)}), flush=True)
    logits, subgraphs = predict(graph, data["x"], data["edge_index"], nodes,
                               prompts, args.batch_graphs, args.device,
                               subgraph_cap=args.subgraph_cap)
    # One repeated forward checks that evaluation has no dropout or state updates.
    repeated, _ = predict(graph, data["x"], data["edge_index"], nodes[:1],
                          prompts, 1, args.device,
                          subgraph_cap=args.subgraph_cap)
    repeat_error = float((repeated[0] - logits[0]).abs().max())
    if repeat_error > 1e-5:
        raise ValueError(f"repeated inference mismatch: {repeat_error}")
    if state_hash(graph) != graph_before or state_hash(text_model) != text_before:
        raise ValueError("frozen checkpoint state changed during inference")
    target = data["y"][nodes].numpy()
    metrics = [score_logits(logits, target, data["stage_classes"], stage)
               for stage in range(len(data["stage_classes"]))]
    sources = [Path(__file__), Path(U.__file__), Path(U.protocol_v3.__file__),
               Path(__file__).with_name("small_comparison_metrics.py"),
               Path(official.__file__), checkpoint, data_path]
    tokenizer_sources = [p for p in args.minilm.iterdir()
                         if p.is_file() and p.suffix in (".json", ".txt")]
    result = {
        "dataset": args.dataset, "arm": "official_graphclip_frozen_checkpoint",
        "status": "complete", "training_steps": 0, "checkpoint_selection": False,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "graph_load_strict": True, "text_load_strict": True,
        "model": "released GPS-12L-1024 + released MiniLM text tower",
        "parameter_counts": parameters, "graph_state_sha256": graph_before,
        "text_state_sha256": text_before, "model_state_unchanged": True,
        "text_cache_mismatches_before_load": mismatches,
        "prompts": data["prompts"], "stage_classes": data["stage_classes"],
        "split": "report_dev: original_node_id % 10 == 3",
        "split_counts": {key: int(data[key].sum()) for key in
                         ("pool_mask", "selector_dev_mask", "report_dev_mask", "test_mask")},
        "inference": {"adapter": "local; not a claim of official protocol reproduction",
                      "source_graph": "complete original TAG", "hops": 2,
                      "subgraph_cap": args.subgraph_cap,
                      "fallback_to_one_hop": False,
                      "rw_pe_length": 32, "batch_graphs": args.batch_graphs,
                      "graph_readout": "official concatenated ego-mean and center",
                      "text_readout": "attention-mask mean pooling + L2 normalization",
                      "extra_text_projection": False, "similarity": "cosine",
                      "stagewise_mode": "slices of one frozen checkpoint, not continual training",
                      "device": args.device, "dtype": "float32"},
        "test_labels_used_for_metrics_or_selection": False,
        "selector_labels_used": False, "report_labels_used_for_selection": False,
        "transductive_full_graph_features_used": True,
        "pretraining_dataset_overlap": "not audited; do not interpret as a strict task upper bound",
        "metrics": metrics, "final": metrics[-1],
        "stage_average": {scope: float(np.mean([m[scope]["cumulative_f1"] for m in metrics]))
                          for scope in ("all", "seen")},
        "predictions": {"node_ids": nodes.tolist(), "targets": target.tolist(),
                        "logits": logits.tolist(), "predicted": logits.argmax(1).tolist()},
        "subgraphs": subgraphs if args.record_subgraphs else {
            "recorded": False,
            "count": len(subgraphs),
            "mean_nodes": float(np.mean([v["nodes"] for v in subgraphs])),
            "mean_edges": float(np.mean([v["edges"] for v in subgraphs])),
            "max_nodes": int(max(v["nodes"] for v in subgraphs)),
            "max_edges": int(max(v["edges"] for v in subgraphs)),
        }, "repeat_first_node_max_abs_error": repeat_error,
        "seconds": time.perf_counter() - started,
        "source_hashes": {str(p): sha(p) for p in sources + tokenizer_sources},
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({k: result[k] for k in ("dataset", "final", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
