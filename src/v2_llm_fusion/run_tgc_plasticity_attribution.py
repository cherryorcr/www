"""P0 attribution for the new-class plasticity gap.

This script is intentionally read-only with respect to sealed R15/R17
artifacts.  It reloads the synthetic memories and the already evaluated
student checkpoints, then measures content margin, real-graph margin,
cross-block normalized message mass, and old-block propagation drift.

No report score is used to construct or select a memory.  The final F1 values
are joined only after all proxy measurements have been computed, so this
script is a diagnosis rather than another selector.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_tgc_r4_relational as R4
import run_tgc_r15_r11_cross as R15
import run_tgc_r11_c0_anchor as R11


R, T, D, B = R15.R, R15.T, R15.D, R15.B


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def scalar(value) -> float:
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=np.float64)
    out[order] = np.arange(len(values), dtype=np.float64)
    return out


def corr(x, y) -> dict:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(len(x)), "pearson": None, "spearman": None}
    return {
        "n": int(len(x)),
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(rank(x), rank(y))[0, 1]),
    }


def row_classes(data: dict, ids: torch.Tensor, memory: dict, old_n: int,
                stage: int, assignment_root: Path, arm: str, device: str) -> torch.Tensor:
    """Recover class labels for synthetic rows from construction assignments.

    C0 assignment is reconstructed from the registered C0 source partition.
    R11 rows use the sealed soft assignment.  This is only an attribution
    label for measuring margins; it is never used to alter a memory.
    """
    if stage == 0:
        assignment = R4.current_assignment(data, ids, memory, 0, stage, device)
        local_classes = data["y"][ids]
        result = []
        for row in range(len(memory["x"])):
            result.append(int(torch.mode(local_classes[assignment == row]).values))
        return torch.tensor(result, dtype=torch.long)

    if arm in {"anchor025", "anchor_content_c0_topology", "message_gate_2", "anchor_full"}:
        path = assignment_root / data["dataset"] / CURRENT_ORDER / (
            f"anchor025_s{stage}_assignment.pt")
        if not path.exists():
            raise FileNotFoundError(f"sealed R11 assignment is missing: {path}")
        p = load(path).float()
        labels = data["y"][ids].long()
        classes = torch.arange(int(data["prompts"].__len__()), dtype=torch.long)
        one_hot = F.one_hot(labels, num_classes=len(classes)).float()
        return (p.T @ one_hot).argmax(1).long()

    assignment = R4.current_assignment(data, ids, memory, old_n, stage, device)
    labels = data["y"][ids]
    result = []
    for row in range(len(memory["x"]) - old_n):
        result.append(int(torch.mode(labels[assignment == row]).values))
    return torch.tensor(result, dtype=torch.long)


def normalize_adj(edge_index: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.cat((edge_index.cpu(), torch.arange(n).repeat(2, 1)), 1)
    values = torch.ones(idx.shape[1], dtype=torch.float32)
    deg = torch.zeros(n, dtype=torch.float32).scatter_add_(0, idx[0], values)
    dinv = deg.clamp_min(1e-9).pow(-0.5)
    values = dinv[idx[0]] * values * dinv[idx[1]]
    return idx, values, deg


def message_mass(memory: dict, old_n: int) -> dict:
    n = len(memory["x"])
    idx, values, _ = normalize_adj(memory["edge_index"], n)
    recv, src = idx[0], idx[1]
    row_total = torch.zeros(n).scatter_add_(0, recv, values.abs())
    new_recv_old = (recv >= old_n) & (src < old_n)
    old_recv_new = (recv < old_n) & (src >= old_n)
    new_rows = torch.arange(old_n, n)
    old_rows = torch.arange(0, old_n)
    new_cross = torch.zeros(n).scatter_add_(0, recv[new_recv_old], values[new_recv_old].abs())
    old_cross = torch.zeros(n).scatter_add_(0, recv[old_recv_new], values[old_recv_new].abs())
    return {
        "cross_edge_entries": int(new_recv_old.sum()),
        "new_receive_old_ratio_mean": scalar((new_cross[new_rows] / row_total[new_rows].clamp_min(1e-9)).mean()) if len(new_rows) else None,
        "old_receive_new_ratio_mean": scalar((old_cross[old_rows] / row_total[old_rows].clamp_min(1e-9)).mean()) if len(old_rows) else None,
        "cross_weight_total": scalar(values[new_recv_old].abs().sum()),
        "all_weight_total": scalar(values.abs().sum()),
        "new_rows": int(len(new_rows)),
        "old_rows": int(old_n),
    }


def propagation_drift(parent: dict | None, memory: dict, old_n: int) -> dict:
    if parent is None or old_n == 0:
        return {"old_adj_fro_abs": None, "old_adj_fro_relative": None,
                "old_edge_weight_mean_abs": None, "old_edge_count": 0}
    pidx, pval, _ = normalize_adj(parent["edge_index"], old_n)
    cidx, cval, _ = normalize_adj(memory["edge_index"], len(memory["x"]))
    parent_dense = torch.zeros((old_n, old_n))
    current_dense = torch.zeros((old_n, old_n))
    parent_dense[pidx[0], pidx[1]] = pval
    keep = (cidx[0] < old_n) & (cidx[1] < old_n)
    current_dense[cidx[0, keep], cidx[1, keep]] = cval[keep]
    delta = current_dense - parent_dense
    denom = parent_dense.norm().clamp_min(1e-9)
    parent_edges = parent_dense != 0
    return {
        "old_adj_fro_abs": scalar(delta.norm()),
        "old_adj_fro_relative": scalar(delta.norm() / denom),
        "old_edge_weight_mean_abs": scalar(delta[parent_edges].abs().mean()) if parent_edges.any() else 0.0,
        "old_edge_count": int(parent_edges.sum()),
    }


def margins(model, memory: dict, data: dict, stage: int, old_n: int,
            row_classes_: torch.Tensor, device: str) -> dict:
    model.eval()
    x = memory["x"].to(device)
    adj = R.C.training_adjacency(memory, device)
    prompt = data["prompt_emb"].to(device)
    with torch.no_grad():
        syn_logits = model.logits(x, adj, prompt).cpu()
        real_logits = model.logits(data["x_device"], data["real_adj"], prompt).cpu()
    new_rows = torch.arange(old_n, len(memory["x"]))
    cls = row_classes_.long()
    correct = syn_logits[new_rows, cls]
    masked = syn_logits[new_rows].clone()
    masked[torch.arange(len(new_rows)), cls] = -torch.inf
    syn_margin = correct - masked.max(1).values
    stage_classes = data["stage_classes"][stage]
    stage_mask = data["report_dev_mask"].cpu() & torch.isin(
        data["y"].cpu(), torch.tensor(stage_classes, dtype=torch.long))
    real_ids = torch.where(stage_mask)[0]
    real_cls = data["y"].cpu()[real_ids].long()
    real_correct = real_logits[real_ids, real_cls]
    real_other = real_logits[real_ids].clone()
    real_other[torch.arange(len(real_ids)), real_cls] = -torch.inf
    real_margin = real_correct - real_other.max(1).values
    return {
        "synthetic_margin_mean": scalar(syn_margin.mean()) if len(syn_margin) else None,
        "synthetic_margin_median": scalar(syn_margin.median()) if len(syn_margin) else None,
        "synthetic_margin_std": scalar(syn_margin.std(unbiased=False)) if len(syn_margin) else None,
        "real_margin_mean": scalar(real_margin.mean()) if len(real_margin) else None,
        "real_margin_median": scalar(real_margin.median()) if len(real_margin) else None,
        "real_margin_std": scalar(real_margin.std(unbiased=False)) if len(real_margin) else None,
        "margin_gap_real_minus_synthetic": scalar(real_margin.mean() - syn_margin.mean()) if len(real_margin) and len(syn_margin) else None,
        "synthetic_rows": int(len(syn_margin)),
        "real_rows": int(len(real_margin)),
    }


def fit_loss(model, x, adj, text, rows, device, sample_seed: int) -> float | None:
    if len(rows) < 2:
        return None
    generator = torch.Generator(device="cpu").manual_seed(int(sample_seed))
    graph_all = model.encode_graph(x.to(device), adj).detach()
    text_all = model.encode_text(text.to(device)).detach()
    values = []
    batch = min(16, len(rows))
    for _ in range(8):
        local = torch.randperm(len(rows), generator=generator)[:batch]
        selected = rows[local].to(device)
        graph = graph_all[selected]
        encoded = text_all[selected]
        logits = model.logit_scale.exp().clamp(max=100) * graph @ encoded.T
        target = torch.arange(len(selected), device=device)
        values.append((F.cross_entropy(logits, target)
                       + F.cross_entropy(logits.T, target)) / 2)
    return scalar(torch.stack(values).mean())


def checkpoint_metrics(root: Path, data: dict, order: str, arm: str, stage: int,
                       seed: int, memory: dict, old_n: int,
                       row_classes_: torch.Tensor, device: str) -> dict:
    ckpt = load(root / data["dataset"] / order / f"{arm}_s{stage}_seed{seed}.pt")
    model = R.G.CLIPStudent(memory["x"].shape[1]).to(device)
    model.load_state_dict(ckpt["state"])
    result = margins(model, memory, data, stage, old_n, row_classes_, device)
    adj = R.C.training_adjacency(memory, device)
    new_rows = torch.arange(old_n, len(memory["x"]), dtype=torch.long)
    sample_seed = 100000 + 1000 * int(stage) + int(seed)
    result["synthetic_infonce_loss"] = fit_loss(
        model, memory["x"], adj, memory["text_emb"], new_rows, device,
        sample_seed)
    ids = torch.where(data["pool_mask"] & torch.isin(
        data["y"], torch.tensor(data["stage_classes"][stage])))[0]
    result["raw_delta_infonce_loss"] = fit_loss(
        model, data["x"], data["real_adj"], data["text_emb"], ids, device,
        sample_seed + 17)
    del model
    return result


def read_f1(root: Path, dataset: str, order: str, arm: str, stage: int, seed: int) -> dict:
    row = json.loads((root / dataset / order / f"{arm}_s{stage}_seed{seed}.json").read_text())
    metrics = row["metrics"]["all"]
    return {"final_f1": float(metrics["cumulative_f1"]),
            "old_f1": float(metrics["old_f1"]) if metrics.get("old_f1") is not None else None,
            "new_f1": float(metrics["new_f1"]) if metrics.get("new_f1") is not None else None}


def analyze_cell(data: dict, order: str, stage: int, seed: int, arm: str,
                 root: Path, assignment_root: Path, device: str) -> dict:
    global CURRENT_ORDER
    CURRENT_ORDER = order
    ids, budgets = D.set_stream(data, order)
    cell = root / data["dataset"] / order
    memory = load(cell / f"{arm}_s{stage}_memory.pt")
    old_n = 0 if stage == 0 else len(load(cell / f"{arm}_s{stage-1}_memory.pt")["x"])
    parent = None if stage == 0 else load(cell / f"{arm}_s{stage-1}_memory.pt")
    classes = row_classes(data, ids[stage], memory, old_n, stage,
                          assignment_root, arm, device)
    result = {"dataset": data["dataset"], "order": order, "stage": int(stage),
              "seed": int(seed), "arm": arm, "rows": int(len(memory["x"])),
              "old_n": int(old_n), "budget": int(budgets[stage])}
    result.update(message_mass(memory, old_n))
    result.update(propagation_drift(parent, memory, old_n))
    result.update(checkpoint_metrics(root, data, order, arm, stage, seed, memory,
                                     old_n, classes, device))
    result.update(read_f1(root, data["dataset"], order, arm, stage, seed))
    result["raw_delta_ids"] = int(len(ids[stage]))
    return result


def write_report(out: Path, rows: list[dict], manifest: dict) -> None:
    fields = sorted({key for row in rows for key in row})
    with (out / "observations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metrics = ["synthetic_margin_mean", "real_margin_mean",
               "margin_gap_real_minus_synthetic", "new_receive_old_ratio_mean",
               "old_receive_new_ratio_mean", "old_adj_fro_relative",
               "synthetic_infonce_loss", "raw_delta_infonce_loss"]
    correlations = {}
    for arm in sorted({row["arm"] for row in rows}):
        subset = [row for row in rows if row["arm"] == arm and row["stage"] > 0]
        correlations[arm] = {metric: corr([r.get(metric, math.nan) for r in subset],
                                           [r.get("new_f1", math.nan) for r in subset])
                             for metric in metrics}
    save_json(out / "correlations.json", correlations)
    lines = [
        "# P0 新类塑性归因报告",
        "",
        "本报告只读取封存的 R15/R17 memory 和 student checkpoint；不重新构造 memory，不用 report F1 选择参数。",
        "分析对象为合成新行 margin、真实报告新类 margin、跨块归一化消息比例、旧块归一化传播漂移，以及它们与 new-F1 的关系。",
        "",
        "## 判读约定",
        "",
        "- `margin_gap_real_minus_synthetic > 0`：真实新类比合成新行有更大 prompt margin，提示合成表示/传播存在塑性缺口。",
        "- `new_receive_old_ratio_mean`：新合成行接收旧块消息的归一化比例。",
        "- `old_receive_new_ratio_mean`：旧行接收新块消息的归一化比例。",
        "- `old_adj_fro_relative`：追加新块后，旧块内部 normalized adjacency 相对 parent 的漂移。",
        "- 相关性只用于机制诊断，不是 selector，也不构成方法选择证据。",
        "",
        "## 结果文件",
        "",
        "- `observations.csv`：逐 dataset/order/stage/seed/arm 的原始观测。",
        "- `correlations.json`：各代理量与 new-F1 的 Pearson/Spearman 相关。",
        "",
        "## 协议",
        "",
        json.dumps(manifest, indent=2, ensure_ascii=False),
    ]
    (out / "P0_PLASTICITY_ATTRIBUTION_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--r15-root", required=True, type=Path)
    ap.add_argument("--r17-root", required=True, type=Path)
    ap.add_argument("--assignment-root", required=True, type=Path)
    ap.add_argument("--dataset", choices=("cora", "citeseer"), nargs="+",
                    default=["cora", "citeseer"])
    ap.add_argument("--order", nargs="+", default=["order19601", "order19602", "order19603"])
    ap.add_argument("--stage", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--seed", nargs="+", type=int, default=list(range(150, 156)))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--text-cache-root", type=Path,
                    default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    arms = [
        ("c0", args.r15_root),
        ("anchor025", args.r15_root),
        ("anchor_content_c0_topology", args.r15_root),
        ("message_gate_2", args.r17_root),
    ]
    manifest = {
        "name": "tgc-p0-plasticity-attribution-v1",
        "datasets": args.dataset, "orders": args.order,
        "stages": args.stage, "seeds": args.seed,
        "arms": [a for a, _ in arms],
        "r15_root": str(args.r15_root), "r17_root": str(args.r17_root),
        "assignment_root": str(args.assignment_root),
        "construction": "read-only analysis of sealed R15/R17 artifacts",
        "report_labels_used_for_construction": False,
        "historical_raw_reads_for_construction": 0,
        "metrics": ["synthetic/real prompt margin", "cross message mass",
                     "old normalized propagation drift", "new-F1", "InfoNCE loss"],
    }
    save_json(args.output / "manifest.json", manifest)
    rows = []
    for dataset in args.dataset:
        cache = args.text_cache_root / f"{dataset}_r005" / "text_emb_pool_only.pt"
        data = R.R.load_data(dataset, args.minilm, args.device, cache)
        for order in args.order:
            for stage in args.stage:
                for arm, root in arms:
                    for seed in args.seed:
                        rows.append(analyze_cell(data, order, stage, seed, arm,
                                                 root, args.assignment_root, args.device))
                        print(json.dumps(rows[-1], ensure_ascii=True), flush=True)
    write_report(args.output, rows, manifest)
    save_json(args.output / "completion.json", {"status": "complete", "rows": len(rows)})
    print(json.dumps({"status": "complete", "rows": len(rows), "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
