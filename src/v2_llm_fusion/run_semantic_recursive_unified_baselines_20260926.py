"""Unified small-graph comparison under the Semantic-Recursive protocol.

This runner keeps the recursive class stream, budgets, C0 topology, original
GCN/InfoNCE Student, report-dev rule, and paired Student seeds fixed.  It
compares the sealed Semantic-Recursive memory against an internal C0 control,
label-free real-node coreset controls, an offline full-raw reference, and two
parameter-stateful Student continual-learning controls.

The coreset arms are not synthetic condensation methods: they select real
current-delta rows and place them in the same current-stage C0 topology.  The
Student continual arms are reported separately because they retain model
parameters across stages and therefore have a different access/state budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve()
GRAPHCLIP_NATIVE = HERE.parents[1] / "graphclip_native"
if str(GRAPHCLIP_NATIVE) not in sys.path:
    sys.path.insert(0, str(GRAPHCLIP_NATIVE))

import protocol_v3
import run_gate1_continual as G
import run_tgc_joint_student_response as V1
import run_tgc_r4_relational as R4
import run_unified_v3_continual as U


PAYLOAD_KEYS = ("x", "text_emb", "texts", "mass", "depth", "stage")
CORESET_METHODS = {
    "random_real": "random_real",
    "k_center_fps": "k_center",
    "herding": "herding",
    "medoid": "kmeans_medoid",
}
ALL_ARMS = (
    "c0",
    "semantic_recursive",
    "random_real",
    "k_center_fps",
    "herding",
    "medoid",
    "raw_full_scratch",
    "student_delta_finetune",
    "student_c0_replay_delta",
)


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                    encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clone_memory(memory: dict) -> dict:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else list(value)
        for key, value in memory.items()
    }


def induced_edges(edge_index: torch.Tensor, ids: torch.Tensor,
                  total_nodes: int) -> torch.Tensor:
    ids = ids.long().cpu()
    remap = torch.full((int(total_nodes),), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids), dtype=torch.long)
    edges = edge_index.long().cpu()
    valid = (remap[edges[0]] >= 0) & (remap[edges[1]] >= 0)
    return torch.stack((remap[edges[0, valid]], remap[edges[1, valid]]))


def raw_support(data: dict, ids: torch.Tensor, stage: int) -> dict:
    ids = ids.long().cpu()
    return {
        "x": data["x"][ids].float(),
        "text_emb": data["text_emb"][ids].float(),
        "texts": [data["texts"][int(i)] for i in ids],
        "edge_index": induced_edges(data["edge_index"], ids, len(data["x"])),
        "mass": torch.ones(len(ids)),
        "weights": torch.ones(len(ids)),
        "depth": torch.zeros(len(ids), dtype=torch.long),
        "stage": torch.full((len(ids),), int(stage), dtype=torch.long),
    }


def concat_support(memory: dict, delta: dict, stage: int) -> dict:
    old_n = len(memory["x"])
    edges_old = memory["edge_index"].long().cpu()
    edges_new = delta["edge_index"].long().cpu() + old_n
    return {
        "x": torch.cat((memory["x"].cpu(), delta["x"].cpu())),
        "text_emb": torch.cat((memory["text_emb"].cpu(), delta["text_emb"].cpu())),
        "texts": list(memory["texts"]) + list(delta["texts"]),
        "edge_index": torch.cat((edges_old, edges_new), dim=1),
        "mass": torch.cat((memory["mass"].cpu(), delta["mass"].cpu())),
        "weights": torch.cat((memory["mass"].cpu(), delta["mass"].cpu())),
        "depth": torch.cat((memory["depth"].cpu(), delta["depth"].cpu())),
        "stage": torch.cat((memory["stage"].cpu(), delta["stage"].cpu())),
    }


def compose_with_c0_topology(c0: dict, parent: dict, block: dict) -> dict:
    """Use the current C0 edge positions but preserve the supplied parent."""
    old_n = len(parent["x"])
    new_n = len(block["x"])
    if len(c0["x"]) != old_n + new_n:
        raise AssertionError(
            f"row mismatch: c0={len(c0['x'])}, parent={old_n}, block={new_n}")
    out = clone_memory(c0)
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        out[key][:old_n] = parent[key]
        out[key][old_n:] = block[key]
    out["texts"] = list(parent["texts"]) + list(block["texts"])
    return out


def load_data(dataset: str, minilm: str, device: str,
              cache_root: Path) -> dict:
    return V1.load_data(dataset, minilm, device, cache_root)


def set_stream(data: dict, order: str):
    ids, budgets = R4.set_stream(data, order)
    return [v.long().cpu() for v in ids], [int(v) for v in budgets]


def build_trajectories(data: dict, source_root: Path,
                       recursive_root: Path, order: str, device: str,
                       arms: list[str]) -> tuple[dict, dict, list[torch.Tensor], list[int]]:
    ids, budgets = set_stream(data, order)
    dataset = data["dataset"]
    source = source_root / dataset / order
    trajectories: dict[str, list[dict]] = {}
    audits: dict[str, list[dict]] = {}

    for arm in arms:
        if arm in {"raw_full_scratch", "student_delta_finetune",
                   "student_c0_replay_delta"}:
            continue
        memories = [load(source / "c0_s0_memory.pt")]
        rows = [{"stage": 0, "memory_rows": len(memories[0]["x"]),
                 "budget": budgets[0], "mode": "sealed_c0_base"}]
        if arm == "semantic_recursive":
            for stage in (1, 2):
                cell = recursive_root / f"{dataset}_{order}_s{stage}"
                memories.append(load(cell / "semantic_perm_memory.pt"))
                rows.append({"stage": stage, "memory_rows": len(memories[-1]["x"]),
                             "budget": budgets[stage], "mode": "sealed_recursive"})
        elif arm == "c0":
            for stage in (1, 2):
                memories.append(load(source / f"c0_s{stage}_memory.pt"))
                rows.append({"stage": stage, "memory_rows": len(memories[-1]["x"]),
                             "budget": budgets[stage], "mode": "sealed_c0"})
        elif arm in CORESET_METHODS:
            method = CORESET_METHODS[arm]
            for stage in (1, 2):
                c0 = load(source / f"c0_s{stage}_memory.pt")
                parent = memories[-1]
                new_rows = len(c0["x"]) - len(parent["x"])
                started = time.perf_counter()
                block, audit = U.build_real_selection_block(
                    data, ids[stage], new_rows, method, stage, device)
                memories.append(compose_with_c0_topology(c0, parent, block))
                rows.append({"stage": stage, "memory_rows": len(memories[-1]["x"]),
                             "budget": budgets[stage], "mode": "real_node_coreset",
                             "method": method, "construction_seconds":
                             time.perf_counter() - started, "audit": audit})
        else:
            raise ValueError(f"unknown memory arm: {arm}")
        trajectories[arm] = memories
        audits[arm] = rows

    return trajectories, audits, ids, budgets


@torch.no_grad()
def score_model(model: torch.nn.Module, data: dict,
                groups: list[list[int]], stage: int) -> dict:
    logits = model.logits(
        data["x_device"], data["real_adj"],
        data["prompt_emb"].to(data["x_device"].device),
    ).cpu()
    y_all = data["y"].cpu()
    report = data["report_dev_mask"].cpu()
    seen = [int(c) for group in groups[:stage + 1] for c in group]
    seen_tensor = torch.tensor(seen, dtype=y_all.dtype)
    mask = report & torch.isin(y_all, seen_tensor)
    y = y_all[mask].numpy()
    pred_all = logits.argmax(1).numpy()[mask.numpy()]
    per_stage = []
    per_class = {}
    for group in groups[:stage + 1]:
        cls = [int(c) for c in group]
        class_mask = np.isin(y, cls)
        yy, pp = y[class_mask], pred_all[class_mask]
        per_stage.append({
            "classes": cls,
            "macro_f1": float(100 * f1_score(
                yy, pp, average="macro", labels=cls, zero_division=0)),
            "accuracy": float(100 * np.mean(yy == pp)) if len(yy) else 0.0,
            "n": int(len(yy)),
        })
        for c in cls:
            cmask = y == c
            tp = int(np.sum((y == c) & (pred_all == c)))
            fp = int(np.sum((y != c) & (pred_all == c)))
            fn = int(np.sum((y == c) & (pred_all != c)))
            denom = 2 * tp + fp + fn
            per_class[str(c)] = float(200 * tp / denom) if denom else 0.0
    old_classes = [int(c) for group in groups[:stage] for c in group]
    new_classes = [int(c) for c in groups[stage]]
    old = [per_class[str(c)] for c in old_classes]
    new = [per_class[str(c)] for c in new_classes]
    return {
        "cumulative_f1": float(np.mean(list(per_class.values())))
        if per_class else 0.0,
        "accuracy": float(100 * np.mean(y == pred_all)) if len(y) else 0.0,
        "old_f1": float(np.mean(old)) if old else None,
        "new_f1": float(np.mean(new)) if new else None,
        "per_class_f1": per_class,
        "per_stage": per_stage,
        "n": int(len(y)),
    }


def train_scratch(memory: dict, data: dict, groups: list[list[int]],
                  stage: int, seeds: list[int], device: str) -> tuple[list[dict], float]:
    steps = int(protocol_v3.student_steps(len(memory["x"])))
    scores = []
    started = time.perf_counter()
    for seed in seeds:
        model = G.train_student(memory, int(seed), device, steps,
                                lr=protocol_v3.STUDENT.lr,
                                batch=protocol_v3.STUDENT.batch)
        scores.append({"seed": int(seed), "steps": steps,
                       **score_model(model, data, groups, stage)})
        del model
    return scores, time.perf_counter() - started


def fine_tune(model: torch.nn.Module, support: dict, seed: int,
              steps: int, device: str) -> torch.nn.Module:
    model.train()
    x = support["x"].to(device)
    t = support["text_emb"].to(device)
    adj = G.norm_adj(support["edge_index"], len(x), device)
    opt = torch.optim.Adam(model.parameters(), lr=protocol_v3.STUDENT.lr)
    generator = torch.Generator().manual_seed(int(seed) * 17 + 5)
    batch = int(protocol_v3.STUDENT.batch)
    for _ in range(int(steps)):
        rows = torch.randperm(len(x), generator=generator)[:min(batch, len(x))]
        rows = rows.to(device)
        g = model.encode_graph(x, adj)[rows]
        tt = model.encode_text(t[rows])
        logits = model.logit_scale.exp().clamp(max=100) * g @ tt.T
        target = torch.arange(len(rows), device=device)
        loss = (F.cross_entropy(logits, target)
                + F.cross_entropy(logits.T, target)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()


def evaluate_stateful(kind: str, data: dict, groups: list[list[int]],
                      ids: list[torch.Tensor], source_root: Path,
                      order: str, seeds: list[int], device: str) -> list[dict]:
    source = source_root / data["dataset"] / order
    c0_memories = [load(source / f"c0_s{stage}_memory.pt") for stage in range(3)]
    delta_supports = [raw_support(data, ids[stage], stage) for stage in range(3)]
    rows = []
    models = []
    started = time.perf_counter()
    base = c0_memories[0]
    base_steps = int(protocol_v3.student_steps(len(base["x"])))
    for seed in seeds:
        models.append(G.train_student(base, int(seed), device, base_steps,
                                       lr=protocol_v3.STUDENT.lr,
                                       batch=protocol_v3.STUDENT.batch))
    for stage in range(3):
        if stage > 0:
            for index, seed in enumerate(seeds):
                if kind == "student_delta_finetune":
                    support = delta_supports[stage]
                elif kind == "student_c0_replay_delta":
                    support = concat_support(c0_memories[stage],
                                             delta_supports[stage], stage)
                else:
                    raise ValueError(kind)
                steps = int(protocol_v3.student_steps(len(support["x"])))
                models[index] = fine_tune(models[index], support, int(seed),
                                           steps, device)
        for seed, model in zip(seeds, models):
            rows.append({"stage": stage, "seed": int(seed),
                         "steps": (base_steps if stage == 0 else
                                   int(protocol_v3.student_steps(
                                       len(delta_supports[stage]["x"]) if kind == "student_delta_finetune"
                                       else len(concat_support(c0_memories[stage], delta_supports[stage], stage)["x"])))),
                         **score_model(model, data, groups, stage)})
    elapsed = time.perf_counter() - started
    for model in models:
        del model
    for row in rows:
        row["stateful_wall_seconds_total"] = elapsed
    return rows


def evaluate_raw_full(data: dict, groups: list[list[int]],
                      ids: list[torch.Tensor], seeds: list[int],
                      device: str) -> list[dict]:
    rows = []
    for stage in range(3):
        source_ids = torch.cat(ids[:stage + 1])
        support = raw_support(data, source_ids, stage)
        scores, elapsed = train_scratch(
            support, data, groups, stage, seeds, device)
        for row in scores:
            row["stage"] = stage
            row["memory_rows"] = int(len(support["x"]))
            row["student_wall_seconds"] = float(elapsed)
        rows.extend(scores)
    return rows


def aggregate_stage_scores(rows: list[dict]) -> dict:
    def mean(key: str):
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else None
    return {
        "mean_f1": mean("cumulative_f1"),
        "std_f1": float(np.std([r["cumulative_f1"] for r in rows])),
        "mean_accuracy": mean("accuracy"),
        "std_accuracy": float(np.std([r["accuracy"] for r in rows])),
        "old_f1": mean("old_f1"),
        "new_f1": mean("new_f1"),
        "per_stage_f1": [mean_from(rows, "per_stage", s, "macro_f1")
                         for s in range(len(rows[0]["per_stage"]))],
        "per_stage_accuracy": [mean_from(rows, "per_stage", s, "accuracy")
                               for s in range(len(rows[0]["per_stage"]))],
    }


def mean_from(rows: list[dict], list_key: str, index: int, key: str) -> float:
    return float(np.mean([r[list_key][index][key] for r in rows]))


def paired_ci(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return {"mean": float(arr.mean()) if len(arr) else None,
                "low": None, "high": None, "n": int(len(arr))}
    mean = float(arr.mean())
    half = float(stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) /
                 math.sqrt(len(arr)))
    return {"mean": mean, "low": mean - half, "high": mean + half,
            "n": int(len(arr))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--recursive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--orders", nargs="+", default=["order19611", "order19612"])
    parser.add_argument("--report-seeds", nargs="+", type=int,
                        default=list(range(150, 156)))
    parser.add_argument("--arms", nargs="+", default=list(ALL_ARMS))
    args = parser.parse_args()
    if set(args.arms) - set(ALL_ARMS):
        raise ValueError(f"unknown arms: {sorted(set(args.arms) - set(ALL_ARMS))}")
    if (args.output_root / "results.json").exists():
        raise FileExistsError(args.output_root / "results.json")
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    protocol = {
        "name": "semantic-recursive-unified-small-baseline-v1",
        "datasets": args.datasets,
        "orders": args.orders,
        "stages": [0, 1, 2],
        "arms": args.arms,
        "report_seeds": args.report_seeds,
        "class_stream": "same R4 order19611/order19612 stream for every arm",
        "budget": "same sealed C0 budgets [43,60,94] or dataset-equivalent",
        "topology": "current-stage C0 edge positions for memory-only arms",
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "student_batch": int(protocol_v3.STUDENT.batch),
        "student_steps": "max(50,min(2048,16*rows))",
        "construction": "current pool delta only; no historical raw reads; report/test labels not used",
        "metric": "all-prompt report-dev per-class Macro-F1 and Accuracy",
        "k_center_fps_note": "k-center and greedy FPS are the same farthest-first control here",
        "stateful_note": "student_delta_finetune and student_c0_replay_delta retain Student parameters across stages",
    }
    save_json(args.output_root / "protocol.json", protocol)
    data_cache = {}
    all_rows = []
    construction_audits = {}
    for dataset in args.datasets:
        data = load_data(dataset, args.minilm, args.device, args.cache_root)
        data["real_adj"] = G.norm_adj(data["edge_index"], len(data["x"]), args.device)
        data["x_device"] = data["x"].float().to(args.device)
        data["prompt_emb"] = data["encoder"].encode(data["prompts"])
        data_cache[dataset] = data
        for order in args.orders:
            trajectories, audits, ids, budgets = build_trajectories(
                data, args.source_root, args.recursive_root, order,
                args.device, args.arms)
            construction_audits[f"{dataset}_{order}"] = {
                "budgets": budgets,
                "arms": audits,
                "source_c0_hashes": {
                    str(stage): sha256(args.source_root / dataset / order /
                                       f"c0_s{stage}_memory.pt")
                    for stage in range(3)
                },
            }
            groups = data["stage_classes"]
            cell_rows = []
            raw_full_scores = None
            for arm in args.arms:
                if arm == "raw_full_scratch":
                    if raw_full_scores is None:
                        raw_full_scores = evaluate_raw_full(
                            data, groups, ids, args.report_seeds, args.device)
                    for stage in range(3):
                        rows = [r for r in raw_full_scores if r["stage"] == stage]
                        cell_rows.append({
                            "dataset": dataset, "order": order,
                            "arm": arm, "stage": stage, "mode": "offline_raw_scratch",
                            "aggregate": aggregate_stage_scores(rows),
                            "per_student": rows,
                            "memory_rows": rows[0]["memory_rows"],
                            "student_wall_seconds": rows[0]["student_wall_seconds"],
                        })
                    continue
                if arm in {"student_delta_finetune", "student_c0_replay_delta"}:
                    stateful = evaluate_stateful(
                        arm, data, groups, ids, args.source_root, order,
                        args.report_seeds, args.device)
                    for stage in range(3):
                        rows = [r for r in stateful if r["stage"] == stage]
                        agg = aggregate_stage_scores(rows)
                        cell_rows.append({
                            "dataset": dataset, "order": order,
                            "arm": arm, "stage": stage, "mode": "stateful",
                            "aggregate": agg, "per_student": rows,
                            "memory_rows": budgets[stage],
                            "student_wall_seconds": rows[0]["stateful_wall_seconds_total"],
                        })
                    continue
                memory_rows = trajectories[arm]
                for stage, memory in enumerate(memory_rows):
                    scores, elapsed = train_scratch(
                        memory, data, groups, stage, args.report_seeds, args.device)
                    cell_rows.append({
                        "dataset": dataset, "order": order,
                        "arm": arm, "stage": stage, "mode": "scratch",
                        "aggregate": aggregate_stage_scores(scores),
                        "per_student": scores,
                        "memory_rows": int(len(memory["x"])),
                        "student_wall_seconds": float(elapsed),
                    })
            all_rows.extend(cell_rows)
            save_json(args.output_root / f"{dataset}_{order}.json",
                      {"protocol": protocol, "rows": cell_rows})
            print(json.dumps({"event": "cell_complete", "dataset": dataset,
                              "order": order, "rows": len(cell_rows)}), flush=True)

    save_json(args.output_root / "results.json", {
        "protocol": protocol,
        "construction_audits": construction_audits,
        "rows": all_rows,
    })
    print(json.dumps({"event": "complete", "rows": len(all_rows),
                      "output": str(args.output_root)}), flush=True)


if __name__ == "__main__":
    main()
