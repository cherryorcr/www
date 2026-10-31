"""Topology-only Student-consistent edge transport for direct outer controls.

This experiment deliberately fixes every memory field except the old-new
cross block.  A candidate pool is formed from the current synthetic X/text
similarity, and the existing bounded two-switch operator uses a short,
report-label-free Student response to accept at most one legal swap.

The script is a diagnostic, not a new main method.  It preserves old-old and
new-new edges, each new row's cross degree, and each old endpoint's capacity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from student_consistent_edge_swap import (
    Config,
    evaluate_one_step,
    select_swaps,
    topology_audit,
)


CELL_SPECS = {
    "cora_canonical": ("cora", "canonical", 25),
    "cora_order19102": ("cora", "order19102", 25),
    "citeseer_canonical": ("citeseer", "canonical", 38),
    "citeseer_order19102": ("citeseer", "order19102", 38),
}


def _pairs(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    edge_index = torch.as_tensor(edge_index, dtype=torch.long).cpu()
    pairs: set[tuple[int, int]] = set()
    for left, right in edge_index.T.tolist():
        left, right = int(left), int(right)
        if left == right:
            continue
        pairs.add(tuple(sorted((left, right))))
    return pairs


def _cross_rows(edge_index: torch.Tensor, old_rows: int) -> dict[int, tuple[int, ...]]:
    result: dict[int, list[int]] = {}
    for left, right in _pairs(edge_index):
        if left < old_rows <= right:
            result.setdefault(right, []).append(left)
    return {node: tuple(sorted(values)) for node, values in result.items()}


def _directed(pairs: Iterable[tuple[int, int]]) -> torch.Tensor:
    pairs = sorted(set(tuple(sorted((int(a), int(b)))) for a, b in pairs
                       if int(a) != int(b)))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    values = pairs + [(right, left) for left, right in pairs]
    return torch.tensor(values, dtype=torch.long).T.contiguous()


def build_candidate_pool(memory: dict, old_rows: int, top_k: int,
                         x_weight: float) -> tuple[torch.Tensor, dict]:
    """Add similarity-ranked old endpoints while retaining every base edge."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    x = F.normalize(torch.as_tensor(memory["x"], dtype=torch.float32), dim=1)
    text = F.normalize(torch.as_tensor(memory["text_emb"], dtype=torch.float32), dim=1)
    old_x, new_x = x[:old_rows], x[old_rows:]
    old_t, new_t = text[:old_rows], text[old_rows:]
    score = float(x_weight) * (new_x @ old_x.T)
    score = score + (1.0 - float(x_weight)) * (new_t @ old_t.T)

    base = _cross_rows(memory["edge_index"], old_rows)
    candidate_pairs: set[tuple[int, int]] = set()
    added = 0
    for new_node, base_old in sorted(base.items()):
        for old_node in base_old:
            candidate_pairs.add((old_node, new_node))
        width = min(int(top_k), old_rows)
        best_old = torch.topk(score[new_node - old_rows], width,
                              largest=True, sorted=True).indices.tolist()
        for old_node in best_old:
            pair = (int(old_node), int(new_node))
            if pair not in candidate_pairs:
                added += 1
            candidate_pairs.add(pair)
    audit = {
        "old_rows": int(old_rows),
        "new_rows_with_cross_edges": int(len(base)),
        "base_cross_pairs": int(sum(len(v) for v in base.values())),
        "candidate_cross_pairs": int(len(candidate_pairs)),
        "added_candidate_pairs": int(added),
        "score": "x_text_cosine",
        "x_weight": float(x_weight),
        "top_k": int(top_k),
    }
    return _directed(candidate_pairs), audit


def clone_with_edge(memory: dict, edge_index: torch.Tensor) -> dict:
    result = dict(memory)
    result["edge_index"] = edge_index.cpu()
    return result


def run_cell(root: Path, output_root: Path, cell: str, device: str,
             top_k: int, x_weight: float, selector_seeds: tuple[int, ...],
             max_swaps: int, base_prefix: str,
             output_prefix: str, student_steps: int,
             min_seed_fraction: float) -> dict:
    dataset, order, old_rows = CELL_SPECS[cell]
    base_path = root / f"{base_prefix}_{cell}" / "memory.pt"
    output = output_root / f"{output_prefix}_{cell}"
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    candidate_edge, pool_audit = build_candidate_pool(
        base, old_rows, top_k=top_k, x_weight=x_weight)
    config = Config(
        max_swaps=int(max_swaps), student_steps=int(student_steps),
        student_lr=1e-3,
        batch=16, new_gain_margin=0.0, max_old_gain_drop=0.0,
        max_total_loss_increase=0.0, new_weight=1.0, old_weight=0.5,
        min_seed_fraction=float(min_seed_fraction))

    def evaluate(memory: dict):
        return evaluate_one_step(
            memory, old_rows, tuple(int(seed) for seed in selector_seeds),
            device=device, config=config)

    result, selection = select_swaps(
        base, candidate_edge, old_rows, config, evaluate)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(result, output / "memory.pt")
    audit = {
        "protocol": {
            "name": output_prefix,
            "dataset": dataset,
            "order": order,
            "fixed": ["X", "text_emb", "rows", "mass", "budget",
                      "old-old edges", "new-new edges", "Student"],
            "editable": "old-new cross edges only",
            "historical_raw_reads": 0,
            "report_labels_used": False,
            "selector_seeds": list(selector_seeds),
            "student_steps": int(student_steps),
            "min_seed_fraction": float(min_seed_fraction),
        },
        "base_file": str(base_path),
        "candidate_pool": pool_audit,
        "selection": selection,
        "topology": topology_audit(base["edge_index"], result["edge_index"],
                                    old_rows),
        "same_content": bool(torch.equal(base["x"], result["x"])
                              and torch.equal(base["text_emb"], result["text_emb"])
                              and base["texts"] == result["texts"]),
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({
        "event": "direct_topology_only_complete",
        "cell": cell,
        "output": str(output),
        "candidate_count": selection["candidate_count"],
        "accepted_count": selection["accepted_count"],
        "fallback_to_input": selection["fallback_to_input"],
        "changed_cross_pairs": audit["topology"]["changed_cross_pairs"],
    }), flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", choices=tuple(CELL_SPECS),
                        default=list(CELL_SPECS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--x-weight", type=float, default=0.5)
    parser.add_argument("--selector-seeds", nargs="+", type=int,
                        default=[17, 23, 29])
    parser.add_argument("--max-swaps", type=int, default=1)
    parser.add_argument("--student-steps", type=int, default=3)
    parser.add_argument("--min-seed-fraction", type=float, default=2 / 3)
    parser.add_argument("--base-prefix", default="control",
                        help="memory directory prefix before _<cell>")
    parser.add_argument("--output-prefix", default="topology_student_v1",
                        help="immutable output directory prefix")
    args = parser.parse_args()
    if not 0.0 <= args.x_weight <= 1.0:
        raise ValueError("x-weight must be in [0, 1]")
    if not 0.0 <= args.min_seed_fraction <= 1.0:
        raise ValueError("min-seed-fraction must be in [0, 1]")
    if args.student_steps < 1:
        raise ValueError("student-steps must be positive")
    for cell in args.cells:
        run_cell(args.root, args.output_root, cell, args.device,
                 args.top_k, args.x_weight,
                 tuple(args.selector_seeds), args.max_swaps,
                 args.base_prefix, args.output_prefix,
                 args.student_steps, args.min_seed_fraction)


if __name__ == "__main__":
    main()
