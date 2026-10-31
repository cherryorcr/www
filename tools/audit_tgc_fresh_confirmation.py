"""Recompute frozen confirmation statistics and audit saved synthetic artifacts.

This script never constructs, selects, or modifies a memory. JSON-only mode
uses the Python standard library; artifact mode imports torch lazily.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

ARMS = ("c0", "identity", "semantic_perm", "role_perm", "anchor_role_perm",
        "message_role_perm", "plasticity_role_perm", "adaptive_role_perm",
        "random_perm")
METRICS = ("cumulative_f1", "old_f1", "new_f1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, where):
    require(math.isfinite(actual) and abs(actual - expected) < 1e-8,
            f"inconsistent value at {where}: {actual} != {expected}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_cell(cell, seeds):
    require(set(cell["arms"]) == set(ARMS), f"missing/extra arm: {cell['cell']}")
    reference = cell["arms"]["c0"]
    for arm, result in cell["arms"].items():
        where = f"{cell['cell']}/{arm}"
        require(result["student_seeds"] == seeds, f"seed mismatch: {where}")
        require([r["seed"] for r in result["per_student"]] == seeds,
                f"duplicate/missing/reordered seed: {where}")
        require(result["student_steps"] == reference["student_steps"],
                f"steps mismatch: {where}")
        for row, base in zip(result["per_student"], reference["per_student"]):
            require(row["steps"] == result["student_steps"], f"row steps: {where}")
            require(row["groups"] == base["groups"] and row["n"] == base["n"],
                    f"evaluation population mismatch: {where}")
            groups = row["groups"]
            require(len(groups) == cell["stage"] + 1, f"stage groups: {where}")
            classes = [str(c) for group in groups for c in group]
            require(len(classes) == len(set(classes)), f"duplicate class: {where}")
            for scope in ("all_prompt", "seen_prompt"):
                values = row[scope]["per_class_f1"]
                require(set(values) == set(classes), f"class coverage: {where}")
                require(all(math.isfinite(v) and 0 <= v <= 100 for v in values.values()),
                        f"invalid F1: {where}")
                close(row[scope]["cumulative_f1"], stats.mean(values.values()), where)
                old = [values[str(c)] for g in groups[:-1] for c in g]
                new = [values[str(c)] for c in groups[-1]]
                close(row[scope]["old_f1"], stats.mean(old), where)
                close(row[scope]["new_f1"], stats.mean(new), where)
        for metric in METRICS:
            close(result[f"{metric}_mean"], stats.mean(
                r["all_prompt"][metric] for r in result["per_student"]), where)


def aggregate(cells):
    out = {}
    for arm in ARMS:
        deltas = {m: [c["arms"][arm][f"{m}_mean"] -
                      c["arms"]["c0"][f"{m}_mean"] for c in cells]
                  for m in METRICS}
        out[arm] = {
            "absolute_mean": {m: stats.mean(c["arms"][arm][f"{m}_mean"]
                                            for c in cells) for m in METRICS},
            "delta_mean_pp": {m: stats.mean(deltas[m]) for m in METRICS},
            "positive_cells": sum(v > 0 for v in deltas["cumulative_f1"]),
            "cells": len(cells),
            "worst_cell_pp": min(deltas["cumulative_f1"]),
            "cell_delta_median_pp": stats.median(deltas["cumulative_f1"]),
            "cell_delta_sd_pp": stats.stdev(deltas["cumulative_f1"])
                if len(cells) > 1 else None,
        }
    return out


def summarize(roots):
    cells, sources = [], []
    seeds = None
    for root in roots:
        path = Path(root) / "results.json"
        data = read_json(path)
        protocol = data["protocol"]
        require(protocol["selection"] is False, f"selection enabled: {path}")
        if seeds is None:
            seeds = protocol["report_seeds"]
        require(seeds == protocol["report_seeds"], f"cross-run seeds: {path}")
        for cell in data["cells"]:
            validate_cell(cell, seeds)
            require(read_json(Path(root) / f"{cell['cell']}.json") == cell,
                    f"cell/aggregate disagreement: {cell['cell']}")
            cells.append(cell)
        sources.append({"path": str(path), "sha256": sha256(path)})
    names = [c["cell"] for c in cells]
    expected = {f"{ds}_order{order}_s{s}" for ds in ("cora", "citeseer")
                for order in (19611, 19612) for s in (1, 2)}
    require(set(names) == expected and len(names) == 8, "expected eight unique cells")
    result = {"sources": sources, "student_seeds": seeds,
              "all_cells": aggregate(cells), "by_dataset": {}, "by_stage": {}}
    for ds in ("cora", "citeseer"):
        result["by_dataset"][ds] = aggregate([c for c in cells if c["cell"].startswith(ds)])
    for stage in (1, 2):
        result["by_stage"][str(stage)] = aggregate([c for c in cells if c["stage"] == stage])
    worst = min(cells, key=lambda c: c["arms"]["semantic_perm"]["cumulative_f1_mean"]
                - c["arms"]["c0"]["cumulative_f1_mean"])
    reference = worst["arms"]["c0"]["per_student"]
    result["tail_cell"] = {"cell": worst["cell"], "arms": {}}
    for arm in ("semantic_perm", "identity", "role_perm", "random_perm"):
        rows = worst["arms"][arm]["per_student"]
        delta = [r["all_prompt"]["cumulative_f1"] - b["all_prompt"]["cumulative_f1"]
                 for r, b in zip(rows, reference)]
        result["tail_cell"]["arms"][arm] = {
            "seed_deltas_pp": dict(zip(seeds, delta)),
            "paired_mean_pp": stats.mean(delta), "paired_sd_pp": stats.stdev(delta),
            "positive_seeds": sum(v > 0 for v in delta),
            "per_class_delta_pp": {cls: stats.mean(
                r["all_prompt"]["per_class_f1"][cls] - b["all_prompt"]["per_class_f1"][cls]
                for r, b in zip(rows, reference))
                for cls in reference[0]["all_prompt"]["per_class_f1"]},
        }
    result["scope"] = {
        "unit": "cell equal-weight mean of paired six-seed means, not pooled node F1",
        "stages": "incremental stages 1/2 only; stage 0 is not included",
        "independence": "new class orders, same datasets/report-dev and Student seed group",
        "causal_claim": False,
    }
    return result, cells


def artifact_audit(memory_root, cells):
    import torch

    tensor_keys = ("x", "text_emb", "mass", "depth", "stage")
    training_keys = ("x", "text_emb", "edge_index")

    def load(path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def equal(a, b, keys, stop=None):
        for key in keys:
            x, y = a[key], b[key]
            if stop is not None:
                x, y = x[:stop], y[:stop]
            if isinstance(x, torch.Tensor):
                if not torch.equal(x, y):
                    return False
            elif x != y:
                return False
        return True

    def row_hash(memory, index):
        digest = hashlib.sha256()
        for key in tensor_keys:
            value = memory[key][index].contiguous()
            digest.update(str((key, value.dtype, tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        digest.update(memory["texts"][index].encode("utf-8"))
        return digest.hexdigest()

    audits = {}
    for cell in cells:
        path = Path(memory_root) / cell["cell"]
        build = read_json(path / "build_audit.json")
        old_n = build["old_rows"]
        memories = {arm: load(path / f"{arm}_memory.pt") for arm in ARMS}
        c0, identity = memories["c0"], memories["identity"]
        remote_parent_path = Path(build["source"]) / f"c0_s{cell['stage'] - 1}_memory.pt"
        parent_path = Path(args_source_root) / cell["cell"].split("_", 1)[0] / \
            cell["cell"].split("_", 1)[1].rsplit("_s", 1)[0] / \
            f"c0_s{cell['stage'] - 1}_memory.pt" if args_source_root else remote_parent_path
        parent = load(parent_path) if parent_path.exists() else None
        keys = (*tensor_keys, "texts")
        if parent is not None:
            require(equal(c0, parent, keys, old_n), f"C0 source prefix: {path}")
        baseline_hashes = Counter(row_hash(identity, i) for i in range(old_n, len(c0["x"])))
        n = len(c0["x"])
        adjacency = torch.zeros((n, n), dtype=torch.float32)
        adjacency[c0["edge_index"][0], c0["edge_index"][1]] = 1
        adjacency.fill_diagonal_(1)
        scale = adjacency.sum(1).clamp_min(1).rsqrt()
        adjacency = scale[:, None] * adjacency * scale[None, :]
        baseline_message = adjacency @ c0["x"].float()
        checked, groups = {}, []
        for arm, memory in memories.items():
            require(sha256(path / f"{arm}_memory.pt") == cell["arms"][arm]["memory_sha256"],
                    f"evaluated artifact replaced: {path}/{arm}")
            same_topology = torch.equal(memory["edge_index"], c0["edge_index"])
            same_prefix = equal(memory, c0, keys, old_n)
            require(n == len(memory["x"]) == build["budget"], f"rows: {path}/{arm}")
            require(cell["arms"][arm]["student_steps"] == max(50, min(2048, 16*n)),
                    f"protocol steps: {path}/{arm}")
            require(same_topology and same_prefix, f"topology/prefix: {path}/{arm}")
            for key in ("x", "text_emb"):
                require(torch.isfinite(memory[key]).all().item(), f"nonfinite: {path}/{arm}")
            if arm != "c0":
                require(Counter(row_hash(memory, i) for i in range(old_n, n)) == baseline_hashes,
                        f"payload loss/duplication: {path}/{arm}")
            if arm in build["mappings"]:
                mapping = {int(k): v for k, v in build["mappings"][arm]["mapping"].items()}
                classes = build["structural_audit"]["c0_row_classes"]
                require(set(mapping) == set(mapping.values()) == set(range(n - old_n)),
                        f"nonbijective mapping: {path}/{arm}")
                require(all(classes[s] == classes[t] for s, t in mapping.items()),
                        f"cross-class mapping: {path}/{arm}")
                require(all(row_hash(identity, old_n+s) == row_hash(memory, old_n+t)
                            for s, t in mapping.items()), f"mapping/payload: {path}/{arm}")
            delta = adjacency @ memory["x"].float() - baseline_message
            checked[arm] = {
                "rows": n, "same_c0_topology": same_topology, "same_c0_parent_prefix": same_prefix,
                "changed_new_x_rows_vs_c0": int((memory["x"][old_n:] != c0["x"][old_n:]).any(1).sum()),
                "old_message_relative_drift": float(delta[:old_n].norm() /
                    baseline_message[:old_n].norm().clamp_min(1e-12)),
            }
            for group in groups:
                if equal(memory, memories[group[0]], training_keys):
                    group.append(arm)
                    break
            else:
                groups.append([arm])
        continued = {}
        if cell["stage"] == 2:
            previous = Path(memory_root) / cell["cell"].replace("_s2", "_s1")
            for arm in ("semantic_perm", "role_perm"):
                continued[arm] = equal(memories[arm], load(previous / f"{arm}_memory.pt"), keys, old_n)
        audits[cell["cell"]] = {
            "parent": str(parent_path), "remote_parent": str(remote_parent_path),
            "parent_loaded": parent is not None,
            "candidate_parent_continued": continued,
            "training_identical_groups": [g for g in groups if len(g) > 1], "arms": checked,
            "plasticity_adaptive_same_mapping": build["mappings"]["plasticity_role_perm"]["mapping"]
                == build["mappings"]["adaptive_role_perm"]["mapping"],
            "semantic_classes": build["mappings"]["semantic_perm"]["audit"]["classes"],
        }
    return audits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-roots", nargs=2, type=Path, required=True)
    parser.add_argument("--memory-root", type=Path)
    parser.add_argument("--source-root", dest="source_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable audit output exists: {args.output}")
    result, cells = summarize(args.eval_roots)
    if args.memory_root:
        global args_source_root
        args_source_root = args.source_root
        result["artifact_checks"] = artifact_audit(args.memory_root, cells)
    result["script_sha256"] = sha256(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cells": len(cells),
                      "summary": result["all_cells"]["semantic_perm"]}))


if __name__ == "__main__":
    main()
