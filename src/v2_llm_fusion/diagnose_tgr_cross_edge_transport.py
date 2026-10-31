"""Read-only diagnosis of routing and degree-constrained edge quantization.

No alternative memory is exported and no report score selects an edge.
The assignment optimum below is for the frozen proxy, not for Student F1.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from scipy.optimize import linear_sum_assignment

import delta_cross_edge_transport as CE
import teacher_guided_message_revision as TGM
import teacher_guided_payload_revision as T


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def diagnose(cell, semantic_root):
    base = load(cell / "c0/tgr_message_memory.pt")
    candidate = load(cell / "message_balanced/tgr_message_memory.pt")
    support = load(cell / "message_balanced/support.pt")
    support_audit = json.loads((cell / "message_balanced/support_audit.json").read_text())
    sealed = json.loads((cell / "message_balanced/assignment.json").read_text())
    old_n = support_audit["old_rows"]
    assignment = T.build_sparse_assignment(base, support, old_n, support["class_id"],
                                           T.Config(top_r=8), "cpu")
    assert assignment["indices"].tolist() == sealed["indices"]
    saved_probs = torch.tensor(sealed["weights"], dtype=torch.float32)
    reconstructed = torch.softmax(assignment["logits"], 1).detach()
    # The serialized weights are the construction artifact. Small CPU/GPU
    # distance rounding can change the last float bits when reconstructing
    # the logits, so only require the saved routing indices to match here.
    if assignment["indices"].tolist() != sealed["indices"]:
        raise AssertionError("sealed routing indices cannot be replayed")
    # Use the saved float32 probabilities. Thread-dependent distance rounding
    # must not silently change the payload routing being diagnosed.
    new_n, raw_n = len(base["x"]) - old_n, len(support["x"]) - old_n
    p = torch.zeros(new_n, raw_n).scatter_add(1, assignment["indices"], saved_probs)
    raw_mass = p.sum(0)
    normalized = p.T / raw_mass.clamp_min(1e-8)[:, None]
    fallback = torch.nn.functional.one_hot(assignment["distance"].argmin(0), new_n).float()
    normalized = torch.where((raw_mass <= 1e-8)[:, None], fallback, normalized)
    q = torch.zeros(len(support["x"]), len(base["x"]))
    q[:old_n, :old_n] = torch.eye(old_n)
    q[old_n:, old_n:] = normalized
    projected = TGM._projected_kernel(TGM._binary_adjacency(
        support["edge_index"], len(support["x"]), "cpu", base["x"].dtype), q).detach()
    edge, _ = CE.materialize_cross_only(base["edge_index"], projected, old_n)
    assert torch.equal(edge, candidate["edge_index"])
    before, after = {}, {}
    for memory, pairs in ((base, before), (candidate, after)):
        for a, b in memory["edge_index"].T.tolist():
            if a < old_n <= b:
                assert b not in pairs, "diagnosis assumes one old neighbor per new slot"
                pairs[b] = a
    new_nodes = sorted(before)
    capacities = Counter(before.values())
    columns = [old for old, count in sorted(capacities.items()) for _ in range(count)]
    assert len(new_nodes) == len(columns)
    scores = projected[columns][:, new_nodes].T.numpy()
    ri, ci = linear_sum_assignment(-scores)
    opt = {new_nodes[int(r)]: columns[int(c)] for r, c in zip(ri, ci)}
    def score_stats(pairs):
        values = [float(projected[old, new]) for new, old in pairs.items()]
        return {"sum_projected_score": sum(values), "zero_score_pairs": sum(v <= 1e-8 for v in values),
                "pair_count": len(values)}
    greedy_stats, opt_stats = score_stats(after), score_stats(opt)
    first = new_nodes[:len(new_nodes) // 2]
    last = new_nodes[len(new_nodes) // 2:]
    class_audit = json.loads((semantic_root / cell.name / "build_audit.json").read_text())
    slot_classes = torch.tensor(class_audit["structural_audit"]["c0_row_classes"])
    raw_class = support["class_id"][old_n:]
    misroute = (slot_classes[None, :] != raw_class[:, None]).float()
    q_new = q[old_n:, old_n:].detach()
    base_adj = CE._norm_adj(base["edge_index"], len(base["x"]))
    cand_adj = CE._norm_adj(candidate["edge_index"], len(candidate["x"]))
    relative = ((cand_adj @ base["x"] - base_adj @ base["x"]).norm(dim=1) /
                (base_adj @ base["x"]).norm(dim=1).clamp_min(1e-8))
    selected_scores = {
        "early_zero_fraction": sum(float(projected[after[n], n]) <= 1e-8 for n in first) / len(first),
        "late_zero_fraction": sum(float(projected[after[n], n]) <= 1e-8 for n in last) / len(last),
    }
    return {"cell": cell.name, "replay_exact": True,
            "rebuilt_probability_max_error": float((reconstructed - saved_probs).abs().max()),
            "rows": len(base["x"]), "old_rows": old_n, "raw_delta_rows": len(raw_class),
            "changed_cross_fraction": sum(before[n] != after[n] for n in new_nodes) / len(new_nodes),
            "raw_fallback_fraction": float((p.sum(0) <= 1e-8).float().mean()),
            "inferred_slot_class_mismatches": int((slot_classes != assignment["slot_class"]).sum()),
            "cross_class_routing_mass_fraction": float((q_new * misroute).sum() / q_new.sum()),
            "base_proxy": score_stats(before), "deployed_greedy_proxy": greedy_stats,
            "optimal_proxy_diagnostic_only": opt_stats,
            "proxy_objective_regret": opt_stats["sum_projected_score"] - greedy_stats["sum_projected_score"],
            "optimal_would_change_pairs": sum(opt[n] != after[n] for n in new_nodes),
            "relative_one_hop_message_change_old": float(relative[:old_n].mean()),
            "relative_one_hop_message_change_new": float(relative[old_n:].mean()),
            **selected_scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--development-semantic-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = []
    for cell in sorted(args.development_root.glob("*_s[12]")):
        rows.append({"cohort": "development", **diagnose(cell, args.development_semantic_root)})
    for dataset in ("cora", "citeseer"):
        root = args.confirmation_root / dataset
        for cell in sorted((root / "topology").glob("*_s[12]")):
            rows.append({"cohort": "confirmation", **diagnose(cell, root / "semantic")})
    if len(rows) != 12:
        raise ValueError("expected 12 sealed stage cells")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cells": rows, "new_memories_exported": 0,
                                      "report_labels_used": False}, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
