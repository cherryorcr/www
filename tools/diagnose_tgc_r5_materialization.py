"""Read-only diagnostics of saved R5 artifacts; never selects candidates."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.snapshot / "code/v2_llm_fusion"))
    import torch
    import torch.nn.functional as F
    import run_tgc_r5_joint as J
    torch.set_num_threads(2)
    protocol = J.read(args.root / "protocol.json")
    assert J.R.source_hashes() == protocol["source_hashes"]
    records = []
    for dataset in protocol["datasets"]:
        for order in protocol["orders"]:
            folder = args.root / dataset / order
            manifest = J.read(folder / "manifest.json")
            for stage in (1, 2):
                c0 = J.load(folder / f"c0_s{stage}_memory.pt")
                old_n = manifest["budgets"][stage-1]
                c0_adj = J.R.C.training_adjacency(c0, "cpu").to_dense()
                for arm in protocol["arms"]:
                    memory = J.load(folder / f"{arm}_s{stage}_memory.pt")
                    edges, n = memory["edge_index"], len(memory["x"])
                    adj = J.R.C.training_adjacency(memory, "cpu").to_dense()
                    degree = torch.bincount(edges[0], minlength=n).float()
                    cross = edges[:, (edges[0] < old_n) & (edges[1] >= old_n)]
                    old_degree = torch.bincount(cross[0], minlength=old_n).float()
                    c0_cross = c0["edge_index"][:, (c0["edge_index"][0] < old_n) & (c0["edge_index"][1] >= old_n)]
                    changed_cross = len(set(map(tuple, cross.T.tolist())) - set(map(tuple, c0_cross.T.tolist())))
                    pair_delta = abs(len(J.edge_set(memory) - J.edge_set(c0))) // 2
                    records.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                                    "old_normalized_block_max_drift_vs_c0": float((adj[:old_n, :old_n] - c0_adj[:old_n, :old_n]).abs().max()),
                                    "max_old_cross_degree": float(old_degree.max()),
                                    "old_cross_coverage": float((old_degree > 0).float().mean()),
                                    "max_degree": float(degree.max()), "isolated_nodes": int((degree == 0).sum()),
                                    "changed_undirected_edges_vs_c0": pair_delta,
                                    "changed_cross_edges_vs_c0": changed_cross,
                                    "new_text_cosine_vs_c0": float(F.cosine_similarity(memory["text_emb"][old_n:], c0["text_emb"][old_n:]).mean()),
                                    "new_text_identical_fraction": sum(a == b for a, b in zip(memory["texts"][old_n:], c0["texts"][old_n:])) / (n-old_n),
                                    "new_x_norm_ratio_vs_c0": float((memory["x"][old_n:].norm(dim=1) / c0["x"][old_n:].norm(dim=1).clamp_min(1e-6)).mean()),
                                    "report_selected": False})
    result = {"scope": "offline descriptive artifact diagnostics, not causal attribution or candidate selection", "records": records}
    (args.root / "materialization_diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "status": "complete"}))


if __name__ == "__main__":
    main()
