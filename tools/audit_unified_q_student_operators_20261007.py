"""Read-only memory audit; write only a separate compact operator report."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import graphtext_prompt_student as P
import run_unified_q_large_smoke_20261007 as L
import run_unified_q_revision_20261007 as V


def compare(old, new):
    difference = (old - new).coalesce().values().abs()
    def digest(adj):
        adj = adj.coalesce().cpu()
        return hashlib.sha256((V.hash_tensor(adj.indices()) + V.hash_tensor(adj.values())).encode()).hexdigest()
    return {"legacy_hash": digest(old), "prompt_hash": digest(new),
            "max_absolute_difference": float(difference.max()) if len(difference) else 0.0,
            "entries_above_1e_7": int((difference > 1e-7).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    a = p.parse_args()
    torch.set_num_threads(2)
    rows = []
    for dataset in ("wikics", "bookhis"):
        source = a.project / f"results/unified_q_revision_verified_20261007/{dataset}_fixed_rate_order9101_seed271006"
        protocol = json.loads((source / "protocol.json").read_text())
        data = L.load_large(dataset, "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
            a.project / f"results/public_baselines_text_cache_20261001/{dataset}", "cpu", protocol)
        edge = data["edge_index"]
        rows.append({"dataset": dataset, "scope": "full raw evaluation graph",
            **compare(L.U.G.norm_adj(edge, len(data["x"]), "cpu"),
                      P.normalized_adjacency(edge, len(data["x"]), "cpu"))})
        for arm in ("cat_cctp", "uq_semantic"):
            memory = torch.load(source / f"latest_{arm}.pt", map_location="cpu", weights_only=False)["memory"]
            for stage, n in enumerate(protocol["budgets"]):
                edge = memory["edge_index"]
                edge = edge[:, (edge < n).all(0)]
                rows.append({"dataset": dataset, "scope": "synthetic training graph", "arm": arm,
                    "stage": stage, **compare(L.U.G.norm_adj(edge, n, "cpu"),
                                               P.normalized_adjacency(edge, n, "cpu"))})
    output = a.project / "results/unified_q_diagnostics_20261007/operator_audit.json"
    output.write_text(json.dumps({"rows": rows}, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
