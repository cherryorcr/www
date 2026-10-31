"""Independent STAD artifact algebra checks and frozen-v4 training regression."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_stad_graphclip_static as B


def normalize_dense(adj):
    adj = adj.double().clone()
    if adj[0, 0] == 0:
        adj += torch.eye(len(adj), dtype=adj.dtype)
    scale = adj.sum(1).pow(-.5)
    scale[torch.isinf(scale)] = 0
    return scale[:, None] * adj * scale[None, :]


def audit_dataset(root, dataset, device):
    data = B.S.load_data(dataset, str(Path(B.U.__file__).parent / "minilm_from_graphclip"), device)
    stages, budgets = B.S.stage_inputs(data)
    encoder = B.TextEncoder(data["encoder"])
    final_ids = torch.cat(stages)
    encoded = encoder.encode([data["texts"][int(i)] for i in final_ids]).cpu()
    cosine = F.cosine_similarity(data["x"][final_ids], encoded)
    alignment = {"rows": len(final_ids), "cosine_mean": float(cosine.mean()),
                 "cosine_min": float(cosine.min()),
                 "original_mean_norm": float(data["x"][final_ids].norm(dim=1).mean()),
                 "encoder_mean_norm": float(encoded.norm(dim=1).mean())}
    checked = []
    for stage, budget in enumerate(budgets):
        base = root / dataset / f"s{stage}"
        memory = torch.load(base / "memory.pt", map_location="cpu", weights_only=False)
        audit = json.loads((base / "construction.json").read_text())
        core = torch.load(base / "core.pt", map_location="cpu", weights_only=False)
        ids = torch.cat(stages[:stage + 1])
        B.validate_sources(data, ids, stage)
        assert ids.tolist() == audit["source_ids"]
        assert len(memory["x"]) == budget
        assert audit["tensor_sha256"] == B.C.tensor_hash(memory)
        assert audit["memory_file_sha256"] == B.C.sha(base / "memory.pt")
        raw = B.C.raw_support(data, ids, stage)
        assert audit["source_support_sha256"] == B.C.tensor_hash(raw)
        clusters = torch.tensor(core["clusters"], dtype=torch.long)
        assert sorted(clusters.unique().tolist()) == list(range(budget))
        membership = F.one_hot(clusters, num_classes=budget).double()
        counts = membership.sum(0)
        membership /= counts
        means = membership.T @ raw["x"].double()
        torch.testing.assert_close(means.float(), core["centers"], atol=2e-6, rtol=2e-5)
        edges = raw["edge_index"]
        adj = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]),
                                     (len(ids), len(ids))).to_dense()
        condensed = membership.T @ normalize_dense(adj) @ membership
        condensed.fill_diagonal_(0)
        expected = normalize_dense(condensed).float()
        actual = B.C.training_adjacency(memory, "cpu").to_dense()
        torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(counts.float(), memory["mass"])
        encoded_text = encoder.encode(memory["texts"]).cpu()
        torch.testing.assert_close(encoded_text, memory["x"], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(F.normalize(encoded_text, dim=1), memory["text_emb"],
                                   atol=2e-5, rtol=2e-5)
        scores = audit["sinkhorn_scores"]
        expected_shortlist = sorted(range(100), key=lambda n: (scores[n], n))[:5]
        assert [r["candidate"] for r in audit["selection"]] == expected_shortlist
        assert all(r["seed"] == 7000 + stage for r in audit["selection"])
        winner = max(audit["selection"], key=lambda r: (r["selector_accuracy"], -r["candidate"]))
        assert winner["candidate"] == audit["winner"]
        for cluster in range(budget):
            payload = json.loads((base / "llm" / f"node{cluster:04d}.json").read_text())
            fingerprint = hashlib.sha256(json.dumps(payload["request"], sort_keys=True).encode()).hexdigest()
            assert payload["request_sha256"] == fingerprint
            assert len(payload["texts"]) == 3
            assert memory["texts"][cluster] in payload["texts"]
            assert payload["request"]["seed"] == 150000 + stage * 1000 + cluster
        checked.append({"stage": stage, "rows": budget,
                        "weighted_adjacency_max_error": float((expected - actual).abs().max()),
                        "synthetic_text_feature_max_error": float((encoded_text - memory["x"]).abs().max()),
                        "llm_requests_checked": budget})
    # A full seed-100 fit verifies the generalized adjacency entrypoint did not
    # alter the current method, not just the shape of its adjacency tensor.
    reference_root = B.C.RESULTS / "small_comparison_audited_r2_2026_09_16" / dataset
    ref = json.loads((reference_root / "v4_s2_seed100.json").read_text())
    mem = torch.load(reference_root / "v4_s2_memory.pt", map_location="cpu", weights_only=False)
    rerun = B.C.fit(mem, 100, B.P.student_steps(budgets[-1]), device, data, 2)
    for key in ("support_sha256", "start_state_sha256", "end_state_sha256"):
        assert ref[key] == rerun[key], f"frozen v4 regression failed: {dataset}/{key}"
    assert B.normalized_metrics(ref["metrics"]) == B.normalized_metrics(rerun["metrics"])
    return {"dataset": dataset, "stages": checked, "source_embedding_alignment": alignment,
            "v4_seed100_state_and_metrics_bitwise_regression": True, "errors": [],
            "v4_reference_sha256": B.C.sha(reference_root / "v4_s2_seed100.json")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.set_num_threads(2)
    result = {"datasets": [audit_dataset(args.root, dataset, args.device)
                           for dataset in ("cora", "citeseer")],
              "auditor_sha256": B.C.sha(__file__), "test_labels_evaluated": False}
    B.C.save_json(args.root / "independent_acceptance.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
