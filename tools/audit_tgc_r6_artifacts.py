"""Read-only R6 projection, text provenance and weighted-operator audit."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/v2_llm_fusion"))
import torch
import run_tgc_r6_mechanisms as V

E, K, R, T, D = V.E, V.K, V.R, V.T, V.D


def audit(args):
    spec = E.read(args.output / "protocol.json")
    E.configure(spec)
    D.verify_seal(args.output)
    assert R.source_hashes() == spec["source_hashes"]
    results = []
    for dataset in spec["datasets"]:
        cache = Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919") / f"{dataset}_r005/text_emb_pool_only.pt"
        data = R.R.load_data(dataset, args.minilm, args.device, cache)
        for order in spec["orders"]:
            ids, budgets = D.set_stream(data, order)
            folder, source = args.output / dataset / order, V.SOURCE / dataset / order
            for stage in spec["evaluation_stages"]:
                c0 = E.load(source / f"c0_s{stage}_memory.pt")
                old0 = E.load(source / f"c0_s{stage-1}_memory.pt")
                old1 = E.load(source / f"recursive_s{stage-1}_memory.pt")
                block = E.load(source / f"c0_s{stage}_block.pt")
                delta = T.raw_delta_block(data, ids[stage], stage)
                old_n = len(old0["x"])
                q = D.current_assignment(data, ids[stage], c0, old_n, stage, args.device)
                for arm in E.arms_at(spec, stage):
                    if arm == "c0" or arm.startswith("frozen_") or stage == 1 and arm == "parent_111":
                        continue
                    detail = E.read(folder / f"{arm}_s{stage}_build_audit.json")
                    memory = E.load(folder / f"{arm}_s{stage}_memory.pt")
                    if arm == "symmetric_recursive":
                        parent = old0 if stage == 1 else E.load(folder / "symmetric_recursive_s1_memory.pt")
                    else:
                        parent = K.blend_parent(old0, old1, detail["parent_bits"])
                    ref_parent = old0 if arm == "reference0_output1" else old1 if arm == "reference1_output0" else parent
                    base = V.V.append_parent(ref_parent, block, args.device)
                    support = K.M.lifted_support(ref_parent, delta, base, q)
                    assert T.memory_hash(support) == detail["reference"]["support_hash"]
                    assert T.memory_hash(ref_parent) == detail["reference"]["parent_hash"]
                    assert detail["current_delta_ids"] == ids[stage].tolist()
                    p = E.load(folder / f"{arm}_s{stage}_assignment.pt").double()
                    pt = p
                    if arm.startswith("decoupled"):
                        pt_raw = E.load(folder / f"{arm}_s{stage}_topology_assignment.pt")
                        sha = R.C.tensor_hash({"p": pt_raw})
                        assert sha == detail["topology_optimization"]["assignment_sha256"]
                        assert sha == detail["materialization"]["topology_assignment_sha256"]
                        pt = pt_raw.double()
                    u = torch.zeros(len(support["x"]), len(memory["x"]), dtype=torch.float64)
                    u[:old_n, :old_n] = torch.eye(old_n, dtype=torch.float64)
                    u[old_n:, old_n:] = pt
                    # Double precision reconstruction independent of stored coarse matrices.
                    n = len(support["x"])
                    indices = torch.cat((support["edge_index"], torch.arange(n).repeat(2, 1)), 1)
                    weights = torch.cat((support["edge_weight"].double(), torch.ones(n, dtype=torch.float64)))
                    adj = torch.sparse_coo_tensor(indices, weights, (n, n)).coalesce()
                    coarse = u.T @ torch.sparse.mm(adj, u)
                    degree = coarse.sum(1)
                    score = coarse / (degree[:, None] * degree[None, :]).sqrt()
                    e = memory["edge_index"]
                    internal = e[:, (e[0] >= old_n) & (e[1] >= old_n) & (e[0] < e[1])]
                    candidates = torch.triu_indices(len(memory["x"])-old_n, len(memory["x"])-old_n, 1) + old_n
                    threshold = score[candidates[0], candidates[1]].sort(descending=True).values[internal.shape[1]-1]
                    assert score[internal[0], internal[1]].min() >= threshold-2e-6
                    cross = e[:, (e[0] < old_n) & (e[1] >= old_n)]
                    assert torch.equal(cross[1], torch.arange(old_n, len(memory["x"])))
                    error = (score[:old_n, old_n:].max(0).values-score[cross[0], cross[1]]).abs().max().item()
                    assert error < 2e-6
                    for j in range(p.shape[1]):
                        members = torch.where(p[:, j] > 0)[0]
                        chosen = members[torch.argsort(p[members, j], descending=True, stable=True)[:V.CFG.text_sources]]
                        per = max(1, V.CFG.text_chars // len(chosen))
                        expected = " ".join(str(delta["texts"][int(i)])[:per] for i in chosen)[:V.CFG.text_chars]
                        assert memory["texts"][old_n+j] == expected
                    weighted_error = None
                    if "weighted" in arm:
                        raw = torch.zeros_like(coarse)
                        raw[e[0], e[1]] = coarse[e[0], e[1]].clamp_min(1e-9)
                        deg = raw.sum(1)
                        canonical = raw / (deg[:, None]*deg[None, :]).sqrt()
                        actual = R.C.training_adjacency(memory, "cpu").to_dense().double()
                        weighted_error = (actual-canonical).abs().max().item()
                        assert weighted_error < 2e-5 and memory["adj_normalized"]
                    reference_arm = "parent_111"
                    reference_memory = E.load(folder / f"{reference_arm}_s{stage}_memory.pt")
                    actual_adj = R.C.training_adjacency(memory, "cpu").to_dense().double()
                    ref_adj = R.C.training_adjacency(reference_memory, "cpu").to_dense().double()
                    source_weights = p / p.sum(0)
                    ref_edges = set(map(tuple, reference_memory["edge_index"].T.tolist()))
                    actual_edges = set(map(tuple, e[:, e[0] != e[1]].T.tolist()))
                    results.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                                    "cross_projection_score_error": error,
                                    "weighted_operator_max_error": weighted_error,
                                    "old_normalized_block_max_change_vs_recursive": float((actual_adj-ref_adj)[:old_n, :old_n].abs().max()),
                                    "mean_effective_sources": float((1/source_weights.square().sum(0)).mean()),
                                    "mean_source_assignment_entropy": float(-(p*p.clamp_min(1e-30).log()).sum(1).mean()),
                                    "mean_new_x_relative_change_vs_recursive": float((memory["x"][old_n:]-reference_memory["x"][old_n:]).norm(dim=1).div(reference_memory["x"][old_n:].norm(dim=1).clamp_min(1e-9)).mean()),
                                    "new_text_rows_changed_vs_recursive": sum(a != b for a, b in zip(memory["texts"][old_n:], reference_memory["texts"][old_n:])),
                                    "directed_edge_symmetric_difference_vs_recursive": len(actual_edges ^ ref_edges)})
    result = {"status": "passed", "artifacts_checked": len(results), "results": results,
              "scope": "support provenance, reference/output intervention, topology-P hashes, double projection ranks, text sources, weighted symmetric operator"}
    E.save(args.output / "projection_artifact_audit.json", result)
    print(json.dumps({k: result[k] for k in ("status", "artifacts_checked", "scope")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    torch.set_num_threads(2)
    audit(parser.parse_args())
