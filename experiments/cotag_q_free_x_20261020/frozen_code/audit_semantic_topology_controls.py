"""Independent audit for semantic topology-control fits and memories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import f1_score

import audit_link_prediction_transfer as LP_AUDIT
import large_lp_support as B
import run_semantic_topology_controls as R
import run_topology_causal_ablation as T
from summarize_semantic_topology_controls import NEW_ARMS, record_key, rows, validate


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def nc_scores(logits, target, groups, stage):
    seen = [c for g in groups[:stage + 1] for c in g]
    old = [c for g in groups[:stage] for c in g]
    active = np.isin(target, seen)
    y = target[active]
    result = {}
    for scope in ("all", "seen"):
        allowed = list(range(logits.shape[1])) if scope == "all" else sorted(seen)
        pred = np.asarray(allowed)[logits[:, allowed].numpy().argmax(1)][active]
        scores = f1_score(y, pred, labels=seen, average=None, zero_division=0) * 100
        by_class = dict(zip(seen, scores.tolist()))
        result[scope] = dict(cumulative_f1=float(scores.mean()), accuracy=float(np.mean(pred == y) * 100),
                             old_f1=float(np.mean([by_class[c] for c in old])) if old else None,
                             new_f1=float(np.mean([by_class[c] for c in groups[stage]])))
    return result


def metric_gap(left, right):
    gaps = []
    for scope, values in left.items():
        for name, value in values.items():
            if value is None:
                assert right[scope][name] is None
            else:
                gaps.append(abs(value - right[scope][name]))
    return max(gaps, default=0.)


def graph_checks(memory, old, keep):
    n = len(memory["x"])
    edge = memory["edge_index"].numpy()
    assert edge.shape[0] == 2 and (edge >= 0).all() and (edge < n).all()
    assert (edge[0] != edge[1]).all()
    keys = set((edge[0] * n + edge[1]).tolist())
    assert len(keys) == edge.shape[1], "duplicate directed edges"
    assert keys == set((edge[1] * n + edge[0]).tolist()), "missing reverse edge"
    matrix = sparse.coo_matrix((np.ones(edge.shape[1]), edge), shape=(n, n)).tocsr()
    deg = np.asarray(matrix.sum(1)).ravel()
    new_matrix = matrix[keep:, keep:]
    load = np.asarray(matrix[:keep, keep:].sum(1)).ravel()
    out = dict(rows=n, pairs=edge.shape[1] // 2, isolated=int((deg == 0).sum()),
               max_degree=int(deg.max()), mean_degree=float(deg.mean()),
               new_components=int(connected_components(new_matrix, directed=False,
                                                       return_labels=False)),
               new_internal_pairs=int(new_matrix.nnz // 2),
               cross_pairs=int(load.sum()),
               max_cross_load=int(load.max()) if keep else 0)
    if old is not None and keep:
        old_e = old["edge_index"].numpy()
        old_matrix = sparse.coo_matrix((np.ones(old_e.shape[1]), old_e), shape=(keep, keep)).tocsr()
        old_degree = np.asarray(old_matrix.sum(1)).ravel() + 1
        new_degree = deg + 1
        old_norm = sparse.diags(old_degree ** -.5) @ (old_matrix + sparse.eye(keep)) @ sparse.diags(old_degree ** -.5)
        new_norm = sparse.diags(new_degree ** -.5) @ (matrix + sparse.eye(n)) @ sparse.diags(new_degree ** -.5)
        drift = np.asarray(abs(new_norm[:keep, :keep] - old_norm).sum(1)).ravel()
        cross_mass = np.asarray(new_norm[:keep, keep:].sum(1)).ravel()
        out.update(old_block_row_l1=float(drift.mean()), full_old_row_l1=float((drift + cross_mass).mean()),
                   cross_norm_mass=float(cross_mass.mean()),
                   max_relative_load=float((load / old_degree).max()))
    else:
        out.update(old_block_row_l1=0., full_old_row_l1=0., cross_norm_mass=0., max_relative_load=0.)
    return out, deg


def check_state(path, evaluation, record, construction):
    assert T.sha(path) == record["checkpoint_sha256"]
    assert T.sha(evaluation) == record["evaluation_sha256"]
    assert record["memory_sha256"] == construction[record["stage"]]["arms"][record["arm"]]["sha256"]
    state = load(path)
    assert all(state["context"][k] == v for k, v in record.items() if k in state["context"])
    return state


def track(info, record, fresh, stored, gap, tolerance):
    task = record["task"]
    score_error = max(float((a - b).abs().max()) for a, b in zip(fresh, stored))
    assert all(torch.allclose(a, b, atol=1e-4, rtol=1e-5) for a, b in zip(fresh, stored))
    info["max_score_reload_abs_error"][task] = max(info["max_score_reload_abs_error"][task], score_error)
    info["max_metric_reload_abs_error"][task] = max(info["max_metric_reload_abs_error"][task], gap)
    if gap > tolerance:
        info["reload_warnings"].append({k: record[k] for k in ("task", "order", "stage", "arm", "seed")}
                                        | {"split": record.get("split"), "gap": gap})
    info["fits"] += 1
    info[task + "_fits"] += 1


def audit(root, baseline_root, minilm, device):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = json.loads((root / "manifest.json").read_text())
    baseline_manifest = json.loads((baseline_root / "manifest.json").read_text())
    records = rows(root)
    expected = validate(records, manifest)
    combined = json.loads((root / "results.json").read_text())
    assert {record_key(r) for r in combined} == {record_key(r) for r in records}
    assert T.sha(baseline_root / "manifest.json") == manifest["baseline_manifest_sha256"]
    assert T.sha(Path(manifest["data"])) == manifest["data_sha256"]
    assert T.sha(Path(manifest["text"])) == manifest["text_sha256"]
    repo = Path(R.__file__).resolve().parents[2]
    for name, expected_hash in manifest["sources"].items():
        assert T.sha(repo / name) == expected_hash, name
    info = dict(dataset=manifest["dataset"], fits=0, nc_fits=0, lp_fits=0, memories=0,
                max_score_reload_abs_error={"nc": 0., "lp": 0.},
                max_metric_reload_abs_error={"nc": 0., "lp": 0.}, reload_warnings=[],
                topology=[], passed=False, source_files_checked=len(manifest["sources"]),
                limits=["This reuses the prior orders/seeds/splits; it is not an independent confirmation.",
                        "Baseline evaluation/split artifacts are reused byte-for-byte.",
                        "Construction time is topology-only because the content block is reused.",
                        "NC remains transductive report-dev; LP negatives remain random stratified non-edges."])
    D = R.U.load_unified(manifest["dataset"], manifest["data"], minilm, device,
                         n_stages=5, split_mode="base_incremental")
    text = load(manifest["text"]).float()
    assert not bool(text[~D["pool_mask"]].any())
    real_x = D["x"].to(device)
    adj = R.U.G.norm_adj(D["edge_index"], len(real_x), device)
    prompts = D["encoder"].encode(D["prompts"])
    old_memories = {}
    for order_text, groups in manifest["orders"].items():
        order = int(order_text)
        folder = root / f"order_{order}"
        base_folder = baseline_root / f"order_{order}"
        construction = json.loads((folder / "construction.json").read_text())
        base_construction = json.loads((base_folder / "construction.json").read_text())
        previous = {arm: None for arm in NEW_ARMS}
        for stage, classes in enumerate(groups):
            base_meta = base_construction[stage]
            base_memory = load(base_folder / base_meta["arms"]["c0"]["path"])
            keep = 0 if stage == 0 else len(previous[NEW_ARMS[0]]["x"])
            block = {key: base_memory[key][keep:] for key in R.FIELDS}
            block["texts"] = base_memory["texts"][keep:]
            old_base = None if stage == 0 else load(base_folder / "memories" / f"c0_s{stage - 1}.pt")
            candidates = T.topology_components(old_base, block, keep, order, stage, device)
            _, _, nearest_cross = T.components(candidates["c0"]["edge_index"], keep, len(base_memory["x"]))
            _, _, capacity_cross = T.components(candidates["t_degree"]["edge_index"], keep, len(base_memory["x"]))
            assert construction[stage]["classes"] == classes
            for arm in NEW_ARMS:
                memory_path = folder / "memories" / f"{arm}_s{stage}.pt"
                memory = load(memory_path)
                assert T.sha(memory_path) == construction[stage]["arms"][arm]["sha256"]
                assert len(memory["x"]) == len(base_memory["x"])
                for key in R.FIELDS:
                    assert torch.equal(memory[key], base_memory[key]), (arm, key)
                assert memory["texts"] == base_memory["texts"]
                T.frozen(previous[arm], memory)
                internal, op_info = R.semantic_internal(block, "knn" if arm.startswith("knn") else "cycles")
                actual_old, actual_internal, actual_cross = T.components(memory["edge_index"], keep, len(memory["x"]))
                expected_cross = capacity_cross if arm.endswith("capacity") else nearest_cross
                assert torch.equal(actual_internal, internal + keep), arm
                assert torch.equal(actual_cross, expected_cross), arm
                topo, degree = graph_checks(memory, previous[arm], keep)
                assert topo["pairs"] == T.degree_stats(base_memory)["pairs"]
                info["topology"].append(dict(order=order, stage=stage, arm=arm, **topo))
                info["memories"] += 1
                previous[arm] = memory
            # Paired cross policy and equal-degree cycle control are explicit.
            for nearest, capacity in (("knn_nearest", "knn_capacity"),
                                      ("cycles_nearest", "cycles_capacity")):
                a = load(folder / "memories" / f"{nearest}_s{stage}.pt")
                b = load(folder / "memories" / f"{capacity}_s{stage}.pt")
                assert torch.equal(T.components(a["edge_index"], keep, len(a["x"]))[1],
                                   T.components(b["edge_index"], keep, len(b["x"]))[1])
            base_previous = old_memories.get(order)
            old_memories[order] = previous

        nc_eval = folder / "nc/evaluation.pt"
        base_nc_eval = base_folder / "nc/evaluation.pt"
        ev = load(nc_eval); base_ev = load(base_nc_eval)
        for key in ev:
            assert torch.equal(ev[key], base_ev[key]) if isinstance(ev[key], torch.Tensor) else ev[key] == base_ev[key]
        assert torch.equal(ev["prompts"], prompts) and ev["stage_classes"] == groups
        for record in (r for r in records if r["order"] == order and r["task"] == "nc"):
            ckpt = folder / f"nc/{record['arm']}_s{record['stage']}_seed{record['seed']}.pt"
            state = check_state(ckpt, nc_eval, record, construction)
            assert record["cost"]["steps"] == 2048 and record["cost"]["batch"] == 16
            stored = nc_scores(state["logits"], ev["y"].numpy(), groups, record["stage"])
            assert metric_gap(stored, record["metrics"]) < 1e-10
            model = R.U.G.CLIPStudent(real_x.shape[1]).to(device); model.load_state_dict(state["state_dict"]); model.eval()
            with torch.no_grad(): logits = model.logits(real_x, adj, ev["prompts"].to(device)).cpu()[ev["ids"]]
            track(info, record, [logits], [state["logits"]], metric_gap(stored, nc_scores(logits, ev["y"].numpy(), groups, record["stage"])), 1e-6)
        for split_seed in manifest["splits"]:
            lp_folder = folder / f"lp_split_{split_seed}"
            base_lp_folder = base_folder / f"lp_split_{split_seed}"
            split = load(lp_folder / "edge_split.pt"); base_split = load(base_lp_folder / "edge_split.pt")
            for key in split:
                assert torch.equal(split[key], base_split[key]) if isinstance(split[key], torch.Tensor) else split[key] == base_split[key]
            for stage in range(len(groups)):
                ev_path = lp_folder / f"evaluation_s{stage}.pt"; base_ev_path = base_lp_folder / f"evaluation_s{stage}.pt"
                ev = load(ev_path); base_ev = load(base_ev_path)
                for key in ev:
                    assert torch.equal(ev[key], base_ev[key]) if isinstance(ev[key], torch.Tensor) else ev[key] == base_ev[key]
                for record in (r for r in records if r["task"] == "lp" and r["order"] == order
                               and r["split"] == split_seed and r["stage"] == stage):
                    ckpt = lp_folder / f"{record['arm']}_s{stage}_seed{record['seed']}.pt"
                    state = check_state(ckpt, ev_path, record, construction)
                    assert record["cost"]["steps"] == 256 and record["cost"]["edge_batch"] == 256
                    stored = LP_AUDIT.recompute_scores(ev["test_pos"], ev["test_neg"], state["pos_scores"],
                                                      state["neg_scores"], ev["arrival"], stage)
                    assert stored == record["metrics"]
                    model = R.L.GAE(R.L.Encoder(ev["x"].shape[1])).to(device); model.load_state_dict(state["state_dict"]); model.eval()
                    with torch.no_grad():
                        z = model.encode(ev["x"].to(device), ev["edge_index"].to(device))
                        ps = model.decoder(z, ev["test_pos"].to(device), sigmoid=False).cpu()
                        ns = model.decoder(z, ev["test_neg"].to(device), sigmoid=False).cpu()
                    fresh = LP_AUDIT.recompute_scores(ev["test_pos"], ev["test_neg"], ps, ns, ev["arrival"], stage)
                    track(info, record, [ps, ns], [state["pos_scores"], state["neg_scores"]],
                          max(abs(fresh[s][m] - stored[s][m]) for s in fresh for m in ("auc", "ap")
                              if fresh[s][m] is not None), 1e-4)
    assert info["fits"] == expected and info["memories"] == 40
    info["passed"] = True
    info["strict_metric_reload_passed"] = not info["reload_warnings"]
    R.save_json(root / "independent_audit.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "topology"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True); p.add_argument("--minilm", required=True)
    p.add_argument("--device", default="cuda:0"); a = p.parse_args()
    audit(a.root, a.baseline_root, a.minilm, a.device)
