"""Post-run artifact audit; does not train, select, or modify experiment inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import f1_score

import run_topology_causal_ablation as R
from audit_link_prediction_transfer import recompute_scores
from summarize_topology_causal_ablation import read_json, read_records, record_key, validate


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


def metric_gap(first, second):
    differences = []
    for scope, metrics in first.items():
        for name, value in metrics.items():
            if value is not None:
                differences.append(abs(value - second[scope][name]))
            else:
                assert second[scope][name] is None
    return max(differences, default=0.)


def check_checkpoint(path, evaluation, record, construction):
    assert R.sha(path) == record["checkpoint_sha256"]
    assert R.sha(evaluation) == record["evaluation_sha256"]
    assert record["memory_sha256"] == construction[record["stage"]]["arms"][record["arm"]]["sha256"]
    state = load(path)
    assert all(record[k] == v for k, v in state["context"].items())
    return state


def track_reload(info, record, fresh, stored, gap, tolerance):
    task = record["task"]
    error = max(float((a - b).abs().max()) for a, b in zip(fresh, stored))
    assert all(torch.allclose(a, b, atol=1e-4, rtol=1e-5) for a, b in zip(fresh, stored))
    info["max_score_reload_abs_error"][task] = max(error, info["max_score_reload_abs_error"][task])
    info["max_metric_reload_abs_error"][task] = max(gap, info["max_metric_reload_abs_error"][task])
    if gap > tolerance:
        info["reload_warnings"].append({k: record[k] for k in ("task", "order", "stage", "arm", "seed")}
                                        | {"split": record.get("split"), "gap": gap})
    info["fits"] += 1
    info[task + "_fits"] += 1
    if info["fits"] % 30 == 0:
        print(json.dumps(dict(audited_fits=info["fits"])), flush=True)


def graph_checks(memory, old, keep):
    n = len(memory["x"])
    edge = memory["edge_index"].numpy()
    assert edge.shape[0] == 2 and (edge >= 0).all() and (edge < n).all()
    assert (edge[0] != edge[1]).all()
    keys = set((edge[0] * n + edge[1]).tolist())
    assert len(keys) == edge.shape[1], "duplicate directed edges"
    assert keys == set((edge[1] * n + edge[0]).tolist()), "missing reverse edge"
    a = sparse.coo_matrix((np.ones(edge.shape[1]), edge), shape=(n, n)).tocsr()
    deg = np.asarray(a.sum(1)).ravel()
    ordered = np.sort(deg)
    gini = ((2 * np.arange(1, n + 1) - n - 1) @ ordered / (n * ordered.sum())
            if ordered.sum() else 0.)
    new_a = a[keep:, keep:]
    new_components = connected_components(new_a, directed=False, return_labels=False)
    load_new = np.asarray(a[:keep, keep:].sum(1)).ravel()
    out = dict(rows=n, pairs=edge.shape[1] // 2, isolated=int((deg == 0).sum()),
               max_degree=int(deg.max()), mean_degree=float(deg.mean()), degree_gini=float(gini),
               new_components=int(new_components), new_internal_pairs=int(new_a.nnz // 2),
               cross_pairs=int(load_new.sum()), max_cross_load=int(load_new.max()) if keep else 0)
    out.update(old_block_row_l1=0., full_old_row_l1=0., cross_norm_mass=0., max_relative_load=0.)
    if keep:
        oe = old["edge_index"].numpy()
        old_a = sparse.coo_matrix((np.ones(oe.shape[1]), oe), shape=(keep, keep)).tocsr()
        od = np.asarray(old_a.sum(1)).ravel() + 1
        norm_old = sparse.diags(od ** -.5) @ (old_a + sparse.eye(keep)) @ sparse.diags(od ** -.5)
        nd = deg + 1
        norm_new = sparse.diags(nd ** -.5) @ (a + sparse.eye(n)) @ sparse.diags(nd ** -.5)
        drift = np.asarray(abs(norm_new[:keep, :keep] - norm_old).sum(1)).ravel()
        cross = np.asarray(norm_new[:keep, keep:].sum(1)).ravel()
        out.update(old_block_row_l1=float(drift.mean()), full_old_row_l1=float((drift + cross).mean()),
                   cross_norm_mass=float(cross.mean()), max_relative_load=float((load_new / od).max()))
    out["tensor_bytes"] = sum(v.numel() * v.element_size() for v in memory.values()
                              if isinstance(v, torch.Tensor))
    out["text_utf8_bytes"] = sum(len(t.encode("utf-8")) for t in memory["texts"])
    return out, deg


def audit(root, device, minilm):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = read_json(root / "manifest.json")
    records = read_records(root)
    expected = validate(records, manifest)
    combined = read_json(root / "results.json")
    validate(combined, manifest)
    assert {record_key(r): r for r in records} == {record_key(r): r for r in combined}
    repo = Path(R.__file__).resolve().parents[2]
    for key in ("data", "text"):
        assert R.sha(manifest[key]) == manifest[key + "_sha256"], key
    for name, expected_hash in manifest["sources"].items():
        assert R.sha(repo / name) == expected_hash, name
    info = dict(dataset=manifest["dataset"], fits=0, nc_fits=0, lp_fits=0, memories=0,
                max_score_reload_abs_error={"nc": 0., "lp": 0.},
                max_metric_reload_abs_error={"nc": 0., "lp": 0.},
                reload_warnings=[], topology=[], passed=False,
                audit_sha256=R.sha(__file__), original_sources_checked=len(manifest["sources"]),
                scope="all checkpoint hashes, stored metrics and fresh inference; memory fields/edges; split regeneration",
                limits=["Original manifest omitted MiniLM path/hash and protocol hash; no retrospective pre-registration claim.",
                        "No saved per-cluster membership: content equality audited, clustering is not independently reconstructed.",
                        "Construction wall time was not recorded. NC is transductive report-dev evaluation.",
                        "Orders and seeds reused from degree confirmation; not independent confirmation."],
                minilm_audit_path=str(minilm))
    D = R.U.load_unified(manifest["dataset"], manifest["data"], minilm, device,
                         n_stages=5, split_mode="base_incremental")
    text = load(manifest["text"]).float()
    assert not bool(text[~D["pool_mask"]].any())
    real_x = D["x"].to(device)
    adj = R.U.G.norm_adj(D["edge_index"], len(real_x), device)
    prompts = D["encoder"].encode(D["prompts"])
    for order_text, groups in manifest["orders"].items():
        order = int(order_text)
        assert groups == R.fresh_groups(manifest["original_class_groups"], order)
        folder = root / f"order_{order}"
        construction = read_json(folder / "construction.json")
        assert len(construction) == len(groups)
        arrival = torch.full((len(real_x),), -1, dtype=torch.long)
        previous = {}
        for stage, classes in enumerate(groups):
            c = construction[stage]
            ids = torch.where(D["pool_mask"] & torch.isin(D["y"], torch.tensor(classes)))[0]
            arrival[ids] = stage
            assert c["stage"] == stage and c["classes"] == classes
            assert c["source_ids"] == ids.tolist() and c["source_count"] == len(ids)
            keep = 0 if stage == 0 else len(previous["c0"]["x"])
            assert c["keep"] == keep
            budget = R.U.protocol_v3.growing_budget(int((arrival >= 0).sum()), rate=.05)
            assert c["budget"] == budget
            memories = {a: load(folder / c["arms"][a]["path"]) for a in R.ARMS}
            base = memories["c0"]
            block = {k: base[k][keep:] for k in R.FIELDS + ("texts",)}
            # Current rules depend on old contents, not its edges. Reconstruct
            # the common new components once, then check each arm's own history.
            rebuilt = R.topology_components(previous.get("c0"), block, keep, order, stage, device)
            degree = {}
            for arm, memory in memories.items():
                path = folder / c["arms"][arm]["path"]
                assert R.sha(path) == c["arms"][arm]["sha256"]
                assert len(memory["x"]) == budget
                assert set(memory) == set(R.FIELDS) | {"texts", "edge_index"}
                for key in R.FIELDS:
                    assert torch.equal(memory[key], base[key]), (arm, key)
                assert memory["texts"] == base["texts"]
                R.frozen(previous.get(arm), memory)
                actual_parts = R.components(memory["edge_index"], keep, budget)
                rebuilt_parts = R.components(rebuilt[arm]["edge_index"], keep, budget)
                assert all(torch.equal(a, b) for a, b in zip(actual_parts[1:], rebuilt_parts[1:])), arm
                topo, degree[arm] = graph_checks(memory, previous.get(arm), keep)
                for key, value in c["arms"][arm]["topology"].items():
                    assert abs(topo[key] - value) < 1e-6, (arm, key, topo[key], value)
                assert topo["pairs"] == c["arms"]["c0"]["topology"]["pairs"]
                topo.update(order=order, stage=stage, arm=arm, file_bytes=path.stat().st_size)
                info["topology"].append(topo)
                info["memories"] += 1
            for arm in ("random_ring", "degree_shuffle"):
                assert np.array_equal(degree[arm], degree["t_degree"]), "per-node degree mismatch"
            # Check the 2x2 factors directly in saved graphs, not just via rebuild.
            parts = {a: R.components(m["edge_index"], keep, budget) for a, m in memories.items()}
            for a, b, which in (("c0", "cross_only", 1), ("internal_only", "t_degree", 1),
                                ("c0", "internal_only", 2), ("cross_only", "t_degree", 2),
                                ("random_ring", "t_degree", 2)):
                assert torch.equal(parts[a][which], parts[b][which]), (a, b, which)
            previous = memories
            print(json.dumps(dict(event="memory_audited", order=order, stage=stage)), flush=True)
        nc_path = folder / "nc/evaluation.pt"
        ev = load(nc_path)
        assert torch.equal(ev["ids"], torch.where(D["report_dev_mask"])[0])
        assert torch.equal(ev["y"], D["y"][ev["ids"]])
        assert torch.allclose(ev["prompts"], prompts, atol=1e-6, rtol=1e-5)
        assert ev["stage_classes"] == groups
        for record in (r for r in records if r["order"] == order and r["task"] == "nc"):
            stage, arm, seed = record["stage"], record["arm"], record["seed"]
            state = check_checkpoint(folder / f"nc/{arm}_s{stage}_seed{seed}.pt", nc_path, record, construction)
            assert record["cost"]["steps"] == 2048 and record["cost"]["batch"] == 16
            stored = nc_scores(state["logits"], ev["y"].numpy(), groups, stage)
            assert metric_gap(stored, record["metrics"]) < 1e-10
            model = R.U.G.CLIPStudent(real_x.shape[1]).to(device)
            model.load_state_dict(state["state_dict"])
            model.eval()
            with torch.no_grad():
                logits = model.logits(real_x, adj, ev["prompts"].to(device)).cpu()[ev["ids"]]
            fresh = nc_scores(logits, ev["y"].numpy(), groups, stage)
            track_reload(info, record, [logits], [state["logits"]], metric_gap(stored, fresh), 1e-6)
        for split_seed in manifest["splits"]:
            lp = folder / f"lp_split_{split_seed}"
            split = load(lp / "edge_split.pt")
            regenerated = R.B.split_edges(D["edge_index"], arrival, split_seed)
            for k, value in regenerated.items():
                assert torch.equal(split[k], value) if isinstance(value, torch.Tensor) else split[k] == value
            R.L.validate_split(split)
            for stage in range(len(groups)):
                path = lp / f"evaluation_s{stage}.pt"
                ev = load(path)
                ids = torch.where((arrival >= 0) & (arrival <= stage))[0]
                assert torch.equal(ev["ids"], ids) and torch.equal(ev["arrival"], arrival[ids])
                assert torch.equal(ev["x"], R.L.features(D["x"][ids], text[ids]))
                assert torch.equal(ev["edge_index"], R.L.bidirectional(R.L.localize(split["train_pos"], ids, len(real_x))))
                for key in ("test_pos", "test_neg"):
                    assert torch.equal(ev[key], R.L.localize(split[key], ids, len(real_x)))
                for record in (r for r in records if r["task"] == "lp" and r["order"] == order
                               and r["split"] == split_seed and r["stage"] == stage):
                    state = check_checkpoint(lp / f"{record['arm']}_s{stage}_seed{record['seed']}.pt",
                                             path, record, construction)
                    assert record["cost"]["steps"] == 256 and record["cost"]["edge_batch"] == 256
                    stored = recompute_scores(ev["test_pos"], ev["test_neg"], state["pos_scores"],
                                              state["neg_scores"], ev["arrival"], stage)
                    assert stored == record["metrics"]
                    model = R.L.GAE(R.L.Encoder(ev["x"].shape[1])).to(device)
                    model.load_state_dict(state["state_dict"])
                    model.eval()
                    with torch.no_grad():
                        z = model.encode(ev["x"].to(device), ev["edge_index"].to(device))
                        ps = model.decoder(z, ev["test_pos"].to(device), sigmoid=False).cpu()
                        ns = model.decoder(z, ev["test_neg"].to(device), sigmoid=False).cpu()
                    fresh = recompute_scores(ev["test_pos"], ev["test_neg"], ps, ns, ev["arrival"], stage)
                    track_reload(info, record, [ps, ns], [state["pos_scores"], state["neg_scores"]],
                                 metric_gap(stored, fresh), 1e-4)
    assert info["fits"] == expected and info["memories"] == 60
    info["passed"] = True
    info["strict_metric_reload_passed"] = not info["reload_warnings"]
    R.save_json(root / "independent_audit.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "topology"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", required=True)
    args = parser.parse_args()
    audit(args.root, args.device, args.minilm)
