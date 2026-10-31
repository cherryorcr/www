"""Reload every checkpoint and verify shared-memory/provenance contracts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

import run_degree_cross_task_confirmation as R
from audit_link_prediction_transfer import recompute_scores
from summarize_degree_confirmation import validate


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def nc_scores(logits, target, groups, stage):
    seen = [c for g in groups[:stage + 1] for c in g]
    active = np.isin(target, seen)
    y = target[active]
    old = [c for g in groups[:stage] for c in g]
    result = {}
    for scope in ("all", "seen"):
        allowed = list(range(logits.shape[1])) if scope == "all" else sorted(seen)
        pred = np.asarray(allowed)[logits[:, allowed].numpy().argmax(1)][active]
        scores = f1_score(y, pred, labels=seen, average=None, zero_division=0) * 100
        by_class = dict(zip(seen, scores.tolist()))
        result[scope] = {"cumulative_f1": float(scores.mean()),
                         "accuracy": float(np.mean(pred == y) * 100),
                         "old_f1": float(np.mean([by_class[c] for c in old])) if old else None,
                         "new_f1": float(np.mean([by_class[c] for c in groups[stage]]))}
    return result


def metric_gap(first, second):
    differences = []
    for scope, metrics in first.items():
        for metric, value in metrics.items():
            if value is not None:
                differences.append(abs(value - second[scope][metric]))
            else:
                assert second[scope][metric] is None
    return max(differences, default=0.)


def audit(root, device):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((root / "manifest.json").read_text())
    records = json.loads((root / "results.json").read_text())
    validate(records)
    for key in ("data", "text"):
        assert R.sha(manifest[key]) == manifest[key + "_sha256"]
    assert R.source_hashes() == manifest["sources"]
    assert R.sha(Path(__file__).resolve().parents[2] / "docs/current" / R.PROTOCOL) == manifest["protocol_sha256"]
    D = R.U.load_unified(manifest["dataset"], manifest["data"], manifest["minilm"], device,
                         n_stages=5, split_mode="base_incremental")
    text = load(manifest["text"]).float()
    assert not bool(text[~D["pool_mask"]].any())
    real_x = D["x"].to(device)
    adj = R.U.G.norm_adj(D["edge_index"], len(real_x), device)
    prompts = D["encoder"].encode(D["prompts"])
    info = {"fits": 0, "nc_fits": 0, "lp_fits": 0, "memories": 0,
            "max_score_reload_abs_error": {"nc": 0., "lp": 0.},
            "max_metric_reload_abs_error": {"nc": 0., "lp": 0.},
            "reload_warnings": [], "topology": [],
            "scope": "data/code/cache hashes, pool sources, shared fields, exact topology reconstruction, all checkpoint reloads",
            "limits": "No reconstruction of per-row clustering membership; original feature preprocessing reused. NC transductive full graph."}
    for order in R.ORDERS:
        folder = root / f"order_{order}"
        construction = json.loads((folder / "construction.json").read_text())
        groups = R.fresh_groups(manifest["original_class_groups"], order)
        assert groups == manifest["orders"][str(order)]
        arrival = torch.full((len(real_x),), -1, dtype=torch.long)
        previous = {}
        for stage, classes in enumerate(groups):
            ids = torch.where(D["pool_mask"] & torch.isin(D["y"], torch.tensor(classes)))[0]
            arrival[ids] = stage
            c = construction[stage]
            assert c["source_ids"] == ids.tolist()
            assert c["classes"] == classes and c["source_count"] == len(ids)
            budget = int(.05 * int((arrival >= 0).sum()))
            assert c["budget"] == budget
            assert sum(c["allocation"].values()) == budget - c["keep"]
            base = load(folder / c["arms"]["c0"]["path"])
            for arm in R.ARMS:
                path = folder / c["arms"][arm]["path"]
                assert R.sha(path) == c["arms"][arm]["sha256"]
                memory = load(path)
                assert len(memory["x"]) == budget
                R.frozen(previous.get(arm), memory)
                for key in R.FIELDS:
                    assert torch.equal(memory[key], base[key])
                assert memory["texts"] == base["texts"]
                assert not any(k in memory for k in ("source_groups", "original_ids", "graph_signature"))
                n = c["keep"]
                block = {key: memory[key][n:] for key in R.FIELDS + ("texts",)}
                rebuilt = (R.U.SCA.append_memory(previous.get(arm), block, n, device=device) if arm == "c0"
                           else R.B.append(previous.get(arm), block, R.WEIGHTS[arm], "degree_balanced"))
                assert torch.equal(rebuilt["edge_index"], memory["edge_index"])
                topo = R.topology(memory)
                assert topo == c["arms"][arm]["topology"]
                assert R.L.pairs(memory["edge_index"], budget).shape == R.L.pairs(base["edge_index"], budget).shape
                info["topology"].append(dict(order=order, stage=stage, arm=arm, **topo))
                info["memories"] += 1
                previous[arm] = memory
        nc_eval_path = folder / "nc/evaluation.pt"
        ev = load(nc_eval_path)
        assert torch.equal(ev["ids"], torch.where(D["report_dev_mask"])[0])
        assert torch.equal(ev["y"], D["y"][ev["ids"]])
        assert torch.allclose(ev["prompts"], prompts, atol=1e-6, rtol=1e-5)
        assert ev["stage_classes"] == groups
        for record in [r for r in records if r["order"] == order and r["task"] == "nc"]:
            stage, arm, seed = record["stage"], record["arm"], record["seed"]
            path = folder / f"nc/{arm}_s{stage}_seed{seed}.pt"
            state = check_checkpoint(path, nc_eval_path, record, construction)
            stored = nc_scores(state["logits"], ev["y"].numpy(), groups, stage)
            assert metric_gap(stored, record["metrics"]) < 1e-10
            model = R.U.G.CLIPStudent(real_x.shape[1]).to(device)
            model.load_state_dict(state["state_dict"])
            model.eval()
            with torch.no_grad():
                logits = model.logits(real_x, adj, ev["prompts"].to(device)).cpu()[ev["ids"]]
            track_reload(info, record, [logits], [state["logits"]],
                         metric_gap(stored, nc_scores(logits, ev["y"].numpy(), groups, stage)), 1e-6)
        for split_seed in R.SPLITS:
            lp_folder = folder / f"lp_split_{split_seed}"
            split = load(lp_folder / "edge_split.pt")
            regenerated = R.B.split_edges(D["edge_index"], arrival, split_seed)
            for key, value in regenerated.items():
                assert torch.equal(split[key], value) if isinstance(value, torch.Tensor) else split[key] == value
            R.L.validate_split(split)
            for stage in range(5):
                path = lp_folder / f"evaluation_s{stage}.pt"
                ev = load(path)
                ids = torch.where((arrival >= 0) & (arrival <= stage))[0]
                assert torch.equal(ev["ids"], ids)
                assert torch.equal(ev["arrival"], arrival[ids])
                assert torch.equal(ev["x"], R.L.features(D["x"][ids], text[ids]))
                assert torch.equal(ev["edge_index"], R.L.bidirectional(R.L.localize(split["train_pos"], ids, len(real_x))))
                for k in ("test_pos", "test_neg"):
                    assert torch.equal(ev[k], R.L.localize(split[k], ids, len(real_x)))
                for record in [r for r in records if r["task"] == "lp" and r["order"] == order
                               and r["split"] == split_seed and r["stage"] == stage]:
                    ckpt = lp_folder / f"{record['arm']}_s{stage}_seed{record['seed']}.pt"
                    state = check_checkpoint(ckpt, path, record, construction)
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
    assert info["nc_fits"] == 180 and info["lp_fits"] == 360 and info["memories"] == 30
    info["passed"] = True
    info["strict_metric_reload_passed"] = not info["reload_warnings"]
    R.save(root / "audit.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "topology"}), flush=True)


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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    audit(args.root, args.device)
