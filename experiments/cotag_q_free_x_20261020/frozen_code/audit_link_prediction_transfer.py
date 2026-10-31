"""Independently reload LP students, splits, and graph-signature provenance."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import average_precision_score, roc_auc_score

import run_link_prediction_transfer as L


def recompute_scores(positive, negative, pscore, nscore, arrival, stage):
    result = {}
    for name in ("all", "old", "new", "cross", "within_new"):
        samples = []
        for edges, scores in ((positive, pscore), (negative, nscore)):
            a, b = arrival[edges]
            mask = {"all": torch.ones(len(a), dtype=torch.bool),
                    "old": (a < stage) & (b < stage),
                    "new": (a == stage) | (b == stage),
                    "cross": (a == stage) ^ (b == stage),
                    "within_new": (a == stage) & (b == stage)}[name]
            samples.append(scores[mask].numpy())
        p, n = samples
        item = {"positives": len(p), "negatives": len(n), "auc": None, "ap": None}
        if len(p) and len(n):
            labels = np.concatenate((np.ones(len(p)), np.zeros(len(n))))
            scores = np.concatenate((p, n))
            item.update(auc=float(roc_auc_score(labels, scores)), ap=float(average_precision_score(labels, scores)))
        result[name] = item
    return result


def topology(memory):
    n = len(memory["x"])
    edges = L.pairs(memory["edge_index"], n)
    bi = L.bidirectional(edges)
    degree = torch.bincount(bi[0], minlength=n).float()
    matrix = coo_matrix((np.ones(bi.shape[1]), bi.numpy()), shape=(n, n)).tocsr()
    count, labels = connected_components(matrix, directed=False)
    return {"rows": n, "pairs": edges.shape[1], "isolated": int((degree == 0).sum()),
            "max_degree": int(degree.max()), "components": int(count),
            "largest_component_fraction": float(np.bincount(labels).max() / n)}


def audit(root, device):
    import run_safe_joint_kernel_experiments as E
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((root / "manifest.json").read_text())
    records = json.loads((root / "results.json").read_text())
    expected = len(manifest["split_seeds"]) * len(manifest["student_seeds"]) * 3 * len(L.ARMS)
    assert len(records) == expected
    keys = {(r["split"], r["stage"], r["arm"], r["seed"]) for r in records}
    assert len(keys) == expected
    summary = {"dataset": manifest["dataset"], "fits": 0, "memories": 0,
               "max_score_reload_abs_error": 0., "max_metric_reload_abs_error": 0.,
               "errors": [], "topology": [], "validation_used": False,
               "rank_metric_warnings": [], "float64_records": [],
               "original_rank_metric_tolerance": 1e-4}
    for split_seed in manifest["split_seeds"]:
        folder = root / f"split_{split_seed}"
        split = torch.load(folder / "edge_split.pt", weights_only=False)
        L.validate_split(split)
        construction = json.loads((folder / "construction.json").read_text())
        dataset = manifest["dataset"]
        raw = torch.load(f"/home/pc/research/GraphCLIP/processed_data/{dataset}.pt", weights_only=False)
        text = torch.load(L.ROOT / f"results/p5_rate_curve_small_20260919/{dataset}_r005/text_emb_pool_only.pt", weights_only=False)
        pool = L.P.split_masks_v4(torch.arange(len(raw.x)))[-1]
        regenerated = L.split_edges(raw.edge_index, split["arrival"], split_seed)
        for key in ("full_pos", "train_pos", "val_pos", "test_pos", "test_neg", "val_neg", "train_neg"):
            assert torch.equal(regenerated[key], split[key])
        old = {}
        for stage in range(3):
            ev = torch.load(folder / f"evaluation_s{stage}.pt", weights_only=False)
            ids = torch.where((split["arrival"] >= 0) & (split["arrival"] <= stage))[0]
            assert torch.equal(ev["ids"], ids)
            assert torch.equal(ev["x"], L.features(raw.x[ids], text[ids]))
            assert torch.equal(ev["edge_index"], L.bidirectional(L.localize(split["train_pos"], ids, len(raw.x))))
            for key in ("test_pos", "test_neg"):
                assert torch.equal(ev[key], L.localize(split[key], ids, len(raw.x)))
            memories = {arm: torch.load(folder / f"{arm}_s{stage}_memory.pt", weights_only=False) for arm in L.ARMS[:-1]}
            archive = L.ROOT / f"results/safe_joint_kernel_v2_{dataset}_20260920/s3_canonical_r05"
            source = json.loads((archive / f"joint_kernel_s{stage}_sources.json").read_text())["source_groups"]
            delta = torch.where(split["arrival"] == stage)[0]
            joint = torch.cat((torch.nn.functional.normalize(raw.x[delta].float(), dim=1), text[delta]), dim=1)
            source_graph = E.J.source_local_graph_summary(
                {"x": raw.x, "edge_index": L.bidirectional(split["train_pos"]), "pool_mask": pool}, delta, joint)
            position = torch.full((len(raw.x),), -1, dtype=torch.long)
            position[delta] = torch.arange(len(delta))
            expected_signature = torch.nn.functional.normalize(torch.stack([source_graph[position[g]].mean(0) for g in source]), dim=1)
            keep = 0 if stage == 0 else len(old["c0"]["x"])
            for arm, memory in memories.items():
                assert E.F.J_hash(memory) == construction["stage_records"][stage]["arms"][arm]["content_sha256"]
                assert not any(k in memory for k in ("source_groups", "audit_source_groups", "original_ids"))
                if stage:
                    E.C.assert_frozen(old[arm], memory)
                if arm != "c0":
                    assert torch.equal(memory["graph_signature"][keep:], expected_signature)
                    for key in ("x", "text_emb", "mass", "depth", "stage"):
                        assert torch.equal(memory[key], memories["c0"][key])
                    block = E.copy_block(memories["c0"], keep)
                    block["graph_signature"] = expected_signature
                    block = L.S.compress_signature(block, "fp32")
                    rebuilt, _ = L.S.append_variant(old.get(arm), block, keep, arm)
                    assert E.F.J_hash(rebuilt) == E.F.J_hash(memory)
                summary["memories"] += 1
                item = {"split": split_seed, "stage": stage, "arm": arm} | topology(memory)
                a = L.pair_keys(L.pairs(memory["edge_index"], len(memory["x"])), len(memory["x"]))
                degree = memories["degree_balanced"]
                b = L.pair_keys(L.pairs(degree["edge_index"], len(degree["x"])), len(degree["x"]))
                item["jaccard_to_degree_balanced"] = len(a & b) / len(a | b)
                summary["topology"].append(item)
            old = memories
            for record in [r for r in records if r["split"] == split_seed and r["stage"] == stage]:
                checkpoint = folder / f"{record['arm']}_s{stage}_seed{record['seed']}.pt"
                assert L.sha(checkpoint) == record["checkpoint_sha256"]
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                assert state["evaluation_sha256"] == L.sha(folder / f"evaluation_s{stage}.pt")
                stored = recompute_scores(ev["test_pos"], ev["test_neg"], state["pos_scores"], state["neg_scores"], ev["arrival"], stage)
                assert stored == record["metrics"], "saved metrics do not reproduce"
                model = L.GAE(L.Encoder(ev["x"].shape[1])).to(device)
                model.load_state_dict(state["state_dict"])
                model.eval()
                with torch.no_grad():
                    z = model.encode(ev["x"].to(device), ev["edge_index"].to(device))
                    ps = model.decoder(z, ev["test_pos"].to(device), sigmoid=False).cpu()
                    ns = model.decoder(z, ev["test_neg"].to(device), sigmoid=False).cpu()
                error = max(float((ps - state["pos_scores"]).abs().max()), float((ns - state["neg_scores"]).abs().max()))
                summary["max_score_reload_abs_error"] = max(summary["max_score_reload_abs_error"], error)
                assert torch.allclose(ps, state["pos_scores"], atol=1e-4, rtol=1e-5)
                assert torch.allclose(ns, state["neg_scores"], atol=1e-4, rtol=1e-5)
                fresh = recompute_scores(ev["test_pos"], ev["test_neg"], ps, ns, ev["arrival"], stage)
                for scope in stored:
                    for metric in ("auc", "ap"):
                        if stored[scope][metric] is not None:
                            gap = abs(stored[scope][metric] - fresh[scope][metric])
                            summary["max_metric_reload_abs_error"] = max(summary["max_metric_reload_abs_error"], gap)
                            if gap > 1e-4:
                                summary["rank_metric_warnings"].append({
                                    "split": split_seed, "stage": stage, "arm": record["arm"],
                                    "seed": record["seed"], "scope": scope, "metric": metric,
                                    "absolute_gap": gap, "tensor_allclose_passed": True})
                # Independent deterministic high-precision inference. This does
                # not alter training, checkpoints, saved scores, or primary metrics.
                high = L.GAE(L.Encoder(ev["x"].shape[1])).double().cpu()
                high.load_state_dict(state["state_dict"])
                high.eval()
                with torch.no_grad():
                    zh = high.encode(ev["x"].double(), ev["edge_index"])
                    ph = high.decoder(zh, ev["test_pos"], sigmoid=False)
                    nh = high.decoder(zh, ev["test_neg"], sigmoid=False)
                summary["float64_records"].append({
                    "split": split_seed, "stage": stage, "arm": record["arm"], "seed": record["seed"],
                    "metrics": recompute_scores(ev["test_pos"], ev["test_neg"], ph, nh, ev["arrival"], stage)})
                summary["fits"] += 1
    summary["passed"] = summary["fits"] == expected
    summary["pass_scope"] = "artifact hashes, leakage controls, frozen memory, exact saved-score metrics, reload tensor allclose"
    summary["strict_rank_metric_check_passed"] = not summary["rank_metric_warnings"]
    L.save_json(root / "audit.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("topology", "float64_records")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    audit(Path(args.root), args.device)
