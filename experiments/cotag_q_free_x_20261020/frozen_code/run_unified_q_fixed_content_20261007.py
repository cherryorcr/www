"""Streaming fixed-Q/X/text attribution; no topology feeds back into construction."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch
import torch.nn.functional as F
import unified_q_multimodal as U
import graphtext_prompt_student as P
import external_continual_mechanisms as M
import run_unified_q_large_smoke_20261007 as L
import run_unified_q_revision_20261007 as V
import run_unified_q_inloop_cat_20261007 as I


ARMS = ("unchanged", "text_blank", "text_cctp", "text_permuted",
        "topology_hard", "topology_weighted", "topology_permuted")


def permutation(labels, seed, same_class):
    rng = torch.Generator().manual_seed(seed)
    result = torch.arange(len(labels))
    groups = torch.unique(labels).tolist() if same_class else [None]
    for cls in groups:
        ids = torch.where(labels == cls)[0] if cls is not None else torch.arange(len(labels))
        result[ids] = ids[torch.randperm(len(ids), generator=rng)]
    return result


def topology_snapshot(state):
    q, delta, block, cfg = state["q"], state["delta"], state["block"], state["config"]
    n = len(block["x"])
    mass = q.sum(0).clamp_min(1e-6)
    target = F.normalize(q.T @ state["target_message"] / mass[:, None], dim=1)
    _, weight, _ = U._soft_topology(q, delta, None, q.new_zeros(len(q), 0))
    hard, audit = U._materialize_edges(q, delta, None, q.new_zeros(len(q), 0),
        replace(cfg, topology_refine_swaps=2), x_new=block["x"].to(q.device), target_message=target)
    perm = permutation(block["labels"], cfg.seed + delta["stage"] * 1009, False)
    shuffled = perm[hard.cpu()]
    x = block["x"].to(q.device)
    operators = dict(self_loop=torch.eye(n, device=q.device),
        hard=P.normalized_adjacency(hard, n, str(q.device)).to_dense(),
        weighted=U._normalize_dense(weight),
        permuted=P.normalized_adjacency(shuffled, n, str(q.device)).to_dense())
    stats = {}
    for name, adj in operators.items():
        first, second = adj @ x, adj @ (adj @ x)
        labels = block["labels"].to(q.device)
        drift = torch.stack([(first[labels == c].mean(0)-x[labels == c].mean(0)).norm()
                             for c in torch.unique(labels)]).mean()
        stats[name] = dict(message_cosine=float(F.cosine_similarity(first, target, dim=1).mean()),
            one_hop_norm=float(first.norm(dim=1).mean()), two_hop_norm=float(second.norm(dim=1).mean()),
            class_mean_drift=float(drift))
    stats["hard_soft_message_error"] = float((operators["hard"] @ x-operators["weighted"] @ x).norm()/
        (operators["weighted"] @ x).norm().clamp_min(1e-8))
    stats["hard_edges"] = int(hard.shape[1])
    stats["hard_homophily"] = (float((block["labels"][hard.cpu()[0]] == block["labels"][hard.cpu()[1]]).float().mean())
                                 if hard.numel() else None)
    stats["construction_hardening"] = audit
    return dict(weight=weight.detach().cpu(), hard=hard.cpu(), permuted=shuffled, stats=stats)


def payload_variants(block, delta, encoder, seed, device):
    variants = {}
    blank = [""] * len(block["x"])
    variants["text_blank"] = dict(block, texts=blank,
        text_emb=torch.as_tensor(encoder.encode(blank)).cpu().float(),
        text_source_ids=[[] for _ in blank])
    source = M.propagated_features(delta, (0., 1.), device)
    attached, attachment_audit = M.attach_text(delta, source, block["x"], block["labels"], encoder,
        M.Config(seed=seed, text_sources=5, text_mode="nearest"))
    variants["text_cctp"] = dict(block, texts=attached["texts"], text_emb=attached["text_emb"],
                                text_source_ids=attachment_audit["text_source_ids"])
    perm = permutation(block["labels"], seed + 1009 * delta["stage"], True)
    variants["text_permuted"] = dict(block, texts=[block["texts"][i] for i in perm.tolist()],
        text_emb=block["text_emb"][perm], text_source_ids=[block["text_source_ids"][i] for i in perm.tolist()])
    return variants


def score_weighted(memory, adjacency, data, stage, device):
    # Only the supplied propagation operator changes, not parameters, losses,
    # steps, batches, seeds, or raw-graph inference.
    original_forward = P.GraphConv.forward
    def forward(layer, x, adj):
        return original_forward(layer, x, adj) if adj.is_sparse else adj @ layer.lin(x)
    def normalized(edge_index, n, requested_device):
        if n != len(memory["x"]):
            raise AssertionError("unexpected adjacency request")
        return adjacency.to(requested_device)
    with patch.object(P, "normalized_adjacency", normalized), patch.object(P.GraphConv, "forward", forward):
        return L.score(memory, data, stage, [150, 151, 152], device, 2048, 24)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--construction-seed", type=int, default=271006)
    p.add_argument("--budget-mode", choices=("fixed_schedule",), default="fixed_schedule")
    args = p.parse_args()
    torch.set_num_threads(2)
    ref = L.read(args.reference)
    data = L.load_large(args.dataset, args.minilm, args.text_cache, args.device, ref)
    ids, budgets = L.stage_ids(data), ref["budgets"]
    protocol = dict(dataset=args.dataset, groups=ref["groups"], budgets=budgets,
        construction_seed=args.construction_seed, order_seed=ref["order_seed"],
        student=P.model_config(), student_steps=2048, student_batch=24, student_seeds=[150,151,152],
        report_dev_mod=ref["report_dev_mod"], report_mask_hash=V.hash_tensor(data["report_dev_mask"]),
        pool_mask_hash=V.hash_tensor(data["pool_mask"]), stage_id_hashes=[V.hash_tensor(i) for i in ids],
        budget_mode=args.budget_mode, arms=list(ARMS), evaluation_split="report-dev",
        experiment="fixed content/operator attribution, NOT reoptimized candidate methods",
        raw_access="each stage delta once; synthetic prefixes frozen; no boundary map",
        source_hashes={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (U,P,M,L,I)},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = args.output / "results.json"
    result = json.loads(path.read_text()) if path.exists() else dict(protocol=protocol, rows=[], status="running")
    if result["protocol"] != protocol:
        raise ValueError("protocol differs")
    V.write_json(args.output / "protocol.json", protocol)
    parents = {arm: None for arm in ARMS}
    weights = []
    cfg = I.config("uq_dm_edgefree", args.construction_seed)
    for stage in range(len(ids)):
        delta = L.delta_view(data, ids[stage], stage)
        parent = parents["unchanged"]
        old_n = 0 if parent is None else len(parent["x"])
        labels = M.row_labels(M.ratio_allocation(delta["y"], budgets[stage]-old_n))
        captured = {}
        def capture(state):
            captured.update(topology_snapshot(state))
        started = time.perf_counter()
        block, audit = U.optimize_stage(delta, parent, labels, data["encoder"], data["prompt_emb"],
            cfg, args.device, use_boundary=False, use_topology=False, diagnostic_callback=capture)
        construction = time.perf_counter()-started
        variants = payload_variants(block, delta, data["encoder"], args.construction_seed, args.device)
        variants.update(unchanged=block,
            topology_hard=dict(block, edge_index=captured["hard"]+old_n),
            topology_permuted=dict(block, edge_index=captured["permuted"]+old_n),
            topology_weighted=block)
        weights.append(captured["weight"])
        for arm in ARMS:
            memory = U.append_memory(parents[arm], variants[arm])
            checks = V.validate_memory(memory, parents[arm], budgets[stage])
            for key in ("x", "labels", "mass"):
                if not torch.equal(variants[arm][key], block[key]):
                    raise AssertionError(f"fixed content changed: {key}")
            if arm.startswith("topology") and (
                not torch.equal(variants[arm]["text_emb"], block["text_emb"])
                or variants[arm]["texts"] != block["texts"]):
                raise AssertionError("topology test changed text")
            parents[arm] = memory
            done = [r for r in result["rows"] if r["arm"] == arm and r["stage"] == stage]
            if done:
                if V.memory_hash(memory) != done[0]["memory_hash"]:
                    raise AssertionError("replay differs")
                continue
            started = time.perf_counter()
            if arm == "topology_weighted":
                adjacency = U._normalize_dense(torch.block_diag(*weights))
                score = score_weighted(memory, adjacency, data, stage, args.device)
                operator_hash = V.hash_tensor(adjacency)
            else:
                score = L.score(memory, data, stage, [150,151,152], args.device, 2048, 24)
                operator_hash = V.hash_tensor(memory["edge_index"])
            V.sync(args.device)
            result["rows"].append(dict(arm=arm, stage=stage, memory_rows=budgets[stage],
                score=score, checks=checks, memory_hash=V.memory_hash(memory), operator_hash=operator_hash,
                student_seconds=time.perf_counter()-started, construction_seconds=construction if arm == "unchanged" else 0,
                diagnostics=captured["stats"] if arm == "unchanged" else {},
                content_hash=V.hash_tensor(memory["x"]), text_hash=V.hash_tensor(memory["text_emb"])))
            V.write_json(path, result)
            print(json.dumps(dict(event="evaluated", arm=arm, stage=stage, f1=score["final_f1"], accuracy=score["accuracy"])), flush=True)
    temporary = args.output / "diagnostic_memory.tmp"
    torch.save(dict(memories=parents, weighted_blocks=weights), temporary)
    temporary.replace(args.output / "diagnostic_memory.pt")
    result["status"] = "complete"
    V.write_json(path, result)


if __name__ == "__main__":
    main()
