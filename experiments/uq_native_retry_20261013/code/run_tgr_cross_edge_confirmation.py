"""Sealed third-order confirmation of the frozen balanced cross-edge operator.

The comparison base is Semantic-Recursive, despite the legacy ``c0`` key in
the topology runner. Stages use shared SR parents, not recursive transport.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

import run_content_gap_attribution as CAG
import run_tgc_r4_relational as D
import run_tgc_selector_confirmation as B
import run_tgc_semantic_recursive_confirmation_20260926 as SR
import run_tgr_cross_edge_eval as E


HERE = Path(__file__).resolve().parent
FIELDS = ("x", "text_emb", "mass", "depth", "stage")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def edge_set(memory):
    return set(map(tuple, memory["edge_index"].T.tolist()))


def audit_transport(base, candidate, parent):
    old_n, total = len(parent["x"]), len(base["x"])
    assert len(candidate["x"]) == total
    for key in FIELDS:
        assert torch.equal(base[key], candidate[key]), key
        assert torch.equal(base[key][:old_n], parent[key]), key
    assert base["texts"] == candidate["texts"]
    assert base["texts"][:old_n] == parent["texts"]
    before, after = edge_set(base), edge_set(candidate)
    assert len(after) == candidate["edge_index"].shape[1] == len(before)
    assert all(a != b and 0 <= a < total and 0 <= b < total for a, b in after)
    assert all((b, a) in after for a, b in after)
    for predicate in (lambda a, b: a < old_n and b < old_n,
                      lambda a, b: a >= old_n and b >= old_n):
        assert {e for e in before if predicate(*e)} == {e for e in after if predicate(*e)}
    assert {e for e in after if max(e) < old_n} == edge_set(parent)
    # Independently check BOTH endpoint degrees; do not trust the legacy
    # old_capacity_overflow field, which counts consumed capacity.
    def cross_degrees(edges):
        return Counter(v for a, b in edges if a < old_n <= b for v in (a, b))
    assert cross_degrees(before) == cross_degrees(after)
    return {"status": "passed", "rows": total, "old_rows": old_n,
            "edge_entries": len(after), "changed_cross_pairs": len(after - before) // 2,
            "content_unchanged": True, "both_endpoint_degrees_preserved": True,
            "old_old_preserved": True, "new_new_preserved": True}


def prepare(args):
    out = args.output / args.dataset
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    prereg = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "order": args.order, "dataset": args.dataset,
        "class_groups": CAG.stage_groups(args.dataset, 3, args.order),
        "arms": ["semantic_recursive", "message_balanced"],
        "legacy_c0_key_means": "semantic_recursive, not original C0",
        "rate": 0.05, "report_seeds": list(range(150, 156)),
        "cross_degree": 1, "message_direct_mix": [0.75, 0.25], "top_r": 8,
        "parent": "shared frozen semantic-recursive parent at each stage",
        "student": "unchanged canonical two-layer GCN + bidirectional InfoNCE",
        "selection_by_report": False, "old_raw_construction_dependency": False,
        "primary": "stage2 pooled all-prompt report-dev Macro-F1 difference",
        "secondary": "true stage0/1/2 average, accuracy, current and old group F1",
        "source_hashes": {p.name: sha(p) for p in sorted(HERE.glob("*.py"))},
    }
    save(out / "preregister.json", prereg)
    data = D.R.R.load_data(args.dataset, args.minilm, args.device,
                          args.cache_root / f"{args.dataset}_r005/text_emb_pool_only.pt")
    ids, budgets = D.set_stream(data, args.order)
    source = out / "source" / args.dataset / args.order
    source.mkdir(parents=True)
    save(source / "manifest.json", {
        "dataset": args.dataset, "order": args.order,
        "stage_classes": data["stage_classes"], "budgets": budgets,
        "stage_source_ids": [v.tolist() for v in ids],
        "pool_hash": D.R.C.tensor_hash({"pool": data["pool_mask"]}),
        "report_hash": D.R.C.tensor_hash({"report": data["report_dev_mask"]}),
    })
    for stage in range(3):
        memory, audit = B.build_c0(data, ids[stage], budgets, stage, source, args.device)
        if args.reference_root is not None:
            reference = load(args.reference_root / args.dataset / args.order /
                             f"c0_s{stage}_memory.pt")
            for key in FIELDS + ("edge_index",):
                assert torch.equal(memory[key], reference[key]), (stage, key)
            assert memory["texts"] == reference["texts"]
        print(json.dumps({"event": "c0_ready", "stage": stage,
                          "rows": len(memory["x"]), "seconds": audit["seconds"]}), flush=True)
    del data
    if args.reference_root is not None:
        save(out / "source_replay_audit.json", {"status": "passed", "stages": 3,
                                                "exact_content_and_topology": True})
        return
    for stage in (1, 2):
        started = time.perf_counter()
        SR.run(SimpleNamespace(
            dataset=args.dataset, order=args.order, stage=stage,
            source_root=out / "source", parent_root=out / "semantic",
            output_root=out / "semantic", cache_root=args.cache_root,
            minilm=args.minilm, device=args.device, parent_arm="semantic_perm",
            random_seed=20260926))
        semantic_seconds = time.perf_counter() - started
        cell = f"{args.dataset}_{args.order}_s{stage}"
        started = time.perf_counter()
        subprocess.run([
            sys.executable, str(HERE / "run_tgr_cross_edge_transport.py"),
            "--dataset", args.dataset, "--order", args.order, "--stage", str(stage),
            "--data", D.R.U.DATA_CFG[args.dataset]["path"],
            "--text-cache", str(args.cache_root / f"{args.dataset}_r005/text_emb_pool_only.pt"),
            "--c0-root", str(out / "source"), "--semantic-root", str(out / "semantic"),
            "--output", str(out / "topology" / cell), "--device", "cpu",
            "--modes", "c0", "message_balanced", "--cross-degree", "1",
            "--random-seed", "9821"], check=True)
        transport_seconds = time.perf_counter() - started
        parent_path = (source / "c0_s0_memory.pt" if stage == 1 else
                       out / "semantic" / f"{args.dataset}_{args.order}_s1/semantic_perm_memory.pt")
        base = load(out / "topology" / cell / "c0/tgr_message_memory.pt")
        candidate = load(out / "topology" / cell / "message_balanced/tgr_message_memory.pt")
        audit = audit_transport(base, candidate, load(parent_path))
        support = json.loads((out / "topology" / cell / "message_balanced/support_audit.json").read_text())
        loads = Counter(v for r in support["selection"]["rows"] for v in r["selected_old"])
        cap = support["selection"]["capacity_cap"]
        assert max(loads.values()) <= cap
        audit.update(semantic_seconds=semantic_seconds, transport_seconds=transport_seconds,
                     support_cap=cap, support_max_load=max(loads.values()),
                     support_top5_fraction=sum(sorted(loads.values(), reverse=True)[:5]) / sum(loads.values()),
                     parent_file=str(parent_path))
        save(out / f"s{stage}_construction_audit.json", audit)
    sealed = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*"))
              if p.is_file() and p.suffix in (".json", ".pt")}
    save(out / "construction_seal.json", {"files": sealed,
         "sealed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed"})


def verify_seal(out):
    seal = json.loads((out / "construction_seal.json").read_text())
    for relative, digest in seal["files"].items():
        assert sha(out / relative) == digest, relative


def evaluation_data(args):
    data = E.U.load_unified(args.dataset, D.R.U.DATA_CFG[args.dataset]["path"],
                           args.minilm, args.device, n_stages=3, split_mode="canonical")
    data["text_emb"] = load(args.cache_root / f"{args.dataset}_r005/text_emb_pool_only.pt")
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = E.U.G.norm_adj(data["edge_index"], len(data["x"]), args.device)
    data["x_device"] = data["x"].float().to(args.device)
    CAG.set_stream(data, CAG.stage_groups(args.dataset, 3, args.order), 0.05)
    return data


def memory_path(args, stage, arm):
    out = args.output / args.dataset
    if stage == 0:
        return out / "source" / args.dataset / args.order / "c0_s0_memory.pt"
    legacy = "c0" if arm == "semantic_recursive" else arm
    return out / "topology" / f"{args.dataset}_{args.order}_s{stage}" / legacy / "tgr_message_memory.pt"


def evaluate(args):
    out = args.output / args.dataset
    verify_seal(out)
    data = evaluation_data(args)
    for stage in range(3):
        target = out / f"eval_s{stage}.json"
        if target.exists():
            continue
        result = {"dataset": args.dataset, "order": args.order, "stage": stage, "arms": {}}
        for arm in ("semantic_recursive", "message_balanced"):
            if stage == 0 and arm == "message_balanced":
                result["arms"][arm] = result["arms"]["semantic_recursive"]
                continue
            memory = load(memory_path(args, stage, arm))
            result["arms"][arm] = E.fit(memory, data, stage, list(range(150, 156)), args.device,
                                                out / "checkpoints" / f"s{stage}_{arm}")
        save(target, result)
    print(json.dumps({"event": "evaluation_complete", "dataset": args.dataset}), flush=True)


def audit(args):
    out = args.output / args.dataset
    verify_seal(out)
    data = evaluation_data(args)
    max_error, count = 0.0, 0
    for stage in range(3):
        result = json.loads((out / f"eval_s{stage}.json").read_text())
        for arm in ("semantic_recursive", "message_balanced"):
            if stage == 0 and arm == "message_balanced":
                continue
            memory = load(memory_path(args, stage, arm))
            for row in result["arms"][arm]["rows"]:
                checkpoint = load(out / "checkpoints" / f"s{stage}_{arm}" / f"seed{row['seed']}.pt")
                model = E.U.G.CLIPStudent(memory["x"].shape[1]).to(args.device)
                model.load_state_dict(checkpoint["state"])
                model.eval()
                score = E.U.G.eval_stagewise(model, data["x_device"], data["real_adj"],
                                             data["prompt_emb"], data["y"], data["report_dev_mask"], stage)
                for key in ("cumulative_macro_f1", "cumulative_accuracy", "current_f1", "old_mean_f1"):
                    if key in score:
                        max_error = max(max_error, abs(score[key] - row[key]))
                for actual, recorded in zip(score["per_stage"], row["per_stage"]):
                    for key in ("macro_f1", "accuracy", "n"):
                        max_error = max(max_error, abs(actual[key] - recorded[key]))
                assert checkpoint["steps"] == E.U.protocol_v3.student_steps(len(memory["x"]))
                count += 1
                del model
    assert count == 30 and max_error < 1e-7
    save(out / "reload_audit.json", {"status": "passed", "unique_checkpoints": count,
                                      "maximum_metric_error_pp": max_error})
    print(json.dumps({"event": "audit_complete", "dataset": args.dataset,
                      "unique_checkpoints": count, "maximum_error": max_error}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "evaluate", "audit"), required=True)
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--order", default="order19613")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--cache-root", type=Path,
                        default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    {"prepare": prepare, "evaluate": evaluate, "audit": audit}[args.phase](args)


if __name__ == "__main__":
    main()
