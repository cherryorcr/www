"""Frozen v5 factorial experiment, with prediction artifacts and complete costs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

import retention_constrained_append as V
import run_p3b_topology_diagnostic as B
import run_small_order_cost_confirmation as O
import small_comparison_metrics as M

C, U, G, P = V.C, V.C.U, V.G, V.C.P
PROTOCOL = "V5_RETENTION_FROZEN_PROTOCOL_ZH_2026-09-19.md"


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def canonical_groups(groups):
    return tuple(tuple(sorted(map(int, group))) for group in groups)


def historical_orders(dataset):
    known = {canonical_groups(v) for v in O.stage_orders(dataset).values()}
    records = []
    # Only the class-group field is used, never any measured score.
    paths = set(C.RESULTS.rglob("manifest.json")) | set(C.RESULTS.rglob("results.json"))
    for path in sorted(paths):
        if dataset not in str(path).lower() or "v5_retention" in str(path):
            continue
        try:
            obj = json.loads(path.read_text())
            groups = obj.get("stage_classes")
            if groups and all(isinstance(g, list) and all(isinstance(c, int) for c in g) for g in groups):
                groups = canonical_groups(groups)
                if len(groups) == 3:
                    known.add(groups)
                    records.append({"path": str(path), "groups": groups})
        except (OSError, ValueError, TypeError):
            continue
    return known, records


def matched_order(counts, reference, excluded):
    reference = canonical_groups(reference)
    sizes = list(map(len, reference))
    targets = [sum(counts[c] for c in group) for group in reference]
    choices = {}
    for order in itertools.permutations(range(sum(sizes))):
        groups, offset = [], 0
        for size in sizes:
            groups.append(tuple(sorted(order[offset:offset + size])))
            offset += size
        groups = tuple(groups)
        if groups in excluded or any(a == b for a, b in zip(groups, reference)):
            continue
        actual = [sum(counts[c] for c in group) for group in groups]
        error = sum(((a - b) / max(1, b)) ** 2 for a, b in zip(actual, targets))
        choices[groups] = (error, actual)
    if not choices:
        raise ValueError("no unused source-size-matched order exists")
    best = min(choices, key=lambda k: (choices[k][0], k))
    return [list(g) for g in best], {"candidate_count": len(choices),
        "objective": choices[best][0], "reference_pool_counts": targets,
        "actual_pool_counts": choices[best][1], "pool_class_counts": counts,
        "selection_uses": "pool counts only; no feature, selector, report or test scores"}


def legacy_from_predictions(target, prediction, groups, stage):
    target, prediction = np.asarray(target), np.asarray(prediction)
    per_stage, all_y, all_p = [], [], []
    for group in groups[:stage + 1]:
        active = np.isin(target, group)
        yy, pp = target[active], prediction[active]
        per_stage.append({"macro_f1": float(100 * f1_score(yy, pp, average="macro", labels=np.unique(yy), zero_division=0)),
                          "accuracy": float(100 * accuracy_score(yy, pp)), "n": int(len(yy))})
        all_y.extend(yy.tolist())
        all_p.extend(pp.tolist())
    result = {"per_stage": per_stage,
              "cumulative_macro_f1": float(100 * f1_score(all_y, all_p, average="macro", labels=np.unique(all_y), zero_division=0)),
              "cumulative_accuracy": float(100 * accuracy_score(all_y, all_p)),
              "current_f1": per_stage[-1]["macro_f1"]}
    if stage:
        result["old_mean_f1"] = float(np.mean([r["macro_f1"] for r in per_stage[:-1]]))
    return result


def prepare(dataset, minilm, device, out, large=False):
    sync(device)
    start = time.perf_counter()
    data = U.load_unified(dataset, U.DATA_CFG[dataset]["path"], minilm, device,
                          n_stages=5 if large else 3,
                          split_mode="base_incremental" if large else "canonical")
    sync(device)
    load_seconds = time.perf_counter() - start
    cache = out / "source_text_embeddings.pt"
    prep_path = out / "preprocessing.json"
    if cache.exists() != prep_path.exists():
        raise ValueError("incomplete source preprocessing cache")
    if cache.exists():
        prep = json.loads(prep_path.read_text())
        if C.sha(cache) != prep["source_embedding_file_sha256"]:
            raise AssertionError("source encoding changed")
        data["text_emb"] = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        data["encoder"].encode(data["prompts"])
        sync(device)
        start = time.perf_counter()
        ids = torch.where(data["pool_mask"])[0]
        emb = data["encoder"].encode([data["texts"][int(i)] for i in ids])
        data["text_emb"] = torch.zeros((len(data["x"]), emb.shape[1]), dtype=emb.dtype)
        data["text_emb"][ids] = emb
        sync(device)
        seconds = time.perf_counter() - start
        if not large:
            frozen = torch.load(C.artifact_dir(dataset, "v4") / "text_emb_pool_only.pt", map_location="cpu", weights_only=False)
            if not torch.equal(frozen, data["text_emb"]):
                raise AssertionError("fresh pool embedding differs from canonical cache")
        torch.save(data["text_emb"], cache)
        prep = {"load_seconds": load_seconds, "source_encoding_seconds": seconds,
                "pool_rows": len(ids), "source_embedding_file_sha256": C.sha(cache),
                "source_data_sha256": C.sha(U.DATA_CFG[dataset]["path"]),
                "canonical_embedding_bitwise_equal": True if not large else None,
                "scope": "shared fresh frozen encoding of legal pool; excludes all holdout texts"}
        C.save_json(prep_path, prep)
    if data["text_emb"][~data["pool_mask"]].abs().max() != 0:
        raise AssertionError("holdout encoding access")
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = G.norm_adj(data["edge_index"], len(data["x"]), device)
    data["x_device"] = data["x"].to(device)
    return data


@torch.no_grad()
def save_predictions(checkpoint, row, data, stage, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if C.tensor_hash(saved["state"]) != row["end_state_sha256"]:
        raise AssertionError("checkpoint state hash mismatch")
    model = G.CLIPStudent(data["x"].shape[1]).to(device)
    model.load_state_dict(saved["state"])
    model.eval()
    logits = model.logits(data["x_device"], data["real_adj"], data["prompt_emb"].to(device))
    mask = data["report_dev_mask"]
    payload = {"logits": logits.detach().cpu()[mask], "target": data["y"][mask],
               "original_ids": data["original_ids"][mask],
               "stage_classes": data["stage_classes"], "stage": stage,
               "split": "report_dev", "test_evaluated": False}
    metrics = M.score_logits(payload["logits"], payload["target"].numpy(), data["stage_classes"], stage)
    if json.loads(json.dumps(metrics)) != json.loads(json.dumps(row["metrics"])):
        raise AssertionError("metrics mismatch on checkpoint reload")
    row["legacy_task_slice"] = legacy_from_predictions(payload["target"], payload["logits"].argmax(1), data["stage_classes"], stage)
    saved["prediction"] = payload
    saved["result"] = row
    torch.save(saved, checkpoint)
    row["checkpoint_sha256"] = C.sha(checkpoint)
    return row


def fit(memory, previous, data, stage, seed, arm, order, out, device):
    key = f"{arm}_s{stage}_seed{seed}"
    row_path, checkpoint = out / f"{key}.json", out / f"{key}.pt"
    if row_path.exists() and checkpoint.exists():
        row = json.loads(row_path.read_text())
        if (row["support_sha256"] != C.tensor_hash(memory)
                or row["checkpoint_sha256"] != C.sha(checkpoint)
                or row["training_sha256"] != V.training_hash(memory)):
            raise AssertionError("stale training cache")
        return row
    steps = P.student_steps(len(memory["x"]))
    sync(device)
    start = time.perf_counter()
    row = C.fit(memory, seed, steps, device, data, stage, checkpoint=checkpoint)
    row.update(arm=arm, stage=stage, order=order, dataset=data["dataset"],
               training_sha256=V.training_hash(memory))
    row["trained_old_embedding_relative_drift"] = B.trained_old_embedding_drift(
        memory, previous, checkpoint, data, device)
    save_predictions(checkpoint, row, data, stage, device)
    sync(device)
    row["fit_evaluation_io_seconds"] = time.perf_counter() - start
    row["nontraining_seconds"] = row["fit_evaluation_io_seconds"] - row["seconds"]
    C.save_json(row_path, row)
    print(json.dumps({"event": "fit_complete", "dataset": data["dataset"], "order": order,
                      "arm": arm, "stage": stage, "seed": seed,
                      "completed_at": datetime.now(timezone.utc).isoformat()}), flush=True)
    return row


def build_memories(data, groups, order, out, device, arms):
    blocks, block_audits = B.build_blocks(data, groups, order, out, device)
    memories, audits = {}, {}
    cfg = V.RetentionConfig()
    for arm in arms:
        chain, records, old = [], [], None
        for stage, block in enumerate(blocks):
            path, audit_path = out / f"{arm}_s{stage}_memory.pt", out / f"{arm}_s{stage}_construction.json"
            if path.exists() and audit_path.exists():
                memory = torch.load(path, map_location="cpu", weights_only=False)
                audit = json.loads(audit_path.read_text())
                if audit["file_sha256"] != C.sha(path):
                    raise AssertionError("memory artifact changed")
            else:
                sync(device)
                started = time.perf_counter()
                memory, details = V.append(old, block, arm, cfg)
                sync(device)
                seconds = time.perf_counter() - started
                audit_started = time.perf_counter()
                checks = V.audit_append(old, block, memory, arm, cfg)
                regression = None
                if arm == "c0" and order == "canonical":
                    frozen_path = C.artifact_dir(data["dataset"], "dispersion") / f"append_fixed_s{stage}_memory.pt"
                    frozen = torch.load(frozen_path, map_location="cpu", weights_only=False)
                    for key in ("x", "text_emb", "mass", "depth", "stage", "edge_index"):
                        if not torch.equal(memory[key], frozen[key]):
                            raise AssertionError(f"canonical C0 {key} regression stage {stage}")
                    if memory["texts"] != frozen["texts"]:
                        raise AssertionError("canonical C0 text regression")
                    regression = {"bitwise_equal": True, "source": str(frozen_path), "source_sha256": C.sha(frozen_path)}
                torch.save(memory, path)
                audit = {"arm": arm, "stage": stage, "details": details, "checks": checks,
                         "append_seconds": seconds, "audit_and_memory_io_seconds": time.perf_counter() - audit_started,
                         "shared_block_seconds": block_audits[stage]["build_seconds"],
                         "file_sha256": C.sha(path), "tensor_sha256": C.tensor_hash(memory),
                         "payload": O.payload_stats(memory), "serialized_bytes": path.stat().st_size,
                         "canonical_regression": regression, "selector_runs": 0, "selector_steps": 0}
                C.save_json(audit_path, audit)
            V.audit_append(old, block, memory, arm, cfg)
            chain.append(memory)
            records.append(audit)
            old = memory
        memories[arm], audits[arm] = chain, records
    return memories, audits, block_audits


def source_hashes():
    modules = [V, C, U, G, P, B, O, M, U.SCA]
    paths = {Path(m.__file__).resolve() for m in modules} | {Path(__file__).resolve(), Path(U.__file__).with_name("run_multiscale_asm.py")}
    return {p.parent.name + "/" + p.name: C.sha(p) for p in sorted(paths)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer", "arxiv"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--large-candidate", choices=list(V.ARMS[1:]))
    args = ap.parse_args()
    large = args.dataset == "arxiv"
    if large != (args.large_candidate is not None):
        raise ValueError("large graph requires the gate-approved candidate")
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    arms = ["c0", args.large_candidate] if large else list(V.ARMS)
    freeze = {"version": "v5-retention-frozen-v1", "dataset": args.dataset,
              "arms": arms, "config": V.RetentionConfig().dict(), "code_sha256": source_hashes(),
              "protocol_sha256": C.sha(args.protocol), "torch": str(torch.__version__),
              "gpu": torch.cuda.get_device_name(args.device), "device": args.device,
              "rate": 0.05, "batch": P.STUDENT.batch, "lr": P.STUDENT.lr,
              "test_labels_evaluated": False, "response_selection": False,
              "metrics_primary": "all-prompt pooled seen support, class-macro-F1",
              "evaluation_graph": "transductive real graph, including unlabeled holdout/future inputs"}
    O.lock_json(out / "freeze.json", freeze)
    data = prepare(args.dataset, args.minilm, args.device, out, large)
    registration_path = out / "registration.json"
    if registration_path.exists():
        registration = json.loads(registration_path.read_text())
    else:
        if large:
            registration = {"orders": {"base4": data["stage_classes"]}, "seeds": {"base4": list(range(9100, 9106))}}
        else:
            canonical = O.stage_orders(args.dataset)["canonical"]
            excluded, records = historical_orders(args.dataset)
            counts = torch.bincount(data["y"][data["pool_mask"]]).tolist()
            matched, details = matched_order(counts, canonical, excluded)
            registration = {"orders": {"canonical": canonical, "matched_unseen": matched},
                            "seeds": {"canonical": list(range(100, 106)), "matched_unseen": list(range(9100, 9106))},
                            "matched_order_selection": details, "excluded_groups": sorted(excluded),
                            "historical_order_sources": records}
        registration.update(created_at=datetime.now(timezone.utc).isoformat(),
                            freeze_sha256=C.sha(out / "freeze.json"),
                            protocol_sha256=C.sha(args.protocol))
        C.save_json(registration_path, registration)
    if registration["freeze_sha256"] != C.sha(out / "freeze.json"):
        raise AssertionError("registration/freeze mismatch")
    if args.prepare_only:
        print(json.dumps({"event": "registered", "dataset": args.dataset,
                          "orders": registration["orders"], "seeds": registration["seeds"]}), flush=True)
        return
    all_results = []
    for order, groups in registration["orders"].items():
        target = out / order
        target.mkdir(exist_ok=True)
        memories, audits, block_audits = build_memories(data, groups, order, target, args.device, arms)
        rows = []
        for stage in range(len(groups)):
            for seed in registration["seeds"][order]:
                offset = seed % len(arms)
                for arm in arms[offset:] + arms[:offset]:
                    previous = memories[arm][stage - 1] if stage else None
                    rows.append(fit(memories[arm][stage], previous, data, stage, seed, arm, order, target, args.device))
        result = {"dataset": args.dataset, "order": order, "stage_classes": data["stage_classes"],
                  "registration_sha256": C.sha(registration_path), "arms": arms, "fits": rows,
                  "construction": audits, "shared_blocks": block_audits,
                  "status": "complete", "test_labels_evaluated": False}
        C.save_json(target / "results.json", result)
        all_results.append({"order": order, "fits": len(rows), "results_sha256": C.sha(target / "results.json")})
    C.save_json(out / "completion.json", {"dataset": args.dataset, "status": "complete",
                "orders": all_results, "fits": sum(r["fits"] for r in all_results),
                "code_sha256": source_hashes(), "freeze_sha256": C.sha(out / "freeze.json")})


if __name__ == "__main__":
    main()
