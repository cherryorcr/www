"""Frozen v6-A counterfactual experiment; C0 history is shared by all candidates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

import v6_local_marginal as V
import run_v5_retention as R5

C, U, SCA, G, P = V.C, V.U, V.SCA, V.G, V.P
O, B, M = R5.O, R5.B, R5.M


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def nested_hash(value):
    def normalize(obj):
        if isinstance(obj, torch.Tensor):
            return {"tensor": C.tensor_hash({"value": obj})}
        if isinstance(obj, dict):
            return {str(k): normalize(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [normalize(v) for v in obj]
        return obj
    return hashlib.sha256(json.dumps(normalize(value), sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def observe(model, data, stage):
    model.eval()
    device = data["x_device"].device
    sync(device)
    started = time.perf_counter()
    logits = model.logits(data["x_device"], data["real_adj"], data["prompt_emb"].to(data["x_device"].device)).cpu()
    sync(device)
    timing = {"forward_seconds": time.perf_counter() - started}
    seen = torch.tensor([c for g in data["stage_classes"][:stage + 1] for c in g])
    result = {}
    for split, key in (("selector", "selector_dev_mask"), ("report", "report_dev_mask")):
        started = time.perf_counter()
        mask = data[key] & torch.isin(data["y"], seen)
        prediction = {"logits": logits[mask], "target": data["y"][mask],
                      "original_ids": data["original_ids"][mask], "stage": stage,
                      "stage_classes": data["stage_classes"], "split": split}
        metrics = M.score_logits(prediction["logits"], prediction["target"].numpy(),
                                 data["stage_classes"], stage)
        result[split] = {"prediction": prediction, "metrics": metrics}
        timing[split + "_seconds"] = time.perf_counter() - started
    result["timing"] = timing
    return result


def train_curve(memory, seed, steps, device, observer=None, checkpoints=None, keep_states=False):
    torch.manual_seed(seed)
    model = G.CLIPStudent(memory["x"].shape[1]).to(device)
    initial = C.tensor_hash(model.state_dict())
    x, text = memory["x"].to(device), memory["text_emb"].to(device)
    adj = C.training_adjacency(memory, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=P.STUDENT.lr)
    generator = torch.Generator().manual_seed(seed * 7 + 1)
    checkpoints = V.horizons(steps) if checkpoints is None else list(checkpoints)
    if sorted(set(checkpoints)) != checkpoints or not checkpoints or checkpoints[-1] != steps:
        raise ValueError("invalid checkpoints")
    cuda_devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    if cuda_devices:
        torch.cuda.reset_peak_memory_stats(device)
    snapshots, states = [], {}
    training_seconds, evaluation_seconds = 0., 0.
    sync(device)
    start = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        rows = torch.randperm(len(x), generator=generator)[:min(P.STUDENT.batch, len(x))].to(device)
        graph, tt = model.encode_graph(x, adj)[rows], model.encode_text(text[rows])
        logits = model.logit_scale.exp().clamp(max=100) * graph @ tt.T
        labels = torch.arange(len(rows), device=device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite student loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            sync(device)
            training_seconds += time.perf_counter() - start
            audit_started = time.perf_counter()
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            snapshot = {"step": step, "state_sha256": C.tensor_hash(state),
                        "optimizer_sha256": nested_hash(optimizer.state_dict()),
                        "generator_sha256": C.tensor_hash({"rng": generator.get_state()}),
                        "last_loss": float(loss.detach().cpu()), "training_seconds": training_seconds}
            if keep_states:
                states[step] = state
            mode = model.training
            # Intermediate observation must not alter the training RNG or mode.
            with torch.random.fork_rng(devices=cuda_devices):
                snapshot["observations"] = observer(model) if observer else {}
            model.train(mode)
            sync(device)
            elapsed = time.perf_counter() - audit_started
            evaluation_seconds += elapsed
            timing = snapshot["observations"].get("timing", {})
            snapshot["observation_timing"] = timing
            snapshot["snapshot_audit_seconds"] = elapsed - sum(timing.values())
            snapshot["evaluation_and_snapshot_seconds"] = evaluation_seconds
            snapshots.append(snapshot)
            start = time.perf_counter()
    result = {"seed": seed, "steps": steps, "rows": len(x), "start_state_sha256": initial,
              "end_state_sha256": C.tensor_hash(state), "memory_sha256": nested_hash(memory),
              "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
              "peak_allocated_mb": (torch.cuda.max_memory_allocated(device) / 2**20 if cuda_devices else None),
              "peak_scope": "training plus full-graph checkpoint evaluations; includes resident data/encoder",
              "snapshots": snapshots, "state": state}
    if keep_states:
        result["states"] = states
    return result


def prefix_audit(memory, data, stage, device):
    seed, steps = V.SELECTION_GROUPS[0][0], P.student_steps(len(memory["x"]))
    curve = train_curve(memory, seed, steps, device, lambda m: observe(m, data, stage))
    checks = []
    for snapshot in curve["snapshots"]:
        end = train_curve(memory, seed, snapshot["step"], device, checkpoints=[snapshot["step"]])
        for key in ("state_sha256", "optimizer_sha256", "generator_sha256", "last_loss"):
            if snapshot[key] != end["snapshots"][-1][key]:
                raise AssertionError(f"prefix evaluation changed {key}")
        checks.append({"steps": snapshot["step"], "bitwise_equal": True})
    reference = G.train_student(G.memory_to_trainable(memory), seed, device, steps)
    if C.tensor_hash(reference.state_dict()) != curve["end_state_sha256"]:
        raise AssertionError("training differs from original student")
    return {"status": "passed", "checks": checks, "legacy_student_bitwise_equal": True,
            "extra_training_runs": len(checks) + 2,
            "extra_optimizer_steps": steps * 2 + sum(r["steps"] for r in checks)}


def source_hashes():
    paths = {Path(p) for p in (V.__file__, __file__)}
    hashes = R5.source_hashes()
    # R5.source_hashes includes R5 itself, all C0 training dependencies and metrics.
    hashes.update({p.parent.name + "/" + p.name: C.sha(p) for p in paths})
    return hashes


def registration(data, out, protocol):
    path = out / "registration.json"
    if path.exists():
        reg = json.loads(path.read_text())
        if reg["freeze_sha256"] != C.sha(out / "freeze.json"):
            raise AssertionError("registration changed")
        return reg
    excluded, sources = R5.historical_orders(data["dataset"])
    # v5 deliberately excludes its own results from historical_orders.
    for prior in sorted(C.RESULTS.glob(f"v5_retention*/{data['dataset']}/registration.json")):
        saved = json.loads(prior.read_text())
        excluded.update(R5.canonical_groups(g) for g in saved["orders"].values())
        sources.append({"path": str(prior), "sha256": C.sha(prior)})
    reference = O.stage_orders(data["dataset"])["canonical"]
    groups, detail = R5.matched_order(torch.bincount(data["y"][data["pool_mask"]]).tolist(), reference, excluded)
    reg = {"orders": {"canonical": reference, "confirmation": groups}, "order_selection": detail,
           "excluded_groups": sorted(excluded), "historical_sources": sources,
           "selection_groups": V.SELECTION_GROUPS, "report_seeds": V.REPORT_SEEDS,
           "freeze_sha256": C.sha(out / "freeze.json"), "protocol_sha256": C.sha(protocol),
           "registered_at": datetime.now(timezone.utc).isoformat()}
    C.save_json(path, reg)
    return reg


def build_candidates(data, groups, order, out, device):
    blocks, block_audits = B.build_blocks(data, groups, order, out, device)
    ids, budgets = O.set_stages(data, groups)
    chains, old = [], None
    for stage, block in enumerate(blocks):
        memory = SCA.append_memory(old, block, 0 if old is None else len(old["x"]), device=device)
        C.assert_frozen(old, memory)
        history_path = out / f"c0_s{stage}_history.pt"
        if history_path.exists():
            saved = torch.load(history_path, map_location="cpu", weights_only=False)
            if nested_hash(saved) != nested_hash(memory):
                raise AssertionError("C0 history cache changed")
        else:
            torch.save(memory, history_path)
        chains.append(memory)
        old = memory
    if order == "canonical":
        for stage, memory in enumerate(chains):
            archived = torch.load(C.artifact_dir(data["dataset"], "dispersion") / f"append_fixed_s{stage}_memory.pt",
                                  map_location="cpu", weights_only=False)
            for key in V.FIELDS + ("edge_index",):
                if isinstance(memory[key], torch.Tensor):
                    equal = torch.equal(memory[key], archived[key])
                else:
                    equal = memory[key] == archived[key]
                if not equal:
                    raise AssertionError("canonical C0 regression")
    records = []
    for stage in (1, 2):
        target = out / f"stage{stage}"
        target.mkdir(exist_ok=True)
        manifest_path = target / "candidates.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if C.sha(target / manifest["old_history_filename"]) != manifest["old_history_sha256"]:
                raise AssertionError("cached parent history changed")
            for item in manifest["candidates"]:
                if (C.sha(target / item["filename"]) != item["file_sha256"]
                        or C.sha(target / item["sources_filename"]) != item["sources_sha256"]):
                    raise AssertionError("candidate cache changed")
            records.append(manifest)
            continue
        sync(device)
        started = time.perf_counter()
        candidates, skipped = V.make_candidates(data, ids[stage], blocks[stage], stage, device)
        sync(device)
        build_seconds = time.perf_counter() - started
        entries = []
        for name, item in candidates.items():
            memory = SCA.append_memory(chains[stage - 1], item["block"], budgets[stage - 1], device=device)
            C.assert_frozen(chains[stage - 1], memory)
            if len(memory["x"]) != budgets[stage]:
                raise AssertionError("candidate budget mismatch")
            path = target / f"{name}_memory.pt"
            source_path = target / f"{name}_sources.pt"
            torch.save(memory, path)
            torch.save(item["records"], source_path)
            entries.append({k: v for k, v in item.items() if k not in ("block", "records")} | {
                "name": name, "filename": path.name, "sources_filename": source_path.name,
                "file_sha256": C.sha(path), "sources_sha256": C.sha(source_path),
                "memory_sha256": nested_hash(memory), "rows": len(memory["x"]),
                "singleton_new_rows": sum(len(r["ids"]) == 1 for r in item["records"]),
                "payload": O.payload_stats(memory)})
        manifest = {"dataset": data["dataset"], "order": order, "stage": stage,
                    "stage_classes": groups, "candidates": entries, "skipped": skipped,
                    "candidate_build_seconds": build_seconds, "source_ids": ids[stage].tolist(),
                    "old_history_filename": f"../c0_s{stage - 1}_history.pt",
                    "old_history_sha256": C.sha(out / f"c0_s{stage - 1}_history.pt"),
                    "shared_blocks": block_audits,
                    "mode": "counterfactual; same C0 parent; not recursive selected chain"}
        C.save_json(manifest_path, manifest)
        records.append(manifest)
    return records


def fit_or_load(item, stage, seed, data, target, device):
    key = f"{item['name']}_seed{seed}"
    path, row_path = target / f"{key}.pt", target / f"{key}.json"
    if row_path.exists() and path.exists():
        row = json.loads(row_path.read_text())
        if row["checkpoint_sha256"] != C.sha(path) or row["memory_sha256"] != item["memory_sha256"]:
            raise AssertionError("stale fit cache")
        return row
    memory = torch.load(target / item["filename"], map_location="cpu", weights_only=False)
    result = train_curve(memory, seed, P.student_steps(len(memory["x"])), device,
                         observer=lambda model: observe(model, data, stage))
    torch.save(result, path)
    row = {k: v for k, v in result.items() if k not in ("state", "snapshots")}
    row.update(candidate=item["name"], family=item["family"], stage=stage,
               checkpoint=path.name, checkpoint_sha256=C.sha(path))
    row["curve"] = [{k: v for k, v in snapshot.items() if k != "observations"} | {
        split: snapshot["observations"][split]["metrics"] for split in ("selector", "report")}
                    for snapshot in result["snapshots"]]
    C.save_json(row_path, row)
    print(json.dumps({"event": "fit_complete", "dataset": data["dataset"], "stage": stage,
                      "candidate": item["name"], "seed": seed}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    freeze = {"version": "v6a-local-marginal-v1", "dataset": args.dataset, "code_sha256": source_hashes(),
              "protocol_sha256": C.sha(args.protocol), "torch": str(torch.__version__),
              "gpu": torch.cuda.get_device_name(args.device), "batch": P.STUDENT.batch, "lr": P.STUDENT.lr,
              "selection_groups": V.SELECTION_GROUPS, "report_seeds": V.REPORT_SEEDS,
              "test_labels_evaluated": False, "history": "C0 only, not selected paths"}
    O.lock_json(out / "freeze.json", json.loads(json.dumps(freeze)))
    data = R5.prepare(args.dataset, args.minilm, args.device, out)
    reg = registration(data, out, args.protocol)
    source_audit = {k: data[k].cpu() for k in ("original_ids", "y", "pool_mask")}
    source_audit["pool_x"] = data["x"][data["pool_mask"]].cpu()
    source_path = out / "source_audit.pt"
    if source_path.exists():
        if nested_hash(torch.load(source_path, weights_only=False)) != nested_hash(source_audit):
            raise AssertionError("audit source changed")
    else:
        torch.save(source_audit, source_path)
    O.lock_json(out / "source_audit.json", {"sha256": C.sha(source_path),
        "scope": "offline audit only; not persistent synthetic memory; no test predictions"})
    manifests = {}
    for order, groups in reg["orders"].items():
        target = out / order
        target.mkdir(exist_ok=True)
        manifests[order] = build_candidates(data, groups, order, target, args.device)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "orders": reg["orders"],
                          "candidates": sum(len(m["candidates"]) for rows in manifests.values() for m in rows)}), flush=True)
        return
    audit_path = out / "prefix_audit.json"
    if not audit_path.exists():
        O.set_stages(data, reg["orders"]["canonical"])
        memory = torch.load(out / "canonical/stage1/c0_memory.pt", map_location="cpu", weights_only=False)
        started = time.perf_counter()
        audit = prefix_audit(memory, data, 1, args.device)
        audit["wall_seconds"] = time.perf_counter() - started
        C.save_json(audit_path, audit)
    seeds = sum((list(g) for g in V.SELECTION_GROUPS), []) + list(V.REPORT_SEEDS)
    count = 0
    for order, entries in manifests.items():
        O.set_stages(data, reg["orders"][order])
        for entry in entries:
            stage = entry["stage"]
            target = out / order / f"stage{stage}"
            fits = []
            for seed in seeds:
                offset = seed % len(entry["candidates"])
                for item in entry["candidates"][offset:] + entry["candidates"][:offset]:
                    fits.append(fit_or_load(item, stage, seed, data, target, args.device))
            C.save_json(target / "results.json", {"status": "complete", "fits": fits,
                        "candidates_sha256": C.sha(target / "candidates.json"),
                        "registration_sha256": C.sha(out / "registration.json")})
            count += len(fits)
    C.save_json(out / "completion.json", {"status": "complete", "fits": count,
                "freeze_sha256": C.sha(out / "freeze.json"), "test_labels_evaluated": False})


if __name__ == "__main__":
    main()
