"""Frozen v7-A: register and select first, then evaluate independent students."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

import run_v6_local_marginal as R6
import v7_gradient_append as V

C, U, SCA, G, P = V.C, V.U, V.SCA, V.G, V.P
R5, O, B, M = R6.R5, R6.O, R6.B, R6.M
sync, nested_hash = R6.sync, R6.nested_hash
SEL = sum((list(g) for g in V.SELECTION_GROUPS), [])
RTOL, ATOL = 1e-4, 1e-5


def read(path):
    return json.loads(Path(path).read_text())


def stable_lock(path, value):
    value = json.loads(json.dumps(value))
    if path.exists():
        if read(path) != value:
            raise ValueError(f"frozen configuration mismatch: {path}")
        return
    C.save_json(path, value)


def source_hashes():
    hashes = R6.source_hashes()
    for path in (Path(__file__), Path(V.__file__)):
        hashes[path.parent.name + "/" + path.name] = C.sha(path)
    return dict(sorted(hashes.items()))


def registration(data, out, protocol):
    path = out / "registration.json"
    if path.exists():
        reg = read(path)
        assert reg["freeze_sha256"] == C.sha(out / "freeze.json")
        return reg
    excluded, sources = R5.historical_orders(data["dataset"])
    for prior in sorted(C.RESULTS.rglob("registration.json")):
        if prior == path or data["dataset"] not in str(prior).lower():
            continue
        saved = read(prior)
        groups_by_name = saved.get("orders", {})
        for groups in groups_by_name.values():
            if len(groups) == 3 and all(isinstance(g, list) for g in groups):
                excluded.add(R5.canonical_groups(groups))
        sources.append({"path": str(prior), "sha256": C.sha(prior)})
    reference = O.stage_orders(data["dataset"])["canonical"]
    groups, detail = R5.matched_order(
        torch.bincount(data["y"][data["pool_mask"]]).tolist(), reference, excluded)
    reg = {"orders": {"canonical": reference, "confirmation": groups},
           "order_selection": detail, "excluded_groups": sorted(excluded),
           "historical_sources": sources, "selection_groups": V.SELECTION_GROUPS,
           "report_seeds": V.REPORT_SEEDS, "freeze_sha256": C.sha(out / "freeze.json"),
           "protocol_sha256": C.sha(protocol), "registered_at": datetime.now(timezone.utc).isoformat()}
    C.save_json(path, reg)
    return reg


def build_candidates(data, groups, order, out, device):
    blocks, block_audits = B.build_blocks(data, groups, order, out, device)
    ids, budgets = O.set_stages(data, groups)
    chains, old = [], None
    for stage, block in enumerate(blocks):
        memory = SCA.append_memory(old, block, 0 if old is None else len(old["x"]), device=device)
        C.assert_frozen(old, memory)
        path = out / f"c0_s{stage}_history.pt"
        if path.exists():
            assert nested_hash(torch.load(path, map_location="cpu", weights_only=False)) == nested_hash(memory)
        else:
            torch.save(memory, path)
        if order == "canonical":
            archive_path = C.artifact_dir(data["dataset"], "dispersion") / f"append_fixed_s{stage}_memory.pt"
            archived = torch.load(archive_path, map_location="cpu", weights_only=False)
            for key in V.V6.FIELDS + ("edge_index",):
                assert (torch.equal(memory[key], archived[key]) if isinstance(memory[key], torch.Tensor)
                        else memory[key] == archived[key]), f"C0 regression: {key}"
        chains.append(memory)
        old = memory
    manifests = []
    for stage in (1, 2):
        target = out / f"stage{stage}"
        target.mkdir(exist_ok=True)
        path = target / "candidates.json"
        if path.exists():
            manifest = read(path)
            assert C.sha(target / manifest["old_history_filename"]) == manifest["old_history_sha256"]
            for item in manifest["candidates"]:
                assert C.sha(target / item["filename"]) == item["file_sha256"]
                assert C.sha(target / item["sources_filename"]) == item["sources_sha256"]
            manifests.append(manifest)
            continue
        sync(device)
        started = time.perf_counter()
        candidates, skipped = V.make_candidates(data, ids[stage], blocks[stage], stage, device)
        sync(device)
        build_seconds = time.perf_counter() - started
        entries = []
        for name, item in candidates.items():
            sync(device)
            started = time.perf_counter()
            memory = SCA.append_memory(chains[stage - 1], item["block"], budgets[stage - 1], device=device)
            sync(device)
            append_seconds = time.perf_counter() - started
            C.assert_frozen(chains[stage - 1], memory)
            assert len(memory["x"]) == budgets[stage]
            memory_path, source_path = target / f"{name}_memory.pt", target / f"{name}_sources.pt"
            torch.save(memory, memory_path)
            torch.save(item["records"], source_path)
            entries.append({k: v for k, v in item.items() if k not in ("block", "records")} | {
                "name": name, "filename": memory_path.name, "sources_filename": source_path.name,
                "file_sha256": C.sha(memory_path), "sources_sha256": C.sha(source_path),
                "memory_sha256": nested_hash(memory), "rows": len(memory["x"]),
                "singleton_new_rows": sum(len(r["ids"]) == 1 for r in item["records"]),
                "append_seconds": append_seconds, "payload": O.payload_stats(memory)})
        manifest = {"dataset": data["dataset"], "order": order, "stage": stage,
                    "stage_classes": groups, "candidates": entries, "skipped": skipped,
                    "candidate_build_seconds": build_seconds, "source_ids": ids[stage].tolist(),
                    "old_history_filename": f"../c0_s{stage - 1}_history.pt",
                    "old_history_sha256": C.sha(out / f"c0_s{stage - 1}_history.pt"),
                    "shared_blocks": block_audits,
                    "mode": "counterfactual; C0 parents only; not a recursive selected chain"}
        C.save_json(path, manifest)
        manifests.append(manifest)
    return manifests


def gradient_or_load(item, seed, target, device):
    key = f"{item['name']}_gradient_seed{seed}"
    path, row_path = target / f"{key}.pt", target / f"{key}.json"
    if row_path.exists():
        row = read(row_path)
        assert C.sha(path) == row["file_sha256"] and row["memory_sha256"] == item["memory_sha256"]
        return torch.load(path, map_location="cpu", weights_only=False), row
    memory = torch.load(target / item["filename"], map_location="cpu", weights_only=False)
    sync(device)
    started = time.perf_counter()
    curve = V.gradient_curve(memory, seed, device)
    sync(device)
    elapsed = time.perf_counter() - started
    torch.save(curve, path)
    row = {"candidate": item["name"], "seed": seed, "filename": path.name,
           "file_sha256": C.sha(path), "memory_sha256": item["memory_sha256"],
           "wall_seconds": elapsed, "optimizer_updates": 2, "gradient_observations": 3,
           "scope": "init, adjacency, 3 backward passes, 2 Adam updates, CPU gradients/hashes; excludes disk I/O"}
    C.save_json(row_path, row)
    return curve, row


def select_or_load(manifest, target, device):
    path = target / "selection.json"
    if path.exists():
        result = read(path)
        assert result["candidates_sha256"] == C.sha(target / "candidates.json")
        for item in result["gradient_runs"]:
            assert item == read(target / f"{item['candidate']}_gradient_seed{item['seed']}.json")
            assert C.sha(target / item["filename"]) == item["file_sha256"]
        return result
    if list(target.glob("*_seed93*.json")):
        raise AssertionError("report fits exist before the selection lock")
    entries = {i["name"]: i for i in manifest["candidates"]}
    names = list(entries)
    scores, rows = {n: {} for n in names}, []
    score_seconds = 0.
    started = time.perf_counter()
    for seed in SEL:
        base, row = gradient_or_load(entries["c0"], seed, target, device)
        rows.append(row)
        score_start = time.perf_counter()
        scores["c0"][seed] = V.gradient_score(base, base, entries["c0"]["geometry_error"], entries["c0"]["geometry_error"])
        score_seconds += time.perf_counter() - score_start
        for name in names[1:]:
            curve, row = gradient_or_load(entries[name], seed, target, device)
            rows.append(row)
            assert curve["start_state_sha256"] == base["start_state_sha256"]
            assert all(torch.equal(a["batch_rows"], b["batch_rows"])
                       for a, b in zip(curve["observations"], base["observations"]))
            score_start = time.perf_counter()
            scores[name][seed] = V.gradient_score(curve, base, entries["c0"]["geometry_error"], entries[name]["geometry_error"])
            score_seconds += time.perf_counter() - score_start
            del curve
        del base
    result = {"selection": V.select_gradient(scores, names), "scores": scores, "gradient_runs": rows,
              "gradient_score_seconds": score_seconds,
              "wall_including_io_seconds": time.perf_counter() - started,
              "candidates_sha256": C.sha(target / "candidates.json"),
              "selected_at": datetime.now(timezone.utc).isoformat(),
              "uses_validation_labels": False, "report_fits_started": False}
    C.save_json(path, result)
    print(json.dumps({"event": "selection_locked", "cell": str(target),
                      "selected": result["selection"]["selected"]}), flush=True)
    return result


def prefix_audit(memory, device):
    seed = SEL[0]
    gradient = V.gradient_curve(memory, seed, device, keep_state=True)
    reference = R6.train_curve(memory, seed, 2, device, checkpoints=[2])
    checks = []
    for key, value in reference["state"].items():
        actual = gradient["state"][key]
        assert torch.allclose(actual, value, rtol=RTOL, atol=ATOL), key
        checks.append({"parameter": key, "bitwise_equal": torch.equal(actual, value),
                       "max_abs": float((actual - value).abs().max())})
    repeat = V.gradient_curve(memory, seed, device)
    for a, b in zip(gradient["observations"], repeat["observations"]):
        assert torch.equal(a["batch_rows"], b["batch_rows"])
        assert torch.allclose(a["gradient"], b["gradient"], rtol=RTOL, atol=ATOL)
    assert gradient["start_state_sha256"] == reference["start_state_sha256"]
    return {"status": "passed", "checks": checks, "seed": seed,
            "gradient_prefix_matches_common_student": True, "repeat_gradient_allclose": True,
            "rtol": RTOL, "atol": ATOL, "optimizer_updates": 6}


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
    O.lock_json = stable_lock
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stable_lock(out / "freeze.json", {
        "version": "v7a-gradient-compatible-v1", "dataset": args.dataset, "code_sha256": source_hashes(),
        "protocol_sha256": C.sha(args.protocol), "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu",
        "batch": P.STUDENT.batch, "lr": P.STUDENT.lr, "selection_groups": V.SELECTION_GROUPS,
        "report_seeds": V.REPORT_SEEDS, "history": "C0 counterfactual only",
        "test_labels_evaluated": False, "gradient_selector_labels": False})
    data = R5.prepare(args.dataset, args.minilm, args.device, out)
    reg = registration(data, out, args.protocol)
    source = {k: data[k].cpu().clone() for k in ("original_ids", "y", "pool_mask")}
    source["y"][source["original_ids"] % 10 == 2] = -1
    source["pool_x"] = data["x"][data["pool_mask"]].cpu()
    source["pool_text_emb"] = data["text_emb"][data["pool_mask"]].cpu()
    source_path = out / "source_audit.pt"
    if source_path.exists():
        assert nested_hash(torch.load(source_path, weights_only=False)) == nested_hash(source)
    else:
        torch.save(source, source_path)
    stable_lock(out / "source_audit.json", {"sha256": C.sha(source_path),
        "scope": "offline source audit; test targets replaced by -1; excluded from deployed memory"})
    manifests = {}
    for order, groups in reg["orders"].items():
        target = out / order
        target.mkdir(exist_ok=True)
        manifests[order] = build_candidates(data, groups, order, target, args.device)
    for order, entries in manifests.items():
        for manifest in entries:
            select_or_load(manifest, out / order / f"stage{manifest['stage']}", args.device)
    audit_path = out / "prefix_audit.json"
    if not audit_path.exists():
        O.set_stages(data, reg["orders"]["canonical"])
        memory = torch.load(out / "canonical/stage1/c0_memory.pt", weights_only=False)
        started = time.perf_counter()
        gradient_check = prefix_audit(memory, args.device)
        curve_check = R6.prefix_audit(memory, data, 1, args.device)
        C.save_json(audit_path, {"status": "passed", "gradient": gradient_check, "student_curve": curve_check,
                                "wall_seconds": time.perf_counter() - started})
    if args.prepare_only:
        print(json.dumps({"status": "prepared_and_selected", "orders": reg["orders"]}), flush=True)
        return
    count = 0
    for order, entries in manifests.items():
        O.set_stages(data, reg["orders"][order])
        for manifest in entries:
            target = out / order / f"stage{manifest['stage']}"
            selection_hash = C.sha(target / "selection.json")
            fits = []
            for seed in V.REPORT_SEEDS:
                candidates = manifest["candidates"]
                offset = seed % len(candidates)
                for item in candidates[offset:] + candidates[:offset]:
                    fits.append(R6.fit_or_load(item, manifest["stage"], seed, data, target, args.device))
            assert C.sha(target / "selection.json") == selection_hash
            C.save_json(target / "results.json", {"status": "complete", "fits": fits,
                "candidates_sha256": C.sha(target / "candidates.json"), "selection_sha256": selection_hash,
                "registration_sha256": C.sha(out / "registration.json")})
            count += len(fits)
    C.save_json(out / "completion.json", {"status": "complete", "fits": count,
                "freeze_sha256": C.sha(out / "freeze.json"), "test_labels_evaluated": False})


if __name__ == "__main__":
    main()
