"""Independent structural/checkpoint audit for the V2 follow-up results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as TF

import run_safe_joint_kernel_experiments as R
import run_frozen_baseline_joint_kernel as F


def edge_set(memory):
    return set(map(tuple, memory["edge_index"].long().T.tolist()))


def check_symmetric(memory):
    edges = edge_set(memory)
    return edges == {(j, i) for i, j in edges}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reevaluate", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    errors, checked_fits, checked_cells, checked_memories = [], 0, 0, 0
    edge_comparisons = []
    freeze = json.loads((args.root / "freeze.json").read_text())
    cache = Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919") / (freeze["dataset"] + "_r005/text_emb_pool_only.pt")
    data = R.load_data(freeze["dataset"], "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", args.device, cache)
    if F.C.sha(cache) != freeze["cache_sha256"] or F.C.sha(F.U.DATA_CFG[freeze["dataset"]]["path"]) != freeze["data_sha256"]:
        errors.append("data/cache fingerprint mismatch")
    expected_cells = {cell["name"] for cell in freeze["cells"]}
    actual_cells = {p.name for p in args.root.iterdir() if p.is_dir() and (p / "results.json").exists()}
    if actual_cells != expected_cells or not (args.root / "completion.json").exists():
        errors.append("incomplete cell matrix or missing completion")
    else:
        completion = json.loads((args.root / "completion.json").read_text())
        if (not completion["complete"] or completion["fits"] != freeze["expected_fits"]
                or completion["freeze_sha256"] != F.C.sha(args.root / "freeze.json")):
            errors.append("completion fingerprint or fit count mismatch")
    if freeze["source_hashes"] != R.source_hashes():
        errors.append("freeze source hashes do not match runner sources")
    for cell in sorted(p for p in args.root.iterdir() if p.is_dir()
                       and (p / "results.json").exists()):
        payload = json.loads((cell / "results.json").read_text())
        manifest, results = payload["manifest"], payload["results"]
        if manifest != json.loads((cell / "manifest.json").read_text()):
            errors.append(f"{cell.name}: manifest mismatch")
        groups = F.stage_groups(data["dataset"], manifest["stage_count"], manifest["order"])
        ids, budgets = F.set_stream(data, groups, manifest["rate"])
        if groups != manifest["stage_classes"] or budgets != manifest["budgets"] or manifest["source_ids_sha256"] != F.U.sha256_json([v.tolist() for v in ids]):
            errors.append(f"{cell.name}: source stream/budgets")
        arms, stages, seeds = manifest["arms"], manifest["stage_count"], manifest["seeds"]
        expected = {(arm, stage) for arm in arms for stage in range(stages)}
        actual = {(row["arm"], row["stage"]) for row in results}
        if actual != expected or len(results) != len(expected):
            errors.append(f"{cell.name}: incomplete arm-stage matrix")
        old_by_arm = {}
        for arm in arms:
            old = None
            for stage in range(stages):
                key = f"{arm}_s{stage}"
                memory_path, audit_path = cell / f"{key}_memory.pt", cell / f"{key}_construction.json"
                if not memory_path.exists() or not audit_path.exists():
                    errors.append(f"{cell.name}/{key}: missing memory or audit")
                    continue
                memory = torch.load(memory_path, map_location="cpu", weights_only=False)
                checked_memories += 1
                audit = json.loads(audit_path.read_text())
                if F.J_hash(memory) != audit.get("content_sha256"):
                    errors.append(f"{cell.name}/{key}: content hash")
                if audit.get("payload", {}).get("rows") != len(memory["x"]):
                    errors.append(f"{cell.name}/{key}: payload rows")
                if R.payload(memory) != audit["payload"]:
                    errors.append(f"{cell.name}/{key}: byte accounting")
                if R.topology_metrics(old, memory) != audit["topology"]:
                    errors.append(f"{cell.name}/{key}: topology diagnostic mismatch")
                if not torch.isfinite(memory["x"]).all() or not torch.isfinite(memory["text_emb"]).all():
                    errors.append(f"{cell.name}/{key}: nonfinite input")
                edge = memory["edge_index"].long()
                keep = 0 if old is None else len(old["x"])
                if edge.shape[1] != len(edge_set(memory)):
                    errors.append(f"{cell.name}/{key}: duplicate edges")
                if edge.shape[0] != 2 or len(edge) and ((edge < 0).any() or (edge >= len(memory["x"])).any()):
                    errors.append(f"{cell.name}/{key}: invalid edge index")
                if len(edge) and (edge[0] == edge[1]).any():
                    errors.append(f"{cell.name}/{key}: explicit self loop")
                if arm != "operator_stable" and not check_symmetric(memory):
                    errors.append(f"{cell.name}/{key}: expected symmetric topology")
                if old is not None:
                    n = len(old["x"])
                    for field in ("x", "text_emb", "mass", "depth", "stage", "graph_signature"):
                        if field in old and field in memory and not torch.equal(old[field], memory[field][:n]):
                            errors.append(f"{cell.name}/{key}: old {field} changed")
                    old_edges = edge[:, (edge < n).all(0)]
                    if not torch.equal(old_edges, old["edge_index"]):
                        errors.append(f"{cell.name}/{key}: old-old edges changed")
                    if old["texts"] != memory["texts"][:n]:
                        errors.append(f"{cell.name}/{key}: old texts changed")
                    if old.get("signature_config") != memory.get("signature_config"):
                        errors.append(f"{cell.name}/{key}: signature coordinate space changed")
                if arm not in {"c0", "k_center"} and not arm.startswith("byte_"):
                    control = torch.load(cell / f"c0_s{stage}_memory.pt", map_location="cpu", weights_only=False)
                    for field in ("x", "text_emb", "mass", "depth", "stage"):
                        if not torch.equal(memory[field], control[field]):
                            errors.append(f"{cell.name}/{key}: C0 {field} mismatch")
                    if memory["texts"] != control["texts"]:
                        errors.append(f"{cell.name}/{key}: C0 texts mismatch")
                    if edge.shape != control["edge_index"].shape:
                        errors.append(f"{cell.name}/{key}: C0 edge count mismatch")
                if arm in ("degree_balanced", "coverage"):
                    internal = edge[:, (edge >= keep).all(0)] - keep
                    if int(torch.bincount(internal[0], minlength=len(memory["x"]) - keep).max()) > 2:
                        errors.append(f"{cell.name}/{key}: internal hard cap")
                    if keep:
                        incoming = edge[:, (edge[0] < keep) & (edge[1] >= keep)]
                        cap = (len(memory["x"]) - keep + keep - 1) // keep
                        if int(torch.bincount(incoming[0], minlength=keep).max()) > cap:
                            errors.append(f"{cell.name}/{key}: old endpoint capacity")
                if arm == "operator_stable" and old is not None:
                    before = F.C.training_adjacency(old, "cpu").to_dense()
                    after = F.C.training_adjacency(memory, "cpu").to_dense()
                    if not torch.equal(before, after[:keep, :keep]) or torch.count_nonzero(after[:keep, keep:]):
                        errors.append(f"{cell.name}/{key}: old propagation not exactly preserved")
                if arm.startswith("byte_"):
                    c0 = torch.load(cell / f"c0_s{stage}_memory.pt", map_location="cpu", weights_only=False)
                    limit = R.payload(c0)["logical_bytes"]
                    attempts = audit["byte_search_attempts"]
                    if audit["payload"]["logical_bytes"] > limit or audit["byte_limit"] != limit:
                        errors.append(f"{cell.name}/{key}: byte limit exceeded")
                    if any(a["bytes"] <= limit for a in attempts[:-1]):
                        errors.append(f"{cell.name}/{key}: descending byte search did not stop first")
                if arm == "k_center":
                    sources = memory["source_ids"]
                    if not torch.isin(sources, torch.cat(ids[:stage+1])).all() or not torch.equal(data["x"][sources].float(), memory["x"]):
                        errors.append(f"{cell.name}/{key}: coreset source legality")
                if "graph_signature" in memory:
                    source = json.loads((cell / f"{key}_sources.json").read_text())
                    parts = source["source_groups"]
                    if sorted(i for g in parts for i in g) != ids[stage].tolist() or source["current_delta"] != ids[stage].tolist():
                        errors.append(f"{cell.name}/{key}: source partition")
                    joint = torch.cat((TF.normalize(data["x"][ids[stage]].float(), dim=1), data["text_emb"][ids[stage]]), dim=1)
                    source_graph = F.J.source_local_graph_summary(data, ids[stage], joint)
                    positions = {int(i): j for j, i in enumerate(ids[stage])}
                    signatures = []
                    for offset, part in enumerate(parts):
                        if not torch.equal(data["x"][part].float().mean(0), memory["x"][keep+offset]) or len(part) != memory["mass"][keep+offset]:
                            errors.append(f"{cell.name}/{key}: synthetic row {offset} does not reconstruct")
                        signatures.append(source_graph[[positions[v] for v in part]].mean(0))
                    original = TF.normalize(torch.stack(signatures), dim=1)
                    reconstructed = R.S.compress_signature({"graph_signature": original}, memory["signature_config"]["mode"])
                    if not torch.equal(reconstructed["graph_signature"], memory["graph_signature"][keep:]):
                        errors.append(f"{cell.name}/{key}: reconstructed signature mismatch")
                    block = R.copy_block(memory, keep)
                    block.update(graph_signature=memory["graph_signature"][keep:], signature_config=memory["signature_config"])
                    if arm == "drift_gated":
                        predicted, extra = R.gated_append(old, block, args.device)
                        if extra != audit["variant_audit"]:
                            errors.append(f"{cell.name}/{key}: gate audit mismatch")
                    else:
                        variant = "joint_kernel" if arm.startswith(("node_", "byte_")) else arm
                        predicted, _ = R.S.append_variant(old, block, keep, variant)
                    if not torch.equal(predicted["edge_index"], edge):
                        errors.append(f"{cell.name}/{key}: deterministic edge reconstruction")
                row = next((r for r in results if r["arm"] == arm and r["stage"] == stage), None)
                if row is None:
                    continue
                if [r["seed"] for r in row["per_seed"]] != seeds:
                    errors.append(f"{cell.name}/{key}: seeds")
                for seed in row["per_seed"]:
                    fit_key = f"{key}_seed{seed['seed']}"
                    fit_json = cell / f"{fit_key}.json"
                    checkpoint = cell / f"{fit_key}.pt"
                    if not fit_json.exists() or not checkpoint.exists():
                        errors.append(f"{cell.name}/{fit_key}: missing fit")
                        continue
                    fit = json.loads(fit_json.read_text())
                    if fit != seed or fit.get("support_sha256") != F.C.tensor_hash(memory):
                        errors.append(f"{cell.name}/{fit_key}: fit/support mismatch")
                    if fit.get("checkpoint_sha256") != F.C.sha(checkpoint):
                        errors.append(f"{cell.name}/{fit_key}: checkpoint hash")
                    if fit["steps"] != manifest["steps"][stage] or fit["effective_batch"] != min(16, len(memory["x"])):
                        errors.append(f"{cell.name}/{fit_key}: steps/batch mismatch")
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    if F.C.tensor_hash(saved["state"]) != fit["end_state_sha256"]:
                        errors.append(f"{cell.name}/{fit_key}: model state hash")
                    prediction = saved["prediction"]
                    mask = data["report_dev_mask"]
                    if not torch.equal(prediction["target"], data["y"][mask]) or not torch.equal(prediction["original_ids"], data["original_ids"][mask]):
                        errors.append(f"{cell.name}/{fit_key}: report split")
                    scored = F.M.score_logits(prediction["logits"], prediction["target"].numpy(), groups, stage)
                    if json.loads(json.dumps(scored)) != fit["metrics"]:
                        errors.append(f"{cell.name}/{fit_key}: saved prediction metrics")
                    if args.reevaluate:
                        model = F.G.CLIPStudent(data["x"].shape[1]).to(args.device)
                        model.load_state_dict(saved["state"])
                        reevaluated = F.M.evaluate(model, data, stage)
                        if json.loads(json.dumps(reevaluated)) != fit["metrics"]:
                            errors.append(f"{cell.name}/{fit_key}: independent checkpoint evaluation")
                        del model
                    checked_fits += 1
                if F.C.aggregate(row["per_seed"]) != row["metrics"]:
                    errors.append(f"{cell.name}/{key}: aggregate mismatch")
                old = memory
                old_by_arm[arm] = old
        for stage in range(stages):
            for candidate, control in (("coverage", "degree_balanced"), ("kernel_t", "c0"),
                    ("node_fp16", "joint_kernel"), ("node_sketch64", "joint_kernel")):
                if candidate not in arms or control not in arms:
                    continue
                left = torch.load(cell / f"{candidate}_s{stage}_memory.pt", map_location="cpu", weights_only=False)
                right = torch.load(cell / f"{control}_s{stage}_memory.pt", map_location="cpu", weights_only=False)
                a, b = edge_set(left), edge_set(right)
                comparison = {"cell": cell.name, "stage": stage, "candidate": candidate,
                    "control": control, "edge_set_equal": a == b,
                    "edge_order_equal": torch.equal(left["edge_index"], right["edge_index"]),
                    "edge_jaccard": len(a & b) / len(a | b) if a | b else 1.}
                if candidate.startswith("node_"):
                    lg = TF.normalize(left["graph_signature"].float(), dim=1)
                    rg = TF.normalize(right["graph_signature"].float(), dim=1)
                    comparison["signature_gram_mean_absolute_error"] = float((lg @ lg.T - rg @ rg.T).abs().mean())
                edge_comparisons.append(comparison)
        checked_cells += 1
    expected_fits = freeze["expected_fits"]
    if checked_fits != expected_fits:
        errors.append(f"fits {checked_fits}/{expected_fits}")
    result = {"dataset": freeze["dataset"], "checked_cells": checked_cells,
              "checked_fits": checked_fits, "expected_fits": expected_fits,
              "checked_memories": checked_memories,
              "error_count": len(errors), "errors": errors,
              "independent_model_reevaluation": args.reevaluate,
              "test_labels_evaluated": False, "auditor_source_hashes": R.source_hashes(),
              "audit_script_sha256": F.C.sha(__file__), "edge_comparisons": edge_comparisons,
              "audit_environment": {"device": args.device, "torch": torch.__version__,
                  "gpu_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None}}
    (args.root / "audit_v2.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
