"""Independent v7-A source, gradient, checkpoint, metric and cost audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import audit_v6_local_marginal as A6
import run_v7_gradient_append as R

V, C, M = R.V, R.C, R.M
read, normalized, paired, rank = R.read, A6.normalized, A6.paired, A6.rank
SEEDS = list(V.REPORT_SEEDS)


def audit_gradients(target, manifest, memories, device, reload_gradients):
    selected = read(target / "selection.json")
    assert selected["candidates_sha256"] == C.sha(target / "candidates.json")
    assert not selected["uses_validation_labels"] and not selected["report_fits_started"]
    entries = {i["name"]: i for i in manifest["candidates"]}
    gradients = {(i["candidate"], i["seed"]): i for i in selected["gradient_runs"]}
    assert len(gradients) == len(selected["gradient_runs"])
    assert set(gradients) == {(n, s) for n in entries for s in R.SEL}
    scores = {n: {} for n in entries}
    reloaded, errors = 0, []
    for seed in R.SEL:
        curves = {}
        for name in entries:
            row = gradients[name, seed]
            assert read(target / f"{name}_gradient_seed{seed}.json") == row
            assert C.sha(target / row["filename"]) == row["file_sha256"]
            assert entries[name]["memory_sha256"] == row["memory_sha256"]
            curve = torch.load(target / row["filename"], map_location="cpu", weights_only=False)
            assert curve["seed"] == seed and curve["optimizer_updates"] == 2
            assert [a["step"] for a in curve["observations"]] == [0, 1, 2]
            generator = torch.Generator().manual_seed(seed * 7 + 1)
            for point in curve["observations"]:
                expected = torch.randperm(entries[name]["rows"], generator=generator)[:R.P.STUDENT.batch]
                assert torch.equal(point["batch_rows"], expected)
                assert torch.isfinite(point["gradient"]).all() and np.isfinite(point["loss"])
                assert point["gradient"].numel() == sum(np.prod(s) for s in curve["parameter_shapes"].values())
            if reload_gradients:
                repeated = V.gradient_curve(memories[name], seed, device)
                assert repeated["start_state_sha256"] == curve["start_state_sha256"]
                for a, b in zip(repeated["observations"], curve["observations"]):
                    assert torch.allclose(a["gradient"], b["gradient"], rtol=R.RTOL, atol=R.ATOL)
                    errors.append(float((a["gradient"] - b["gradient"]).abs().max()))
                reloaded += 1
            curves[name] = curve
        base = curves["c0"]
        for name, curve in curves.items():
            assert curve["start_state_sha256"] == base["start_state_sha256"]
            assert curve["parameter_shapes"] == base["parameter_shapes"]
            scores[name][seed] = V.gradient_score(curve, base, entries["c0"]["geometry_error"], entries[name]["geometry_error"])
            saved = selected["scores"][name][str(seed)]
            assert normalized(scores[name][seed]) == saved
            distances = []
            for point, reference in zip(curve["observations"], base["observations"]):
                # Recompute with the same float32 kernel used by the selector.
                # A float64 audit-only dot product can differ by >1e-6 on the
                # full parameter vector and would falsely reject a valid lock.
                a, b = point["gradient"], reference["gradient"]
                cosine = float(F.cosine_similarity(a[None], b[None]).item())
                distances.append(float(np.clip(1. - cosine, 0., 2.)))
            independent_q = ((entries["c0"]["geometry_error"] - entries[name]["geometry_error"])
                             / max(abs(entries["c0"]["geometry_error"]), 1e-8)
                             - .5 * float(np.mean(distances)))
            assert abs(independent_q - saved["score"]) <= 1e-7
        del curves
    assert normalized(V.select_gradient(scores, list(entries))) == selected["selection"]
    return selected, reloaded, max(errors, default=0.)


def audit_geometry(target, manifest, source):
    pool = source["pool_mask"]
    position = torch.full_like(source["y"], -1)
    position[pool] = torch.arange(int(pool.sum()))
    joint = torch.cat((F.normalize(source["pool_x"].float(), dim=1), source["pool_text_emb"].float()), dim=1)
    entries = {i["name"]: i for i in manifest["candidates"]}
    for name, item in entries.items():
        records = torch.load(target / item["sources_filename"], weights_only=False)
        count, error = 0, 0.
        for record in records:
            points = joint[position[record["ids"]]]
            count += len(points)
            error += float((points - points.mean(0)).square().sum())
        assert abs(error / count - item["geometry_error"]) < 1e-7
        if name == "multi_prototype":
            assert item["allocation"] == entries["c0"]["allocation"]
        if name == "recluster":
            expected = dict(entries["c0"]["allocation"])
            expected[str(item["edit"]["donor"])] -= 1
            expected[str(item["edit"]["recipient"])] += 1
            assert expected == item["allocation"]


def audit_fits(target, manifest, source, data, device):
    results = read(target / "results.json")
    assert results["status"] == "complete"
    assert results["candidates_sha256"] == C.sha(target / "candidates.json")
    assert results["registration_sha256"] == C.sha(target.parent.parent / "registration.json")
    assert results["selection_sha256"] == C.sha(target / "selection.json")
    entries = {i["name"]: i for i in manifest["candidates"]}
    indexed, reloaded, errors = {}, 0, []
    for row in results["fits"]:
        name, seed = row["candidate"], row["seed"]
        assert (name, seed) not in indexed and name in entries and seed in SEEDS
        assert row == read(target / f"{name}_seed{seed}.json")
        assert row["memory_sha256"] == entries[name]["memory_sha256"]
        path = target / row["checkpoint"]
        assert C.sha(path) == row["checkpoint_sha256"]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert C.tensor_hash(saved["state"]) == row["end_state_sha256"]
        assert row["steps"] == R.P.student_steps(entries[name]["rows"])
        assert [s["step"] for s in row["curve"]] == V.V6.horizons(row["steps"])
        assert len(saved["snapshots"]) == len(row["curve"]) == 4
        assert row["curve"][-1]["state_sha256"] == row["end_state_sha256"]
        for snap, point in zip(saved["snapshots"], row["curve"]):
            for key in ("step", "state_sha256", "optimizer_sha256", "generator_sha256"):
                assert snap[key] == point[key]
            for split, modulo in (("selector", 1), ("report", 3)):
                pred = snap["observations"][split]["prediction"]
                seen = [c for g in manifest["stage_classes"][:manifest["stage"] + 1] for c in g]
                active = ((source["original_ids"] % 10 == modulo)
                          & torch.isin(source["y"], torch.tensor(seen)))
                assert torch.equal(pred["original_ids"], source["original_ids"][active])
                assert torch.equal(pred["target"], source["y"][active])
                assert pred["stage_classes"] == manifest["stage_classes"]
                assert pred["stage"] == manifest["stage"] and pred["split"] == split
                assert pred["logits"].shape == (int(active.sum()), sum(map(len, manifest["stage_classes"])))
                assert torch.isfinite(pred["logits"]).all()
                metrics = normalized(M.score_logits(pred["logits"], pred["target"].numpy(),
                                                     manifest["stage_classes"], manifest["stage"]))
                assert metrics == point[split] == normalized(snap["observations"][split]["metrics"])
        if data is not None:
            model = R.G.CLIPStudent(data["x"].shape[1]).to(device)
            model.load_state_dict(saved["state"])
            observed = R.R6.observe(model, data, manifest["stage"])
            for split in ("selector", "report"):
                final = saved["snapshots"][-1]["observations"][split]
                current, previous = observed[split]["prediction"]["logits"], final["prediction"]["logits"]
                assert torch.allclose(current, previous, rtol=R.RTOL, atol=R.ATOL)
                assert normalized(observed[split]["metrics"]) == normalized(final["metrics"])
                errors.append(float((current - previous).abs().max()))
            reloaded += 1
            del model
        indexed[name, seed] = row
    assert set(indexed) == {(n, s) for n in entries for s in SEEDS}
    for seed in SEEDS:
        assert len({indexed[n, seed]["start_state_sha256"] for n in entries}) == 1
    return indexed, reloaded, max(errors, default=0.)


def summarize_cell(manifest, fits, selection):
    entries = {i["name"]: i for i in manifest["candidates"]}
    names = list(entries)
    metrics = ("cumulative_f1", "old_f1", "new_f1")
    def values(name, metric="cumulative_f1"):
        return np.asarray([fits[name, s]["curve"][-1]["report"]["all"][metric] for s in SEEDS])
    def gain(name, metric="cumulative_f1"):
        return paired(values(name, metric) - values("c0", metric))
    scores = {n: float(np.mean([selection["scores"][n][str(s)]["score"] for s in R.SEL])) for n in names}
    reports = {n: float(values(n).mean()) for n in names}
    selected = selection["selection"]["selected"]
    oracle = max(names, key=lambda n: (reports[n], n == "c0", n))
    geometry = min(names, key=lambda n: (entries[n]["geometry_error"], n != "c0", n))
    base_build = manifest["shared_blocks"][manifest["stage"]]["build_seconds"]
    base_append = entries["c0"]["append_seconds"]
    base_train = float(np.mean([fits["c0", s]["training_seconds"] for s in SEEDS]))
    chosen_train = float(np.mean([fits[selected, s]["training_seconds"] for s in SEEDS]))
    gradient_seconds = sum(x["wall_seconds"] for x in selection["gradient_runs"])
    accounted = (base_build + manifest["candidate_build_seconds"] + sum(i["append_seconds"] for i in entries.values())
                 + gradient_seconds + selection["gradient_score_seconds"] + chosen_train)
    cell = {k: manifest[k] for k in ("dataset", "order", "stage", "stage_classes")}
    cell.update(rows=entries["c0"]["rows"], candidate_count=len(entries), selection=selection["selection"],
                gain=gain(selected), old_gain=gain(selected, "old_f1"), new_gain=gain(selected, "new_f1"),
                report={m: float(values(selected, m).mean()) for m in metrics},
                c0={m: float(values("c0", m).mean()) for m in metrics},
                ranking=rank([scores[n] for n in names], [reports[n] for n in names]),
                oracle=oracle, oracle_headroom=reports[oracle] - reports["c0"],
                regret=reports[oracle] - reports[selected])
    cell["candidates"] = {n: {"report": {m: float(values(n, m).mean()) for m in metrics}, "gain": gain(n),
        "score": scores[n], "coverage_gain": selection["scores"][n][str(R.SEL[0])]["coverage_gain"],
        "dynamic_distance": float(np.mean([selection["scores"][n][str(s)]["dynamic_distance_mean"] for s in R.SEL])),
        "geometry_error": entries[n]["geometry_error"], "payload_bytes": entries[n]["payload"]["logical_payload_bytes"],
        "singleton_new_rows": entries[n]["singleton_new_rows"], "changed_slots": len(entries[n]["changed_slots"])} for n in names}
    cell["controls"] = {"geometry_only": geometry, "geometry_gain": gain(geometry)["mean"],
        "uniform_all_gain": float(np.mean([gain(n)["mean"] for n in names])),
        "uniform_non_c0_gain": float(np.mean([gain(n)["mean"] for n in names if n != "c0"])) if len(names) > 1 else 0.,
        "q_vs_geometry_selected_gain": gain(selected)["mean"] - gain(geometry)["mean"]}
    cell["cost"] = {"gradient_runs": len(selection["gradient_runs"]),
        "gradient_optimizer_updates": 2 * len(selection["gradient_runs"]),
        "gradient_observations": 3 * len(selection["gradient_runs"]), "gradient_seconds": gradient_seconds,
        "score_seconds": selection["gradient_score_seconds"], "shared_c0_block_seconds": base_build,
        "candidate_build_seconds": manifest["candidate_build_seconds"], "single_student_seconds": chosen_train,
        "c0_single_student_seconds": base_train, "accounted_single_use_seconds": accounted,
        "accounted_c0_seconds": base_build + base_append + base_train,
        "accounted_ratio_to_c0": accounted / (base_build + base_append + base_train),
        "selection_wall_with_artifact_io_seconds": selection["wall_including_io_seconds"],
        "scope": "current block + candidate construction + all appends + gradient/score + one student; excludes shared preprocessing/history, audit/report evaluation and artifact I/O"}
    cell["diagnostic_training_seconds"] = sum(f["training_seconds"] for f in fits.values())
    cell["diagnostic_evaluation_seconds"] = sum(f["evaluation_seconds"] for f in fits.values())
    cell["diagnostic_optimizer_steps"] = sum(f["steps"] for f in fits.values())
    return cell


def decision(cells, audit_complete):
    assert len(cells) == 8
    mean_gain = float(np.mean([c["gain"]["mean"] for c in cells]))
    rho = float(np.mean([c["ranking"]["spearman"] for c in cells]))
    confirmation = {d: float(np.mean([c["gain"]["mean"] for c in cells
                    if c["dataset"] == d and c["order"] == "confirmation"])) for d in ("cora", "citeseer")}
    gates = {"ranking": rho >= .3, "selected_gain": mean_gain >= .5,
             "new_order_stability": min(confirmation.values()) >= 0.,
             "retention": all(c["gain"]["mean"] >= -2. and c["old_gain"]["mean"] >= -3. for c in cells),
             "cost_le_2x": all(c["cost"]["accounted_ratio_to_c0"] <= 2. for c in cells),
             "artifact_audit": bool(audit_complete)}
    return {"continue_v7b": all(gates.values()), "gates": gates, "selected_mean_gain": mean_gain,
        "mean_spearman": rho, "confirmation_gains": confirmation,
        "accepted_cells": sum(c["selection"]["accepted"] for c in cells),
        "mean_regret": float(np.mean([c["regret"] for c in cells])),
        "geometry_mean_gain": float(np.mean([c["controls"]["geometry_gain"] for c in cells])),
        "scope": "eight counterfactual cells; reused report-dev; no recursive or final-test promotion"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--reload-final", action="store_true")
    parser.add_argument("--reload-gradients", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-name", default="audit_summary.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root)
    cells, total, reloads, gradient_reloads, max_logit, max_gradient = [], 0, 0, 0, 0., 0.
    started = time.perf_counter()
    for dataset in ("cora", "citeseer"):
        folder = root / dataset
        freeze, reg = read(folder / "freeze.json"), read(folder / "registration.json")
        assert freeze["code_sha256"] == R.source_hashes()
        assert freeze["protocol_sha256"] == C.sha(args.protocol) == reg["protocol_sha256"]
        assert reg["freeze_sha256"] == C.sha(folder / "freeze.json")
        assert reg["selection_groups"] == [list(g) for g in V.SELECTION_GROUPS]
        assert reg["report_seeds"] == SEEDS and not set(R.SEL) & set(SEEDS)
        assert R.R5.canonical_groups(reg["orders"]["confirmation"]) not in {
            R.R5.canonical_groups(g) for g in reg["excluded_groups"]}
        expected, detail = R.R5.matched_order(reg["order_selection"]["pool_class_counts"],
            reg["orders"]["canonical"], {R.R5.canonical_groups(g) for g in reg["excluded_groups"]})
        assert expected == reg["orders"]["confirmation"] and detail == reg["order_selection"]
        prefix = read(folder / "prefix_audit.json")
        assert prefix["status"] == "passed" and prefix["gradient"]["gradient_prefix_matches_common_student"]
        assert prefix["student_curve"]["legacy_student_bitwise_equal"]
        assert C.sha(folder / "source_audit.pt") == read(folder / "source_audit.json")["sha256"]
        source = torch.load(folder / "source_audit.pt", map_location="cpu", weights_only=False)
        assert (source["y"][source["original_ids"] % 10 == 2] == -1).all()
        data = R.R5.prepare(dataset, "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
                             args.device, folder) if args.reload_final else None
        dataset_count = 0
        for order, groups in reg["orders"].items():
            if data is not None:
                R.O.set_stages(data, groups)
            for stage in (1, 2):
                target = folder / order / f"stage{stage}"
                manifest = read(target / "candidates.json")
                assert (manifest["dataset"], manifest["order"], manifest["stage"]) == (dataset, order, stage)
                assert manifest["stage_classes"] == groups
                memories = A6.audit_memory(target, manifest, source)
                audit_geometry(target, manifest, source)
                selected, n, error = audit_gradients(target, manifest, memories, args.device, args.reload_gradients)
                gradient_reloads += n
                max_gradient = max(max_gradient, error)
                fits, n, error = audit_fits(target, manifest, source, data, args.device)
                reloads += n
                max_logit = max(max_logit, error)
                dataset_count += len(fits)
                cells.append(summarize_cell(manifest, fits, selected))
                print(json.dumps({"audited": f"{dataset}/{order}/stage{stage}", "fits": len(fits)}), flush=True)
                del memories
        completed = read(folder / "completion.json")
        assert completed["status"] == "complete" and completed["fits"] == dataset_count
        assert completed["freeze_sha256"] == C.sha(folder / "freeze.json") and not completed["test_labels_evaluated"]
        total += dataset_count
        del data
    summary = {"status": "passed", "fits": total, "checkpoint_observations": total * 8,
        "final_model_logit_reloads": reloads, "gradient_reloads": gradient_reloads,
        "max_reload_logit_abs_error": max_logit, "max_reload_gradient_abs_error": max_gradient,
        "cells": cells, "decision": decision(cells, args.reload_final and args.reload_gradients),
        "auditor_sha256": C.sha(__file__), "inherited_auditor_sha256": C.sha(A6.__file__),
        "audit_wall_seconds": time.perf_counter() - started, "test_evaluated": False,
        "inference": "seed intervals descriptive, not population CI; independent seeds/order, reused report-dev nodes"}
    C.save_json(root / args.output_name, summary)
    print(json.dumps(summary["decision"]), flush=True)


if __name__ == "__main__":
    main()
