"""Independent artifact, prediction, selection and cost audit for frozen v6-A."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy.stats import spearmanr, t
import torch

import run_v6_local_marginal as R

V, C, M = R.V, R.C, R.M
SEL = sum((list(g) for g in V.SELECTION_GROUPS), [])
SEEDS = SEL + list(V.REPORT_SEEDS)
RELOAD_RTOL = 1e-4
RELOAD_ATOL = 1e-5


def read(path):
    return json.loads(Path(path).read_text())


def normalized(value):
    return json.loads(json.dumps(value))


def paired(values):
    values = np.asarray(values, dtype=float)
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.
    margin = float(t.ppf(.975, len(values) - 1) * se) if len(values) > 1 else 0.
    mean = float(values.mean())
    return {"mean": mean, "seed_sd": float(values.std(ddof=1)),
            "descriptive_95pct_t_interval": [mean - margin, mean + margin], "values": values.tolist()}


def rank(left, right):
    constant = len(set(left)) < 2 or len(set(right)) < 2
    return {"spearman": 0. if constant else float(spearmanr(left, right).statistic),
            "constant_rank": constant,
            "top1_agreement": int(int(np.argmax(left)) == int(np.argmax(right)))}


def row_equal(a, b, index):
    return a[index] == b[index] if isinstance(a, list) else torch.equal(a[index], b[index])


def audit_memory(target, manifest, source):
    old_path = target / manifest["old_history_filename"]
    assert C.sha(old_path) == manifest["old_history_sha256"]
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    old_n = len(old["x"])
    stage, groups = manifest["stage"], manifest["stage_classes"]
    pool, labels = source["pool_mask"], source["y"]
    ids = torch.where(pool & torch.isin(labels, torch.tensor(groups[stage])))[0]
    assert ids.tolist() == manifest["source_ids"]
    seen = [c for g in groups[:stage + 1] for c in g]
    budget = int(.05 * int((pool & torch.isin(labels, torch.tensor(seen))).sum()))
    assert not torch.isin(source["original_ids"][pool] % 10, torch.tensor([1, 2, 3])).any()
    position = torch.full_like(labels, -1)
    position[pool] = torch.arange(int(pool.sum()))
    memories, records_by_name = {}, {}
    for item in manifest["candidates"]:
        assert C.sha(target / item["filename"]) == item["file_sha256"]
        assert C.sha(target / item["sources_filename"]) == item["sources_sha256"]
        memory = torch.load(target / item["filename"], map_location="cpu", weights_only=False)
        records = torch.load(target / item["sources_filename"], map_location="cpu", weights_only=False)
        assert R.nested_hash(memory) == item["memory_sha256"]
        C.assert_frozen(old, memory)
        assert len(memory["x"]) == budget == item["rows"]
        V.validate_partition({"pool_mask": pool, "y": labels}, ids, records, groups[stage])
        assert len(records) == budget - old_n
        assert sum(len(r["ids"]) == 1 for r in records) == item["singleton_new_rows"]
        quotas = {str(cls): sum(r["class"] == cls for r in records) for cls in groups[stage]}
        assert quotas == item["allocation"]
        for slot, record in enumerate(records):
            expected = source["pool_x"][position[record["ids"]]].float().mean(0)
            assert torch.equal(memory["x"][old_n + slot], expected)
            assert memory["mass"][old_n + slot] == len(record["ids"])
            assert memory["stage"][old_n + slot] == stage
        assert normalized(R.O.payload_stats(memory)) == item["payload"]
        memories[item["name"]], records_by_name[item["name"]] = memory, records
    base = memories["c0"]
    base_records = records_by_name["c0"]
    for item in manifest["candidates"]:
        name = item["name"]
        records = records_by_name[name]
        actual_changed = [i for i, (a, b) in enumerate(zip(base_records, records))
                          if a["class"] != b["class"] or not torch.equal(a["ids"], b["ids"])]
        assert actual_changed == item["changed_slots"]
        for i in set(range(len(records))) - set(actual_changed):
            assert all(row_equal(memories[name][k], base[k], old_n + i) for k in V.FIELDS)
        if item["family"] == "local":
            assert len(actual_changed) == 3
            other = records_by_name[name.replace("local_", "recluster_", 1)]
            assert [r["class"] for r in other] == [r["class"] for r in records]
    return memories


def audit_fits(target, manifest, source, data=None, device="cpu"):
    results = read(target / "results.json")
    assert results["status"] == "complete"
    assert results["candidates_sha256"] == C.sha(target / "candidates.json")
    assert results["registration_sha256"] == C.sha(target.parent.parent / "registration.json")
    entries = {i["name"]: i for i in manifest["candidates"]}
    indexed = {}
    reloaded = 0
    reload_errors = []
    for row in results["fits"]:
        name, seed = row["candidate"], row["seed"]
        key = (name, seed)
        assert key not in indexed and seed in SEEDS and name in entries
        item = entries[name]
        assert row == read(target / f"{name}_seed{seed}.json")
        assert row["memory_sha256"] == item["memory_sha256"]
        path = target / row["checkpoint"]
        assert C.sha(path) == row["checkpoint_sha256"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert C.tensor_hash(checkpoint["state"]) == row["end_state_sha256"]
        assert [x["step"] for x in row["curve"]] == V.horizons(R.P.student_steps(item["rows"]))
        assert row["curve"][-1]["state_sha256"] == row["end_state_sha256"]
        for snap, point in zip(checkpoint["snapshots"], row["curve"]):
            for field in ("step", "state_sha256", "optimizer_sha256", "generator_sha256"):
                assert snap[field] == point[field]
            for split, modulo in (("selector", 1), ("report", 3)):
                prediction = snap["observations"][split]["prediction"]
                seen = [c for g in manifest["stage_classes"][:manifest["stage"] + 1] for c in g]
                active = ((source["original_ids"] % 10 == modulo)
                          & torch.isin(source["y"], torch.tensor(seen)))
                assert torch.equal(prediction["original_ids"], source["original_ids"][active])
                assert torch.equal(prediction["target"], source["y"][active])
                assert prediction["stage_classes"] == manifest["stage_classes"]
                assert prediction["stage"] == manifest["stage"] and prediction["split"] == split
                logits = prediction["logits"]
                assert logits.shape == (int(active.sum()), len(source["y"].unique()))
                assert torch.isfinite(logits).all()
                score = normalized(M.score_logits(logits, prediction["target"].numpy(),
                                                   manifest["stage_classes"], manifest["stage"]))
                assert score == point[split] == normalized(snap["observations"][split]["metrics"])
        if data is not None:
            model = R.G.CLIPStudent(data["x"].shape[1]).to(device)
            model.load_state_dict(checkpoint["state"])
            observed = R.observe(model, data, manifest["stage"])
            for split in ("selector", "report"):
                previous = checkpoint["snapshots"][-1]["observations"][split]
                current = observed[split]["prediction"]["logits"]
                saved_logits = previous["prediction"]["logits"]
                delta = (current - saved_logits).abs()
                assert torch.allclose(current, saved_logits, rtol=RELOAD_RTOL, atol=RELOAD_ATOL)
                reload_errors.append({"candidate": name, "seed": seed, "split": split,
                    "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                    "max_relative": float((delta / (saved_logits.abs() + 1e-8)).max())})
            reloaded += 1
            del model
        indexed[key] = row
    assert set(indexed) == {(n, s) for n in entries for s in SEEDS}
    for seed in SEEDS:
        assert len({indexed[(n, seed)]["start_state_sha256"] for n in entries}) == 1
    return indexed, reloaded, reload_errors


def summarize_cell(manifest, fits):
    entries = {i["name"]: i for i in manifest["candidates"]}
    def mean(name, seeds, split="report", horizon=3, metric="cumulative_f1"):
        return float(np.mean([fits[name, s]["curve"][horizon][split]["all"][metric] for s in seeds]))
    def gains(name, metric="cumulative_f1"):
        return [fits[name, s]["curve"][3]["report"]["all"][metric]
                - fits["c0", s]["curve"][3]["report"]["all"][metric] for s in V.REPORT_SEEDS]
    def maximum(names, values):
        return max(names, key=lambda n: (values[n], n == "c0", n))
    cell = {k: manifest[k] for k in ("dataset", "order", "stage", "stage_classes")}
    cell["candidate_count"] = len(entries)
    cell["c0"] = {m: mean("c0", V.REPORT_SEEDS, metric=m) for m in ("cumulative_f1", "old_f1", "new_f1")}
    cell["rows"] = entries["c0"]["rows"]
    cell["candidates"] = {n: {
        "family": item["family"], "geometry_error": item["geometry_error"],
        "changed_slots": len(item["changed_slots"]), "singleton_new_rows": item["singleton_new_rows"],
        "report": {m: mean(n, V.REPORT_SEEDS, metric=m) for m in ("cumulative_f1", "old_f1", "new_f1")},
        "gain": paired(gains(n)), "payload_bytes": item["payload"]["logical_payload_bytes"]}
        for n, item in entries.items()}
    cell["families"] = {}
    for family in V.FAMILIES:
        names = ["c0"] + sorted(n for n, item in entries.items() if item["family"] == family)
        observations = {n: {s: [p["selector"]["all"] for p in fits[n, s]["curve"]] for s in SEL} for n in names}
        selection = V.select_paired_prefix(observations, names)
        selected = selection["selected"]
        full_sel = {n: mean(n, SEL, "selector") for n in names}
        full_report = {n: mean(n, V.REPORT_SEEDS) for n in names}
        geometry = min(names, key=lambda n: (entries[n]["geometry_error"], n != "c0", n))
        random_seed = 6400 + manifest["stage"] + 100 * (manifest["order"] == "confirmation") + 1000 * (manifest["dataset"] == "citeseer")
        random_name = random.Random(random_seed).choice(names[1:]) if len(names) > 1 else "c0"
        oracle = maximum(names, full_report)
        full_winner = maximum(names, full_sel)
        def ranked(a_seeds, a_split, a_horizon, b_seeds, b_split, b_horizon):
            return rank([mean(n, a_seeds, a_split, a_horizon) for n in names],
                        [mean(n, b_seeds, b_split, b_horizon) for n in names])
        ranks = {
            "half_selector_to_independent_full_report": ranked(SEL, "selector", 2, V.REPORT_SEEDS, "report", 3),
            "half_to_full_same_seed_selector": ranked(SEL, "selector", 2, SEL, "selector", 3),
            "full_same_seed_selector_to_report": ranked(SEL, "selector", 3, SEL, "report", 3),
            "full_report_selection_to_independent_seeds": ranked(SEL, "report", 3, V.REPORT_SEEDS, "report", 3)}
        prefix_seconds = sum(fits[n, s]["curve"][2]["training_seconds"]
            + sum(sum(fits[n, s]["curve"][h]["observation_timing"][k]
                      for k in ("forward_seconds", "selector_seconds")) for h in (1, 2))
            for n in names for s in SEL)
        student_seconds = float(np.mean([fits[selected, s]["training_seconds"] for s in V.REPORT_SEEDS]))
        c0_student = float(np.mean([fits["c0", s]["training_seconds"] for s in V.REPORT_SEEDS]))
        shared_build = manifest["shared_blocks"][manifest["stage"]]["build_seconds"]
        cost = {"selection_fits": len(names) * len(SEL),
                "selection_optimizer_steps": len(names) * len(SEL) * fits[selected, SEL[0]]["curve"][2]["step"],
                "selection_half_prefix_seconds": prefix_seconds,
                "single_student_seconds": student_seconds, "c0_single_student_seconds": c0_student,
                "current_shared_block_build_seconds": shared_build,
                "both_family_candidate_build_seconds": manifest["candidate_build_seconds"],
                "accounted_single_use_seconds": shared_build + manifest["candidate_build_seconds"] + prefix_seconds + student_seconds,
                "scope": "estimated single use: current C0 block + both-family proposal construction (overcounts chosen family) + four-seed half-prefix selection + one final student; append/serialization/preprocessing excluded; not full lifecycle cost"}
        cost["accounted_ratio_to_c0"] = cost["accounted_single_use_seconds"] / (shared_build + c0_student)
        cell["families"][family] = {"selection": selection, "report": cell["candidates"][selected]["report"],
            "gain": paired(gains(selected)), "old_gain": paired(gains(selected, "old_f1")),
            "new_gain": paired(gains(selected, "new_f1")), "ranks": ranks, "cost": cost,
            "oracle": oracle, "oracle_headroom": full_report[oracle] - full_report["c0"],
            "regret": full_report[oracle] - full_report[selected],
            "controls": {"geometry": geometry, "geometry_gain": float(np.mean(gains(geometry))),
                "random": random_name, "random_gain": float(np.mean(gains(random_name))),
                "uniform_gain": float(np.mean([np.mean(gains(n)) for n in names])),
                "full_selector": full_winner, "full_selector_gain": float(np.mean(gains(full_winner)))}}
    cell["diagnostic_training_seconds"] = sum(f["training_seconds"] for f in fits.values())
    cell["diagnostic_evaluation_seconds"] = sum(f["evaluation_seconds"] for f in fits.values())
    cell["diagnostic_optimizer_steps"] = sum(f["steps"] for f in fits.values())
    return cell


def decision(cells):
    local = [c["families"]["local"] for c in cells]
    confirm = [c for c in cells if c["order"] == "confirmation"]
    mean_gain = float(np.mean([x["gain"]["mean"] for x in local]))
    geometry = float(np.mean([x["controls"]["geometry_gain"] for x in local]))
    uniform = float(np.mean([x["controls"]["uniform_gain"] for x in local]))
    rho = float(np.mean([x["ranks"]["half_selector_to_independent_full_report"]["spearman"] for x in local]))
    confirmation_gains = {d: float(np.mean([c["families"]["local"]["gain"]["mean"]
        for c in confirm if c["dataset"] == d])) for d in ("cora", "citeseer")}
    oracle_cells = sum(c["families"]["local"]["oracle_headroom"] >= 1. for c in confirm)
    accepted_confirmation = sum(c["families"]["local"]["selection"]["accepted"] for c in confirm)
    gates = {"oracle_headroom": oracle_cells >= 2, "ranking": rho >= .3,
             "selected_gain_and_controls": mean_gain >= .5 and mean_gain >= geometry and mean_gain >= uniform,
             "new_order_stability": min(confirmation_gains.values()) >= 0. and accepted_confirmation >= 1,
             "retention": all(x["gain"]["mean"] >= -2. and x["old_gain"]["mean"] >= -3. for x in local),
             "artifact_audit": True}
    return {"continue_v6b": all(gates.values()), "gates": gates,
            "oracle_cells_ge_1pp": oracle_cells, "mean_spearman": rho,
            "selected_mean_gain": mean_gain, "geometry_mean_gain": geometry, "uniform_mean_gain": uniform,
            "confirmation_gains": confirmation_gains, "accepted_confirmation_cells": accepted_confirmation,
            "scope": "counterfactual stage screen; not a recursive continual-method promotion"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--reload-final", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-name", default="audit_summary.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root)
    cells, total, reloaded, reload_errors = [], 0, 0, []
    started = time.perf_counter()
    for dataset in ("cora", "citeseer"):
        folder = root / dataset
        freeze, registration = read(folder / "freeze.json"), read(folder / "registration.json")
        assert freeze["code_sha256"] == R.source_hashes()
        assert freeze["protocol_sha256"] == C.sha(args.protocol) == registration["protocol_sha256"]
        assert registration["freeze_sha256"] == C.sha(folder / "freeze.json")
        assert registration["selection_groups"] == [list(g) for g in V.SELECTION_GROUPS]
        assert registration["report_seeds"] == list(V.REPORT_SEEDS)
        assert registration["orders"]["confirmation"] not in registration["excluded_groups"]
        prefix = read(folder / "prefix_audit.json")
        assert prefix["status"] == "passed" and prefix["legacy_student_bitwise_equal"]
        assert len(prefix["checks"]) == 4 and all(x["bitwise_equal"] for x in prefix["checks"])
        assert C.sha(folder / "source_audit.pt") == read(folder / "source_audit.json")["sha256"]
        source = torch.load(folder / "source_audit.pt", map_location="cpu", weights_only=False)
        data = R.R5.prepare(dataset, "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
                            args.device, folder) if args.reload_final else None
        dataset_count = 0
        for order, groups in registration["orders"].items():
            if data is not None:
                R.O.set_stages(data, groups)
            for stage in (1, 2):
                target = folder / order / f"stage{stage}"
                manifest = read(target / "candidates.json")
                assert (manifest["dataset"], manifest["order"], manifest["stage"]) == (dataset, order, stage)
                assert manifest["stage_classes"] == groups
                audit_memory(target, manifest, source)
                fits, count, errors = audit_fits(target, manifest, source, data, args.device)
                reloaded += count
                reload_errors.extend(errors)
                dataset_count += len(fits)
                cells.append(summarize_cell(manifest, fits))
                print(json.dumps({"audited": f"{dataset}/{order}/stage{stage}", "fits": len(fits)}), flush=True)
        completion = read(folder / "completion.json")
        assert completion["status"] == "complete" and completion["fits"] == dataset_count
        assert completion["freeze_sha256"] == C.sha(folder / "freeze.json")
        assert completion["test_labels_evaluated"] is False
        total += dataset_count
        del data
    summary = {"status": "passed", "fits": total, "checkpoint_observations": total * 4 * 2,
               "final_model_logit_reloads": reloaded, "cells": cells, "decision": decision(cells),
               "reload_logit_tolerance": {"rtol": RELOAD_RTOL, "atol": RELOAD_ATOL},
               "reload_error_summary": ({"count": len(reload_errors),
                   "max_abs": max((x["max_abs"] for x in reload_errors), default=0.),
                   "mean_abs": float(np.mean([x["mean_abs"] for x in reload_errors]))
                   if reload_errors else 0.} if reload_errors else None),
               "auditor_sha256": C.sha(__file__), "audit_wall_seconds": time.perf_counter() - started,
               "test_evaluated": False, "inference": "seed intervals descriptive; reused report-dev; no literature novelty claim"}
    C.save_json(root / args.output_name, summary)
    print(json.dumps(summary["decision"]), flush=True)


if __name__ == "__main__":
    main()
