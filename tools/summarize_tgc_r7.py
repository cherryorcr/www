"""Verify sealed R7 results and summarize paired effects without selecting runs."""
import argparse
import itertools
import json
from pathlib import Path
import statistics as st

from summarize_tgc_r4_relational import read, digest, interval, csv_file


def ranks(values):
    return [1 + sum(x < v for x in values) + (sum(x == v for x in values)-1)/2 for v in values]


def correlation(a, b):
    return st.correlation(a, b) if len(a) > 1 and len(set(a)) > 1 and len(set(b)) > 1 else None


def summarize(root):
    spec, seal = read(root / "protocol.json"), read(root / "selection_seal.json")
    assert read(root / "pipeline_status.json")["status"] == "complete"
    seal_hash = digest(root / "selection_seal.json")
    assert seal["protocol_sha256"] == digest(root / "protocol.json")
    for path, sha in seal["decisions"].items():
        assert digest(root / path) == sha, path
    rows, cells, diagnostics, audited, costs = [], [], [], [], []
    for dataset, order in itertools.product(spec["datasets"], spec["orders"]):
        folder = root / dataset / order
        audit, manifest = read(folder / "audit.json"), read(folder / "manifest.json")
        assert audit["status"] == "passed" and audit["maximum_metric_error_pp"] < 1e-7
        audited.append({"dataset": dataset, "order": order, **audit})
        for stage in spec["evaluation_stages"]:
            for arm in spec.get("stage_arms", {}).get(str(stage), spec["arms"]):
                group = []
                for seed in spec["report_seeds"]:
                    item = read(folder / f"{arm}_s{stage}_seed{seed}.json")
                    base = read(folder / f"c0_s{stage}_seed{seed}.json")
                    assert item["selection_seal_sha256"] == seal_hash
                    assert item["start_state_sha256"] == base["start_state_sha256"]
                    assert item["steps"] == base["steps"] and item["rows"] == manifest["budgets"][stage]
                    m, b = item["metrics"]["all"], base["metrics"]["all"]
                    value = {"dataset": dataset, "order": order, "stage": stage, "arm": arm, "seed": seed,
                             "f1": m["cumulative_f1"], "old_f1": m["old_f1"], "new_f1": m["new_f1"],
                             "delta_f1": m["cumulative_f1"]-b["cumulative_f1"],
                             "delta_old": m["old_f1"]-b["old_f1"] if stage else None,
                             "delta_new": m["new_f1"]-b["new_f1"],
                             "executed": item["executed_here"], "seconds": item["seconds"],
                             "checkpoint_sha256": item["checkpoint_sha256"], "source_sha256": digest(folder / f"{arm}_s{stage}_seed{seed}.json")}
                    rows.append(value)
                    group.append(value)
                ci = interval([v["delta_f1"] for v in group])
                cell = {"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                        **{key: st.mean(v[key] for v in group) if all(v[key] is not None for v in group) else None
                           for key in ("f1", "old_f1", "new_f1", "delta_f1", "delta_old", "delta_new", "seconds")},
                        "f1_sd": st.stdev(v["f1"] for v in group), "paired_ci_low": ci[0], "paired_ci_high": ci[1]}
                cells.append(cell)
                path = folder / f"{arm}_s{stage}_build_audit.json"
                if path.exists():
                    build = read(path)
                    diag = {"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                            "f1": cell["f1"], "seconds": build.get("seconds"),
                            "actual_reconstruction": build.get("actual_reconstruction"),
                            **{k: build.get("response", {}).get(k) for k in ("gradient_cosine", "relation_kl", "legal_one_step_gain")}}
                    diagnostics.append(diag)
                    if "recursive" in spec["name"]:
                        costs.append({"dataset": dataset, "order": order, "stage": stage, "arm": arm,
                            "build_seconds": build["seconds"] + (build.get("c0_block_seconds", 0.) if stage else 0.),
                            "student_seconds": cell["seconds"], "logical_bytes": build["payload"]["logical_bytes"],
                            "edge_entries": build["payload"]["edge_entries"],
                            "text_utf8_bytes": build["payload"]["text_utf8_bytes"]})
        complete = read(folder / "completion.json")
        selected = [v for v in rows if v["dataset"] == dataset and v["order"] == order]
        assert len(selected) == complete["fit_slots"] == audit["fit_slots"]
        assert sum(v["executed"] for v in selected) == complete["executed_fits"]
    methods = []
    for arm in spec["arms"]:
        final = [v for v in cells if v["arm"] == arm and v["stage"] == max(spec["evaluation_stages"])]
        if not final:
            continue
        all_stages = [v for v in cells if v["arm"] == arm]
        methods.append({"arm": arm, "f1": st.mean(v["f1"] for v in final),
                        "delta_f1": st.mean(v["delta_f1"] for v in final),
                        "delta_old": st.mean(v["delta_old"] for v in final),
                        "delta_new": st.mean(v["delta_new"] for v in final),
                        "stage_mean_delta": st.mean(v["delta_f1"] for v in all_stages),
                        "worst_delta": min(v["delta_f1"] for v in final),
                        "positive_conditions": sum(v["delta_f1"] > 0 for v in final),
                        "conditions": len(final), "cell_descriptive_ci": interval([v["delta_f1"] for v in final])})
    repeated = []
    historical = root.parent / "tgc_r5_joint_20260923"
    if "r5_frozen" in spec["arms"]:
        for dataset, order in itertools.product(spec["datasets"], spec["orders"]):
            for arm, original in (("c0", "c0"), ("r5_frozen", "recursive")):
                diffs = []
                for seed in spec["report_seeds"]:
                    now = read(root / dataset / order / f"{arm}_s2_seed{seed}.json")
                    before = read(historical / dataset / order / f"{original}_s2_seed{seed}.json")
                    for key in ("support_memory_sha256", "start_state_sha256", "steps", "rows"):
                        assert now[key] == before[key]
                    diffs.append(now["metrics"]["all"]["cumulative_f1"]-before["metrics"]["all"]["cumulative_f1"])
                repeated.append({"dataset": dataset, "order": order, "arm": arm,
                                 "mean_pp": st.mean(diffs), "max_abs_pp": max(map(abs, diffs))})
    correlations = []
    if "alpha05" in spec["arms"]:
        for dataset, order in itertools.product(spec["datasets"], spec["orders"]):
            group = [v for v in diagnostics if v["dataset"] == dataset and v["order"] == order
                     and v["arm"] in ("dense0", "alpha0", "alpha025", "alpha05")]
            for key, direction in (("actual_reconstruction", -1), ("relation_kl", -1),
                                   ("gradient_cosine", 1), ("legal_one_step_gain", 1)):
                x, y = [direction*v[key] for v in group], [v["f1"] for v in group]
                correlations.append({"dataset": dataset, "order": order, "metric": key,
                    "spearman": correlation(ranks(x), ranks(y)), "n": len(group),
                    "top1_agreement": group[max(range(len(group)), key=lambda i: x[i])]["arm"]
                                      == group[max(range(len(group)), key=lambda i: y[i])]["arm"]})
    family_methods, family_cells, paired = [], [], []
    if "recursive" in spec["name"]:
        families = {kind: [a for a in spec["arms"] if a.startswith(kind)] for kind in ("original", "dense", "bounded")}
        families.update({a: [a] for a in ("c0", "k_center") if a in spec["arms"]})
        indexed = {(v["dataset"], v["order"], v["stage"], v["seed"], v["arm"]): v for v in rows}
        averaged = {}
        for dataset, order, stage, seed in itertools.product(spec["datasets"], spec["orders"], spec["evaluation_stages"], spec["report_seeds"]):
            for family, arms in families.items():
                parts = [indexed[dataset, order, stage, seed, a] for a in arms]
                averaged[dataset, order, stage, seed, family] = {
                    key: st.mean(v[key] for v in parts) if all(v[key] is not None for v in parts) else None
                    for key in ("f1", "old_f1", "new_f1", "delta_f1", "delta_old", "delta_new")}
        for dataset, order, stage, family in itertools.product(spec["datasets"], spec["orders"], spec["evaluation_stages"], families):
            parts = [averaged[dataset, order, stage, seed, family] for seed in spec["report_seeds"]]
            family_cells.append({"dataset": dataset, "order": order, "stage": stage, "family": family,
                **{key: st.mean(v[key] for v in parts) if all(v[key] is not None for v in parts) else None
                   for key in parts[0]}, "construction_groups": len(families[family])})
        for family, arms in families.items():
            final = [v for v in family_cells if v["family"] == family and v["stage"] == 2]
            evaluated = [v for v in family_cells if v["family"] == family]
            bill = [v for v in costs if v["arm"] in arms and v["stage"] > 0]
            family_methods.append({"family": family,
                **{key: st.mean(v[key] for v in final) for key in ("f1", "delta_f1", "delta_old", "delta_new")},
                "stage_mean_delta": st.mean(v["delta_f1"] for v in evaluated),
                "worst_delta": min(v["delta_f1"] for v in final),
                "worst_old_delta": min(v["delta_old"] for v in final),
                "positive_conditions": sum(v["delta_f1"] > 0 for v in final),
                "mean_build_seconds": st.mean(v["build_seconds"] for v in bill),
                "mean_student_seconds": st.mean(v["student_seconds"] for v in bill),
                "mean_final_logical_bytes": st.mean(v["logical_bytes"] for v in bill if v["stage"] == 2),
                "cora_f1": st.mean(v["f1"] for v in final if v["dataset"] == "cora"),
                "citeseer_f1": st.mean(v["f1"] for v in final if v["dataset"] == "citeseer")})
        for candidate, baseline in (("bounded", "c0"), ("bounded", "dense"), ("bounded", "original"),
                                    ("dense", "c0"), ("original", "c0"), ("bounded", "k_center")):
            if baseline not in families:
                continue
            for dataset, order, stage in itertools.product(spec["datasets"], spec["orders"], (1, 2)):
                item = {"dataset": dataset, "order": order, "stage": stage, "candidate": candidate, "baseline": baseline}
                for metric in ("f1", "old_f1", "new_f1"):
                    differences = [averaged[dataset, order, stage, seed, candidate][metric]
                                   -averaged[dataset, order, stage, seed, baseline][metric] for seed in spec["report_seeds"]]
                    ci = interval(differences)
                    item.update({metric + "_delta": st.mean(differences), metric + "_ci_low": ci[0], metric + "_ci_high": ci[1]})
                paired.append(item)
    comparisons = {f"{d}/{o}": read(root / d / o / "construction_comparisons.json")
                   for d, o in itertools.product(spec["datasets"], spec["orders"])
                   if (root / d / o / "construction_comparisons.json").exists()}
    result = {"protocol_sha256": digest(root / "protocol.json"), "selection_seal_sha256": seal_hash,
              "fit_slots": len(rows), "executed_fits": sum(v["executed"] for v in rows),
              "audits": audited, "methods": methods, "construction_comparisons": comparisons,
              "frozen_student_repeats": repeated, "correlations": correlations,
              "family_methods": family_methods,
              "student_actual_seconds": sum(v["seconds"] for v in rows if v["executed"]),
              "ci_scope": "paired student seeds within fixed condition; across-order intervals descriptive; no best-realization selection",
              "diagnostics_only": "tracing and response-probe time is not deployed algorithm cost"}
    for name, values in (("per_seed", rows), ("cells", cells), ("diagnostics", diagnostics), ("correlations", correlations),
                         ("family_cells", family_cells), ("family_contrasts", paired), ("costs", costs)):
        if values:
            csv_file(root / f"{name}.csv", values)
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("fit_slots", "executed_fits", "methods", "correlations", "family_methods")}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    summarize(parser.parse_args().root)
