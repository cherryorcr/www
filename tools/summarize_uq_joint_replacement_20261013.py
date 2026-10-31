"""Audit the one-cell joint-distribution replacement without selecting on test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DATASETS = ("arxiv", "sportsfit", "bookhis")
METHOD = "uq_dm_edgefree"
CHANGE = {"lambda_distribution": (1.0, 0.0),
          "lambda_joint_distribution": (0.0, 1.0)}
SCORES = ("final_f1", "accuracy", "old_f1", "new_f1")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(dataset: str, root: Path):
    cp, rp = root / f"{dataset}_candidate.json", root / f"{dataset}_reference.json"
    candidate, reference = read(cp), read(rp)
    c, r = candidate["protocol"], reference["protocol"]
    require(candidate["status"] == reference["status"] == "complete", f"{dataset}: incomplete")
    excluded = {"arms", "runner_hash", "joint_replacement", "source_hashes"}
    require({k: v for k, v in c.items() if k not in excluded} ==
            {k: v for k, v in r.items() if k not in excluded}, f"{dataset}: protocol differs")
    require(set(c["source_hashes"]) - set(r["source_hashes"]) ==
            {"unified_q_pair_distribution.py"} and
            all(c["source_hashes"][key] == value for key, value in r["source_hashes"].items()),
            f"{dataset}: shared source differs")
    require(c["arms"] == ["uq_joint_replace"] and METHOD in r["arms"], f"{dataset}: arms")
    require(c["joint_replacement"]["changes"] ==
            {k: list(v) for k, v in CHANGE.items()}, f"{dataset}: declared changes")
    base, joint = c["q_config"], c["joint_replacement"]["config"]
    require({k: (base[k], joint[k]) for k in base if base[k] != joint[k]} == CHANGE,
            f"{dataset}: effective constructor config differs")
    require(r["q_config"] == base, f"{dataset}: reference constructor config")
    require(c["order_seed"] == 9101 and c["construction_seed"] == 271006 and
            c["student_seeds"] == [150, 151, 152], f"{dataset}: cell identity")
    cumulative, expected = 0, []
    for count in c["stage_raw_rows"]:
        cumulative += count
        expected.append(round(cumulative * .05))
    require(c["budget_mode"] == "fixed_rate_current_legal_pool" and
            c["memory_rate"] == .05 and c["budgets"] == expected and
            abs(c["actual_final_rate"] - c["budgets"][-1] / cumulative) < 1e-12,
            f"{dataset}: memory schedule")
    require(c["method_selection_from_test"] is False and c["official_reproduction"] is False,
            f"{dataset}: test/official flag")
    cr = {row["stage"]: row for row in candidate["rows"]}
    rr = {row["stage"]: row for row in reference["rows"] if row["arm"] == METHOD}
    require(set(cr) == set(rr) == set(range(len(c["budgets"]))), f"{dataset}: stages")
    stage_rows = []
    for stage in sorted(cr):
        a, b = cr[stage], rr[stage]
        require(a["arm"] == "uq_joint_replace" and
                all(a[k] == b[k] for k in ("stage", "raw_rows", "new_rows", "memory_rows")),
                f"{dataset}/{stage}: stage or budget differs")
        require(a["memory_rows"] == c["budgets"][stage] and
                a["new_rows"] == a["memory_rows"] - (cr[stage-1]["memory_rows"] if stage else 0),
                f"{dataset}/{stage}: memory accounting")
        for row in (a, b):
            require(all(row["checks"].get(k) is True for k in
                        ("budget_ok", "prefix_ok", "finite_ok", "sinkhorn_feasible")),
                    f"{dataset}/{stage}: failed constraints")
            require(row["audit"]["historical_raw_reads"] == 0 and
                    row["audit"]["old_update_count"] == 0 and
                    not row["audit"]["uses_q_topology"] and
                    not row["audit"]["uses_boundary_sketch"], f"{dataset}/{stage}: access/topology")
            require(set(row["scores"]) == {"report_dev", "sealed_test"},
                    f"{dataset}/{stage}: split missing")
            for split in row["scores"].values():
                require([x["seed"] for x in split["per_student"]] == c["student_seeds"],
                        f"{dataset}/{stage}: Student seeds differ")
        diag = a["audit"]
        grad = diag.get("trace", [{}])[0].get("joint_distribution_q_gradient_norm")
        gap = diag.get("text_hard_forward_max_error")
        require(grad is not None and grad > 0 and gap is not None and gap < 1e-5,
                f"{dataset}/{stage}: joint gradient or text hard-forward diagnostic")
        require(diag["joint_pair_distribution"]["enabled"] and
                not diag["distribution_matching"]["enabled"], f"{dataset}/{stage}: objective")
        stage_rows.append(dict(dataset=dataset, stage=stage, memory_rows=a["memory_rows"],
                               joint_q_gradient=grad, hard_forward_error=gap,
                               q_entropy=diag.get("q_entropy"),
                               unique_text_ratio=diag.get("unique_text_ratio"),
                               peak_gib=a["construction_peak_mb"] / 1024,
                               construction_s=a["construction_seconds"],
                               joint_initial_loss=diag["joint_pair_distribution"].get("initial_loss"),
                               joint_final_loss=diag["joint_pair_distribution"].get("final_loss")))
    final = []
    for split in ("report_dev", "sealed_test"):
        a, b = cr[max(cr)]["scores"][split], rr[max(rr)]["scores"][split]
        final.append(dict(dataset=dataset, split=split, order_seed=c["order_seed"],
                          construction_seed=c["construction_seed"], student_seeds="150/151/152",
                          **{f"{key}_{arm}": item[key] for arm, item in
                             (("joint", a), ("ef", b)) for key in SCORES},
                          **{f"delta_{key}": a[key] - b[key] for key in SCORES},
                          joint_construction_s=sum(x["construction_seconds"] for x in cr.values()),
                          ef_construction_s=sum(x["construction_seconds"] for x in rr.values()),
                          joint_peak_gib=max(x["construction_peak_mb"] for x in cr.values()) / 1024,
                          ef_peak_gib=max(x["construction_peak_mb"] for x in rr.values()) / 1024,
                          candidate_sha256=digest(cp), reference_sha256=digest(rp)))
    return final, stage_rows


def csv_out(path: Path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] /
                        "results/uq_joint_replacement_20261013")
    args = parser.parse_args()
    final, stages = [], []
    for dataset in DATASETS:
        f, s = audit(dataset, args.root)
        final.extend(f)
        stages.extend(s)
    csv_out(args.root / "paired_results.csv", final)
    csv_out(args.root / "stage_audit.csv", stages)
    lines = ["# Joint-distribution replacement: independent audit", "",
             "Post-hoc, report-development candidate; one matched order/construction cell per dataset,",
             "three Student seeds per cell. This is not a blind sealed confirmation or a new main method.",
             "The effective change is marginal DM weight 1 -> 0 and paired product-kernel",
             "joint DM weight 0 -> 1. Identical current-delta legal pools, masks, stage IDs, rounded 5%",
             "budget schedules, Student, seeds, core source hashes and frozen-prefix interface",
             "were asserted against the original confirmation JSONs. The runner differs by design.", "",
             "| Dataset | Dev joint F1 | Dev EF F1 | Delta F1 (pp) | Joint Acc | EF Acc | Delta Acc | Joint Old/New | EF Old/New | Construct s (joint/EF) | Peak GiB (joint/EF) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |"]
    for row in final:
        if row["split"] != "report_dev":
            continue
        lines.append("| {dataset} | {final_f1_joint:.2f} | {final_f1_ef:.2f} | {delta_final_f1:+.2f} | {accuracy_joint:.2f} | {accuracy_ef:.2f} | {delta_accuracy:+.2f} | {old_f1_joint:.2f}/{new_f1_joint:.2f} | {old_f1_ef:.2f}/{new_f1_ef:.2f} | {joint_construction_s:.1f}/{ef_construction_s:.1f} | {joint_peak_gib:.2f}/{ef_peak_gib:.2f} |".format(**row))
    lines += ["", "Sealed-test readings below are *post-hoc transparency only*. This candidate followed",
              "earlier negative additive paired-MMD report-dev results, so test observations must",
              "not select its weight or justify claiming an independent held-out improvement.", "",
              "| Dataset | Sealed joint F1 | Sealed EF F1 | Delta F1 (pp) | Joint Acc | EF Acc |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in final:
        if row["split"] == "sealed_test":
            lines.append("| {dataset} | {final_f1_joint:.2f} | {final_f1_ef:.2f} | {delta_final_f1:+.2f} | {accuracy_joint:.2f} | {accuracy_ef:.2f} |".format(**row))
    lines += ["", "All stages passed budget, frozen-prefix, finite and Sinkhorn checks; historical",
              "raw reads and old-memory updates were zero. Every stage had nonzero initial",
              "joint-loss gradient into Q and hard-text forward error below 1e-5; see",
              "`stage_audit.csv` for per-stage values. These checks establish that the joint",
              "objective ran, not that it improved transfer. The one-cell BookHis regression",
              "rules out promoting this replacement now. No significance interval is calculated",
              "from three Student seeds in one construction cell. Previous additive paired-MMD",
              "control also declined on BookHis, SportsFit and WikiCS:",
              "`results/uq_gecc_gap_20261009/summary.md` (different development protocol).",
              "Original sealed confirmation tables and paper method are not changed.", ""]
    (args.root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.root / "summary.md")


if __name__ == "__main__":
    main()
