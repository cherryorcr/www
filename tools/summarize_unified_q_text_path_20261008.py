"""Keep diagnostic mechanisms, external baselines and internal comparisons separate."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/uq_text_path_20261008"
ARMS = ("uq_qmedoid", "uq_dm_edgefree", "cat_cctp", "gecc_hard")
NAMES = dict(uq_qmedoid="Q-medoid (single change)", uq_dm_edgefree="Original In-loop CaT-Q",
             cat_cctp="CaT + nearest5 (adapted)", gecc_hard="GECC-hard + TAG (adapted)")
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1")


def paired_summary(values, orders, seeds):
    matrix = np.asarray([[values[(o, s)] for s in seeds] for o in orders], dtype=float)
    rng = np.random.default_rng(8102026)
    oo = rng.integers(len(orders), size=(10000, len(orders)))
    ss = rng.integers(len(seeds), size=(10000, len(seeds)))
    draws = matrix[oo[:, :, None], ss[:, None, :]].mean(axis=(1, 2))
    return dict(mean=float(matrix.mean()), std=float(matrix.std(ddof=1)),
        ci95_crossed_bootstrap=np.quantile(draws, [.025, .975]).tolist(),
        positive_cells=int((matrix > 0).sum()), cells=int(matrix.size), worst_cell=float(matrix.min()),
        per_order={str(o): float(matrix[i].mean()) for i, o in enumerate(orders)},
        per_construction_seed={str(s): float(matrix[:, i].mean()) for i, s in enumerate(seeds)},
        warning="Exploratory crossed-factor bootstrap, only two order and two construction-seed levels; not a strong significance claim")


def validate_plan(cells, plan):
    expected = {(v["dataset"], v["order"], v["seed"]) for v in plan["tasks"]}
    if len(expected) != len(plan["tasks"]):
        raise AssertionError("duplicate planned cell")
    seen, order_protocols, dataset_contracts, common = set(), {}, {}, None
    for cell in cells:
        p = cell["protocol"]
        key = (p["dataset"], p["order_seed"], p["construction_seed"])
        if key not in expected or key in seen:
            raise AssertionError("unplanned or duplicate completed cell")
        seen.add(key)
        if (p["student_seeds"] != plan["student_seeds"] or p["student_steps"] != plan["steps"]
                or p["arms"] != plan["arms"] or sorted(p["students"]) != sorted(plan["students"])
                or p["max_stages"] != 5):
            raise AssertionError("confirmation protocol differs from frozen plan")
        signature = {k: p[k] for k in ("source_hashes", "runner_sha256", "students", "student_batch")}
        if common is None:
            common = signature
        elif signature != common:
            raise AssertionError("code or Student settings changed between cells")
        contract = dict(budgets=p["budgets"], group_sizes=list(map(len, p["groups"])),
                        classes=sorted(c for g in p["groups"] for c in g),
                        pool_mask_hash=p["pool_mask_hash"], report_mask_hash=p["report_mask_hash"])
        if p["dataset"] in dataset_contracts and dataset_contracts[p["dataset"]] != contract:
            raise AssertionError("fixed budget, class set or split changed between orders")
        dataset_contracts[p["dataset"]] = contract
        order_signature = {k: p[k] for k in ("groups", "budgets", "pool_mask_hash", "report_mask_hash", "stage_id_hashes")}
        order_key = key[:2]
        if order_key in order_protocols and order_protocols[order_key] != order_signature:
            raise AssertionError("data protocol changed between construction seeds")
        order_protocols[order_key] = order_signature
    return dict(expected_cells=len(expected), completed_cells=len(seen),
                missing_cells=[dict(dataset=d, order=o, seed=s) for d, o, s in sorted(expected - seen)],
                code_and_student_consistent=True, same_order_data_consistent=True,
                fixed_budget_and_split_across_orders=True)


def diagnostics(root):
    output = {}
    for path in sorted(root.glob("diagnostic_v2_*/results.json")):
        item = json.loads(path.read_text())
        if item["status"] != "complete":
            continue
        stages = item["stages"]
        probes = [v["gradients"] for s in stages for v in s["measurements"]["snapshots"]]
        keys = stages[0]["measurements"]["final"]["macro"].keys()
        output[item["protocol"]["dataset"]] = dict(
            stage_equal_mean={k: float(np.mean([s["measurements"]["final"]["macro"][k] for s in stages])) for k in keys},
            medoid_unique_source_fraction=float(np.mean([s["measurements"]["final"]["medoid_unique_source_fraction"] for s in stages])),
            mean_direct_text_dm_norm_ratio=float(np.mean([p["text_to_dm_norm_ratio"] for p in probes])),
            text_dm_negative_cosine=sum(p["text_dm_cosine"] is not None and p["text_dm_cosine"] < 0 for p in probes),
            gradient_probes=len(probes), stages=len(stages), observer=stages[0]["observer"],
            mean_step_source_switch_fraction=float(np.mean([r["changed_fraction"] for s in stages
                for r in s["measurements"]["switches"] if r["changed_fraction"] is not None])))
    return output


def collect_cells(root, audit_archives=False):
    cells, audit = [], []
    if audit_archives:
        import torch
        sys.path.insert(0, str(ROOT / "code/v2_llm_fusion"))
        from run_unified_q_revision_20261007 import memory_hash
    for path in sorted(root.glob("confirm_*/results.json")):
        cell = json.loads(path.read_text())
        if cell["status"] != "complete":
            continue
        protocol = cell["protocol"]
        if len(cell["rows"]) != 8 or len(cell["stages"]) != 20:
            raise AssertionError(f"incomplete declared complete cell: {path}")
        old_count, new_count = sum(map(len, protocol["groups"][:-1])), len(protocol["groups"][-1])
        keys = [(r["arm"], r["student"]) for r in cell["rows"]]
        if len(set(keys)) != 8:
            raise AssertionError("duplicate readout")
        changed = {k for k in protocol["q_config"] if protocol["q_config"][k] != protocol["candidate_config"][k]}
        if changed != {"text_materialization"}:
            raise AssertionError("not a single-factor method change")
        for arm in ARMS:
            stages = [s for s in cell["stages"] if s["arm"] == arm]
            if [s["stage"] for s in stages] != list(range(5)):
                raise AssertionError("stage chain incomplete")
            for stage in stages:
                for key in ("budget_ok", "prefix_ok", "finite_ok", "current_delta_text_sources", "same_class_text_sources"):
                    if not stage["checks"][key]:
                        raise AssertionError(f"failed protocol check {key}")
                if stage["memory_rows"] != protocol["budgets"][stage["stage"]]:
                    raise AssertionError("budget differs")
            rows = [r for r in cell["rows"] if r["arm"] == arm]
            if len({r["memory_hash"] for r in rows}) != 1 or rows[0]["memory_hash"] != stages[-1]["memory_hash"]:
                raise AssertionError("architectures did not see identical memory")
            if audit_archives:
                archive = path.parent / f"final_{arm}.pt.gz"
                with gzip.open(archive, "rb") as stream:
                    memory = torch.load(stream, map_location="cpu", weights_only=False)
                if memory_hash(memory) != rows[0]["memory_hash"]:
                    raise AssertionError("archived memory differs from evaluated memory")
                if memory["edge_index"].numel() or len(memory["x"]) != protocol["budgets"][-1]:
                    raise AssertionError("archive topology/budget mismatch")
                if len(memory["texts"]) != len(memory["text_source_ids"]) or max(map(len, memory["texts"])) > 320:
                    raise AssertionError("archive payload contract mismatch")
                audit.append(dict(path=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                                  memory_hash=rows[0]["memory_hash"]))
        for row in cell["rows"]:
            scores = row["score"]["per_student"]
            if [v["seed"] for v in scores] != protocol["student_seeds"]:
                raise AssertionError("Student seeds differ")
            for metric in METRICS:
                if abs(np.mean([v[metric] for v in scores]) - row["score"][metric]) > 1e-8:
                    raise AssertionError("metric aggregation differs")
            for score in scores:
                reconstructed = (old_count * score["old_f1"] + new_count * score["new_f1"]) / (old_count + new_count)
                if abs(reconstructed - score["final_f1"]) > 1e-6:
                    raise AssertionError("old/new macro decomposition differs")
        cells.append(cell)
    return cells, audit


def build(root, audit_archives=False):
    cells, audit = collect_cells(root, audit_archives)
    plan_path = root / "confirmation_feasible_plan.json"
    plan_audit = validate_plan(cells, json.loads(plan_path.read_text())) if plan_path.exists() else None
    diag = diagnostics(root)
    summary = dict(completed_cells=len(cells), expected_cells=12, diagnosis=diag, datasets={}, paired={},
                   original_vs_external={}, plan_audit=plan_audit,
                   readout_records=sum(len(c["rows"]) for c in cells),
                   student_trainings=sum(len(r["score"]["per_student"]) for c in cells for r in c["rows"]))
    for dataset in ("wikics", "sportsfit", "bookhis"):
        current = [c for c in cells if c["protocol"]["dataset"] == dataset]
        if not current:
            continue
        summary["datasets"][dataset] = {}
        for student in ("prompt_gcn", "prompt_sgc"):
            table = []
            indexed = {}
            for arm in ARMS:
                rows = [r for c in current for r in c["rows"] if r["arm"] == arm and r["student"] == student]
                table.append(dict(arm=arm, cells=len(rows),
                    **{metric: dict(mean=float(np.mean([r["score"][metric] for r in rows])),
                        std=float(np.std([r["score"][metric] for r in rows], ddof=1)) if len(rows)>1 else None) for metric in METRICS},
                    construction_seconds=float(np.mean([sum(s["construction_seconds"] for s in c["stages"] if s["arm"] == arm) for c in current])),
                    peak_mb=float(np.mean([max(s["construction_peak_mb"] for s in c["stages"] if s["arm"] == arm) for c in current])),
                    student_seconds=float(np.mean([r["student_seconds"] for r in rows])),
                    archive_bytes=float(np.mean([r["archive_bytes"] for r in rows])),
                    unique_texts=float(np.mean([r["unique_texts"] for r in rows])), memory_rows=rows[0]["memory_rows"]))
                indexed[arm] = {(c["protocol"]["order_seed"], c["protocol"]["construction_seed"]):
                    next(r["score"] for r in c["rows"] if r["arm"] == arm and r["student"] == student) for c in current}
            summary["datasets"][dataset][student] = table
            orders = sorted({k[0] for k in indexed[ARMS[0]]})
            seeds = sorted({k[1] for k in indexed[ARMS[0]]})
            if len(current) == len(orders) * len(seeds) and len(orders) > 1 and len(seeds) > 1:
                for baseline in ARMS[1:]:
                    summary["paired"][f"{dataset}/{student}/vs_{baseline}"] = {
                        m: paired_summary({k: indexed[ARMS[0]][k][m] - indexed[baseline][k][m] for k in indexed[ARMS[0]]}, orders, seeds)
                        for m in METRICS}
                for baseline in ("cat_cctp", "gecc_hard"):
                    summary["original_vs_external"][f"{dataset}/{student}/vs_{baseline}"] = {
                        m: paired_summary({k: indexed["uq_dm_edgefree"][k][m] - indexed[baseline][k][m]
                                          for k in indexed["uq_dm_edgefree"]}, orders, seeds)
                        for m in METRICS}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if audit_archives:
        (root / "archive_audit.json").write_text(json.dumps(dict(archives=len(audit), all_pass=True, rows=audit), indent=2))
    lines = ["# Q-medoid 单一修改：独立确认", "", f"完整组合：{len(cells)}/12；正式 Student 训练：{summary['student_trainings']}。",
        "", "每个方法均为完整五阶段构造；仅最终 memory 评价。报告 report-dev，不是 sealed test。",
        "", "每 cell 先平均三个 Student seeds，再对已完成 cell 等权汇总；完整矩阵每图为 2 个顺序 × 2 个构造种子。± 为 cell 标准差；单 cell 不估计标准差。",
        "不能将 Student seeds 当作独立构造；两个随机因素仅各两级，区间仅为探索性双因素 bootstrap。", "",
        "原主方法是内部 anchor，CaT/GECC 才是外部论文机制适配，均非官方完整协议复现。", ""]
    for ds, students in summary["datasets"].items():
        lines += [f"## {ds}", ""]
        for student, table in students.items():
            lines += [f"### {student}", "", "| 方法 | cells | Final F1 | Accuracy | Old F1 | New F1 | 构造秒 |", "|---|---:|---:|---:|---:|---:|---:|"]
            for row in table:
                cells_text = " | ".join((f"{row[m]['mean']:.2f} ± {row[m]['std']:.2f}" if row[m]['std'] is not None else f"{row[m]['mean']:.2f} (n=1)") for m in METRICS)
                lines.append(f"| {NAMES[row['arm']]} | {row['cells']} | {cells_text} | {row['construction_seconds']:.1f} |")
            lines += [""]
    lines += ["## 候选减原主方法", "", "| 数据集 / Student | F1 差值 | 探索性 95% CI | 正收益 cell | 最差差值 |", "|---|---:|---:|---:|---:|"]
    for key, value in summary["paired"].items():
        if not key.endswith("/vs_uq_dm_edgefree"):
            continue
        v = value["final_f1"]
        low, high = v["ci95_crossed_bootstrap"]
        lines.append(f"| {key.split('/vs_')[0]} | {v['mean']:+.2f} | [{low:+.2f}, {high:+.2f}] | {v['positive_cells']}/{v['cells']} | {v['worst_cell']:+.2f} |")
    lines += ["", "## 原主方法减外部适配基线", "", "| 数据集 / Student | 外部基线 | F1 差值 | Accuracy 差值 | F1 探索性 95% CI | F1 正收益 cell | 最差 F1 差值 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for key, value in summary["original_vs_external"].items():
        label, baseline = key.split("/vs_")
        v, a = value["final_f1"], value["accuracy"]
        low, high = v["ci95_crossed_bootstrap"]
        lines.append(f"| {label} | {NAMES[baseline]} | {v['mean']:+.2f} | {a['mean']:+.2f} | [{low:+.2f}, {high:+.2f}] | {v['positive_cells']}/{v['cells']} | {v['worst_cell']:+.2f} |")
    lines += ["", "构造时间来自共享服务器且不含首次加载；不是独占硬件加速比。SGC 更改架构，不更改文本分支/训练目标。",
        "memory rows 相同不等于字节/token 信息量相同。完整存储、成本、逐 cell 配对和诊断见 summary.json。", ""]
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def plot(root, summary):
    if not summary["datasets"]:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#c34462", "#286f9c", "#469b78", "#a27e22"]
    for metric, title in (("final_f1", "Final Macro-F1 (%)"), ("accuracy", "Final Accuracy (%)")):
        fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.5), constrained_layout=True)
        for col, ds in enumerate(("wikics", "sportsfit", "bookhis")):
            for row, student in enumerate(("prompt_gcn", "prompt_sgc")):
                ax = axes[row, col]
                table = summary["datasets"].get(ds, {}).get(student)
                if table is None:
                    ax.set_visible(False)
                    continue
                means = [v[metric]["mean"] for v in table]
                stds = [v[metric]["std"] or 0 for v in table]
                bars = ax.bar(range(4), means, yerr=stds, color=colors, capsize=3)
                ax.set_xticks(range(4), ["Q-medoid", "CaT-Q", "CaT", "GECC"], fontsize=9)
                display = dict(wikics="WikiCS", sportsfit="SportsFit", bookhis="BookHis")
                ax.set_title(f"{display[ds]} | {student.removeprefix('prompt_').upper()}")
                ax.set_ylim(0, 100)
                ax.grid(axis="y", alpha=.2)
                ax.set_axisbelow(True)
                if col == 0:
                    ax.set_ylabel(title)
                for bar, mean, std in zip(bars, means, stds):
                    ax.text(bar.get_x() + bar.get_width()/2, mean + std + 2, f"{mean:.1f}", ha="center", fontsize=9)
        fig.suptitle(title + f" | {summary['completed_cells']}/12 cells; report-dev", fontsize=13)
        fig.supxlabel("CaT-Q: original method; CaT/GECC: TAG protocol adaptations. Error bars: order/seed cell SD.", fontsize=10)
        fig.savefig(root / (metric + ".png"), dpi=170)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT)
    p.add_argument("--audit-archives", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    summary = build(args.root, args.audit_archives)
    if not args.no_plots:
        plot(args.root, summary)
    print(json.dumps(dict(completed_cells=summary["completed_cells"], student_trainings=summary["student_trainings"])))
