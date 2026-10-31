"""Separate external comparisons from same-run component ablations."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from summarize_unified_q_validation_20261007 import pair_protocol


DATASETS = ("wikics", "sportsfit", "bookhis")
EXTERNAL = ("uq_dm_edgefree", "cat_cctp", "gecc_hard", "puma_tag",
            "random_coreset", "kcenter_coreset")
ABLATIONS = ("uq_dm_edgefree", "uq_no_dm", "uq_no_prop", "uq_no_residual",
             "uq_frozen_text", "uq_detached_text", "uq_no_text_init",
             "uq_no_q_anchor", "uq_no_x_anchor", "uq_dm_only")
NAMES = {
    "uq_dm_edgefree": "In-loop CaT-Q (full)",
    "cat_cctp": "CaT-CCTP adapter",
    "gecc_hard": "GECC-hard TAG adapter",
    "puma_tag": "PUMA-inspired TAG adapter",
    "random_coreset": "Random coreset",
    "kcenter_coreset": "Graph-text k-center",
    "tagsam_response_adapter": "TAGSAM-response adapter",
    "uq_no_dm": "w/o CaT DM",
    "uq_no_prop": "w/o propagation loss",
    "uq_no_residual": "w/o X residual",
    "uq_frozen_text": "Frozen Q0 text routing",
    "uq_detached_text": "Hard text, stop text gradient",
    "uq_no_text_init": "w/o text initialization",
    "uq_no_q_anchor": "w/o Q anchor",
    "uq_no_x_anchor": "w/o residual penalty (bound kept)",
    "uq_dm_only": "DM-only simplification",
}
METRICS = ("final_f1", "accuracy", "stage_f1", "stage_acc", "old_f1", "new_f1", "worst_stage")


def read_cell(data, arm, path):
    protocol = data["protocol"]
    rows = sorted((r for r in data["rows"] if r["arm"] == arm), key=lambda r: r["stage"])
    if [r["stage"] for r in rows] != list(range(len(protocol["groups"]))):
        return None
    for row in rows:
        if row["memory_rows"] != protocol["budgets"][row["stage"]]:
            raise ValueError("budget mismatch")
        if not all(row["checks"].get(k) for k in ("budget_ok", "prefix_ok", "finite_ok")):
            raise ValueError("memory checks failed")
        if arm.startswith("uq_") and not row["checks"].get("sinkhorn_feasible"):
            raise ValueError("assignment infeasible")
        score = row["score"]
        if [s["seed"] for s in score["per_student"]] != protocol["student_seeds"]:
            raise ValueError("student seed mismatch")
        for metric in ("final_f1", "accuracy", "old_f1", "new_f1"):
            values = [s[metric] for s in score["per_student"]]
            if metric == "old_f1" and row["stage"] == 0:
                if any(v is not None for v in values):
                    raise ValueError("base stage cannot have old F1")
                continue
            if any(not math.isfinite(v) for v in values) or abs(mean(values)-score[metric]) > 1e-8:
                raise ValueError("incorrect metric mean")
    final = rows[-1]["score"]
    return dict(dataset=protocol["dataset"], arm=arm,
        order_seed=protocol["order_seed"], construction_seed=protocol["construction_seed"],
        protocol=protocol, source=str(path.resolve()),
        **{key:final[key] for key in ("final_f1", "accuracy", "old_f1", "new_f1")},
        stage_f1=mean(r["score"]["final_f1"] for r in rows),
        stage_acc=mean(r["score"]["accuracy"] for r in rows),
        worst_stage=min(r["score"]["final_f1"] for r in rows),
        student_f1_sd=final["final_f1_std"], student_acc_sd=final["accuracy_std"],
        memory_rows=rows[-1]["memory_rows"], per_student=final["per_student"],
        construction_seconds=sum(r["construction_seconds"] for r in rows),
        student_seconds=sum(r["student_seconds"] for r in rows),
        peak_mb=max(r["construction_peak_mb"] or 0 for r in rows),
        memory_hash=rows[-1]["memory_hash"],
        activation=[dict(stage=r["stage"],
            q_updates=r["audit"].get("q_update_count"),
            r_updates=r["audit"].get("residual_update_count"),
            old_updates=r["audit"].get("old_update_count"),
            cat_warm_start=r["audit"].get("cat_anchor_used"),
            topology_enabled=r["audit"].get("uses_q_topology"),
            text_assignment=r["audit"].get("text_assignment"),
            text_error=r["audit"].get("text_hard_forward_max_error"),
            residual_ratio=r["audit"].get("x_residual_ratio_max"),
            unique_text_count=r["audit"].get("unique_text_count"),
            new_rows=r.get("new_rows"),
            sampled_q_grad_min=min((t["q_gradient_norm"] for t in r["audit"].get("trace",[])),default=None),
            sampled_r_grad_min=min((t["residual_gradient_norm"] for t in r["audit"].get("trace",[])),default=None))
            for r in rows if r.get("audit",{}).get("constructor")=="unified_q_multimodal"],
        numerical_warnings=[dict(stage=r["stage"], checks=r["checks"])
                            for r in rows if r["checks"].get("numerical_warning")])


def t_interval(values):
    # This matrix has three paired order means (or three paired Student
    # seeds for conditional diagnostics). Never treat 3x3 as 9 independent.
    if len(values) != 3:
        raise ValueError("three paired units required")
    center = mean(values)
    radius = 4.302652729911275 * stdev(values) / math.sqrt(3)
    return [center-radius, center+radius]


def key(cell):
    return (cell["dataset"], cell["order_seed"], cell["construction_seed"], cell["arm"])


def split_incomplete(pending, completed):
    unresolved, recovered = [], []
    for attempt in pending:
        destination = completed.get(key(attempt))
        if destination is None:
            unresolved.append(attempt)
        else:
            recovered.append(dict(attempt, completed_source=destination["source"]))
    return unresolved, recovered


def collect(root, archived_path):
    archived = json.loads(archived_path.read_text(encoding="utf-8"))
    old = {key(c): c for c in archived["cells"]}
    new = {}
    pending = []
    partial_files = []
    for path in sorted(root.glob("*/results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "complete":
            partial_files.append(str(path))
        for arm in data["protocol"]["arms"]:
            if arm not in NAMES:
                raise ValueError(f"undeclared method {arm}")
            cell = read_cell(data, arm, path)
            if cell is None:
                protocol = data["protocol"]
                pending.append(dict(source=str(path), arm=arm,
                    dataset=protocol["dataset"], order_seed=protocol["order_seed"],
                    construction_seed=protocol["construction_seed"]))
                continue
            if key(cell) in new:
                raise ValueError(f"duplicate completed cell; do not select the better repeat: {key(cell)}")
            new[key(cell)] = cell
    pending, recovered = split_incomplete(pending, new)
    # The new full-method replay belongs only to the new ablation table.
    external = {k:c for k,c in old.items() if k[2] == 271006 and k[3] in EXTERNAL}
    external.update({k:c for k,c in new.items() if c["arm"] in EXTERNAL[1:]})
    aggregates, pairs, missing_external = [], [], []
    for dataset in DATASETS:
        ref = [external.get((dataset,o,271006,"uq_dm_edgefree")) for o in (9101,9102,9103)]
        for arm in EXTERNAL:
            cells = [external.get((dataset,o,271006,arm)) for o in (9101,9102,9103)]
            if any(c is None for c in cells):
                missing_external.append(dict(dataset=dataset, arm=arm))
                continue
            for candidate, control in zip(cells, ref):
                pair_protocol(candidate["protocol"], control["protocol"])
            aggregates.append(dict(dataset=dataset, arm=arm, n_orders=3,
                **{m:mean(c[m] for c in cells) for m in METRICS},
                f1_order_sd=stdev(c["final_f1"] for c in cells),
                acc_order_sd=stdev(c["accuracy"] for c in cells),
                memory_rows=cells[0]["memory_rows"],
                construction_seconds=mean(c["construction_seconds"] for c in cells),
                student_seconds=mean(c["student_seconds"] for c in cells),
                sources=[c["source"] for c in cells]))
            if arm != "uq_dm_edgefree":
                df = [a["final_f1"]-b["final_f1"] for a,b in zip(ref,cells)]
                da = [a["accuracy"]-b["accuracy"] for a,b in zip(ref,cells)]
                pairs.append(dict(dataset=dataset, comparison=f"main - {arm}",
                    delta_f1=mean(df), ci95_f1=t_interval(df), delta_accuracy=mean(da),
                    ci95_accuracy=t_interval(da), positive_orders=sum(v>0 for v in df),
                    worst_delta=min(df)))
    ablations, missing_ablations = [], []
    for dataset in DATASETS:
        main = new.get((dataset,9101,271006,"uq_dm_edgefree"))
        for arm in ABLATIONS:
            cell = new.get((dataset,9101,271006,arm))
            if main is None or cell is None:
                missing_ablations.append(dict(dataset=dataset, arm=arm))
                continue
            pair_protocol(cell["protocol"], main["protocol"])
            df = [a["final_f1"]-b["final_f1"] for a,b in zip(cell["per_student"],main["per_student"])]
            ablations.append(dict(cell, delta_f1=cell["final_f1"]-main["final_f1"],
                delta_accuracy=cell["accuracy"]-main["accuracy"],
                conditional_student_ci95_f1=t_interval(df)))
    tagsam=[]
    for dataset in DATASETS:
        cell = new.get((dataset,9101,271006,"tagsam_response_adapter"))
        if cell:
            pair_protocol(cell["protocol"], old[(dataset,9101,271006,"uq_dm_edgefree")]["protocol"])
            tagsam.append(cell)
    return dict(output_root=str(root.resolve()),
        external=aggregates, external_pairs=pairs, ablations=ablations,
        tagsam=tagsam, new_cells=list(new.values()),
        archived_no_gm=[p for p in archived["paired"]
                        if p["candidate"] == "uq_dm_edgefree_no_gm" and p["control"] == "uq_dm_edgefree"],
        archived_fixed_q=[c for c in archived["cells"] if c["arm"]=="uq_dm_edgefree_fixed_q"],
        archived_reference=[c for c in archived["cells"] if c["arm"] in EXTERNAL
                            and c["order_seed"]==9101 and c["construction_seed"]==271006],
        started_incomplete=pending, recovered_attempts=recovered,
        partial_files=partial_files,
        missing_external=missing_external, missing_ablations=missing_ablations,
        audit=dict(new_complete_cells=len(new), expected_new_core_cells=54,
                   expected_tagsam_cells=3,
                   numeric_warning_stages=sum(len(c["numerical_warnings"]) for c in new.values()),
                   seal_test_used=False, main_structure_changed=False))


def render(result):
    lines = ["# In-loop CaT-Q：外部基线与组件消融汇总", "", "日期：2026-10-08。数值单位为百分数；全部为 report-dev，不是封存 test。", "",
        "## 1. 完成范围", "", f"本批完整方法 cell：{result['audit']['new_complete_cells']}。预注册核心矩阵 54 cell，TAGSAM 额外 3 cell。",
        f"外部三顺序表缺失方法：{len(result['missing_external'])}；新消融缺失 cell：{len(result['missing_ablations'])}。",
        f"TAGSAM 适配完成 {len(result['tagsam'])}/3；未补齐的已开始 arm：{len(result['started_incomplete'])}；保留但已由新目录补齐的 arm：{len(result['recovered_attempts'])}。",
        "每个 cell 为五阶段、三个相同 Student seeds。原失败文件保留；只纳入完整 arm，不把失败前的阶段均值当最终结果。", "",
        "## 2. 实际启用的主方法", "",
        "Q=class-masked Sinkhorn(Z)；X=mass-normalized Q.T@raw_X+R；文本采用真实 top-1 前向与软池化直通反向。",
        "有效目标：GM + DM + 0.5 propagation + 0.1 Q-anchor + 0.1 R-penalty；R 每行限制在当前 pooled X 范数的 5%。",
        "CaT 预训练 anchor、old anchor、旧 memory 更新、boundary、合成边、EM 都未启用。配置中保留某个系数不代表该项实际非零。",
        "跨类别的 -1e4 仅是 Z 占位；最终 Q 跨类别为精确 0。梯度更新 Z/R，不更新字符串或 MiniLM。", "",
        "## 3. 外部方法与适配", "",
        "| 原方法 | 本表名称 | 适配内容 | 状态 |", "|---|---|---|---|",
        "| CaT | CaT-CCTP | 当前 delta 内 class-wise MMD 优化自由 X；top-5 文本；冻结前缀 | 机制适配 |",
        "| GECC | GECC-hard TAG | hop 传播、hard Lloyd/restarts、附着文本；不回读历史 raw | 非完整 fuzzy 官方实现 |",
        "| PUMA | PUMA-inspired TAG | pool 内 withheld 子集模拟伪标签、一次传播、筛选与凝聚、附着文本 | 非官方完整实现；监督分配有改动 |",
        "| TAGSAM | TAGSAM-response adapter | 直接调用固定官方 commit 的 wBCELoss；10-step unroll；新 X/学习率优化；旧前缀冻结；同一最终 Student | 构造 proxy/初始化/预算是适配，不是官方原样复现 |",
        "| Random / k-center | Coreset | 同样类别配额，选择真实 X/text，self-loop memory | 通用参考 |",
        "CaT 的多种文本附着规则不是多篇独立方法；no-GM 等也不属于外部基线。", "",
        "## 4. 外部基线：三顺序", "",
        "所有行使用 9101/9102/9103，construction seed=271006。先平均三个 Student seeds，再计算三个顺序的 mean±SD。不能与以前 3×3 构造确认均值混为同一表。",
        "不同顺序采用同一参考 memory rows；相同 pool/report/stage hash、Student 与训练预算均逐项配对核对。",
        "各方法内部构造预算遵循各自声明，不强行等同为相同训练成本；时间可能受共享 GPU 影响。", ""]
    for dataset in DATASETS:
        lines += [f"### {dataset}", "", "| 方法 | Final F1±SD | Acc±SD | Stage F1 | Stage Acc | Old F1 | New F1 | 平均最差阶段 F1 | Rows |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c in result["external"]:
            if c["dataset"] == dataset:
                lines.append(f"| {NAMES[c['arm']]} | {c['final_f1']:.2f}±{c['f1_order_sd']:.2f} | {c['accuracy']:.2f}±{c['acc_order_sd']:.2f} | {c['stage_f1']:.2f} | {c['stage_acc']:.2f} | {c['old_f1']:.2f} | {c['new_f1']:.2f} | {c['worst_stage']:.2f} | {c['memory_rows']} |")
        lines += [""]
    lines += ["### 配对差值", "", "95% CI 是基于三个顺序均值的配对 t 区间，样本很少；不把 Student seeds 当额外顺序。", "",
        "| 数据集 | main − baseline | F1 差值 [95% CI] | Acc 差值 | 正收益顺序 | 最差顺序差值 |", "|---|---|---:|---:|---:|---:|"]
    for p in result["external_pairs"]:
        lines.append(f"| {p['dataset']} | {p['comparison']} | {p['delta_f1']:+.2f} [{p['ci95_f1'][0]:+.2f}, {p['ci95_f1'][1]:+.2f}] | {p['delta_accuracy']:+.2f} | {p['positive_orders']}/3 | {p['worst_delta']:+.2f} |")
    lines += ["", "## 5. TAGSAM 补充参考顺序", "", "本段只有一个顺序和构造 seed，SD 来自三个 Student seeds，不能和上面的顺序 SD 混用。", "",
        "| 数据集 | 方法 | F1±Student SD | Accuracy±Student SD | Old F1 | New F1 | Build s | Student s |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        cells = [c for c in result["archived_reference"] if c["dataset"]==dataset and c["arm"]=="uq_dm_edgefree"]
        cells += [c for c in result["tagsam"] if c["dataset"]==dataset]
        for c in cells:
            lines.append(f"| {dataset} | {NAMES[c['arm']]} | {c['final_f1']:.2f}±{c['student_f1_sd']:.2f} | {c['accuracy']:.2f}±{c['student_acc_sd']:.2f} | {c['old_f1']:.2f} | {c['new_f1']:.2f} | {c['construction_seconds']:.1f} | {c['student_seconds']:.1f} |")
    lines += ["", "## 6. 组件消融：重新构造完整链", "", "本批各消融与本批 full 比较；旧同 seed 的 full 不混入配对。Δ=消融−full，正数表示删除/替换后更好。只有参考顺序/一个构造 seed，不证明普遍可删除。", ""]
    for dataset in DATASETS:
        lines += [f"### {dataset}", "", "| 版本 | F1±Student SD | Accuracy±Student SD | ΔF1 | ΔAcc | Stage F1 | Old F1 | New F1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c in result["ablations"]:
            if c["dataset"] == dataset:
                lines.append(f"| {NAMES[c['arm']]} | {c['final_f1']:.2f}±{c['student_f1_sd']:.2f} | {c['accuracy']:.2f}±{c['student_acc_sd']:.2f} | {c['delta_f1']:+.2f} | {c['delta_accuracy']:+.2f} | {c['stage_f1']:.2f} | {c['old_f1']:.2f} | {c['new_f1']:.2f} |")
        lines += [""]
    lines += ["### 已有多顺序 no-GM 证据", "", "以下为已有 3 顺序×3 构造 seeds 的配对确认，不是本批单 cell 重跑。", "", "| 数据集 | no-GM − full F1 | 95% CI |", "|---|---:|---:|"]
    for p in result["archived_no_gm"]:
        lines.append(f"| {p['dataset']} | {p['delta_final_f1']:+.2f} | [{p['ci95_final_f1'][0]:+.2f}, {p['ci95_final_f1'][1]:+.2f}] |")
    lines += ["", "### 已有固定 Q 对照", "",
        "只与同一旧批次的参考 full 配对，不与本批 full 混合。此对照冻结 Q，因而同时限制 X 聚合和文本路由，不能把差异单独归因为文本梯度。",
        "| 数据集 | 旧批 full F1 | 固定 Q F1 | 固定 Q Acc | 固定 Q − full F1 |", "|---|---:|---:|---:|---:|"]
    references={key(c):c for c in result["archived_reference"]}
    for c in result["archived_fixed_q"]:
        reference=references.get((c["dataset"],c["order_seed"],c["construction_seed"],"uq_dm_edgefree"))
        if reference is None:
            continue
        pair_protocol(c["protocol"],reference["protocol"])
        lines.append(f"| {c['dataset']} | {reference['final_f1']:.2f} | {c['final_f1']:.2f} | {c['accuracy']:.2f} | {c['final_f1']-reference['final_f1']:+.2f} |")
    lines += ["", "no-GM 仍有 DM/propagation/anchor 对 Q/R 的梯度，但失去直接依赖文本的 GM 梯度。删 GM 后不能继续声称显式多模态梯度匹配。",
        "DM-only 同时删除多个损失，仅是整体简化诊断；不能将其收益归因于某一个删项。",
        "冻结路由与 stop-text-gradient 不同：前者前向也固定 Q0 文本，后者前向继续随 Q 硬选，只切断反向文本路径。",
        "去文本初始化不等于完全无文本；类别 prompt 和后续 GM 文本分支仍在。", "",
        "## 7. 审计和限制", "",
        f"新结果中带 full-matrix 浮点容差警告的阶段：{result['audit']['numeric_warning_stages']}。类内 Sinkhorn 阈值仍为 1e-4；零填充全矩阵求和允许另加 8×float32 epsilon，并记录原始误差。",
        "首次 WikiCS 去 Q-anchor 在汇总矩阵误差检查停止，未产生最终指标；保留日志并在新目录补跑，不根据 F1 重试。原失败未保存误差数值，不能声称已精确确认其数值根因。",
        "本批 compact runner 只保存 JSON/hash，不保存完整 memory。旧批次有 checkpoint 审计；本批 hash 不能恢复输入，未来正式封存实验需恢复完整归档。",
        "current-delta-only 是构造函数输入边界与代码审计，不是对整个进程系统调用的隔离证明：评估器仍在完整原图上推理。历史 raw 访问计数字段不能被解释为独立的 OS 级追踪证据。",
        "旧批次与本批同 seed CUDA 构造仍可能有数值/非确定性差异，因此本批 full 只作为本批消融参考，不合并成额外独立确认样本。",
        "TAGSAM-response 是官方损失适配，不是原样官方运行。STAD 官方文本生成需要 Qwen 配置；当前未完成其原生流水线。不能把缺失记成 0，也不能声称已补齐所有公开强基线。",
        "官方原生 TAGSAM 的编码器、全图 buffer、按 5 类分组评估与当前协议不同；native 结果若后续运行，只能单列 reference，不参与严格持续排名。",
        "仍需简化候选独立顺序/构造 seed 确认，才能冻结删项；不能按每个数据集挑赢家拼接主方法。", ""]
    lines += ["### 完整主方法的实际更新", "",
        "以下来自本批完整链的构造审计。梯度下界仅覆盖保存的 trace 采样步骤，不宣称覆盖所有优化步骤。",
        "| 数据集 | 阶段数 | Q/R 每阶段更新次数 | 旧 X 更新次数 | 最小采样 Q 梯度范数 | 最小采样 R 梯度范数 | 最大文本前向误差 | 最大 R 相对范数 |",
        "|---|---:|---|---:|---:|---:|---:|---:|"]
    for c in result["ablations"]:
        if c["arm"] != "uq_dm_edgefree":
            continue
        records=c["activation"]
        if not records:
            continue
        updates=sorted(set((r["q_updates"],r["r_updates"]) for r in records))
        lines.append(f"| {c['dataset']} | {len(records)} | {updates} | {sum(r['old_updates'] for r in records)} | {min(r['sampled_q_grad_min'] for r in records):.3g} | {min(r['sampled_r_grad_min'] for r in records):.3g} | {max(r['text_error'] for r in records):.3g} | {max(r['residual_ratio'] for r in records):.6f} |")
    lines += ["", "### 实测成本", "",
        "外部主表三顺序的平均完整链时间，包含每阶段三个 Student seeds。Student 时间包含训练与评估；共享设备时不能据此给出严格独占加速比。",
        "| 数据集 | 方法 | 构造 s | Student 总训练/评估 s |",
        "|---|---|---:|---:|"]
    for c in result["external"]:
        lines.append(f"| {c['dataset']} | {NAMES[c['arm']]} | {c['construction_seconds']:.1f} | {c['student_seconds']:.1f} |")
    root=Path(result["output_root"]).as_posix()
    lines += ["", "## 8. 可视化", "",
        "外部条形图误差线为三个顺序均值的 SD；消融图为同批次消融减 full。",
        f"![External Macro-F1]({root}/external_final_f1.png)", "",
        f"![External Accuracy]({root}/external_accuracy.png)", "",
        f"![Ablation Macro-F1]({root}/ablation_f1.png)", "",
        f"![Ablation Accuracy]({root}/ablation_accuracy.png)", ""]
    return "\n".join(lines)


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size":9, "savefig.dpi":160})
    for metric,label in (("final_f1","Macro-F1 (%)"),("accuracy","Accuracy (%)")):
        fig, axes = plt.subplots(1,3,figsize=(14,5.4),layout="constrained")
        for axis,dataset in zip(axes,DATASETS):
            cells=[c for c in result["external"] if c["dataset"]==dataset]
            colors=["#007f73" if c["arm"]=="uq_dm_edgefree" else "#90969c" for c in cells]
            axis.barh(range(len(cells)),[c[metric] for c in cells],color=colors,
                xerr=[c["f1_order_sd" if metric=="final_f1" else "acc_order_sd"] for c in cells],capsize=2)
            for y,c in enumerate(cells):
                sd=c["f1_order_sd" if metric=="final_f1" else "acc_order_sd"]
                axis.text(c[metric]+sd+1.2,y,f"{c[metric]:.2f}",va="center",fontsize=7.5)
            axis.set_yticks(range(len(cells)),[NAMES[c["arm"]] for c in cells],fontsize=8)
            axis.invert_yaxis(); axis.set_xlim(0,100); axis.set_title(dataset); axis.set_xlabel(label)
        fig.suptitle("Same protocol: 3 class orders, 1 construction seed, 3 Student seeds")
        fig.savefig(output/f"external_{metric}.png"); plt.close(fig)
    for suffix,field,label in (("f1","delta_f1","Macro-F1"),
                               ("accuracy","delta_accuracy","Accuracy")):
        values=np.full((len(ABLATIONS)-1,3),np.nan)
        for r in result["ablations"]:
            if r["arm"] != "uq_dm_edgefree":
                values[ABLATIONS[1:].index(r["arm"]),DATASETS.index(r["dataset"])]=r[field]
        fig,axis=plt.subplots(figsize=(9.5,6.4),layout="constrained")
        limit=max(1,float(np.nanmax(abs(values)))) if np.isfinite(values).any() else 1
        image=axis.imshow(values,aspect="auto",cmap="RdBu",vmin=-limit,vmax=limit)
        axis.set_xticks(range(3),DATASETS); axis.set_yticks(range(len(ABLATIONS)-1),[NAMES[a] for a in ABLATIONS[1:]])
        for y in range(values.shape[0]):
            for x in range(3):
                if np.isfinite(values[y,x]):
                    axis.text(x,y,f"{values[y,x]:+.2f}",ha="center",va="center",
                        color="white" if abs(values[y,x]) > .6*limit else "black")
        axis.set_title(f"Component ablation minus same-batch full: {label} pp\nOne order/construction seed; positive is better after ablation")
        fig.colorbar(image,ax=axis,label=f"{label} percentage points")
        fig.savefig(output/f"ablation_{suffix}.png"); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("results/unified_q_matrix_20261008"))
    parser.add_argument("--archive",type=Path,default=Path("results/unified_q_inloop_validation_20261007/summary.json"))
    args=parser.parse_args()
    result=collect(args.root,args.archive)
    args.root.mkdir(parents=True,exist_ok=True)
    (args.root/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (args.root/"summary.md").write_text(render(result),encoding="utf-8")
    fields=["dataset","arm","order_seed","construction_seed",*METRICS,
            "student_f1_sd","student_acc_sd","memory_rows","construction_seconds","student_seconds","peak_mb","source"]
    with (args.root/"cells.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader(); writer.writerows(result["new_cells"])
    plot(result,args.root)
    print(json.dumps(result["audit"],indent=2))


if __name__=="__main__":
    main()
