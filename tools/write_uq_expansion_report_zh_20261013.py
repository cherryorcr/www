"""Write dataset-first Chinese tables from the verified expansion bundle."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from summarize_uq_expansion_20261013 import DATASETS, DISPLAY, METHODS, QUALIFIED_REFERENCES, markdown


IDENTITY = {
    "proposed": "主方法",
    "mechanism adaptation": "公开机制适配",
    "native-core adaptation": "官方核心适配",
    "selection control": "选样对照",
    "inspired reference": "启发式参考",
    "budget-limited reference": "计算预算受限参考",
    "different-supervision reference": "监督/拓扑不同的参考",
}
STATUS = {"complete": "完成", "running": "运行中", "queued": "排队中",
          "failed / queued retry": "失败，待重试", "not registered": "未注册"}
ADAPTATIONS = {
    "uq_dm_edgefree": "共享 Q 联合决定特征聚合与真实文本分配，循环内 DM/GM 优化；历史前缀冻结，合成 memory 无显式边。",
    "cat_cctp": "在当前增量图上做 CaT 风格分布匹配；按合成特征附着同类最近 5 个当前文本，合计 320 字符。不是官方原任务分数复现。",
    "gecc_hard": "对传播后的特征做类内硬聚类；附着同类特征最近 5 个文本。硬分支与原生 FCM 软分支分开列出。",
    "random_coreset": "按类别预算无放回抽取真实节点，保留所选节点自己的文本。",
    "kcenter_coreset": "在归一化 X 与文本 embedding 拼接空间做精确 greedy farthest-point selection；保留所选节点文本。",
    "cluster_medoid": "MiniBatchKMeans 后选择每个簇离中心最近的真实成员；空簇采用确定性补足。不是精确 k-medoids。",
    "herding_coreset": "在相同图文拼接空间，以保持类均值为目标做精确 greedy herding；保留所选节点文本。",
    "puma_tag": "使用项目中的伪教师适配器；仅为 PUMA-inspired reference，不是官方 PUMA 核心复现。",
    "gcond_x_core": "从固定官方提交加载 SGC 与梯度距离函数；当前增量真实梯度、1000 次外层特征更新、无 inner training、合成邻接为单位阵；统一附着 nearest-5 文本。",
    "sfgc_core": "从固定官方提交加载 GCN/ReparamModule，保留多步教师轨迹匹配目标；5 个 expert × 800 epoch，200 次外层 × 200 次 unroll；统一附着 nearest-5 文本。官方示例为 200 expert/2000 外层，故标为 bounded core。",
    "gecc_soft": "保留 GraphSlim 原生 fuzzy C-means：fuzziness 1.3、20 restarts、300 iterations、1e-8 tolerance；附着共同的特征近邻文本。",
    "tagsam_selector_response": "保留官方文本 selector/response 机制，但蒸馏预算由 5000 × 15 降为 100 × 10；独立参考表，不能据其低分推断完整官方方法差。",
    "stad_native_local": "保留 STAD 核心拓扑与伪标签流程；10% labels/10% 内部验证/80% pseudo，用本地 Qwen2.5-1.5B 代替原 LLM。监督与合成拓扑不同，独立列为参考。",
}


def fmt(value, deviation=None):
    if value is None:
        return "--"
    return f"{value:.2f}" if deviation is None else f"{value:.2f} ± {deviation:.2f}"


def csv_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(bundle, output):
    checks = bundle["checks"]
    if not checks["no_duplicate_cells"] or not checks["stored_student_means_and_stds_verified"]:
        raise ValueError("Evidence must pass the expansion verifier before reporting")
    cells = [row for row in bundle["matched"] if row["split"] == "sealed_test"]
    main = [row for row in cells if row["arm"] not in QUALIFIED_REFERENCES]
    completed = sum(row["status"] == "complete" for row in bundle["coverage"])
    lines = ["# Unified-Q-EF：五数据集基线扩展与结果汇总", "",
        f"证据快照：`{bundle['collected_at']}`。完成 {completed}/{len(bundle['coverage'])} 个注册方法/数据集组合；"
        f"覆盖 {checks['snapshot_sources']} 份来源文件、{checks['completed_unique_cells']} 个已完成独立实验单元。", "",
        "## 统一协议与读表方式", "",
        "主表为 order=9101、construction seed=271006，Student seeds=150/151/152；"
        "每个方法必须完成全部阶段才入表。F1、Acc、Old/New、阶段均值均以百分数计；"
        "主表的 ± 是三个 Student seed 的样本标准差，不能当作 construction seed 的置信区间。", "",
        "所有主表方法共用当前合法构造池的累计 5% 预算（整数舍入）、current-delta-only、"
        "冻结历史合成前缀、MiniLM、prompt GCN（2048 steps/batch 24）及相同构造/评估划分。"
        "5% 是固定凝聚比例，合成行数会随累计合法池增长，不是固定容量内存。"
        "测试推理共用原始 transductive 图；无边 memory 只指不保存合成边，不代表推理时不用原始图结构。", "",
        "Cora/Citeseer 沿用项目 canonical split，不是 Planetoid 的每类 20 标签设置。"
        "主表各方法的 Student 训练预算相同，构造预算各异，已在适配表公开。"
        "这些分数是本任务适配结果，不是官方论文原设置分数的 reproduction。", ""]
    for dataset in DATASETS:
        rows = [row for row in main if row["dataset"] == dataset]
        rows.sort(key=lambda row: list(METHODS).index(row["arm"]))
        lines.extend(["## " + DISPLAY[dataset], ""])
        table = [[row["label"], IDENTITY[row["identity"]],
                  fmt(row["final_f1"], row["f1_student_std"]),
                  fmt(row["accuracy"], row["acc_student_std"]),
                  fmt(row["old_f1"]), fmt(row["new_f1"]), fmt(row["stage_average_f1"]),
                  fmt(row["construction_seconds"]), fmt(row["construction_peak_gib"])] for row in rows]
        lines.extend([markdown(["方法", "身份", "最终 Macro-F1", "Accuracy", "Old F1", "New F1",
                                "阶段平均 F1", "构造秒", "CUDA peak GiB"], table), ""])
        ours = next(row for row in rows if row["arm"] == "uq_dm_edgefree")
        comparators = [row for row in rows if row["arm"] != "uq_dm_edgefree"]
        leader = max(comparators, key=lambda row: row["final_f1"])
        cat = next(row for row in rows if row["arm"] == "cat_cctp")
        gecc = next(row for row in rows if row["arm"] == "gecc_hard")
        lines.extend([f"本扩展单元中，主方法相对 CaT-CCTP 为 {ours['final_f1']-cat['final_f1']:+.2f} F1 点，"
            f"相对 GECC-hard 为 {ours['final_f1']-gecc['final_f1']:+.2f} 点；"
            f"已完成对照中最高 F1 是 {leader['label']} 的 {leader['final_f1']:.2f}，"
            f"主方法与它相差 {ours['final_f1']-leader['final_f1']:+.2f} 点。"
            "这描述的是一个顺序/构造种子，稳定性判断见后面的九单元表。", ""])
        csv_rows(output / f"{dataset}_sealed_table.csv", [
            {key: row[key] for key in ("dataset", "arm", "label", "identity", "final_f1", "f1_student_std",
             "accuracy", "acc_student_std", "old_f1", "new_f1", "stage_average_f1", "construction_seconds",
             "construction_peak_gib", "memory_rows", "actual_rate", "source", "source_sha256")} for row in rows])
    references = [row for row in cells if row["arm"] in QUALIFIED_REFERENCES]
    lines.extend(["## TAG 原生机制参考（单独列出）", "",
        "共同 Student 与评估掩码可支持横向参考；但 TAGSAM 的构造预算显著缩减，STAD 的监督、LLM 与拓扑不同，"
        "因此下面的结果不参与主表同预算排名，也不能证明完整官方算法不如主方法。", ""])
    lines.extend([markdown(["数据集", "方法", "Macro-F1", "Accuracy", "构造秒", "CUDA peak GiB"], [
        [DISPLAY[row["dataset"]], row["label"], fmt(row["final_f1"], row["f1_student_std"]),
         fmt(row["accuracy"], row["acc_student_std"]), fmt(row["construction_seconds"]),
         fmt(row["construction_peak_gib"])] for row in references]), "",
        "## 九单元稳定性确认", "",
        "以下只统计主方法、CaT-CCTP、GECC-hard：3 个类别顺序 × 3 个 construction seeds，"
        "每个单元平均 3 个 Student seeds，共 27 次 Student 评估/方法/数据集。"
        "± 为九个单元的样本标准差；区间为九个配对单元的 95% t 区间，不能泛化为跨数据集保证。", ""])
    stable = [row for row in bundle["stability"] if row["split"] == "sealed_test"]
    lines.extend([markdown(["数据集", "方法", "F1 ± 单元 SD", "Acc ± 单元 SD", "最差单元 F1"], [
        [DISPLAY[row["dataset"]], row["label"], fmt(row["final_f1_mean"], row["final_f1_cell_std"]),
         fmt(row["accuracy_mean"], row["accuracy_cell_std"]), fmt(row["worst_f1"])] for row in stable]), ""])
    paired = [row for row in bundle["paired"] if row["split"] == "sealed_test" and row["metric"] == "final_f1"]
    lines.extend([markdown(["数据集", "EF 减去", "F1 差", "95% paired CI", "正单元", "最差差值"], [
        [DISPLAY[row["dataset"]], METHODS[row["comparator"]][0], f"{row['mean_delta']:+.2f}",
         f"[{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]", f"{row['positive_cells']}/{row['cells']}",
         f"{row['worst_delta']:+.2f}"] for row in paired]), "",
        "## 原方法—适配内容", "",
        markdown(["方法", "身份", "具体实现与适配"], [
            [METHODS[arm][0], IDENTITY[METHODS[arm][1]], ADAPTATIONS[arm]] for arm in METHODS]), "",
        "Random、K-center、Cluster-medoid、Herding 是四种选样对照，不算四篇独立的图凝聚论文。"
        "GECC-hard/soft 是同一论文的两个分支；PUMA-inspired 也不能按官方 PUMA 复现计数。"
        "组件消融不在这张基线表中，继续保留原归因报告。", "",
        "GCond 官方提交：`e0065cfdfd95963d988c359266a4fb0a4bdf60a1`；"
        "SFGC 官方提交：`b034621b8e39eba7aa58ec009963a3071cd87079`。"
        "构造入口核对提交、源码字节哈希，再加载原模型/目标定义；固定使用最后一次构造迭代，"
        "不按 report-dev 或 sealed test 选择 checkpoint。", "",
        "## 原生核心的有效更新诊断", "",
        "GCond-x/SFGC 的分类代理只包含当前增量类别。当该阶段只有一个类别时，单输出 softmax "
        "的分类交叉熵恒为零，没有判别性特征梯度。以下阶段的最终特征更新量为零，所有已记录"
        "梯度样本也为零；目标值非零不能单独证明特征被有效优化。SFGC 的教师参数仍可能因"
        "weight decay 位移，导致轨迹目标保持为 1。", "",
        "这是 current-delta-only 分类代理适配的限制，不能由这些阶段推断完整官方算法无效。"
        "其他阶段的更新情况保存在逐阶段诊断表中；原始结果、预注册预算和方法参数保持原样。", ""])
    zero_rows = [row for row in bundle["native_diagnostics"] if row["zero_feature_feedback"]]
    lines.extend([markdown(["数据集", "方法", "阶段（从 0 开始）", "类别数", "特征更新范数",
                            "零梯度/采样数", "初始/最终目标"], [
        [DISPLAY[row["dataset"]], METHODS[row["arm"]][0], row["stage"], row["stage_class_count"],
         row["feature_update_norm"], f"{row['zero_gradient_samples']}/{row['gradient_samples']}",
         f"{row['objective_first']:.4g} / {row['objective_last']:.4g}"] for row in zero_rows]), "",
        "## 完成情况", ""])
    lines.extend([markdown(["方法"] + [DISPLAY[d] for d in DATASETS], [[METHODS[arm][0]] + [
        STATUS[next((row["status"] for row in bundle["coverage"]
                     if row["dataset"] == dataset and row["arm"] == arm), "not registered")]
        for dataset in DATASETS] for arm in METHODS]), "",
        "## 对论文的含义", "",
        "当前最可靠的收益证据集中在 BookHis：相对 GECC-hard 的九单元增益为正，配对区间不跨零。"
        "ArXiv 与 GECC-hard 基本持平，SportsFit 没有稳定领先；不能由单扩展单元的正差值声称三大图普遍 SOTA。"
        "PUMA-inspired 在部分数据集上的较高分也必须保留，同时明确它是启发式参考。", "",
        "扩展到 Cora/Citeseer 补充了小图覆盖，并增加梯度匹配、轨迹匹配和选样对照。"
        "这些是有效性证据；共享 Q 的因果收益仍需原来的控制消融支持，不能用新增弱基线替代。", "",
        "计算开销应与精度一起呈现：CUDA peak 为 PyTorch allocated peak，包含加载后的共同数据张量，"
        "不是 CPU RAM、nvidia-smi 总占用或隔离吞吐；本轮有并发任务，wall time 不宜用于精细速度排名。"
        "GECC-soft 运行时间长，应保留 pending 状态，不能当作零分或用 hard 分支替代。", "",
        "本轮未引入历史 full-history report-dev 的保留率，因为其调度/划分与当前 sealed exact-5% 不能直接混用。"
        "主方法也没有根据本次测试结果重新调参。", "",
        "## 表格、图和来源", "",
        "- [完整英文表及覆盖状态](summary.md)",
        "- [所有匹配单元，包括 report-dev 与 sealed test](matched.csv)",
        "- [逐阶段结果](stages.csv)",
        "- [原生核心逐阶段更新与梯度诊断](native_diagnostics.csv)",
        "- [九单元配对统计](paired.csv)",
        "- [来源文件与 SHA-256](sources.csv)",
        "- [协议核验](verification.json)", "",
        "![五数据集 F1 与 Student SD](matched_final_f1.png)", "",
        "![五数据集 Accuracy 与 Student SD](matched_accuracy.png)", "",
        "![九单元主方法相对 GECC-hard 的配对差值](nine_cell_paired.png)", ""])
    (output / "summary_zh.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    write_report(bundle, args.bundle.parent)
    print(json.dumps(dict(output=str(args.bundle.parent / "summary_zh.md")), ensure_ascii=True))


if __name__ == "__main__":
    main()
