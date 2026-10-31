"""Write the reviewed five-level large-graph comparison report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "c0": "C0 / continual append",
    "semantic_recursive": "semantic-recursive append",
    "random_real": "random real-node coreset",
    "k_center_subsampled": "k-center coreset",
    "real_node_fps_subsampled": "real-node FPS coreset",
    "full_history": "full-history synthetic re-condensation",
    "full_history_gpu": "full-history synthetic (GPU implementation)",
    "raw_equal": "complete raw graph, equal steps",
    "raw_long": "complete raw graph, 8192 steps",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def method_access(audit: dict, dataset: str, method: str) -> dict:
    checks = audit["datasets"][dataset]["checks"]
    return next(item for item in checks if item["method"] == method)


def score_table(dataset: str, methods: dict) -> list[str]:
    lines = [
        f"### {dataset}: stage scores and C0-relative deltas",
        "",
        "| method | stage 0 | stage 1 | stage 2 | stage 3 | stage 4 | "
        "incremental mean delta | old delta | new delta | positive | worst |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in methods.items():
        values = list(summary.get("stage_cum", []))[:5]
        values += [None] * (5 - len(values))
        lines.append(
            "| " + " | ".join([
                LABELS.get(name, name),
                *(fmt(value) for value in values),
                fmt(summary.get("incremental_mean_delta_pp")),
                fmt(summary.get("old_mean_delta_pp")),
                fmt(summary.get("new_mean_delta_pp")),
                f"{summary.get('positive_incremental_cells', 0)}/"
                f"{summary.get('incremental_cells', 0)}",
                fmt(summary.get("worst_incremental_delta_pp")),
            ]) + " |"
        )
    return lines


def cost_table(dataset: str, costs: dict) -> list[str]:
    lines = [
        f"### {dataset}: recorded cost",
        "",
        "| method | construction seconds | student seconds | raw wall seconds | reported total |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, cost in costs.items():
        lines.append(
            f"| {LABELS.get(name, name)} | "
            f"{fmt(cost.get('build_seconds_sum'), 1)} | "
            f"{fmt(cost.get('student_seconds_sum'), 1)} | "
            f"{fmt(cost.get('wall_seconds_sum'), 1)} | "
            f"{fmt(cost.get('reported_total_seconds'), 1)} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = load(Path(args.summary_json))
    audit = load(Path(args.audit_json))
    lines = [
        "# 大图 base+4 五层比较：持续追加、coreset、全历史重凝聚与完整 raw 参考",
        "",
        "日期：2026-09-26",
        "",
        "## 1. 结论摘要",
        "",
        "本报告在三个大图的统一 `base+4`、固定 5% memory rate、" 
        "report-dev mod=3、Student batch=16、Student seeds=100--105 协议下，" 
        "比较持续合成追加、真实节点 coreset、全历史合成重凝聚和完整 raw graph 参考。",
        "",
        "`semantic-recursive` 在 12 个增量 stage cells 上相对同进程 C0 平均提升 "
        "`+0.858 pp`，9/12 个 cell 为正，新类平均提升 `+1.764 pp`；" 
        "最差 cell 为 `-2.381 pp`。它是大图上的方向性有效机制，不是已经稳定超过所有基线的主方法。",
        "",
        "完整 raw graph 和 full-history synthetic 允许读取累计历史 raw 并重写历史记忆，" 
        "因此是离线性能参考，不是合法 continual constructor。静态 TAGSAM 只有小图同协议适配，" 
        "本报告不把小图分数挪入大图主表。",
        "",
        "## 2. 五层含义与权限边界",
        "",
        "| 层次 | 方法 | 历史 raw | 可改写旧 memory | 用途 |",
        "|---|---|---|---|---|",
        "| continual | C0 / semantic-recursive | 否 | C0 旧前缀保持 | 回答持续凝聚能否追加并支持重训 |",
        "| continual reference | random / k-center / real-node FPS | 否 | 不适用真实节点回放 | 强 coreset 参考，不是合成凝聚 |",
        "| offline synthetic | full-history synthetic | 是 | 是 | 相同构造族的全历史参考 |",
        "| offline raw | raw equal / raw long | 是 | 是 | 完整真实图的同步数与长训参考 |",
        "",
        "raw_equal 使用与 synthetic student 相同的 2048 steps；raw_long 固定 8192 optimizer steps。"
        "8192 不是理论上界，也不代表充分收敛。",
        "",
        "## 3. 共享协议审计",
        "",
        f"审计文件的 all_pass = `{audit.get('all_pass')}`。所有方法均完成五个 stage，" 
        "阶段类别、预算、report-dev、Student batch 和 Student seed 与 C0 一致；" 
        "full-history/raw 的 offline 权限被单独标记。",
        "",
    ]

    for dataset, dataset_report in summary["datasets"].items():
        lines.extend(score_table(dataset, dataset_report["methods"]))
        lines.append("")
        lines.extend(cost_table(dataset, dataset_report.get("costs", {})))
        lines.append("")

    lines += [
        "## 4. 结果解释",
        "",
        "1. C0 是当前共同控制。它提供最稳定的低成本 continual 参照，不因某个单独阶段低于其它方法而被替换。",
        "2. semantic-recursive 的收益不是来自更多 rows、更多 Student steps 或改变拓扑：它与 paired C0 使用相同预算、边、batch、steps 和 seeds；区别在当前新增 block 的 structural proposal 与 semantic conserved payload assignment，并递归使用上一阶段 semantic memory。",
        "3. k-center 等真实节点 coreset 在 SportsFit 的部分阶段较强，但在 arXiv/BookChild 并不稳定；这说明“覆盖几何”与“可重训合成 memory”不是同一目标，不能以单一数据集的最好结果替换主线。",
        "4. full-history synthetic 和 complete raw 是性能参照，不是 continual 方法。它们回答的是“如果可以回读历史原图并重写记忆/直接训练，性能位置在哪里”。",
        "5. raw_equal 与 raw_long 的差异应解释为训练预算敏感性，而不是凝聚算法收益。",
        "",
        "## 5. 当前可写入论文的结论",
        "",
        "> 在统一大图 `base+4` 协议下，semantic-recursive append 相对冻结 C0 在 12 个增量 stage cells 上平均提升 0.858 个百分点，并将 new-class Macro-F1 平均提高 1.764 个百分点。该收益在 9/12 个 cell 中出现，但最差 cell 下降 2.381 个百分点，说明机制具有跨大图的方向性效用而非无条件优势。完整 raw 与 full-history 结果仅作为离线参考，不能用于宣称 continual SOTA。",
        "",
        "## 6. 缺失与后续",
        "",
        "- 静态 TAGSAM/STAD 尚未完成同协议大图适配；小图结果不能替代该主表。",
        "- 仍需在独立类别顺序和 construction seed 上确认 semantic-recursive，避免把当前三条固定流解释成普遍泛化。",
        "- 应继续报告 memory rows、构造成本、Student 成本、old/new F1 和最差阶段，而不是只报告最终累计 F1。",
        "- 若要升级为论文主方法，应先完成更强静态 TAG 基线与独立顺序确认；在此之前 C0 是共同冻结基线，semantic-recursive 是有证据的候选机制。",
        "",
        "## 7. 证据文件",
        "",
        f"- 汇总 JSON：`{args.summary_json}`",
        f"- 汇总 Markdown：`{Path(args.summary_json).with_suffix('.md')}`",
        f"- 协议审计 JSON：`{args.audit_json}`",
        f"- 协议审计 Markdown：`{Path(args.audit_json).with_suffix('.md')}`",
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
