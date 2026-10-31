"""Build dataset-first tables from existing experiments without rerunning them.

Protocol panels remain separate. Every numerical row keeps its input source,
uncertainty unit, and adaptation identity; mirrored exports are inventoried.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/baseline_master_by_dataset_20261013"
FINAL = "results/uq_final_evidence_report_20261013_132047"
EXPANSION = "results/uq_baseline_expansion_20261013/report_expansion_utc_20261014_021125_595059"
DATASETS = ("arxiv", "sportsfit", "bookhis", "wikics", "cora", "citeseer", "bookchild")
DISPLAY = dict(zip(DATASETS, ("ArXiv", "SportsFit", "BookHis", "WikiCS", "Cora", "Citeseer", "BookChild")))
PGCN = "GCN / multi-positive Prompt Student"
LEGACY = "GCN / legacy row-wise InfoNCE"
PANELS: dict[str, dict] = {}
SOURCES: dict[str, dict] = {}
ROWS: list[dict] = []
CONTROLS: list[dict] = []
PAIRS: list[dict] = []
CHECKS: list[dict] = []
SKIPPED: list[dict] = []

METHODS = {
    "Unified-Q-EF": ("Unified-Q-EF", "main", "proposed",
        "共享类别内 Q 池化 X、路由真实文本；循环内 GM/DM/传播匹配；冻结旧前缀；不存合成边"),
    "CaT-CCTP": ("CaT", "external_baseline", "mechanism adaptation",
        "保留类级随机投影分布匹配并优化 X；current-delta/frozen-prefix；附着图特征近邻 top-5 文本并重编码；共同 Student"),
    "GECC-hard": ("GECC", "external_baseline", "mechanism adaptation",
        "hop 加权传播 + 类内 Lloyd 硬聚类；current-delta/frozen-prefix；附着图特征近邻 top-5 文本；共同 Student"),
    "GECC-soft (native FCM)": ("GECC", "external_baseline", "native-core adaptation",
        "调用固定版本 GraphSlim 原生 FCM，fuzziness=1.3、20 次重启；共同持续封装、文本附着和 Student；非完整官方协议复现"),
    "TAGSAM (native selector)": ("TAGSAM", "external_baseline", "native-core adaptation",
        "接入官方文本选择机制与响应匹配核心；当前增量构造、冻结旧前缀、共同 Student；不是官方端到端持续复现"),
    "TAGSAM (wBCE response)": ("TAGSAM", "external_baseline", "mechanism adaptation",
        "保留官方 wBCE 响应损失；缩短 unroll；固定图近邻 top-5 文本；共同最终 Student；不等同原生文本选择版本"),
    "TAGSAM (legacy mechanism)": ("TAGSAM", "external_baseline", "mechanism adaptation",
        "旧版 wBCE/响应匹配持续封装，edge-free 记忆与原生机制有差别；不同 Student/快照单独列示"),
    "TAGSAM (static self-loop reference)": ("TAGSAM", "external_baseline", "adapted reference",
        "早期静态 TAGSAM 机制参考、自环合成记忆；协议和 Student 不同"),
    "TAGSAM (static semantic reference)": ("TAGSAM", "external_baseline", "adapted reference",
        "早期静态 TAGSAM 机制参考、语义连接合成记忆；协议和 Student 不同"),
    "STAD (native core, local Qwen)": ("STAD", "external_baseline", "native-core adaptation",
        "保留 STAD 核心与文本生成路径；使用本地 Qwen、当前增量/冻结前缀与共同 Student；非原论文完整复现"),
    "STAD (local canonical reference)": ("STAD", "external_baseline", "adapted reference",
        "官方组件/本地 canonical 文本生成参考；早期任务封装与 Student"),
    "STAD-inspired": ("STAD", "external_baseline", "inspired reference",
        "简化的文本生成/蒸馏机制参考；text-kNN 记忆拓扑、1024-step 单正例 Student；非 STAD 完整实现"),
    "DANCE-inspired": ("DANCE", "external_baseline", "inspired reference",
        "简化动态/邻居机制的集中式持续封装；没有完整联邦 DANCE 流程；1024-step 单正例 Student"),
    "SR-GM-inspired": ("SR-GM", "external_baseline", "inspired reference",
        "结构正则梯度匹配机制的简化封装；text-kNN 记忆拓扑、1024-step 单正例 Student；非官方完整复现"),
    "PUMA-inspired": ("PUMA", "external_baseline", "inspired reference",
        "伪教师/置信度筛选/传播的简化 TAG 封装；current-delta/frozen-prefix；缺少完整原生 PUMA 管线"),
    "Random real coreset": ("Random coreset", "coreset", "selection control",
        "当前合法池按预算随机保留真实节点及其文本；冻结旧前缀"),
    "K-center/FPS": ("K-center/FPS", "coreset", "selection control",
        "当前合法池的 farthest-point/k-center 选择；保留选中节点的真实文本"),
    "K-center (class-wise)": ("K-center/FPS", "coreset", "selection control",
        "类别内 farthest-point 选择；保留真实特征和文本"),
    "Medoid": ("Medoid coreset", "coreset", "selection control",
        "选择代表真实节点；保持特征和原文本对应"),
    "Herding": ("Herding", "coreset", "selection control",
        "用均值逼近的 herding 选择真实节点与文本"),
    "Semantic-recursive-inspired": ("Semantic-recursive reference", "external_baseline", "inspired reference",
        "早期语义递归机制参考；不是新增的已复现公开强基线"),
    "GraphCLIP official checkpoint": ("GraphCLIP", "official_checkpoint_reference", "official checkpoint inference",
        "加载官方预训练 checkpoint 做预测；未构造 5% 凝聚记忆；不是持续凝聚基线"),
    "GraphCLIP official checkpoint (512-cap)": ("GraphCLIP", "official_checkpoint_reference", "official checkpoint inference",
        "官方 checkpoint 在至多 512 个 ego 样本上推断；样本数不是 synthetic memory 行数；无持续凝聚训练"),
    "Full-history raw (self-loops)": ("Full-history raw", "full_history_reference", "raw-data reference",
        "每阶段重用全部已见合法 raw 特征/文本；Student 训练传播为自环；允许历史 raw 访问"),
    "Full-history raw (real graph)": ("Full-history raw", "full_history_reference", "raw-data reference",
        "每阶段使用全部已见合法 raw 特征/文本与真实图传播；允许历史 raw 访问；与自环参考分别报告"),
    "Static-TAG full-history": ("Static TAG reference", "full_history_reference", "adapted reference",
        "每阶段重建全部已见 raw 的静态 TAG 记忆与 text-kNN 边；重写旧记忆；不满足历史访问限制"),
    "Full-history synthetic (early)": ("Full-history synthetic control", "full_history_reference", "internal reference",
        "早期全历史合成记忆参考；不同构造/训练预算"),
    "Full raw (equal steps, early)": ("Full raw reference", "full_history_reference", "raw-data reference",
        "早期原图训练、与当时记忆训练步数对照；只记录 F1"),
    "Full raw (long training, early)": ("Full raw reference", "full_history_reference", "raw-data reference",
        "早期较长 raw 训练参考；不是当前同训练预算比较；只记录 F1"),
    "Adaptive coherent": ("Adaptive coherent", "internal_candidate", "internal research candidate",
        "早期内部文本构造候选；保留历史结果作上下文，不计入外部基线数"),
    "Cluster-medoid": ("Cluster-medoid", "coreset", "selection control",
        "当前增量按类选真实 medoid，连同原文本保存；共同持续封装和 Student"),
    "GCond-x (core)": ("GCond-x", "external_baseline", "native-core adaptation",
        "保留官方 SGC/梯度距离核心，当前增量、冻结前缀、单位合成邻接；top-5 近邻文本，统一 Student；非完整官方持续复现"),
    "SFGC (bounded core)": ("SFGC", "external_baseline", "bounded native-core adaptation",
        "保留官方 GCN/ReparamModule 教师轨迹匹配核心，但构造预算缩减；top-5 近邻文本，统一 Student；BookHis 末阶段单类无判别梯度"),
    "TAGSAM (bounded core)": ("TAGSAM", "external_baseline", "compute-bounded native-core reference",
        "保留文本 selector/response，构造预算从 5000x15 缩至 100x10；共同 Student；非官方完整复现"),
    "STAD (local Qwen)": ("STAD", "external_baseline", "supervision/topology-qualified reference",
        "保留拓扑与伪标签路径，使用本地 Qwen2.5-1.5B；10% labels/10% 验证/80% pseudo，与主方法监督和拓扑不同"),
}


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def track(path: Path, disposition="included", note="") -> bytes:
    data = path.read_bytes()
    key = relpath(path)
    SOURCES.setdefault(key, {"path": key, "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data), "disposition": disposition, "note": note})
    return data


def read_csv(rel: str) -> list[dict]:
    path = ROOT / rel
    track(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(track(path).decode("utf-8-sig"))


def number(value):
    if value is None or str(value).strip() in {"", "None", "nan", "--"}:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite value: {value}")
    return result


def mean(values):
    vals = [number(v) for v in values if number(v) is not None]
    return stats.mean(vals) if vals else None


def sd(values):
    vals = [number(v) for v in values if number(v) is not None]
    return stats.stdev(vals) if len(vals) > 1 else None


def panel(key, title, note, student, split, budget, priority):
    PANELS[key] = {"panel_id": key, "title": title, "note": note, "student": student,
        "split": split, "budget": budget, "priority": priority}


def add(key, dataset, method, f1, acc, *, source, **extra):
    if dataset not in DATASETS:
        raise ValueError(f"Unmapped dataset {dataset}")
    meta = METHODS.get(method, (method, "internal_ablation", "internal control",
        "内部组件消融/候选，不能计为公开方法基线"))
    spec = PANELS[key]
    row = {"panel_id": key, "dataset": dataset, "method": method,
        "original_method": meta[0], "role": meta[1], "fidelity": meta[2],
        "adaptation": meta[3], "student": spec["student"], "split": spec["split"],
        "budget": spec["budget"], "final_f1": number(f1), "accuracy": number(acc),
        "source": source, "official_condensation_reproduction": False}
    row.update(extra)
    target = CONTROLS if row["role"] in {"internal_candidate", "internal_ablation"} else ROWS
    target.append(row)
    return row


def init_panels():
    specs = [
        ("00_expansion_sealed", "五图扩展：同一 sealed cell 的十配置", "o9101/c271006，每配置三 Student seeds；每图主方法加九个对照配置。单 cell SD 仅反映 Student 随机性，不是九 cell 稳定性；构造计算预算因方法而异", PGCN, "sealed_test", "cumulative legal-pool 5% (integer rounding)", 0),
        ("00_expansion_dev", "五图扩展：同一 report-dev cell", "与 sealed 同一记忆和种子，development split；不混合 split 取最大值", PGCN, "report_dev", "cumulative legal-pool 5% (integer rounding)", 0.1),
        ("00_qualified_sealed", "五图扩展：受限原生参考 sealed cell", "同一读出单元，但 TAGSAM 构造预算缩减、STAD 监督/拓扑不同；BookHis GECC-soft 完成，ArXiv/SportsFit GECC-soft 停止重试；不并入十配置排名", PGCN, "sealed_test", "source-specific qualifications; target 5%", 0.2),
        ("00_qualified_dev", "五图扩展：受限原生参考 report-dev cell", "与上述受限参考同记忆；不并入主十配置排名", PGCN, "report_dev", "source-specific qualifications; target 5%", 0.3),
        ("01_sealed", "最终确认：sealed test，真实累计 5%", "3 orders × 3 construction seeds × 3 Student seeds；SD 为九个 cell 均值的样本标准差；当前论文主表优先使用本组", PGCN, "sealed_test", "cumulative legal-pool 5% (integer rounding)", 1),
        ("02_dev", "最终确认：report-dev，真实累计 5%", "与最终确认同一记忆/网格的 development split；不得与 sealed 合并", PGCN, "report_dev", "cumulative legal-pool 5% (integer rounding)", 2),
        ("03_native_sealed", "原生核心适配：同一个 sealed cell", "固定 o9101/c271006，三 Student seeds；先列该 cell 主方法/CaT/GECC，再列原生核心适配；不使用九 cell 均值替换此处对照", PGCN, "sealed_test", "cumulative legal-pool 5% (integer rounding)", 3),
        ("04_native_dev", "原生核心适配：同一个 report-dev cell", "与上一组同记忆和种子，development split；单 cell 不能支持跨顺序显著性结论", PGCN, "report_dev", "cumulative legal-pool 5% (integer rounding)", 4),
        ("05_matrix", "2026-10-08 共同 Student 外部矩阵", "o9102/o9103，各三 Student seeds；使用当时 fixed stage schedule；最终行数相同不代表每阶段调度相同", PGCN, "report_dev", "legacy fixed 5%-target stage schedule", 5),
        ("06_tagsam_response", "2026-10-08 TAGSAM wBCE 响应适配", "o9101/c271006、三 Student seeds；共同最终 Student；构造使用旧 proxy 和缩短 unroll；不是最新 native selector 版本", PGCN, "report_dev", "legacy fixed 5%-target stage schedule", 6),
        ("07_pre_prompt", "2026-10-10 三顺序共同 Prompt Student", "较早记忆/调度快照；三个顺序的均值与 order SD；保留已有结论的上下文，不替换最终主表", PGCN, "report_dev", "legacy fixed 5%-target stage schedule", 7),
        ("08_pre_legacy", "2026-10-10 匹配算子的 Legacy Student", "仍是 GCN，变化主要是 row-wise InfoNCE 训练协议；不是跨架构；不与 Prompt Student 横向混排", LEGACY, "report_dev", "legacy fixed 5%-target stage schedule", 8),
        ("09_extra", "2026-10-11 扩展数据集确认", "ArXiv 单顺序，旧预算 6773/118539=5.71%；Cora 三顺序；源表单顺序 SD=0 不代表稳定性，显示为空", PGCN, "report_dev", "legacy fixed stage schedule; ArXiv actual 5.71%", 9),
        ("10_sage", "GraphSAGE 跨架构：固定既有记忆", "3 orders × 3 construction seeds，各三 Student seeds；九 cell 的样本 SD；当时版本的 report-dev，未重新调构造参数", "GraphSAGE / Prompt Student", "report_dev", "frozen existing 5%-target memories", 10),
        ("11_formal", "外部交接记忆：六 Student seeds 重评估", "早期冻结构造记忆，Student seeds 150–155；原始方法、调度和训练版本保留；不与最终九 cell 均值合并", "canonical GCN/MiniLM Student (handoff snapshot)", "legacy_evaluation", "cumulative 5%-target schedule; actual rate not audited here", 11),
        ("12_tagsam_legacy", "早期 TAGSAM 四图适配参考", "交接包 wBCE 机制适配；Student seeds 150–155；不等同最新原生文本选择适配", "canonical GCN/MiniLM Student (handoff TAGSAM snapshot)", "legacy_evaluation", "5%-target continual wrapper", 12),
        ("13_small_native", "小图 STAD/TAGSAM 早期适配参考", "Student seeds 100–105；与当前主方法协议不相同；保留为 adapted reference", "early canonical TAG Student", "legacy_evaluation", "source-specific (not fully recorded)", 13),
        ("14_simplified", "大图 DANCE/SR-GM/STAD 简化参考", "1024-step 单正例 Student + text-kNN synthetic topology；未完整保留所有官方核心；不能把低分解释成主方法已超过官方 SOTA", "GCN / 1024-step single-positive Student", "adapter_report_dev", "source-specific 5%-target continual budget", 14),
        ("15_coherent_era", "2026-10-06 Adaptive coherent 时期", "六 Student seeds 的较早四图快照；Adaptive coherent 在内部候选表；基线数不包括 CaT 文本附着组件", "legacy GCN/MiniLM continual Student", "legacy_evaluation", "legacy cumulative 5%-target schedule", 15),
        ("16_small_strict", "2026-10-04 小图严格三阶段外部比较", "Student seeds 150–155；各方法覆盖顺序数量不同；均值先按 Student 再按顺序等权，SD 重算为 order sample SD", "canonical scratch GCN/MiniLM bidirectional InfoNCE", "legacy_evaluation", "5% cumulative legal-pool budget", 16),
        ("17_small_cctp", "2026-10-01 小图 CCTP/近邻矩阵", "四顺序；部分实验与后续 master 表重叠，不应当成额外独立复现；CaT text-center 与 nearest 为附着变体", "canonical scratch GCN/MiniLM bidirectional InfoNCE", "legacy_evaluation", "5% cumulative legal-pool budget", 17),
        ("18_large_legacy", "2026-10-04 旧大图 base+4 外部比较", "Student seeds 100–105、单构造顺序；BookChild 与 BookHis 是不同数据集；不能拿旧分数与新 Student 直接相减", "legacy GCN/MiniLM Student (100–105)", "legacy_evaluation", "legacy cumulative 5%-target schedule", 18),
        ("19_large_corrected", "2026-10-03 corrected 大图外部结果", "只保留外部/coreset 行；与 master 相同数值/实验的行记为重复，不重复展示", "legacy GCN/MiniLM Student (100–105)", "legacy_evaluation", "legacy cumulative 5%-target schedule", 19),
        ("20_bounded", "2026-10-01 bounded-XP 时期外部对照", "保留当时原始数据；内部 XP/edge 控制另存；该批不与 corrected 或新 Student 合并", "legacy GCN/MiniLM Student", "legacy_evaluation", "source-specific legacy 5%-target schedule", 20),
        ("21_early_small", "2026-09-17 小图早期外部/原图参考", "早期长训练预算；仅记录 F1，没有 Accuracy；SD 为原表提供的 Student population SD", "early canonical GCN/text Student", "legacy_evaluation", "source-specific node ratios", 21),
        ("22_raw_self", "完整历史 raw 自环参考与当时主方法", "同一 report-dev 对照快照，三 Student seeds；主方法和 raw 分数沿用当时配对；ArXiv 5.71%；不能用最新 sealed 分数重算保留率", PGCN, "report_dev", "full legal raw vs historical synthetic budget", 22),
        ("23_raw_graph", "完整历史 raw 真实图参考", "共同 Prompt Student、真实图传播；历史 raw 可重访问；不是同存储预算/同构造访问的凝聚基线", PGCN, "report_dev", "full legal raw (100%)", 23),
        ("24_static", "静态 TAG 全历史重建参考", "旧记忆重写、可访问全部历史 raw、text-kNN topology；与当前持续协议不同", "legacy GCN/MiniLM continual Student", "legacy_evaluation", "source-specific offline synthetic budget", 24),
        ("25_clip_large", "官方 GraphCLIP checkpoint：大图样本参考", "最多 512 ego 样本；无凝聚过程；不同预训练、任务和评估样本，不能纳入 5% 排名", "GraphCLIP official frozen pretrained model", "native_checkpoint_reference", "no condensation memory", 25),
        ("26_clip_small", "官方 GraphCLIP checkpoint：小图参考", "官方 checkpoint 推断；完整图/ego 输入；无持续凝聚、无共同 Student 训练", "GraphCLIP official frozen pretrained model", "native_checkpoint_reference", "no condensation memory", 26),
        ("90_controls", "最终共享 Q 组件归因（内部）", "内部组件消融，单类别顺序、最多三构造种子；不是外部公开基线", PGCN, "mixed (see row)", "current cumulative 5%", 90),
    ]
    for spec in specs:
        panel(*spec)


def latest():
    summary = read_csv(f"{FINAL}/summary.csv")
    cells = read_csv(f"{FINAL}/cells.csv")
    students = read_csv(f"{FINAL}/student_rows.csv")
    read_csv(f"{FINAL}/pairs.csv")
    read_json(ROOT / FINAL / "provenance.json")
    by_cell = defaultdict(list)
    by_summary = defaultdict(list)
    for r in cells:
        key = (r["category"], r["dataset"], r["arm"], r["split"])
        by_summary[key].append(r)
    for r in students:
        key = (r["category"], r["dataset"], r["order_seed"], r["construction_seed"], r["arm"], r["split"])
        by_cell[key].append(r)
    final_students = {}
    for key, vals in by_cell.items():
        stage = max(int(r["stage"]) for r in vals)
        final = [r for r in vals if int(r["stage"]) == stage]
        if len({r["student_seed"] for r in final}) != len(final):
            raise AssertionError(f"Duplicate final Student observations: {key}")
        final_students[key] = final
    for key, vals in by_summary.items():
        for r in vals:
            skey = (r["category"], r["dataset"], r["order_seed"], r["construction_seed"], r["arm"], r["split"])
            final = final_students[skey]
            for metric in ("final_f1", "accuracy", "old_f1", "new_f1"):
                value = mean(x[metric] for x in final)
                if value is not None and abs(value - float(r[metric])) > 1e-8:
                    raise AssertionError(f"Cell does not match final Students: {skey}/{metric}")
    names = {"uq_dm_edgefree": "Unified-Q-EF", "cat_cctp": "CaT-CCTP", "gecc_hard": "GECC-hard",
        "gecc_soft": "GECC-soft (native FCM)", "tagsam_selector_response": "TAGSAM (native selector)",
        "stad_native_local": "STAD (native core, local Qwen)"}
    for r in summary:
        key = (r["category"], r["dataset"], r["arm"], r["split"])
        vals = by_summary[key]
        for metric, col, sdcol in (("final_f1", "final_f1_mean", "final_f1_cell_std"),
                                  ("accuracy", "accuracy_mean", "accuracy_cell_std")):
            if abs(mean(v[metric] for v in vals) - float(r[col])) > 1e-8:
                raise AssertionError(f"Summary mean mismatch: {key}/{metric}")
            observed = number(r[sdcol])
            computed = sd(v[metric] for v in vals)
            if observed is not None and abs(computed - observed) > 1e-8:
                raise AssertionError(f"Summary SD mismatch: {key}/{metric}")
        if r["category"] == "confirmation":
            assert len(vals) == 9
            assert len({v["order_seed"] for v in vals}) == 3
            assert len({v["construction_seed"] for v in vals}) == 3
            assert all(v["student_seeds"] == "150,151,152" for v in vals)
            pkey = "01_sealed" if r["split"] == "sealed_test" else "02_dev"
        elif r["category"] == "controls":
            pkey = "90_controls"
        else:
            continue
        add(pkey, r["dataset"], names.get(r["arm"], r["label"]), r["final_f1_mean"], r["accuracy_mean"],
            source=f"{FINAL}/summary.csv", arm=r["arm"], split=r["split"],
            f1_sd=number(r["final_f1_cell_std"]), accuracy_sd=number(r["accuracy_cell_std"]),
            sd_unit="sample SD of Student-averaged order/construction cells",
            n_cells=len(vals), n_orders=int(r["orders"]), n_construction_seeds=int(r["construction_seeds"]),
            students_per_cell=3, n_student_runs=int(r["student_runs"]), student_seeds="150,151,152",
            student_steps=2048, student_batch=24, text_encoder="frozen MiniLM",
            memory_rows=int(r["memory_rows"]), actual_rate=float(r["actual_rate"]),
            old_f1=number(r["old_f1_mean"]), new_f1=number(r["new_f1_mean"]),
            stage_average_f1=number(r["stage_average_f1_mean"]), worst_final_f1=number(r["worst_final_f1"]),
            construction_seconds=number(r["construction_seconds"]), construction_peak_gib=number(r["construction_peak_gib"]),
            underlying_sources=" | ".join(v["source"] for v in vals))
    # Restrict the native-core table's reference methods to the very same cell.
    native_datasets = {r["dataset"] for r in summary if r["category"] in {"baselines", "stad"}}
    for r in cells:
        wanted = r["category"] in {"baselines", "stad"} or (
            r["category"] == "confirmation" and r["order_seed"] == "9101" and r["construction_seed"] == "271006")
        if not wanted or r["dataset"] not in native_datasets:
            continue
        key = (r["category"], r["dataset"], r["order_seed"], r["construction_seed"], r["arm"], r["split"])
        final = final_students[key]
        assert len(final) == 3
        add("03_native_sealed" if r["split"] == "sealed_test" else "04_native_dev",
            r["dataset"], names[r["arm"]], r["final_f1"], r["accuracy"], source=f"{FINAL}/cells.csv",
            arm=r["arm"], f1_sd=sd(v["final_f1"] for v in final), accuracy_sd=sd(v["accuracy"] for v in final),
            sd_unit="sample SD of 3 Student seeds within a single cell", n_cells=1, n_orders=1,
            n_construction_seeds=1, students_per_cell=3, n_student_runs=3, student_seeds="150,151,152",
            student_steps=2048, student_batch=24, text_encoder="frozen MiniLM", order_seeds=r["order_seed"],
            construction_seeds=r["construction_seed"], memory_rows=int(r["memory_rows"]), actual_rate=float(r["actual_rate"]),
            old_f1=number(r["old_f1"]), new_f1=number(r["new_f1"]), stage_average_f1=number(r["stage_average_f1"]),
            construction_seconds=number(r["construction_seconds"]), construction_peak_gib=number(r["construction_peak_gib"]),
            underlying_sources=r["source"], underlying_sha256=r["source_sha256"])
    # Paired inference uses nine cell means, not the 27 individual Student runs.
    for dataset in DATASETS[:3]:
        for split in ("sealed_test", "report_dev"):
            main = {(r["order_seed"], r["construction_seed"]): r for r in cells
                if r["category"] == "confirmation" and r["dataset"] == dataset and r["split"] == split and r["arm"] == "uq_dm_edgefree"}
            for arm in ("gecc_hard", "cat_cctp"):
                other = {(r["order_seed"], r["construction_seed"]): r for r in cells
                    if r["category"] == "confirmation" and r["dataset"] == dataset and r["split"] == split and r["arm"] == arm}
                assert main.keys() == other.keys() and len(main) == 9
                for metric in ("final_f1", "accuracy"):
                    diff = [float(main[k][metric]) - float(other[k][metric]) for k in sorted(main)]
                    half = 2.306004135204166 * stats.stdev(diff) / math.sqrt(9)
                    PAIRS.append({"dataset": dataset, "split": split, "main": "Unified-Q-EF", "baseline": names[arm],
                        "metric": metric, "n_cells": 9, "mean_delta_pp": stats.mean(diff),
                        "ci95_low": stats.mean(diff) - half, "ci95_high": stats.mean(diff) + half,
                        "positive_cells": sum(v > 0 for v in diff), "positive_cell_ratio": sum(v > 0 for v in diff) / 9,
                        "worst_delta_pp": min(diff), "unit": "Student-averaged paired order/construction cell"})
    CHECKS.append({"check": "latest summary / cells / final-stage Students", "status": "passed",
        "detail": f"Verified {len(summary)} summary rows and {len(cells)} cells against {len(students)} stage/Student observations; all final confirmation grids 9×3 complete."})


def expansion():
    matched = read_csv(f"{EXPANSION}/matched.csv")
    registry = {r["path"]: r for r in read_csv(f"{EXPANSION}/sources.csv")}
    verified = read_json(ROOT / EXPANSION / "independent_verification.json")
    receipt = read_json(ROOT / "results/uq_baseline_expansion_20261013/latest_report_receipt.json")
    assert verified["status"] == receipt["independent_verification"] == "passed"
    assert verified["matched_rows_verified"] == len(matched) == 114
    assert receipt["report"] == EXPANSION and receipt["completed"] == 57 and receipt["registered"] == 59
    assert {r["student_source_hash"] for r in matched} == {receipt["checks"]["student_source_hash"]}
    confirmation = {(r["dataset"], r["arm"], r["split"]): r for r in read_csv(f"{FINAL}/cells.csv")
        if r["category"] == "confirmation" and r["order_seed"] == "9101" and r["construction_seed"] == "271006"}
    names = {"cluster_medoid": "Cluster-medoid", "gcond_x_core": "GCond-x (core)",
        "herding_coreset": "Herding", "kcenter_coreset": "K-center/FPS", "puma_tag": "PUMA-inspired",
        "random_coreset": "Random real coreset", "sfgc_core": "SFGC (bounded core)",
        "uq_dm_edgefree": "Unified-Q-EF", "cat_cctp": "CaT-CCTP", "gecc_hard": "GECC-hard",
        "gecc_soft": "GECC-soft (native FCM)", "tagsam_selector_response": "TAGSAM (bounded core)",
        "stad_native_local": "STAD (local Qwen)"}
    seen = set()
    for r in matched:
        key = (r["dataset"], r["arm"], r["split"])
        assert key not in seen and r["arm"] in names and r["split"] in {"sealed_test", "report_dev"}
        seen.add(key)
        assert r["order_seed"] == "9101" and r["construction_seed"] == "271006"
        assert r["category"] in {"confirmation", "expansion", "native_reference"}
        assert registry[r["source"]]["source_sha256"] == r["source_sha256"]
        assert registry[r["source"]]["status"] == "complete"
        original = ROOT / r["source"]
        if original.is_file():
            track(original)
            assert SOURCES[relpath(original)]["sha256"] == r["source_sha256"]
        if r["category"] == "confirmation":
            ref = confirmation[key]
            for metric in ("final_f1", "accuracy", "old_f1", "new_f1", "memory_rows", "actual_rate"):
                assert abs(float(r[metric]) - float(ref[metric])) < 1e-8, (key, metric)
        pkey = ("00_qualified_" if r["category"] == "native_reference" else "00_expansion_") + (
            "sealed" if r["split"] == "sealed_test" else "dev")
        add(pkey, r["dataset"], names[r["arm"]], r["final_f1"], r["accuracy"],
            source=f"{EXPANSION}/matched.csv", arm=r["arm"], category=r["category"],
            f1_sd=number(r["f1_student_std"]), accuracy_sd=number(r["acc_student_std"]),
            sd_unit="sample SD of 3 Student seeds within a single order/construction cell",
            n_cells=1, n_orders=1, n_construction_seeds=1, students_per_cell=3, n_student_runs=3,
            student_seeds="150,151,152", student_steps=2048, student_batch=24,
            text_encoder="frozen MiniLM", order_seeds=r["order_seed"], construction_seeds=r["construction_seed"],
            memory_rows=int(r["memory_rows"]), actual_rate=float(r["actual_rate"]),
            old_f1=number(r["old_f1"]), new_f1=number(r["new_f1"]),
            stage_average_f1=number(r["stage_average_f1"]), construction_seconds=number(r["construction_seconds"]),
            construction_peak_gib=number(r["construction_peak_gib"]),
            underlying_sources=r["source"], underlying_sha256=r["source_sha256"])
    assert len(seen) == 114
    for split in ("sealed_test", "report_dev"):
        for dataset in ("arxiv", "sportsfit", "bookhis", "cora", "citeseer"):
            group = [r for r in matched if r["dataset"] == dataset and r["split"] == split]
            assert len([r for r in group if r["category"] != "native_reference"]) == 10
            assert len([r for r in group if r["category"] == "native_reference"]) == (3 if dataset == "bookhis" else 2 if dataset in {"arxiv", "sportsfit"} else 0)
    CHECKS.append({"check": "five-dataset expansion against immutable source", "status": "passed",
        "detail": "114 matched rows (57 method/dataset cells x 2 splits), 50 main configurations and 7 qualified references; source SHA-256 matched the immutable registry (remote originals not necessarily local); overlapping final confirmation cells verified; two GECC-soft cells remain absent."})


def current_matrix():
    buckets = defaultdict(list)
    arms = {"random_coreset": "Random real coreset", "kcenter_coreset": "K-center/FPS",
        "gecc_hard": "GECC-hard", "puma_tag": "PUMA-inspired", "tagsam_response_adapter": "TAGSAM (wBCE response)"}
    paths = sorted((ROOT / "results/unified_q_matrix_20261008").glob("baseline_*_o*/results.json"))
    paths += sorted((ROOT / "results/unified_q_matrix_20261008").glob("tagsam_*/results.json"))
    for path in paths:
        payload = read_json(path)
        protocol = payload["protocol"]
        final_stage = len(protocol["groups"]) - 1
        for r in payload["rows"]:
            if r.get("stage") == final_stage and r.get("arm") in arms:
                buckets[(protocol["dataset"], r["arm"])].append((r, protocol, relpath(path)))
    for (dataset, arm), vals in buckets.items():
        scores = [v[0]["score"] for v in vals]
        multi = len(vals) > 1
        add("06_tagsam_response" if arm == "tagsam_response_adapter" else "05_matrix", dataset, arms[arm],
            mean(s["final_f1"] for s in scores), mean(s["accuracy"] for s in scores),
            source=" | ".join(v[2] for v in vals),
            f1_sd=sd(s["final_f1"] for s in scores) if multi else number(scores[0].get("final_f1_std")),
            accuracy_sd=sd(s["accuracy"] for s in scores) if multi else number(scores[0].get("accuracy_std")),
            sd_unit="sample SD of order means" if multi else "source-provided Student-seed SD",
            n_cells=len(vals), n_orders=len({v[1]["order_seed"] for v in vals}), n_construction_seeds=1,
            students_per_cell=3, n_student_runs=3 * len(vals), student_steps=2048, student_batch=24,
            student_seeds="150,151,152", text_encoder="frozen MiniLM", memory_rows=vals[0][0]["memory_rows"],
            order_seeds=",".join(str(v[1]["order_seed"]) for v in vals),
            construction_seeds=",".join(str(v) for v in sorted({v[1]["construction_seed"] for v in vals})),
            old_f1=mean(s.get("old_f1") for s in scores), new_f1=mean(s.get("new_f1") for s in scores))


def earlier_prompt_and_sage():
    path = "results/pre_submission_evidence_20261010/primary_matched.csv"
    names = {"Unified-Q (edge-free)": "Unified-Q-EF", "CaT-CCTP adapter": "CaT-CCTP",
        "GECC-hard TAG adapter": "GECC-hard", "Adaptive coherent attribution": "Adaptive coherent"}
    for r in read_csv(path):
        key = "07_pre_prompt" if r["student"] == "Prompt Student" else "08_pre_legacy"
        add(key, r["dataset"], names[r["method"]], r["final_f1"], r["accuracy"], source=path,
            f1_sd=number(r["order_sd_f1"]), sd_unit="source-provided SD of order means", n_orders=int(r["n_orders"]),
            old_f1=number(r["old_f1"]), new_f1=number(r["new_f1"]), underlying_sources=r["source"])
    path = "results/extended_evidence_20261011/additional_dataset_summary.csv"
    names = {"uq_dm_edgefree": "Unified-Q-EF", "cat_cctp": "CaT-CCTP", "gecc_hard": "GECC-hard"}
    for r in read_csv(path):
        multi = int(r["n_orders"]) > 1
        add("09_extra", r["dataset"], names[r["method"]], r["final_f1_mean"], r["accuracy_mean"], source=path,
            f1_sd=number(r["final_f1_sd"]) if multi else None,
            accuracy_sd=number(r["accuracy_sd"]) if multi else None, sd_unit="source-provided SD of order means" if multi else "not estimable across a single order",
            n_orders=int(r["n_orders"]), order_seeds=r["orders"], student_steps=2048,
            old_f1=number(r["old_f1_mean"]), new_f1=number(r["new_f1_mean"]),
            memory_rows=6773 if r["dataset"] == "arxiv" else None,
            raw_pool_rows=118539 if r["dataset"] == "arxiv" else None,
            actual_rate=6773 / 118539 if r["dataset"] == "arxiv" else None)
    path = "results/unified_q_overnight_20261012/sage_cells.csv"
    vals = defaultdict(list)
    names = {"Unified-Q-EF": "Unified-Q-EF", "GECC-hard-TAG": "GECC-hard", "CaT-CCTP": "CaT-CCTP",
        "Unified-Q-QMedoid": "Unified-Q-QMedoid (internal)"}
    for r in read_csv(path):
        vals[(r["dataset"], r["method"])].append(r)
    for (dataset, method), cells in vals.items():
        assert len(cells) == 9
        add("10_sage", dataset, names[method], mean(r["final_f1"] for r in cells), mean(r["accuracy"] for r in cells),
            source=path, f1_sd=sd(r["final_f1"] for r in cells), accuracy_sd=sd(r["accuracy"] for r in cells),
            sd_unit="sample SD of Student-averaged order/construction cells", n_cells=9,
            n_orders=len({r["order"] for r in cells}), n_construction_seeds=len({r["construction_seed"] for r in cells}),
            students_per_cell=3, n_student_runs=27, student_seeds="150,151,152", student_steps=2048,
            old_f1=mean(r["old_f1"] for r in cells), new_f1=mean(r["new_f1"] for r in cells),
            underlying_sources=" | ".join(r["source"] for r in cells))


def registry_and_broad_adapters():
    root = "results/final_baseline_registry_20261011"
    names = {"CaT-CCTP": "CaT-CCTP", "GECC-hard-TAG": "GECC-hard", "K-center/FPS": "K-center/FPS",
        "Medoid": "Medoid", "PUMA-inspired-TAG": "PUMA-inspired", "Random real coreset": "Random real coreset",
        "TAGSAM mechanism adapter": "TAGSAM (legacy mechanism)", "STAD": "STAD (local canonical reference)",
        "TAGSAM": "TAGSAM (legacy mechanism)", "GraphCLIP official frozen checkpoint": "GraphCLIP official checkpoint"}
    for filename, pkey in (("formal_external_150155.csv", "11_formal"),
        ("mechanism_adapted_large.csv", "12_tagsam_legacy"), ("mechanism_adapted_small.csv", "13_small_native"),
        ("official_reference.csv", "26_clip_small")):
        path = f"{root}/{filename}"
        for r in read_csv(path):
            if number(r["final_f1"]) is None:
                SKIPPED.append({"source": path, "method": r["method"], "reason": "inventory only; no numerical experiment"})
                continue
            row = add(pkey, r["dataset"], names[r["method"]], r["final_f1"], r["accuracy"], source=path,
                f1_sd=number(r["final_f1_sd"]), accuracy_sd=number(r["accuracy_sd"]),
                sd_unit="source-provided Student-seed SD" if r["final_f1_sd"] else "not recorded",
                student_seeds=r["student_seeds"], orders_description=r["orders"], old_f1=number(r["old_f1"]),
                new_f1=number(r["new_f1"]), underlying_sources=r["source"], source_note=r["note"],
                source_adaptation=r["adaptation"])
            if pkey in {"11_formal", "12_tagsam_legacy", "13_small_native"}:
                row.update(n_orders=1, students_per_cell=6, n_student_runs=6)
    path = "results/large_external_adapters_20261011/large_external_comparison.csv"
    names = {"dance_adapter": "DANCE-inspired", "sr_gm_adapter": "SR-GM-inspired", "stad_adapter": "STAD-inspired",
        "GraphCLIP official frozen (512-cap)": "GraphCLIP official checkpoint (512-cap)"}
    for r in read_csv(path):
        if r["method"] not in names:
            SKIPPED.append({"source": path, "dataset": r["dataset"], "method": r["method"], "reason": "mirrored earlier primary table"})
            continue
        official = "GraphCLIP" in r["method"]
        add("25_clip_large" if official else "14_simplified", r["dataset"], names[r["method"]], r["f1"], r["accuracy"], source=path,
            f1_sd=number(r["f1_sd"]), accuracy_sd=number(r["accuracy_sd"]),
            sd_unit="not applicable (frozen checkpoint)" if official else "source-provided Student-seed SD",
            memory_rows=None if official else number(r["rows"]),
            evaluation_samples=number(r["rows"]) if official else None, student_seeds=r["seeds"],
            n_student_runs=0 if official else 3, n_orders=None if official else 1,
            student_steps=0 if official else 1024, underlying_sources=r["source"])


def legacy_and_small():
    path = "results/puma_static_unified_comparison_20261006/strict_continual_cells.csv"
    names = {"Adaptive coherent": "Adaptive coherent", "CaT-MMD-TAG + CCTP top-5": "CaT-CCTP",
        "GECC-hard-TAG adapted": "GECC-hard", "K-center/FPS": "K-center/FPS", "Medoid coreset": "Medoid",
        "PUMA-inspired TAG": "PUMA-inspired", "Random real coreset": "Random real coreset"}
    for r in read_csv(path):
        add("15_coherent_era", r["dataset"], names[r["method"]], r["final_f1"], r["accuracy"], source=path,
            f1_sd=number(r["final_f1_std"]), accuracy_sd=number(r["accuracy_std"]),
            sd_unit="source-provided Student-seed SD", n_orders=1, students_per_cell=6, n_student_runs=6,
            old_f1=number(r["old_f1"]), new_f1=number(r["new_f1"]), stage_average_f1=number(r["stage_avg_f1"]),
            construction_seconds=number(r["construction_seconds"]), underlying_sources=r["source"])
    path = "results/puma_static_unified_comparison_20261006/offline_reference_cells.csv"
    for r in read_csv(path):
        add("24_static", r["dataset"], "Static-TAG full-history", r["final_f1"], r["accuracy"], source=path,
            f1_sd=number(r["final_f1_std"]), accuracy_sd=number(r["accuracy_std"]), sd_unit="source-provided Student-seed SD",
            old_f1=number(r["old_f1"]), new_f1=number(r["new_f1"]), underlying_sources=r["source"])
    path = "results/master_protocol_table_20261004/master_protocol_table_20261004.csv"
    small = defaultdict(list)
    names = {"CaT-MMD-TAG": "CaT (nearest-text adapter)", "GECC-hard-TAG": "GECC-hard",
        "k-center/FPS": "K-center/FPS", "medoid": "Medoid", "PUMA-inspired": "PUMA-inspired",
        "semantic-recursive": "Semantic-recursive-inspired", "CaT + CCTP top-5": "CaT-CCTP",
        "K-center": "K-center/FPS", "Medoid": "Medoid", "Random real-node coreset": "Random real coreset"}
    METHODS["CaT (nearest-text adapter)"] = ("CaT", "external_baseline", "mechanism adaptation",
        "CaT 类级分布匹配 + 最近真实文本附着；早期三阶段协议")
    master = read_csv(path)
    for r in master:
        if r["protocol_id"] == "strict_small_3stage_seed150_155":
            small[(r["dataset"], r["method"])].append(r)
        elif r["protocol_id"] == "large_base4_seed100_105_314159" and r["method"] in names:
            add("18_large_legacy", r["dataset"], names[r["method"]], r["macro_f1"], r["accuracy"], source=path,
                old_f1=number(r["old_f1"]), new_f1=number(r["new_f1"]), memory_rows=number(r["memory_rows"]),
                n_orders=1, n_construction_seeds=1, students_per_cell=6, n_student_runs=6,
                student_seeds=r["student_seeds"], construction_seeds=r["construction_seed"], order_seeds=r["order"],
                sd_unit="not recorded in this export", underlying_sources=r["source"])
        else:
            SKIPPED.append({"source": path, "dataset": r["dataset"], "method": r["method"],
                "reason": "internal ablation / protocol-pooled row / small reference mirrored in registry"})
    for (dataset, method), vals in small.items():
        if method not in names:
            raise AssertionError(f"Unmapped small external method: {method}")
        assert len({r["order"] for r in vals}) == len(vals)
        add("16_small_strict", dataset, names[method], mean(r["macro_f1"] for r in vals), mean(r["accuracy"] for r in vals),
            source=path, f1_sd=sd(r["macro_f1"] for r in vals), accuracy_sd=sd(r["accuracy"] for r in vals),
            sd_unit="sample SD of Student-averaged order means", n_orders=len(vals), n_cells=len(vals),
            n_construction_seeds=len({r["construction_seed"] for r in vals}), students_per_cell=6,
            n_student_runs=len(vals) * 6, student_seeds=vals[0]["student_seeds"],
            order_seeds=",".join(r["order"] for r in vals),
            old_f1=mean(r["old_f1"] for r in vals), new_f1=mean(r["new_f1"] for r in vals),
            memory_rows=mean(r["memory_rows"] for r in vals), underlying_sources=" | ".join(r["source"] for r in vals))
    path = "results/cctp_baseline_matrix_20261001/pooled_by_dataset.csv"
    names = {"CaT-TAG CCTP (text-center)": "CaT (text-center adapter)", "CaT-TAG nearest": "CaT (nearest-text adapter)",
        "GECC-hard-TAG adapter": "GECC-hard", "k-center native append": "K-center/FPS", "medoid native append": "Medoid"}
    METHODS["CaT (text-center adapter)"] = ("CaT", "external_baseline", "text-attachment variant",
        "CaT 类级分布匹配 + text-center 附着；与 nearest/top-5 是同一基线的附着变体，不计新增公开方法")
    for r in read_csv(path):
        add("17_small_cctp", r["dataset"], names[r["method"]], r["final_f1"], r["final_accuracy"], source=path,
            f1_sd=number(r["final_f1_order_std"]), sd_unit="source-provided SD of order means", n_orders=int(r["cells"]),
            memory_rows=number(r["memory_rows"]), old_f1=number(r["old_f1"]), new_f1=number(r["new_f1"]),
            stage_average_f1=number(r["stage_avg_f1"]))
    path = "results/public_baselines_large_corrected_summary_20261003/dataset_cells.csv"
    # Resolve the source's human labels as well as arms: arm spelling changed.
    for r in read_csv(path):
        label = r["method"]
        method = next((name for prefix, name in (("Random real", "Random real coreset"), ("K-center real", "K-center/FPS"),
            ("Medoid real", "Medoid"), ("GECC-TAG hard", "GECC-hard"), ("CaT-TAG CCTP top-5", "CaT-CCTP")) if label.startswith(prefix)), None)
        if method is None:
            SKIPPED.append({"source": path, "dataset": r["dataset"], "method": label, "reason": "internal component/control"})
            continue
        duplicate = next((v for v in ROWS if v["panel_id"] == "18_large_legacy" and v["dataset"] == r["dataset"] and v["method"] == method
            and abs(v["final_f1"] - float(r["final_f1"])) < 1e-6 and abs(v["accuracy"] - float(r["final_accuracy"])) < 1e-6), None)
        if duplicate:
            duplicate["mirrored_exports"] = path
            SKIPPED.append({"source": path, "dataset": r["dataset"], "method": label, "reason": "same results as master old-large panel"})
            continue
        add("19_large_corrected", r["dataset"], method, r["final_f1"], r["final_accuracy"], source=path,
            f1_sd=number(r["final_f1_std"]), accuracy_sd=number(r["final_accuracy_std"]),
            sd_unit="source-provided Student-seed SD", n_orders=1, students_per_cell=6, n_student_runs=6,
            student_seeds="100,101,102,103,104,105", old_f1=number(r["old_f1"]), new_f1=number(r["new_f1"]))
    path = "results/bounded_xp_final_evidence_20261001/large_strict_rows.csv"
    names = {"CaT-MMD-TAG adapted": "CaT (nearest-text adapter)", "GECC-hard-TAG adapted": "GECC-hard",
        "k-center": "K-center/FPS", "medoid": "Medoid", "random-real": "Random real coreset"}
    for r in read_csv(path):
        if r["method"] not in names:
            SKIPPED.append({"source": path, "dataset": r["dataset"], "method": r["method"], "reason": "internal XP/edge control"})
            continue
        add("20_bounded", r["dataset"], names[r["method"]], r["final_f1"], r["final_accuracy"], source=path,
            f1_sd=number(r["final_f1_std"]), sd_unit="source-provided Student-seed SD",
            memory_rows=number(r["memory_rows"]), old_f1=number(r["old_f1"]), new_f1=number(r["new_f1"]),
            stage_average_f1=number(r["stage_average_f1"]), underlying_sources=r["source"])
    path = "results/external_tag_comparison_2026_09_17/summary.csv"
    names = {"random_real": "Random real coreset", "k_center": "K-center/FPS", "k_center_class": "K-center (class-wise)",
        "herding": "Herding", "kmeans_medoid": "Medoid", "stad_local_canonical": "STAD (local canonical reference)",
        "tagsam_static_selfloop": "TAGSAM (static self-loop reference)", "tagsam_static_semantic": "TAGSAM (static semantic reference)",
        "full_history_syn": "Full-history synthetic (early)", "raw_equal": "Full raw (equal steps, early)",
        "raw_long_selected": "Full raw (long training, early)"}
    for r in read_csv(path):
        if r["arm"] not in names:
            SKIPPED.append({"source": path, "method": r["arm"], "dataset": r["dataset"], "reason": "early internal main candidate"})
            continue
        add("21_early_small", r["dataset"], names[r["arm"]], r["final_f1"], None, source=path,
            f1_sd=number(r["std_population"]), sd_unit="source-provided Student population SD", n_orders=1,
            students_per_cell=len(json.loads(r["final_per_seed"])), n_student_runs=len(json.loads(r["final_per_seed"])),
            student_steps=number(r["student_steps"]), memory_rows=number(r["rows"]),
            actual_rate=number(r["node_ratio_percent"]) / 100 if number(r["node_ratio_percent"]) is not None else None,
            old_f1=number(r["final_old_f1"]), new_f1=number(r["final_new_f1"]), stage_average_f1=number(r["stage_average_f1"]))


def full_history():
    path = "results/full_raw_prompt_reference_20261011/full_raw_vs_unified_q.csv"
    paired = read_csv(path)
    for r in paired:
        add("22_raw_self", r["dataset"], "Unified-Q-EF", r["unified_q_f1"], r["unified_q_accuracy"], source=path,
            memory_rows=int(r["budget_rows"]), raw_pool_rows=int(r["full_raw_rows"]),
            actual_rate=int(r["budget_rows"]) / int(r["full_raw_rows"]),
            n_orders=1, students_per_cell=3, n_student_runs=3, student_seeds="150,151,152", student_steps=2048,
            sd_unit="not recorded in paired rounded export", f1_retention_pct=float(r["f1_retention_pct"]),
            accuracy_retention_pct=float(r["accuracy_retention_pct"]), source_precision="two-decimal published comparison")
        for edgefree in (True, False):
            filename = f"{r['dataset']}_full_raw_prompt_student{'_edgefree' if edgefree else ''}.json"
            rawpath = ROOT / "results/full_raw_prompt_reference_20261011" / filename
            payload = read_json(rawpath)
            final = max(payload["rows"], key=lambda v: int(v["stage"]))
            score = final["score"]
            add("22_raw_self" if edgefree else "23_raw_graph", r["dataset"],
                "Full-history raw (self-loops)" if edgefree else "Full-history raw (real graph)",
                score["final_f1"], score["accuracy"], source=relpath(rawpath),
                f1_sd=number(score.get("final_f1_std")), accuracy_sd=number(score.get("accuracy_std")),
                sd_unit="source-provided Student-seed SD", n_orders=1, students_per_cell=3, n_student_runs=3,
                student_seeds=",".join(map(str, payload["student_seeds"])), student_steps=payload["student_steps"],
                student_batch=payload["student_batch"], memory_rows=final["raw_rows"], raw_pool_rows=final["raw_rows"], actual_rate=1.0,
                old_f1=number(score.get("old_f1")), new_f1=number(score.get("new_f1")))
            if edgefree:
                assert round(float(score["final_f1"]), 2) == float(r["full_raw_f1"])
                assert round(float(score["accuracy"]), 2) == float(r["full_raw_accuracy"])
    write_csv(OUT / "full_history_comparison.csv", [dict(r, actual_rate=int(r["budget_rows"]) / int(r["full_raw_rows"]),
        split="report_dev", note="historical paired snapshot; do not substitute latest sealed scores") for r in paired])


def inventory_mirrors():
    mirrors = {
        "results/final_baseline_registry_20261011/strict_latest_prompt.csv": "mirrors pre_submission/extended panel",
        "results/final_baseline_registry_20261011/large_external_appendix.csv": "mirrors large_external_comparison",
        "results/large_baseline_panel_20261011/large_graph_baseline_panel.csv": "older integrated export; original inputs selected instead",
        "results/multidataset_f1_accuracy_comparison_20261006/dataset_method_cells.csv": "mirrors PUMA/coherent-era strict cells",
        "results/expanded_baseline_comparison_20261006/all_cells.csv": "contains mirrored baselines and many CaT text-attachment ablations, not distinct public methods",
        "results/small_comparison_summary_2026_09_16/summary.csv": "overlaps 2026-09-17 small external report and internal candidates",
        "results/public_baselines_large_corrected_summary_20261003/pooled_methods.csv": "pooled across datasets; not a per-dataset comparison",
        "results/bounded_xp_final_evidence_20261001/large_pooled.csv": "pooled across datasets; not a per-dataset comparison",
    }
    for rel, note in mirrors.items():
        p = ROOT / rel
        if p.exists():
            track(p, disposition="excluded_mirrored_or_pooled", note=note)
        else:
            SKIPPED.append({"source": rel, "reason": "optional mirrored export absent; canonical sources used"})


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, decimals=2):
    return "未记录" if value is None or value == "" else f"{float(value):.{decimals}f}"


def metric(row, value_key, sd_key):
    value = fmt(row.get(value_key))
    if row.get(sd_key) is not None:
        value += " ± " + fmt(row[sd_key])
    return value


def count_label(r):
    runs = r.get("n_student_runs")
    run_label = f"{runs} runs" if runs is not None else "Student runs 未完整记录"
    if r.get("n_cells"):
        return f"{r['n_cells']} cells / {run_label}"
    if r.get("n_orders"):
        return f"{r['n_orders']} orders / {run_label}"
    return r.get("orders_description", "未完整记录")


def sortrow(r):
    return (PANELS[r["panel_id"]]["priority"], 0 if r["method"] == "Unified-Q-EF" else 1,
        {"CaT-CCTP": 0, "GECC-hard": 1}.get(r["method"], 2), r["method"], r.get("split", ""))


def table(rows, show_panel=False):
    headers = (["组"] if show_panel else []) + ["方法", "Macro-F1 (%)", "Accuracy (%)", "覆盖", "证据身份", "来源 ID"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in sorted(rows, key=sortrow):
        values = ([r["panel_id"]] if show_panel else []) + [r["method"], metric(r, "final_f1", "f1_sd"),
            metric(r, "accuracy", "accuracy_sd"), count_label(r), r["fidelity"], r["record_id"]]
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in values) + " |")
    return lines


def validate_and_write():
    unique = set()
    for collection, prefix in ((ROWS, "B"), (CONTROLS, "C")):
        collection.sort(key=lambda r: (DATASETS.index(r["dataset"]), *sortrow(r)))
        for ix, r in enumerate(collection, 1):
            key = (r["panel_id"], r["dataset"], r["method"], r["split"])
            if key in unique:
                raise AssertionError(f"Duplicate aggregate identity: {key}")
            unique.add(key)
            for field in ("final_f1", "accuracy", "old_f1", "new_f1", "stage_average_f1"):
                value = r.get(field)
                if value is not None and not 0 <= float(value) <= 100:
                    raise AssertionError(f"Metric outside [0,100]: {key}/{field}={value}")
            r["record_id"] = f"{prefix}{ix:03}"
    CHECKS.append({"check": "aggregate identity / metric range / missingness", "status": "passed",
        "detail": f"{len(ROWS)} baseline/main/reference rows; {len(CONTROLS)} internal rows; no duplicate panel identities; missing Accuracy left blank."})
    write_csv(OUT / "baseline_results.csv", ROWS)
    write_csv(OUT / "internal_candidates_and_ablations.csv", CONTROLS)
    write_csv(OUT / "paired_comparisons.csv", PAIRS)
    write_csv(OUT / "protocol_panels.csv", list(PANELS.values()))
    write_csv(OUT / "source_inventory.csv", list(SOURCES.values()))
    write_csv(OUT / "excluded_rows.csv", SKIPPED)
    write_csv(OUT / "validation.csv", CHECKS)
    methods = []
    for name in sorted({r["method"] for r in ROWS}):
        m = METHODS[name]
        methods.append({"method": name, "original_method": m[0], "role": m[1], "fidelity": m[2], "adaptation": m[3],
            "datasets_with_numbers": ",".join(d for d in DATASETS if any(r["method"] == name and r["dataset"] == d for r in ROWS)),
            "official_condensation_reproduction": False})
    write_csv(OUT / "method_adaptations.csv", methods)
    coverage = []
    for method in sorted({r["method"] for r in ROWS if r["role"] in {"external_baseline", "coreset"}}):
        for dataset in DATASETS:
            found = [r for r in ROWS if r["dataset"] == dataset and r["method"] == method]
            coverage.append({"dataset": dataset, "method": method, "numerical_panels": len(found),
                "has_latest_sealed_9cell": any(r["panel_id"] == "01_sealed" for r in found),
                "has_native_core_current_student": any(r["panel_id"] == "03_native_sealed"
                    and r["fidelity"] == "native-core adaptation" for r in found),
                "panels": ",".join(r["panel_id"] for r in found)})
    write_csv(OUT / "coverage.csv", coverage)


def reports():
    intro = ["# 已完成基线比较：按数据集总表（2026-10-13）", "",
        "本报告汇总既有数值结果，不启动新实验、不覆盖 sealed 文件。Macro-F1 与 Accuracy 均为百分数；缺失数据保留为空。", "",
        "**当前横向表与稳定性表：**先列五图十配置同一 cell 的 sealed 横向比较（每图主方法加九个对照配置），再单列三大图 EF/CaT/GECC 的九 cell 稳定性。新增对照只有单 cell，不能把它们的 Student SD 当作九 cell SD。受限原生参考、旧 Student 和旧调度均另列，不合并排名。", "",
        "**统计口径：**扩展横向表的 ± 是同一顺序/构造种子下三个 Student seeds 的样本 SD。九 cell 表先平均同一 cell 的 Student seeds，再计算 cell 均值/样本 SD；paired CI 基于九个配对 cell。历史表另见分组说明。", "",
        "**方法身份：**CCTP 是本项目的图特征近邻文本附着规则，不是独立公开基线，也不是随机选五篇。CaT 的附着变体不算多个公开方法；PUMA/DANCE/SR-GM 简化参考不标为官方复现；GraphCLIP 是预训练推断参考而非凝聚算法。", "",
        "[全部数据集完整总表](all_results_by_dataset.md) 包含下方各数据集的全部分组。所有数值行及其来源见 [baseline_results.csv](baseline_results.csv)，[原方法与适配说明](method_adaptations.md) 逐项对应方法名称。内部候选/消融见 [单独的消融表](internal_candidates_and_ablations.md)。", ""]
    complete = ["# 全部已收录基线比较：按数据集分组", "",
        "[当前主表与结论](summary.md) · [方法适配说明](method_adaptations.md) · [带来源的 CSV](baseline_results.csv)", "",
        "Macro-F1 和 Accuracy 均为百分数。每个数据集按实验版本、Student、split 和预算分组；各组中的主方法列在首行。不同组不能合并排名。缺失值标为未记录，组件消融另列。九 cell 结果的 SD 以先平均三个 Student seeds 后的九个 cell 为单位；单 cell 的 SD 只反映 Student seeds 波动。", ""]
    internal_report = ["# 内部候选与组件消融", "",
        "[返回主表](summary.md) · [带元数据的 CSV](internal_candidates_and_ablations.csv)", "",
        "这些行不计为外部公开方法。按数据集与协议分组，保留已有负结果；不同构造版本和 split 的结果不能混合计算增益。", ""]
    adaptations = ["# 原方法与本任务适配的对应关系", "",
        "名称后缀对应实际实现，不把同一原论文的多个文本附着版本累计为多个公开方法。native-core adaptation 表示调用或保留原生核心，同时修改持续封装或 Student；它不等于完整官方复现。", "",
        "| 表中方法名 | 原方法 | 实现身份 | 实际做法 | 已有数值的数据集 |",
        "| --- | --- | --- | --- | --- |"]
    for name in sorted({r["method"] for r in ROWS}):
        original, _, fidelity, adaptation = METHODS[name]
        datasets = ", ".join(DISPLAY[d] for d in DATASETS
            if any(r["method"] == name and r["dataset"] == d for r in ROWS))
        adaptations.append("| " + " | ".join((name, original, fidelity, adaptation, datasets)) + " |")
    (OUT / "method_adaptations.md").write_text("\n".join(adaptations) + "\n", encoding="utf-8")
    for dataset in DATASETS:
        own = [r for r in ROWS if r["dataset"] == dataset]
        title = DISPLAY[dataset]
        detail = [f"# {title}：全部已收录基线比较", "", "[返回总表](../summary.md)", "",
            "每组对应独立的协议/版本；行内的来源 ID 可在 baseline_results.csv 中定位精确来源和元数据。不能在不同组之间按最高分重新排名。", ""]
        complete.extend([f"## {title}", "", f"[独立表格及来源索引](by_dataset/{dataset}.md)", ""])
        intro.extend([f"## {title}", "", f"[完整分组表（{len(own)} 行）](by_dataset/{dataset}.md)", ""])
        primary = [r for r in own if r["panel_id"] == "00_expansion_sealed"]
        if not primary:
            priority = next((p for p in ("07_pre_prompt", "09_extra", "16_small_strict", "18_large_legacy") if any(r["panel_id"] == p for r in own)), None)
            primary = [r for r in own if r["panel_id"] == priority]
            intro.extend(["该数据集尚无五图扩展同一 sealed cell 比较。以下是已完成的较早同组结果；完整表保留其它版本。", ""])
        if primary:
            intro.extend(table(primary))
            intro.extend(["", f"协议：{PANELS[primary[0]['panel_id']]['title']}；同一 cell 的三个 Student seeds。", ""])
        stable = [r for r in own if r["panel_id"] == "01_sealed"]
        if stable:
            intro.extend(["**九 cell 稳定性确认（只覆盖三方法）**", ""])
            intro.extend(table(stable))
            intro.append("")
        qualified = [r for r in own if r["panel_id"] == "00_qualified_sealed"]
        if qualified:
            intro.extend(["**受限参考，不参与十配置同协议排名**", ""])
            intro.extend(table(qualified))
            intro.append("")
        for pkey in sorted({r["panel_id"] for r in own}, key=lambda p: PANELS[p]["priority"]):
            p = PANELS[pkey]
            panelrows = [r for r in own if r["panel_id"] == pkey]
            detail.extend([f"## {p['title']}", "", p["note"], "",
                f"Student：{p['student']}；split：`{p['split']}`；budget：{p['budget']}。", ""])
            detail.extend(table(panelrows))
            detail.extend(["", "SD 口径：" + "；".join(sorted({r.get("sd_unit", "未记录") for r in panelrows})) + "。", ""])
            complete.extend([f"### {p['title']}", "", p["note"], "",
                f"Student：{p['student']}；split：`{p['split']}`；budget：{p['budget']}。", ""])
            complete.extend(table(panelrows))
            complete.extend(["", "SD 口径：" + "；".join(sorted({r.get("sd_unit", "未记录") for r in panelrows})) + "。", ""])
            if any(r.get("actual_rate") for r in panelrows):
                detail.append("已记录的实际行数比例：" + "；".join(f"{r['method']} {float(r['actual_rate']) * 100:.3f}%" for r in panelrows if r.get("actual_rate") is not None) + "。")
                detail.append("")
        internal = [r for r in CONTROLS if r["dataset"] == dataset]
        if internal:
            detail.extend(["## 内部候选与组件消融（不计外部基线）", "",
                "这些结果单独存档，保留版本与 split；不能用来增加公开基线数量。", ""])
            detail.extend(table(internal, show_panel=True))
            detail.append("")
            internal_report.extend([f"## {title}", ""])
            for pkey in sorted({r["panel_id"] for r in internal}, key=lambda p: PANELS[p]["priority"]):
                p = PANELS[pkey]
                internal_report.extend([f"### {p['title']}", "", p["note"], ""])
                internal_report.extend(table([r for r in internal if r["panel_id"] == pkey]))
                internal_report.append("")
        detail.extend(["## 来源索引", ""])
        for r in own:
            source = r["source"].split(" | ")[0]
            target = "../../../" + source.replace("\\", "/")
            detail.append(f"- {r['record_id']}：[{source}]({target})；`{r['panel_id']}`，{r['method']}。")
        path = OUT / "by_dataset" / f"{dataset}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(detail) + "\n", encoding="utf-8")
        write_csv(OUT / "by_dataset" / f"{dataset}.csv", own)
    (OUT / "all_results_by_dataset.md").write_text("\n".join(complete) + "\n", encoding="utf-8")
    (OUT / "internal_candidates_and_ablations.md").write_text("\n".join(internal_report) + "\n", encoding="utf-8")
    intro.extend(["## 最终同协议配对差值", "", "以下为 sealed test 上的 Macro-F1 差值；相同顺序、构造种子和 Student seeds 配对。", "",
        "| 数据集 | Unified-Q-EF − baseline | ΔF1 (pp) | 95% paired CI | positive cells | worst ΔF1 |",
        "| --- | --- | --- | --- | --- | --- |"])
    for r in PAIRS:
        if r["split"] == "sealed_test" and r["metric"] == "final_f1":
            intro.append(f"| {DISPLAY[r['dataset']]} | {r['baseline']} | {r['mean_delta_pp']:+.2f} | [{r['ci95_low']:.2f}, {r['ci95_high']:.2f}] | {r['positive_cells']}/9 | {r['worst_delta_pp']:+.2f} |")
    intro.extend(["", "BookHis 对 GECC-hard 的收益最明确；ArXiv/SportsFit 的配对区间覆盖零。三个数据集均优于本协议的 CaT-CCTP 适配，但没有证据支持‘普遍超过 GECC 约 2 个点’。", "",
        "## 完整历史参考", "", "沿用 2026-10-11 的同一 report-dev 配对，不将新 sealed 分数放入旧 retention 分母。raw 训练自环版本与真实图版本分别报告。保留率为指标之比，不是理论上界。", "",
        "| 数据集 | 合成 / raw 行数 | 实际比例 | 当时 EF F1 / Acc | raw 自环 F1 / Acc | F1 保留率 | Acc 保留率 |",
        "| --- | --- | --- | --- | --- | --- | --- |"])
    for r in read_csv("results/full_raw_prompt_reference_20261011/full_raw_vs_unified_q.csv"):
        rate = int(r["budget_rows"]) / int(r["full_raw_rows"]) * 100
        intro.append(f"| {DISPLAY[r['dataset']]} | {r['budget_rows']} / {r['full_raw_rows']} | {rate:.2f}% | {float(r['unified_q_f1']):.2f} / {float(r['unified_q_accuracy']):.2f} | {float(r['full_raw_f1']):.2f} / {float(r['full_raw_accuracy']):.2f} | {float(r['f1_retention_pct']):.2f}% | {float(r['accuracy_retention_pct']):.2f}% |")
    intro.extend(["", "## 覆盖与缺口", "",
        "五图均完成十配置同一 sealed cell；其中公开机制适配、真实节点选样和 inspired reference 不是九篇官方复现。TAGSAM bounded 与 STAD local Qwen 在三大图完成，但因计算预算或监督/拓扑不同单列；GECC-soft 仅 BookHis 完成。依用户决定，ArXiv/SportsFit 的 GECC-soft 停止重试，不把缺失分数记为零。", "",
        "GCond、SFGC、GCondenser 只有源码/协议检查记录，没有可用的本任务 F1/Accuracy，未填造数值。GraphCLIP 使用官方 checkpoint，但未复现训练、更不是本任务官方持续凝聚。", "",
        "历史 CaT 附着变体、Adaptive coherent、Q/GM/DM 消融以及 topology 分支不能累计为独立公开方法。数值覆盖矩阵见 [coverage.csv](coverage.csv)，重复/排除项见 [excluded_rows.csv](excluded_rows.csv)。", ""])
    (OUT / "summary.md").write_text("\n".join(intro), encoding="utf-8")
    audit = ["# 总表核查", "", f"已收录 {len(ROWS)} 个主方法/外部比较/参考汇总行、{len(CONTROLS)} 个内部候选/组件行，覆盖 {len(DATASETS)} 个数据集。", "",
        "0. 最新扩展快照 57/59 完成；核对五图十配置各一 sealed/report-dev 行、7 个受限参考、57 个快照登记 SHA-256 与 matched 表一致、九 cell 对应单元；通过。远端原始 JSON 不全在本地，独立核验只复算了快照中存储的 Student 指标；未完成两项为 GECC-soft ArXiv/SportsFit，停止重试。",
        "1. 重新计算最终三图 F1/Acc cell 均值与 sample SD，并逐 cell 对照最终阶段 Student 行；通过。",
        "2. 每个最终主方法/CaT/GECC 确认 3×3×3 完成；两个 split 分开，paired CI 以九个 cell 为单位；通过。",
        "3. 原生核心适配只与 o9101/c271006 的主方法行同表；通过。",
        "4. 缺失 Accuracy 和不可估计的单 cell/order SD 留空，不填 0；通过。",
        "5. 不使用历史最高分代替当前结果，不混淆 BookChild 与 BookHis；通过。",
        "6. Source SHA-256、协议分组及原方法/适配描述均有独立 CSV；旧 sealed 文件只读。", "",
        "汇总行数不等于独立实验次数：dev/sealed 是同一记忆的不同评估，原生核心表复用一个配对 cell，小图早期矩阵也有重叠。", "",
        "部分历史 source 已是汇总导出，只有该导出的值/协议字段可本地核查；不把缺失的 source-level seeds、Accuracy SD 或实际预算补成推测值。", ""]
    (OUT / "audit.md").write_text("\n".join(audit), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    init_panels()
    latest()
    expansion()
    current_matrix()
    earlier_prompt_and_sage()
    registry_and_broad_adapters()
    legacy_and_small()
    full_history()
    inventory_mirrors()
    validate_and_write()
    reports()
    manifest = {"generator": relpath(Path(__file__)), "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "existing-results-only; no reruns or test-based selection", "sources": list(SOURCES.values()),
        "rows": len(ROWS), "internal_rows": len(CONTROLS), "dataset_counts": {d: sum(r["dataset"] == d for r in ROWS) for d in DATASETS},
        "checks": CHECKS, "outputs": {relpath(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob("*") if p.is_file() and p.name != "manifest.json"}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "rows": len(ROWS), "internal_rows": len(CONTROLS),
        "dataset_counts": manifest["dataset_counts"], "validation": "passed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
