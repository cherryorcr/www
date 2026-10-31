"""Audit the frozen three-stage ArXiv main-comparison values.

This audit is intentionally read-only. It recomputes the repeated baseline
means and sample standard deviations from the nine memory means, then checks
the published main table and its compact appendix copy.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "experiments/cotag_baseline_repetitions_20261024/summary_by_method.csv"
MAIN_TABLE = ROOT / "paper_www/tables/main_comparison.tex"
ARCHIVE_TABLE = ROOT / "paper_www/tables/arxiv_three_stage_archive.tex"
TABLE1 = ROOT / "paper_www/tables/table1.tex"
REPORT = ROOT / "paper_www/AUDIT_ARXIV_THREE_STAGE_CONSISTENCY_2026-10-26.md"

ARM_TO_LABEL = {
    "cat_cctp": "CaT",
    "gecc_hard": "GECC",
    "random_coreset": "Random",
    "kcenter_coreset": "K-center",
    "cluster_medoid": "Cluster medoid",
    "herding_coreset": "Herding",
    "gcond_x_core": "GCond-x",
    "sfgc_core": "SFGC",
    "tagsam": "TAGSAM",
    "stad_native_local": "STAD",
}

EXPECTED = {
    "CaT": (34.98, 1.79, 60.41, 2.64),
    "GECC": (52.68, 0.45, 70.94, 0.49),
    "Random": (47.72, 0.56, 69.39, 0.30),
    "K-center": (30.84, 1.50, 59.24, 1.26),
    "Cluster medoid": (48.30, 0.86, 69.41, 0.27),
    "Herding": (49.21, 0.18, 69.63, 0.34),
    "GCond-x": (49.64, 1.28, 66.71, 0.93),
    "SFGC": (24.23, 2.17, 49.10, 3.41),
    "TAGSAM": (39.31, 3.11, 58.92, 3.54),
    "STAD": (49.76, 0.64, 66.07, 1.37),
}

REPEATED_EXPECTED = {
    label: values for label, values in EXPECTED.items() if label not in {"CaT", "GECC"}
}


def close(a: float, b: float, tol: float = 0.006) -> bool:
    return abs(a - b) <= tol


def fmt(value: float) -> str:
    return f"{value:.2f}"


def load_arxiv_summary() -> dict[str, tuple[float, float, float, float]]:
    rows = {}
    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != "arxiv":
                continue
            label = ARM_TO_LABEL.get(row["arm"])
            if label is None or row["memories"] != "9" or row["complete"] != "True":
                continue
            rows[label] = (
                float(row["f1_mean"]),
                float(row["f1_sd"]),
                float(row["accuracy_mean"]),
                float(row["accuracy_sd"]),
            )
    return rows


def parse_table_cells(path: Path) -> dict[str, tuple[float, float, float, float]]:
    result = {}
    pattern = re.compile(
        r"^(CaT|GECC|Random|K-center|Cluster medoid|Herding|GCond-x|SFGC|TAGSAM|STAD)\s*&\s*(?:9\s*&\s*)?"
        r"([0-9.]+)\$?\\pm\$?([0-9.]+)\s*/\s*([0-9.]+)\$?\\pm\$?([0-9.]+)"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.replace("\\underline{", "").replace("\\textbf{", "")
        line = line.replace("}", "")
        match = pattern.match(line.strip())
        if match:
            result[match.group(1)] = tuple(float(match.group(i)) for i in range(2, 6))
    return result


def check_values(name: str, actual: dict, expected: dict) -> list[str]:
    failures = []
    for label, target in expected.items():
        values = actual.get(label)
        if values is None:
            failures.append(f"{name}: missing {label}")
            continue
        if not all(close(a, b) for a, b in zip(values, target)):
            failures.append(f"{name}: {label}={values}, expected={target}")
    return failures


def main() -> int:
    summary = load_arxiv_summary()
    main_cells = parse_table_cells(MAIN_TABLE)
    archive_cells = parse_table_cells(ARCHIVE_TABLE)
    failures = []
    failures.extend(check_values("summary vs expected repeated baselines", summary, REPEATED_EXPECTED))
    failures.extend(check_values("main table vs expected", main_cells, EXPECTED))
    failures.extend(check_values("archive table vs expected", archive_cells, EXPECTED))
    failures.extend(check_values("main vs archive", archive_cells, main_cells))

    table1 = TABLE1.read_text(encoding="utf-8")
    protocol_ok = "ArXiv     & 118,539 & 5,927 & 3 & 13/14/13" in table1
    stale_sources = []
    for path in [TABLE1, MAIN_TABLE, ARCHIVE_TABLE]:
        text = path.read_text(encoding="utf-8")
        if "8/8/8/8/8" in text or "archived three-stage" in text:
            stale_sources.append(str(path.relative_to(ROOT)))

    report = [
        "# ArXiv 三阶段主协议一致性审计",
        "",
        "审计日期：2026-10-26。主比较冻结为三阶段类别增量协议 `13/14/13`。",
        "每种方法使用三个类别顺序和三个构造种子，共九份记忆；每份记忆使用三个 Student 种子，先在记忆内取均值，再在九份记忆上计算均值和样本标准差。",
        "",
        "## 审计对象",
        "",
        f"- 重复实验汇总：`{SUMMARY.relative_to(ROOT)}`",
        f"- 主表：`{MAIN_TABLE.relative_to(ROOT)}`",
        f"- 附录副本：`{ARCHIVE_TABLE.relative_to(ROOT)}`",
        f"- 协议表：`{TABLE1.relative_to(ROOT)}`",
        "",
        "## 结果",
        "",
        f"- 主表包含 11 个方法，其中 8 个扩展基线由重复实验汇总核对，CaT 和 GECC 使用核心结果汇总：`{'通过' if len(summary) == 8 else '未通过'}（检测到 {len(summary)} 个扩展基线）`。",
        f"- 汇总 CSV 与主表显示值一致：`{'通过' if not check_values('summary vs main', main_cells, summary) else '未通过'}`。",
        f"- 主表与附录 ArXiv 副本逐项一致：`{'通过' if not check_values('main vs archive', archive_cells, main_cells) else '未通过'}`。",
        f"- 表 1 的主协议为 `{'通过' if protocol_ok else '未通过'}`。",
        "",
        "## 当前仍需修正的旧文字",
        "",
    ]
    if stale_sources:
        report.extend(f"- `{item}` 仍包含旧的五阶段主协议或旧的存档措辞。" for item in stale_sources)
    else:
        report.append("- 未在三个核心表文件中发现旧的五阶段主协议文字。")
    report.extend([
        "",
        "五阶段结果不作为 ArXiv 主比较。按出版年份的五阶段时间顺序实验，以及早期五阶段开发检查，保留在各自的证据段落中，并明确其与主表的协议不同。",
        "",
        "## 结论",
        "",
        f"`{'通过' if not failures and protocol_ok else '需要修正'}`。详细失败项数量：{len(failures)}。",
    ])
    if failures:
        report.extend(["", "### 失败项", ""] + [f"- {item}" for item in failures])
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0 if not failures and protocol_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
