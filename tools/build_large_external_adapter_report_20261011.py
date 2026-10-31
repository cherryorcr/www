"""Assemble the 2026-10-11 large external adapter appendix."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "large_external_adapters_20261011"
REG = ROOT / "results" / "final_baseline_registry_20261011" / "BASELINE_REGISTRY_20261011.json"
DATASETS = ["arxiv", "sportsfit", "bookhis", "wikics"]


def strict_rows() -> list[dict]:
    registry = json.loads(REG.read_text(encoding="utf-8"))
    wanted = {"Unified-Q", "GECC-hard-TAG", "CaT-CCTP"}
    rows = []
    for row in registry["strict_latest_prompt"]:
        if row["dataset"] in DATASETS and row["method"] in wanted:
            rows.append({
                "dataset": row["dataset"], "method": row["method"],
                "tier": "strict-latest-prompt", "status": "comparable",
                "f1": row["final_f1"], "accuracy": row["accuracy"],
                "f1_sd": row.get("final_f1_sd"), "accuracy_sd": row.get("accuracy_sd"),
                "rows": None, "seeds": row.get("student_seeds"),
                "source": row.get("source"),
            })
    return rows


def adapter_rows() -> list[dict]:
    rows = []
    for ds in DATASETS:
        payload = json.loads((OUT / ds / "results.json").read_text(encoding="utf-8"))
        for row in payload["results"]:
            final = row["final"]
            rows.append({
                "dataset": ds, "method": row["method"],
                "tier": "mechanism-adapted-large", "status": "not-comparable",
                "f1": final["macro_f1"], "accuracy": final["accuracy"],
                "f1_sd": final["macro_f1_std"], "accuracy_sd": final["accuracy_std"],
                "rows": final["rows"], "seeds": ",".join(map(str, payload["manifest"]["student_seeds"])),
                "source": str(OUT / ds / "results.json"),
            })
    return rows


def graphclip_rows() -> list[dict]:
    rows = []
    for path in sorted(OUT.glob("graphclip_large_reference_20261011_*512cap.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        final = payload["final"]["all"]
        rows.append({
            "dataset": payload["dataset"], "method": "GraphCLIP official frozen (512-cap)",
            "tier": "official-reference-large", "status": "not-comparable",
            "f1": final["cumulative_f1"], "accuracy": final["accuracy"],
            "f1_sd": None, "accuracy_sd": None, "rows": 512,
            "seeds": "frozen checkpoint", "source": str(path),
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = strict_rows() + adapter_rows() + graphclip_rows()
    frame = pd.DataFrame(rows).sort_values(["dataset", "tier", "method"])
    frame.to_csv(OUT / "large_external_comparison.csv", index=False)

    lines = [
        "# Large External Baseline Appendix (2026-10-11)", "",
        "本附录补充 STAD、DANCE、SR-GM 和 GraphCLIP 的大图实验。", "",
        "## Protocol Tiers", "",
        "- `strict-latest-prompt`: 已有 Unified-Q、GECC-hard-TAG、CaT-CCTP，使用同一 Prompt Student，可进入严格主表。",
        "- `mechanism-adapted-large`: STAD、DANCE、SR-GM 保留可识别的原方法机制，但原论文没有 current-delta-only + frozen-prefix 接口，不能称 official reproduction。",
        "- `official-reference-large`: GraphCLIP 使用官方 frozen checkpoint 和完整原图 ego context；512 节点 deterministic report-dev reference，不是持续凝聚结果。",
        "",
        "## Results", "",
        "数值格式为 `Macro-F1 / Accuracy`，adapter 的 `+-` 是 3 个 Student seeds 的标准差。", "",
    ]
    for ds in DATASETS:
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| Tier | Method | Final Macro-F1 | Accuracy | F1 SD | Acc SD | Status |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for r in frame[frame.dataset == ds].to_dict("records"):
            f1_sd = "-" if pd.isna(r["f1_sd"]) else f"{r['f1_sd']:.2f}"
            acc_sd = "-" if pd.isna(r["accuracy_sd"]) else f"{r['accuracy_sd']:.2f}"
            lines.append(f"| {r['tier']} | {r['method']} | {r['f1']:.2f} | {r['accuracy']:.2f} | {f1_sd} | {acc_sd} | {r['status']} |")
        lines.append("")
    lines += [
        "## Interpretation", "",
        "1. 这些结果补齐了大图覆盖，但不能把机制适配结果写成原论文的官方复现。",
        "2. STAD/DANCE/SR-GM adapter 使用共同 Prompt GraphText Student、5% cumulative memory、current-delta-only 和 frozen synthetic prefix；adapter 构造使用阶段类别标签进行 quota 分配，但不读取 report/test labels。",
        "3. GraphCLIP reference 的输入是完整原图，且使用官方 GPS-12L checkpoint；它是模型参考或上界型参照，不是 memory condensation baseline。",
        "4. 新增结果应放在论文的 external reference / adaptation appendix，不应改写 strict continual 主表的可比性结论。",
        "",
        "## Provenance", "",
        "- Runner: `code/v2_llm_fusion/run_external_large_adapters_20261011.py`",
        "- GraphCLIP runner: `code/v2_llm_fusion/eval_official_graphclip_report.py`",
        "- Machine-readable table: `large_external_comparison.csv`",
    ]
    (OUT / "LARGE_EXTERNAL_BASELINE_APPENDIX_20261011.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_frame = frame.copy()
    plot_frame["label"] = plot_frame["method"].str.replace(" official frozen (512-cap)", "", regex=False)
    for metric, ylabel, name in [("f1", "Final Macro-F1", "large_external_f1.png"),
                                 ("accuracy", "Accuracy", "large_external_accuracy.png")]:
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(18, 5), constrained_layout=True)
        for ax, ds in zip(axes, DATASETS):
            sub = plot_frame[plot_frame.dataset == ds]
            colors = ["#1f77b4" if s == "comparable" else "#ff7f0e" for s in sub.status]
            ax.bar(range(len(sub)), sub[metric], color=colors)
            ax.set_title(ds)
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub.label, rotation=55, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=.25)
        fig.savefig(OUT / "figures" / name, dpi=180)
        plt.close(fig)

    print(frame.to_string(index=False))


if __name__ == "__main__":
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    main()
