from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "assets" / "adaptive_coherent"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save_flow():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    boxes = [
        (0.03, "当前阶段 raw delta\n节点、边、文本、标签"),
        (0.22, "CaT class-wise MMD\n梯度优化合成特征 X"),
        (0.41, "图文离散度\n每类分配 1/2/3 slots"),
        (0.60, "有限候选池\n真实文本、320 字符预算"),
        (0.79, "X-conditioned routing\nsoft assignment -> hard payload"),
    ]
    for x, label in boxes:
        ax.text(
            x,
            0.54,
            label,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#EAF2F8", edgecolor="#3776AB", linewidth=1.4),
        )
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 0.075
        x2 = boxes[i + 1][0] - 0.075
        ax.annotate("", xy=(x2, 0.54), xytext=(x1, 0.54), arrowprops=dict(arrowstyle="->", lw=1.8, color="#555555"))
    ax.text(0.5, 0.13, "最终：只追加新 memory；历史 raw 不回读，旧 synthetic prefix 不修改；当前版本不宣称拓扑创新", ha="center", fontsize=10, color="#444444")
    fig.tight_layout()
    fig.savefig(OUT / "method_flow.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_large_results():
    datasets = ["Arxiv", "SportsFit", "WikiCS", "BookHis"]
    ours = np.array([21.93, 47.31, 73.06, 14.71])
    cctp = np.array([18.93, 25.02, 50.44, 14.74])
    kcenter = np.array([17.50, 29.54, 50.29, 18.08])
    medoid = np.array([18.19, 29.04, 55.01, 17.33])
    x = np.arange(len(datasets))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(x - 1.5 * width, ours, width, label="Adaptive coherent", color="#1f77b4")
    ax.bar(x - 0.5 * width, cctp, width, label="CCTP top-5", color="#ff7f0e")
    ax.bar(x + 0.5 * width, kcenter, width, label="K-center/FPS", color="#2ca02c")
    ax.bar(x + 1.5 * width, medoid, width, label="Medoid", color="#9467bd")
    ax.set_ylabel("Final Macro-F1 (%)")
    ax.set_xticks(x, datasets)
    ax.set_title("独立顺序与 construction seed 汇总：9 个 cell 等权均值")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "large_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_protocol():
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    rows = [
        ("历史 raw graph", "不可回读", "避免把持续凝聚变成每阶段重凝聚"),
        ("当前 raw delta", "可访问", "候选文本和 CaT 当前阶段构造来源"),
        ("旧 synthetic memory", "只读 prefix", "不修改，保证 continual invariants"),
        ("memory budget", "5% cumulative", "所有方法统一"),
        ("Student", "Prompt zero-shot", "无 learned classifier head"),
        ("拓扑", "edge-free/self-loop-only", "当前版本不宣称拓扑创新"),
    ]
    table = ax.table(cellText=rows, colLabels=["协议项", "设置", "含义"], loc="center", cellLoc="left", colLoc="center", colWidths=[0.22, 0.25, 0.53])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#B8C2CC")
        if r == 0:
            cell.set_facecolor("#EAF2F8")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#FAFAFA")
    fig.tight_layout()
    fig.savefig(OUT / "protocol_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    save_flow()
    save_large_results()
    save_protocol()
    print(OUT)
