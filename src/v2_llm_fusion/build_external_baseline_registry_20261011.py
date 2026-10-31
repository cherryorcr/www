"""Build a protocol-aware external baseline registry for the TAG continual study.

The registry is intentionally tiered.  It keeps the newest matched Prompt
Student rows separate from the older formal external panel, adapted TAG
methods, native small-graph references, and methods for which a legal
continual constructor is unavailable.  No protocol-mismatched score is used
to create a single leaderboard.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HANDOFF = Path(r"D:/Aprojects/LARGE_GRAPH_BASELINE_HANDOFF_20261008/large_graph_baseline_20261008")
TAG_BASELINE = Path(r"D:/Aprojects/tag_baseline/EXTERNAL_BASELINE_HANDOFF_PACKAGE_20261004/EXTERNAL_BASELINE_HANDOFF_PACKAGE_20261004")
OUT = ROOT / "results/final_baseline_registry_20261011"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(value):
    if value in (None, "", "-"):
        return None
    return float(value)


def record(*, tier: str, method: str, dataset: str, f1=None, acc=None,
           old_f1=None, new_f1=None, f1_sd=None, acc_sd=None, orders="",
           seeds="", comparison: str, source: str, adaptation: str,
           note: str = "") -> dict:
    return {
        "tier": tier,
        "method": method,
        "dataset": dataset,
        "final_f1": f1,
        "accuracy": acc,
        "old_f1": old_f1,
        "new_f1": new_f1,
        "final_f1_sd": f1_sd,
        "accuracy_sd": acc_sd,
        "orders": orders,
        "student_seeds": seeds,
        "directly_comparable": comparison,
        "source": source,
        "adaptation": adaptation,
        "note": note,
    }


def latest_primary() -> list[dict]:
    rows = read_csv(ROOT / "results/pre_submission_evidence_20261010/primary_matched.csv")
    out = []
    keep = {"Unified-Q (edge-free)": "Unified-Q",
            "GECC-hard TAG adapter": "GECC-hard-TAG",
            "CaT-CCTP adapter": "CaT-CCTP"}
    for row in rows:
        if row["student"] != "Prompt Student" or row["method"] not in keep:
            continue
        out.append(record(
            tier="strict-latest-prompt",
            method=keep[row["method"]],
            dataset=row["dataset"],
            f1=number(row["final_f1"]),
            acc=number(row["accuracy"]),
            old_f1=number(row["old_f1"]),
            new_f1=number(row["new_f1"]),
            f1_sd=number(row["order_sd_f1"]),
            orders=row["n_orders"],
            seeds="150-155",
            comparison="yes",
            source="results/pre_submission_evidence_20261010/primary_matched.csv",
            adaptation=("Unified-Q shared assignment with in-loop distribution matching" if keep[row["method"]] == "Unified-Q"
                        else "current-delta-only mechanism adapter with legal TAG text attachment"),
            note="Newest matched Prompt Student panel; do not merge with legacy Student rows.",
        ))

    # ArXiv was added as a one-order current-protocol transfer cell.
    extra = read_csv(ROOT / "results/extended_evidence_20261011/additional_dataset_summary.csv")
    names = {"uq_dm_edgefree": "Unified-Q", "gecc_hard": "GECC-hard-TAG", "cat_cctp": "CaT-CCTP"}
    for row in extra:
        if row["dataset"] != "arxiv" or row["method"] not in names:
            continue
        out.append(record(
            tier="strict-latest-prompt",
            method=names[row["method"]], dataset="arxiv",
            f1=number(row["final_f1_mean"]), acc=number(row["accuracy_mean"]),
            old_f1=number(row["old_f1_mean"]), new_f1=number(row["new_f1_mean"]),
            f1_sd=number(row["final_f1_sd"]), acc_sd=number(row["accuracy_sd"]),
            orders=row["orders"], seeds="150-155", comparison="yes",
            source="results/extended_evidence_20261011/additional_dataset_summary.csv",
            adaptation=("Unified-Q shared assignment with in-loop distribution matching" if names[row["method"]] == "Unified-Q"
                        else "current-delta-only mechanism adapter with legal TAG text attachment"),
            note="One preregistered ArXiv transfer order; supplementary within the latest tier.",
        ))
    return out


def formal_external() -> list[dict]:
    path = HANDOFF / "server_snapshot_20261008/formal_150155/FORMAL_LARGE_GRAPH_BASELINE_20261008.json"
    payload = read_json(path)
    labels = {
        "random_real": "Random real coreset",
        "k_center": "K-center/FPS",
        "medoid": "Medoid",
        "gecc_hard_tag": "GECC-hard-TAG",
        "cat_mmd_tag_cctp_top5": "CaT-CCTP",
        "puma_full": "PUMA-inspired-TAG",
    }
    out = []
    for row in payload["records"]:
        method = labels[row["method"]]
        out.append(record(
            tier="formal-external-150155",
            method=method,
            dataset=row["dataset"],
            f1=row["final_macro_f1"], acc=row["accuracy"],
            old_f1=row["old_f1"], new_f1=row["new_f1"],
            f1_sd=row["final_macro_f1_std"], acc_sd=row["accuracy_std"],
            orders="frozen base+4", seeds="150-155", comparison="conditional",
            source=str(path),
            adaptation=("official PUMA-inspired mechanism adapted to current-delta-only TAG stream"
                        if method == "PUMA-inspired-TAG" else
                        "original mechanism adapted to 5% cumulative TAG stream and common Prompt Student"),
            note=("Formal external panel; construction memory was sealed from an earlier source seed and Student was reevaluated with 150-155. Keep this panel separate from the newest matched primary panel."
                  if method != "PUMA-inspired-TAG" else
                  "Mechanism adaptation, not official PUMA end-to-end reproduction."),
        ))
    return out


def tagsam_large() -> list[dict]:
    path = HANDOFF / "server_snapshot_20261008/tagsam_formal/TAGSAM_LARGE_GRAPH_20261008.json"
    payload = read_json(path)
    out = []
    for row in payload["rows"]:
        if row["arm"] != "tagsam_selfloop":
            continue
        final = row["final"]
        out.append(record(
            tier="mechanism-adapted-large",
            method="TAGSAM mechanism adapter",
            dataset=row["dataset"], f1=final["macro_f1_mean"],
            acc=final["accuracy_mean"], old_f1=final["old_f1_mean"],
            new_f1=final["new_f1_mean"], f1_sd=final["macro_f1_std"],
            orders="base+4", seeds="150-155", comparison="no",
            source=str(path),
            adaptation="official TAGSAM wBCE mechanism wrapped with frozen-prefix continual blocks, common GCN/MiniLM Prompt Student, and edge-free memory",
            note="Four-dataset mechanism adaptation; not official end-to-end TAGSAM reproduction and not ranked with strict rows.",
        ))
    return out


def adapted_small() -> list[dict]:
    # These rows are the audited small-graph adapted/reference table.
    rows = [
        ("STAD", "citeseer", 28.9085, 34.4828, 39.0430, 8.6393),
        ("STAD", "cora", 33.9941, 42.9889, 45.1248, 19.1530),
        ("TAGSAM", "citeseer", 56.1442, 61.0240, 64.0506, 40.3314),
        ("TAGSAM", "cora", 49.8096, 59.4096, 67.7595, 25.8764),
    ]
    out = []
    for method, dataset, f1, acc, old_f1, new_f1 in rows:
        out.append(record(
            tier="mechanism-adapted-small", method=method, dataset=dataset,
            f1=f1, acc=acc, old_f1=old_f1, new_f1=new_f1,
            orders="order-unspecified", seeds="100-105", comparison="no",
            source="D:/Aprojects/tag_baseline/.../EXTERNAL_BASELINE_ADAPTED_REFERENCE_TABLE_20261003.md",
            adaptation=("official STAD components with local canonical text generation and common TAG Student"
                        if method == "STAD" else
                        "TAGSAM graph/text condensation mechanism with common continual evaluation wrapper"),
            note="Adapted/reference only; Student seeds and protocol differ from the latest strict panel.",
        ))
    return out


def large_external_appendix() -> list[dict]:
    """Load the newly completed large-graph adapter/reference experiments."""
    base = ROOT / "results/large_external_adapters_20261011"
    out = []
    names = {
        "stad_adapter": "STAD mechanism adapter",
        "dance_adapter": "DANCE mechanism adapter",
        "sr_gm_adapter": "SR-GM mechanism adapter",
    }
    for dataset in ("arxiv", "sportsfit", "bookhis", "wikics"):
        payload = read_json(base / dataset / "results.json")
        for row in payload["results"]:
            final = row["final"]
            out.append(record(
                tier="mechanism-adapted-large-20261011",
                method=names[row["method"]], dataset=dataset,
                f1=final["macro_f1"], acc=final["accuracy"],
                f1_sd=final["macro_f1_std"], acc_sd=final["accuracy_std"],
                orders="base+4", seeds="150-152", comparison="no",
                source=str(base / dataset / "results.json"),
                adaptation=(payload["manifest"]["method_status"][row["method"]]),
                note="Large-graph mechanism adapter; not official end-to-end reproduction.",
            ))
    for path in sorted(base.glob("graphclip_large_reference_20261011_*512cap.json")):
        payload = read_json(path)
        final = payload["final"]["all"]
        out.append(record(
            tier="official-reference-large-20261011",
            method="GraphCLIP official frozen (512-cap)", dataset=payload["dataset"],
            f1=final["cumulative_f1"], acc=final["accuracy"],
            orders="512 report-dev nodes", seeds="frozen checkpoint", comparison="no",
            source=str(path), adaptation="official frozen checkpoint inference",
            note="Full original TAG ego context with deterministic 512-node cap; not continual condensation.",
        ))
    return out


def official_references() -> list[dict]:
    out = []
    for dataset in ("cora", "citeseer"):
        path = ROOT / f"results/official_graphclip_report_2026_09_17/{dataset}.json"
        payload = read_json(path)
        metrics = payload.get("metrics", [])
        final = metrics[-1]["all"] if metrics else {}
        out.append(record(
            tier="official-reference",
            method="GraphCLIP official frozen checkpoint",
            dataset=dataset,
            f1=final.get("cumulative_f1"), acc=final.get("accuracy"),
            orders="stage slices of one frozen checkpoint", seeds="none",
            comparison="no", source=str(path), adaptation="none; official checkpoint inference",
            note="Reference-only: complete original TAG and transductive ego-subgraph features; no synthetic memory or continual constructor.",
        ))
    for method, note in (
        ("GCond", "Official static graph condensation reference; no audited current-delta-only TAG adapter."),
        ("SFGC", "Official static structure-free condensation reference; trajectory-buffer interface conflicts with frozen-prefix continual construction."),
        ("GCondenser", "Official benchmark toolkit reference; no single native continual TAG constructor."),
    ):
        out.append(record(
            tier="official-reference",
            method=method, dataset="Cora/Citeseer native task", comparison="no",
            source="baseline cards / official repository audit", adaptation="none",
            note=note,
        ))
    return out


def inventory_rows() -> list[dict]:
    audit = read_json(OUT / "fidelity_audit/summary.json")
    rows = []
    for row in audit["rows"]:
        # The bundle card is metadata for several arms, not an additional method.
        if row["method"] == "external_continual_adapters_card.json":
            continue
        rows.append({
            "method": row["method"],
            "category": row["category"],
            "strict_status": row["strict_status"],
            "mechanism": row["mechanism"],
            "adaptation": row["adaptation"],
            "access": row["access"],
            "student": row["student"],
            "student_seeds": row["student_seeds"],
            "orders": row["orders"],
            "native_reproduction": row["native_reproduction"],
            "eligibility": row["eligibility"],
        })
    return rows


def adaptation_rows() -> list[dict]:
    return [
        {"original_method": "Unified-Q", "current_name": "Unified-Q", "category": "ours",
         "retained_mechanism": "shared Q assignment, in-loop distribution matching, real-text materialization",
         "adaptation": "main method; no external-method adaptation",
         "status": "strict latest primary", "official_reproduction": "not applicable", "numeric_coverage": "ArXiv/SportsFit/BookHis/WikiCS"},
        {"original_method": "CaT", "current_name": "CaT-CCTP adapter", "category": "continual graph condensation",
         "retained_mechanism": "class-wise feature distribution matching / MMD",
         "adaptation": "current-delta-only constructor, frozen synthetic prefix, legal TAG text attachment, common Prompt Student",
         "status": "mechanism-adapted", "official_reproduction": "no", "numeric_coverage": "strict latest + formal four-graph panel"},
        {"original_method": "GECC", "current_name": "GECC-hard-TAG adapter", "category": "continual graph condensation",
         "retained_mechanism": "hard hop-weight propagation and Lloyd-style structure-free clustering",
         "adaptation": "current delta only, frozen prefix, TAG payload attachment, common Prompt Student",
         "status": "mechanism-adapted", "official_reproduction": "no", "numeric_coverage": "strict latest + formal four-graph panel"},
        {"original_method": "PUMA", "current_name": "PUMA-inspired-TAG", "category": "continual graph condensation",
         "retained_mechanism": "pseudo-teacher, confidence filtering, one-pass propagation",
         "adaptation": "TAG data, frozen-prefix stream, edge-free memory, common Prompt Student and MiniLM payload",
         "status": "mechanism-adapted", "official_reproduction": "no", "numeric_coverage": "formal four-graph panel"},
        {"original_method": "TAGSAM", "current_name": "TAGSAM mechanism adapter", "category": "static TAG condensation",
         "retained_mechanism": "official wBCE graph/text condensation mechanism",
         "adaptation": "wrapped in base+4 frozen-prefix continual blocks and evaluated with common Prompt Student",
         "status": "adapted/reference", "official_reproduction": "no", "numeric_coverage": "four large graphs + small-graph reference"},
        {"original_method": "STAD", "current_name": "STAD local-canonical adapter", "category": "static/semi-supervised TAG distillation",
         "retained_mechanism": "official STAD components and text-generation path",
         "adaptation": "canonical TAG Student, local generated text, protocol wrapper; no strict large-graph continual score",
         "status": "adapted/reference", "official_reproduction": "no", "numeric_coverage": "Cora/Citeseer only"},
        {"original_method": "k-center / FPS", "current_name": "K-center/FPS", "category": "coreset control",
         "retained_mechanism": "farthest-point selection",
         "adaptation": "legal current pool, cumulative 5% budget, real text retained, common Prompt Student",
         "status": "mechanism-adapted control", "official_reproduction": "not claimed", "numeric_coverage": "formal four-graph panel"},
        {"original_method": "Medoid", "current_name": "Medoid", "category": "coreset control",
         "retained_mechanism": "representative real-node selection",
         "adaptation": "legal current pool, cumulative 5% budget, real text retained, common Prompt Student",
         "status": "mechanism-adapted control", "official_reproduction": "not claimed", "numeric_coverage": "formal four-graph panel"},
        {"original_method": "GraphCLIP", "current_name": "GraphCLIP official frozen checkpoint", "category": "graph-text reference",
         "retained_mechanism": "official graph-text pretrained model",
         "adaptation": "none; evaluated as a native frozen checkpoint reference",
         "status": "official/reference only", "official_reproduction": "yes for checkpoint load, no for continual condensation", "numeric_coverage": "Cora/Citeseer native reference"},
        {"original_method": "GCond", "current_name": "GCond", "category": "static graph condensation",
         "retained_mechanism": "gradient matching over synthetic features and adjacency",
         "adaptation": "none; no legal current-delta-only frozen-prefix TAG constructor was identified",
         "status": "official/reference only", "official_reproduction": "compile/source audit only", "numeric_coverage": "no comparable score"},
        {"original_method": "SFGC", "current_name": "SFGC", "category": "static structure-free condensation",
         "retained_mechanism": "trajectory-buffer imitation and feature distillation",
         "adaptation": "none; native trajectory-buffer pipeline conflicts with the continual access boundary",
         "status": "official/reference only", "official_reproduction": "compile/source audit only", "numeric_coverage": "no comparable score"},
        {"original_method": "GCondenser", "current_name": "GCondenser", "category": "benchmark toolkit",
         "retained_mechanism": "static condensation benchmark wrapper",
         "adaptation": "none; no single native frozen-prefix continual TAG constructor",
         "status": "official/reference only", "official_reproduction": "compile/source audit only", "numeric_coverage": "no comparable score"},
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["method"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return "-" if value is None else f"{float(value):.2f}"


def markdown_table(rows: list[dict], title: str) -> list[str]:
    lines = [f"## {title}", "", "| Dataset | Method | F1 | Acc. | Old F1 | New F1 | Seeds | Orders | Comparable |", "|---|---|---:|---:|---:|---:|---|---|---|"]
    for row in rows:
        lines.append("| {dataset} | {method} | {f1} | {acc} | {old} | {new} | {seeds} | {orders} | {comparison} |".format(
            dataset=row["dataset"], method=row["method"], f1=fmt(row["final_f1"]),
            acc=fmt(row["accuracy"]), old=fmt(row["old_f1"]), new=fmt(row["new_f1"]),
            seeds=row["student_seeds"], orders=row["orders"], comparison=row["directly_comparable"]))
    return lines


def plot_group(rows: list[dict], methods: list[str], datasets: list[str], name: str, metric: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(datasets))
    width = 0.8 / max(1, len(methods))
    for i, method in enumerate(methods):
        vals = []
        for dataset in datasets:
            match = next((r for r in rows if r["method"] == method and r["dataset"] == dataset), None)
            vals.append(np.nan if match is None or match[metric] is None else float(match[metric]))
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width, label=method)
    ax.set_xticks(x, [d.capitalize() for d in datasets])
    ax.set_ylabel("Percent")
    ax.set_title(title)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / name, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    latest = latest_primary()
    formal = formal_external()
    tagsam = tagsam_large()
    small = adapted_small()
    official = official_references()
    large_external = large_external_appendix()
    inventory = inventory_rows()
    adaptations = adaptation_rows()

    write_csv(OUT / "strict_latest_prompt.csv", latest)
    write_csv(OUT / "formal_external_150155.csv", formal)
    write_csv(OUT / "mechanism_adapted_large.csv", tagsam)
    write_csv(OUT / "mechanism_adapted_small.csv", small)
    write_csv(OUT / "official_reference.csv", official)
    write_csv(OUT / "large_external_appendix.csv", large_external)
    write_csv(OUT / "method_inventory.csv", inventory)
    write_csv(OUT / "method_adaptation_table.csv", adaptations)

    plot_group(latest, ["Unified-Q", "GECC-hard-TAG", "CaT-CCTP"],
               ["arxiv", "sportsfit", "bookhis", "wikics"],
               "strict_latest_f1.png", "final_f1", "Newest matched Prompt Student: final Macro-F1")
    plot_group(latest, ["Unified-Q", "GECC-hard-TAG", "CaT-CCTP"],
               ["arxiv", "sportsfit", "bookhis", "wikics"],
               "strict_latest_accuracy.png", "accuracy", "Newest matched Prompt Student: accuracy")
    plot_group(formal, ["Random real coreset", "K-center/FPS", "Medoid", "GECC-hard-TAG", "CaT-CCTP", "PUMA-inspired-TAG"],
               ["arxiv", "sportsfit", "wikics", "bookhis"],
               "formal_external_f1.png", "final_f1", "Formal external panel (150-155): final Macro-F1")
    plot_group(tagsam, ["TAGSAM mechanism adapter"], ["arxiv", "sportsfit", "wikics", "bookhis"],
               "tagsam_adapted_f1.png", "final_f1", "TAGSAM mechanism adaptation: final Macro-F1")

    report = {
        "name": "protocol-aware-external-baseline-registry-20261011",
        "tiers": {
            "strict_latest_prompt": "newest matched Prompt Student rows; directly comparable within this tier",
            "formal_external_150155": "formal external panel with the common 150-155 Student, kept separate because its sealed construction generation differs from the newest primary matched panel",
            "mechanism_adapted_large": "TAGSAM four-dataset mechanism adaptation; not native official reproduction",
            "mechanism_adapted_small": "STAD/TAGSAM small-graph adapted/reference rows; not directly comparable",
            "official_reference": "native official checkpoint/code references; no continual ranking",
            "large_external_appendix": "2026-10-11 large-graph mechanism adapters and GraphCLIP reference; kept outside strict ranking",
        },
        "strict_latest_prompt": latest,
        "formal_external_150155": formal,
        "mechanism_adapted_large": tagsam,
        "mechanism_adapted_small": small,
        "official_reference": official,
        "large_external_appendix": large_external,
        "method_inventory": inventory,
        "method_adaptation_table": adaptations,
        "limitations": [
            "Mechanism adaptation is not official end-to-end reproduction.",
            "Native static methods do not receive a fabricated current-delta-only score.",
            "The newest primary panel and the formal external panel are retained as separate protocol generations.",
            "WikiCS is shown as supplementary evidence because it is a boundary case in the primary study.",
        ],
    }
    (OUT / "BASELINE_REGISTRY_20261011.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Protocol-Aware External Baseline Registry",
        "",
        "本报告把外部方法分成四类：最新同 Student 严格对照、正式外部适配面板、机制适配附表、官方原生/reference。不同协议代际不合并排名。",
        "",
        "## 适配规则",
        "",
        "- 可保留原方法核心机制、同时满足 current-delta-only、frozen synthetic prefix、5% cumulative memory 和共同 Prompt Student 的，标为 mechanism-adapted。",
        "- 原论文是静态凝聚、联邦凝聚或图文预训练，不能合法提供持续构造接口的，不强行改写为 continual 数字。",
        "- 官方仓库/官方 checkpoint 的结果只称 official/reference，不称同协议主表结果。",
        "- 最新 primary panel 和 formal external panel 的 construction provenance 不同，故分开报告。",
        "",
    ]
    lines += markdown_table(latest, "Latest Matched Prompt Student")
    lines += ["", "说明：这张表是当前最适合论文主文引用的外部比较。Unified-Q、GECC 和 CaT 使用同一 Prompt Student；ArXiv 是一个预注册转移单元，其余主数据集为三顺序面板。", ""]
    lines += markdown_table(formal, "Formal External Panel (Student Seeds 150-155)")
    lines += ["", "说明：该面板补齐 Random、K-center、Medoid 和 PUMA，并覆盖四图。核心外部机制可比较，但构造 memory 来自已封存的另一代 formal run，不能与上表直接合并成一个无条件 leaderboard。", ""]
    lines += markdown_table(tagsam, "Mechanism-Adapted TAGSAM on Four Large Graphs")
    lines += ["", "说明：保留官方 TAGSAM wBCE 机制，替换为当前 frozen-prefix 和共同 Student 接口；这是机制适配，不是官方端到端 TAGSAM reproduction。", ""]
    lines += markdown_table(small, "Small-Graph Adapted/Reference Results")
    lines += ["", "说明：STAD 和早期 TAGSAM 结果使用小图 adapted/reference 协议，不能和最新大图 Prompt Student 结果直接排序。", ""]
    lines += markdown_table(official, "Official/Native References")
    lines += ["", "说明：GraphCLIP 是官方 frozen checkpoint reference；GCond、SFGC、GCondenser 当前没有通过审计的合法 continual TAG 构造接口，因此只记录官方方法身份和不适配原因，不填造数值。", ""]
    lines += ["## Figures", "", "- `figures/strict_latest_f1.png`", "- `figures/strict_latest_accuracy.png`", "- `figures/formal_external_f1.png`", "- `figures/tagsam_adapted_f1.png`", "", "## Method Inventory", ""]
    lines.append("| Method | Category | Strict status | Adaptation / reason | Native reproduction |")
    lines.append("|---|---|---|---|---|")
    for row in inventory:
        lines.append("| {method} | {category} | {status} | {adaptation} | {native} |".format(
            method=row["method"], category=row["category"], status=row["strict_status"],
            adaptation=(row["adaptation"] or "not recorded"), native=row["native_reproduction"]))
    lines += ["", "## Original Method to Current Adaptation", "", "| Original method | Current name | Retained mechanism | TAG continual adaptation | Status | Numeric coverage |", "|---|---|---|---|---|---|"]
    for row in adaptations:
        lines.append("| {original_method} | {current_name} | {retained_mechanism} | {adaptation} | {status} | {numeric_coverage} |".format(**row))
    lines += ["", "## Paper-facing conclusion", "", "当前论文主文应使用 Latest Matched Prompt Student 表；Formal External Panel、TAGSAM adapted/reference 和 Official/Native References 放在补充材料或单独小节。这样既补齐了外部方法覆盖，又不会把机制适配、静态官方结果和严格 continual 结果混成不公平排名。", ""]
    (OUT / "BASELINE_REGISTRY_20261011.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "latest": len(latest), "formal": len(formal), "tagsam": len(tagsam), "small": len(small), "official": len(official), "inventory": len(inventory)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
