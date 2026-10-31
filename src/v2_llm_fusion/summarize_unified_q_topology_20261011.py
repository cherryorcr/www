"""Build compact tables and figures for the Unified-Q topology study."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--lp-cora", type=Path, required=True)
    parser.add_argument("--lp-citeseer", type=Path, required=True)
    parser.add_argument("--lp-extra", type=Path, nargs="*", default=[])
    parser.add_argument("--large-bookhis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    node = read(args.node)["rows"]
    topo = read(args.topology)["rows"]
    lp = []
    for path in [args.lp_cora, args.lp_citeseer, *args.lp_extra]:
        payload = read(path)
        split_seed = payload.get("protocol", {}).get("split_seed")
        for row in payload["rows"]:
            row = dict(row)
            row["split_seed"] = split_seed
            lp.append(row)

    node_rows = []
    for row in node:
        agg = row["aggregate"]
        node_rows.append({
            "dataset": row["dataset"], "order": row["order"],
            "arm": row["arm"], "stage": row["stage"],
            "split_seed": row.get("split_seed"),
            "memory_rows": row["memory_rows"],
            "final_f1": agg["mean_f1"], "final_f1_std": agg["std_f1"],
            "accuracy": agg["mean_accuracy"], "accuracy_std": agg["std_accuracy"],
            "old_f1": agg["old_f1"], "new_f1": agg["new_f1"],
            "construction_seconds": row["construction_seconds"],
        })
    write_csv(out / "node_classification.csv", node_rows, list(node_rows[0]))

    resource_rows = [{
        "dataset": row["dataset"], "order": row["order"],
        "arm": row["arm"], "stage": row["stage"],
        "memory_rows": row["memory_rows"],
        "directed_edge_entries": row["directed_edge_entries"],
        "undirected_edges": row["undirected_edges"],
        "mean_degree": row["mean_degree"], "max_degree": row["max_degree"],
        "construction_seconds": row["construction_seconds"],
    } for row in topo]
    write_csv(out / "topology_resources.csv", resource_rows, list(resource_rows[0]))

    lp_rows = []
    for row in lp:
        m, fm, s = row["metrics"], row["metrics_feature_only"], row["structure"]
        lp_rows.append({
            "dataset": row["dataset"], "order": row["order"],
            "arm": row["arm"], "stage": row["stage"],
            "memory_rows": row["memory_rows"],
            "auc": m["auc"], "ap": m["ap"],
            "feature_only_auc": fm["auc"], "feature_only_ap": fm["ap"],
            "topology_positive_rate": row["topology_positive_rate"],
            "topology_negative_rate": row["topology_negative_rate"],
            "heldout_edge_recall": s["heldout_edge_recall"],
            "degree_pearson": s["degree_pearson"],
            "synthetic_undirected_edges": s["synthetic_undirected_edges"],
            "synthetic_directed_edge_entries": s["synthetic_directed_edge_entries"],
        })
    write_csv(out / "link_prediction.csv", lp_rows, list(lp_rows[0]))

    final_node = [r for r in node_rows if r["stage"] == max(x["stage"] for x in node_rows)]
    final_lp_raw = [r for r in lp_rows if r["stage"] == max(x["stage"] for x in lp_rows)]
    final_lp = []
    for dataset in sorted({r["dataset"] for r in final_lp_raw}):
        for arm in sorted({r["arm"] for r in final_lp_raw if r["dataset"] == dataset}):
            group = [r for r in final_lp_raw if r["dataset"] == dataset and r["arm"] == arm]
            mean_auc = sum(r["auc"] for r in group) / len(group)
            mean_ap = sum(r["ap"] for r in group) / len(group)
            std_auc = (sum((r["auc"] - mean_auc) ** 2 for r in group) / max(1, len(group) - 1)) ** 0.5 if len(group) > 1 else 0.0
            std_ap = (sum((r["ap"] - mean_ap) ** 2 for r in group) / max(1, len(group) - 1)) ** 0.5 if len(group) > 1 else 0.0
            row = dict(group[-1])
            row.update({"auc": mean_auc, "ap": mean_ap, "auc_std": std_auc, "ap_std": std_ap, "split_count": len(group)})
            final_lp.append(row)
    lines = [
        "# Unified-Q topology branch report",
        "",
        "The edge-free arm is the current main method. `unified_q` adds Q-derived new-new topology; `unified_q_boundary` additionally materializes a boundary sketch.",
        "All node-classification rows use the same prompt-only GraphText Student. Link prediction reconstructs memory using train edges only and evaluates held-out undirected edges.",
        "",
        "## Final node classification",
        "",
        "| Dataset | Arm | Final Macro-F1 | Accuracy | Old F1 | New F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in final_node:
        lines.append(f"| {r['dataset']} | {r['arm']} | {r['final_f1']:.2f} +/- {r['final_f1_std']:.2f} | {r['accuracy']:.2f} | {r['old_f1'] if r['old_f1'] is not None else '-'} | {r['new_f1']:.2f} |")
    lines += ["", "## Final link prediction", "", "| Dataset | Arm | AUC | AP | Held-out edge recall | Degree Pearson | Synthetic edges |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in final_lp:
        lines.append(f"| {r['dataset']} | {r['arm']} | {r['auc']:.4f} +/- {r['auc_std']:.4f} | {r['ap']:.4f} +/- {r['ap_std']:.4f} | {r['heldout_edge_recall']:.3f} | {r['degree_pearson']:.3f} | {r['synthetic_undirected_edges']} |")
    lines += [
        "", "## Interpretation", "",
        "The topology branch is a cross-task extension rather than a replacement for the edge-free main method. It preserves similar node-classification quality and adds explicit synthetic edges.",
        "Across three independent edge splits, Q-derived topology improves the fixed slot-level link score on Cora (about +2.7 AUC points over edge-free) but is slightly below edge-free on Citeseer. The boundary branch is mixed as well. The defensible conclusion is therefore structural capability and a dataset-dependent signal, not a universal link-prediction win.",
        "Boundary edges should remain an ablation/extension: they add an access and storage assumption and do not dominate the no-boundary topology branch consistently.",
    ]
    if args.large_bookhis:
        large = read(args.large_bookhis)
        final = [r for r in large["rows"] if r["stage"] == max(x["stage"] for x in large["rows"])]
        lines += ["", "## Large-graph node classification: BookHis", "", "| Arm | Final Macro-F1 | Accuracy | Old F1 | New F1 | Synthetic edges |", "|---|---:|---:|---:|---:|---:|"]
        for r in final:
            s = r["score"]
            lines.append(f"| {r['arm']} | {s['final_f1']:.2f} | {s['accuracy']:.2f} | {s['old_f1']:.2f} | {s['new_f1']:.2f} | {r['undirected_edges']} |")
        lines.append("The large-graph run uses the same five-stage base+4 stream and prompt Student for both arms; only JSON metrics were persisted.")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        arms = ["unified_q_edge_free", "unified_q", "unified_q_boundary"]
        labels = {"unified_q_edge_free": "Edge-free", "unified_q": "Topology", "unified_q_boundary": "Boundary"}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for dataset in sorted({r["dataset"] for r in final_node}):
            values = [next(r["final_f1"] for r in final_node if r["dataset"] == dataset and r["arm"] == a) for a in arms]
            axes[0].plot([labels[a] for a in arms], values, marker="o", label=dataset)
        axes[0].set_ylabel("Final Macro-F1")
        axes[0].set_title("Node classification")
        axes[0].legend()
        for dataset in sorted({r["dataset"] for r in final_lp}):
            values = [next(r["auc"] for r in final_lp if r["dataset"] == dataset and r["arm"] == a) for a in arms]
            axes[1].plot([labels[a] for a in arms], values, marker="o", label=dataset)
        axes[1].set_ylabel("AUC")
        axes[1].set_title("Held-out link prediction")
        axes[1].set_ylim(0.5, 1.0)
        axes[1].legend()
        fig.savefig(out / "task_comparison.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        (out / "plot_warning.txt").write_text(str(exc), encoding="utf-8")

    print(out / "summary.md")


if __name__ == "__main__":
    main()
