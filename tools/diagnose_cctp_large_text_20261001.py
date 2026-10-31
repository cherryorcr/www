"""Diagnose CCTP source concentration in the large-graph result artifacts."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ARMS = {
    "cat_mmd_tag_cctp_top3": "CCTP top-3",
    "cat_mmd_tag_cctp_top5": "CCTP top-5",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def diagnose(dataset: str, path: Path):
    payload = read(path)
    rows = []
    for row in payload["rows"]:
        arm = row["arm"]
        if arm not in ARMS:
            continue
        audit = row["construction_audit"]
        source_rows = audit["text_source_ids"]
        row_classes = [str(v) for v in audit["row_classes"]]
        by_class = defaultdict(list)
        for cls, sources in zip(row_classes, source_rows):
            by_class[cls].append(list(sources))

        total_selected = sum(len(sources) for sources in source_rows)
        unique_selected = len({int(source) for sources in source_rows for source in sources})
        class_stats = []
        for cls, class_rows in sorted(by_class.items()):
            class_unique = len({int(source) for sources in class_rows for source in sources})
            class_selected = sum(len(sources) for sources in class_rows)
            class_stats.append({
                "class": cls,
                "rows": len(class_rows),
                "unique_sources": class_unique,
                "selected_sources": class_selected,
                "reuse_factor": class_selected / max(class_unique, 1),
            })

        raw_source_ids = audit.get("source_ids", [])
        raw_source_count = len(raw_source_ids) if isinstance(raw_source_ids, list) else int(raw_source_ids or 0)
        rows.append({
            "dataset": dataset,
            "stage": int(row["stage"]),
            "arm": arm,
            "method": ARMS[arm],
            "memory_rows": int(row["memory_rows"]),
            "raw_source_count": raw_source_count,
            "class_count": len(by_class),
            "total_selected_sources": total_selected,
            "unique_selected_sources": unique_selected,
            "global_reuse_factor": total_selected / max(unique_selected, 1),
            "mean_unique_sources_per_class": sum(v["unique_sources"] for v in class_stats) / max(len(class_stats), 1),
            "mean_class_reuse_factor": sum(v["reuse_factor"] for v in class_stats) / max(len(class_stats), 1),
            "text_chars_max": int(audit.get("text_chars_max", 0)),
            "text_diversity": audit.get("text_diversity"),
            "text_adapter": audit.get("text_adapter"),
            "allocation": audit.get("allocation", {}),
            "class_stats": class_stats,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv", type=Path, required=True)
    parser.add_argument("--bookchild", type=Path, required=True)
    parser.add_argument("--sportsfit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for dataset, path in (("arxiv", args.arxiv), ("bookchild", args.bookchild), ("sportsfit", args.sportsfit)):
        rows.extend(diagnose(dataset, path))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "text_diagnostics.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# CCTP large-graph text-source diagnostics",
        "",
        "The table measures source concentration in the constructed memory. It does not use downstream labels.",
        "",
        "|dataset|stage|method|memory rows|raw candidates|classes|unique selected|selected sources|reuse factor|mean unique/class|mean class reuse|chars|",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"|{row['dataset']}|{row['stage']}|{row['method']}|{row['memory_rows']}|{row['raw_source_count']}|"
            f"{row['class_count']}|{row['unique_selected_sources']}|{row['total_selected_sources']}|"
            f"{row['global_reuse_factor']:.1f}|{row['mean_unique_sources_per_class']:.2f}|"
            f"{row['mean_class_reuse_factor']:.1f}|{row['text_chars_max']}|"
        )

    lines += [
        "",
        "Interpretation:",
        "- A high reuse factor means many synthetic rows share the same class prototype sources.",
        "- `raw candidates` is the current-delta source count recorded by the construction audit.",
        "- These are construction diagnostics, not evidence that a higher or lower reuse factor is intrinsically better.",
    ]
    (args.output / "text_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
