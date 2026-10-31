"""Build a high-fidelity audit of external baseline adaptations.

This is a provenance/audit report, not a rerunner.  It deliberately keeps
official/native, mechanism-adapted, and reference-only results separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CARD_FIELDS = (
    "method", "classification", "paper_title", "paper_year", "paper_venue",
    "paper_url", "paper_or_repo", "upstream", "official_commit", "commit",
    "access", "memory", "mechanism", "adaptation", "student", "student_seeds",
    "construction_seed", "construction_seeds", "orders", "coverage",
    "audits", "native_reproduction", "main_table_eligibility", "main_table",
    "result_root", "result_roots", "metadata_note",
)

INFERRED_MECHANISMS = {
    "CaT-MMD-TAG": "class-balanced MMD condensation of synthetic features against current raw/teacher statistics",
    "GECC-hard-TAG": "hard hop-weight propagation followed by Lloyd-style structure-free clustering",
    "k-center/FPS": "farthest-point/k-center coreset selection in the feature/text space",
    "medoid": "medoid coreset selection in the feature/text space",
    "TAGSAM": "text-attributed graph condensation with graph/text-aware synthetic memory",
    "semantic-recursive": "recursive semantic append of synthetic memory with class-aware text/topology payloads",
}


def load_cards(root: Path) -> list[dict]:
    cards = []
    for path in sorted(root.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["card_file"] = path.name
        cards.append(value)
    if not cards:
        raise FileNotFoundError(f"no baseline cards under {root}")
    return cards


def as_text(value) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(as_text(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}={as_text(v)}" for k, v in value.items())
    return str(value)


def category(card: dict) -> str:
    classification = str(card.get("classification", ""))
    eligibility = str(card.get("main_table_eligibility", ""))
    if "reference-only" in classification or "reference-only" in eligibility:
        return "reference-only"
    if "adapted" in classification or "adapted" in eligibility:
        return "mechanism-adapted"
    return "unclassified"


def strict_status(card: dict) -> tuple[str, list[str]]:
    reasons = []
    access = str(card.get("access", ""))
    if "current delta" not in access.lower() and "current legal pool" not in access.lower():
        reasons.append("current-delta locality not explicit")
    seeds = card.get("student_seeds")
    if seeds and list(seeds) != [150, 151, 152, 153, 154, 155]:
        reasons.append("Student seed set differs from 150-155 strict baseline")
    if not card.get("audits"):
        reasons.append("audit bundle not recorded")
    if card.get("native_reproduction") is True:
        reasons.append("native result has not been independently separated")
    if category(card) == "reference-only":
        return "exclude", reasons + ["reference-only classification"]
    eligibility = str(card.get("main_table_eligibility", ""))
    if "eligible" not in eligibility:
        return "supplementary", reasons + ["card does not declare strict eligibility"]
    if reasons:
        return "conditional", reasons
    return "eligible", []


def normalize(card: dict) -> dict:
    status, reasons = strict_status(card)
    method = card.get("method", card["card_file"])
    mechanism = card.get("mechanism") or INFERRED_MECHANISMS.get(method)
    return {
        "method": method,
        "card_file": card["card_file"],
        "category": category(card),
        "strict_status": status,
        "strict_reasons": reasons,
        "paper": card.get("paper_title"),
        "paper_year": card.get("paper_year"),
        "paper_venue": card.get("paper_venue"),
        "upstream": card.get("upstream", card.get("paper_or_repo")),
        "mechanism": mechanism,
        "adaptation": card.get("adaptation"),
        "access": card.get("access"),
        "memory": card.get("memory"),
        "student": card.get("student"),
        "student_seeds": card.get("student_seeds", card.get("seeds", {}).get("student")),
        "construction_seed": card.get("construction_seed", card.get("construction_seeds")),
        "orders": card.get("orders", card.get("coverage")),
        "audits": card.get("audits"),
        "native_reproduction": card.get("native_reproduction", False),
        "eligibility": card.get("main_table_eligibility", card.get("main_table")),
        "metadata_note": card.get("metadata_note"),
    }


def md_cell(value) -> str:
    return as_text(value).replace("|", "\\|").replace("\n", " ")


def build_markdown(rows: list[dict]) -> str:
    lines = [
        "# External Baseline Fidelity Audit",
        "",
        "本表核对的是方法身份与协议可比性，不把机制适配结果写成官方复现。",
        "严格主表的默认要求：current-delta-only、不可读取历史 raw、相同 memory budget、相同 Prompt Student/文本编码器、同一 Student seed 集、同一 report/test 划分。",
        "",
        "## Classification",
        "",
        "| Method | Category | Strict status | Original mechanism | Current adaptation | Data access | Memory/topology | Student | Seeds/orders | Caveat |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        caveat = "; ".join(row["strict_reasons"]) or "none recorded"
        seeds_orders = f"seeds={md_cell(row['student_seeds'])}; orders={md_cell(row['orders'])}"
        lines.append("| " + " | ".join([
            md_cell(row["method"]), md_cell(row["category"]), md_cell(row["strict_status"]),
            md_cell(row["mechanism"]), md_cell(row["adaptation"]), md_cell(row["access"]),
            md_cell(row["memory"]), md_cell(row["student"]), md_cell(seeds_orders),
            md_cell(caveat),
        ]) + " |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- `eligible` 只表示 card 中记录的适配版本满足进入严格比较的元数据条件，不表示它是官方 native reproduction。",
        "- `conditional` 表示方法机制可比较，但种子、顺序覆盖或访问描述仍不完整；应进入补充表或补跑后再进主表。",
        "- `reference-only` 不参与严格排序，包括静态/离线方法、官方 checkpoint 或 Student/loss 不同的方法。",
        "- CaT、GECC、PUMA、k-center 和 medoid 的当前结果属于 mechanism-adapted；TAGSAM/STAD 有额外的 seed/order 警告；GCond/SFGC/GCondenser/GraphCLIP 等 reference-only 不应和 Unified-Q 直接横排。",
        "",
        "## Per-card Detail",
        "",
    ]
    for row in rows:
        lines += [f"### {row['method']}", "", f"- Card: `{row['card_file']}`",
                   f"- Paper/repository: {md_cell(row['paper'])}; upstream `{md_cell(row['upstream'])}`",
                   f"- Mechanism: {md_cell(row['mechanism'])}",
                   f"- Adaptation: {md_cell(row['adaptation'])}",
                   f"- Access/memory: {md_cell(row['access'])}; {md_cell(row['memory'])}",
                   f"- Student: {md_cell(row['student'])}",
                   f"- Audits: {md_cell(row['audits'])}",
                   f"- Strict status: **{row['strict_status']}**",
                   ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cards = load_cards(args.cards)
    rows = [normalize(card) for card in cards]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "name": "external-baseline-fidelity-audit-20261009",
        "card_root": str(args.cards.resolve()),
        "card_count": len(rows),
        "rows": rows,
        "rules": {
            "strict_access": "current delta only; no historical raw constructor access",
            "strict_student": "same Prompt Student, frozen MiniLM, same seeds and steps",
            "strict_split": "same selector/report/sealed-test masks",
            "official_reproduction": "false unless separately verified; current cards are adapted/reference",
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    (args.output / "summary.md").write_text(
        build_markdown(rows), encoding="utf-8")
    print(json.dumps({"status": "complete", "card_count": len(rows),
                      "output": str(args.output.resolve())}, ensure_ascii=True))


if __name__ == "__main__":
    main()
