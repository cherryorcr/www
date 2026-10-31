"""Run a frozen-memory GCN/SAGE transfer audit for the available ArXiv cells."""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path("/home/pc/research/TAG_condensation")
EVAL_DIR = ROOT / "experiments/uq_final_evidence_20261012/code"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(1, str(ROOT / "code/v2_llm_fusion"))

import evaluate_uq_sealed_test_20261009 as evaluator  # noqa: E402


def discover_current_cells(roots: list[str]):
    """Reuse only the nine registered ArXiv confirmation cells."""
    cells = []
    for raw in roots:
        cell = Path(raw)
        protocol_path = cell / "protocol.json"
        result_path = cell / "results.json"
        if not protocol_path.exists() or not result_path.exists():
            continue
        protocol = evaluator.read_json(protocol_path)
        result = evaluator.read_json(result_path)
        if result.get("status") != "complete":
            continue
        cells.append((cell, protocol))
    return cells


def main() -> None:
    confirmation = ROOT / "results/uq_final_evidence_20261012/confirmation"
    roots = sorted(str(path) for path in confirmation.glob("arxiv_o*_c*"))
    if len(roots) != 9:
        raise RuntimeError(f"expected nine ArXiv cells, found {len(roots)}")

    # The current sealed archive contains these three methods.  The historical
    # q-medoid arm is intentionally excluded rather than reconstructed.
    evaluator.ARMS = ("uq_dm_edgefree", "cat_cctp", "gecc_hard")
    evaluator.discover_cells = discover_current_cells
    args = Namespace(
        source_root=roots,
        cache=[
            "arxiv=/home/pc/research/TAG_condensation/results/"
            "public_baselines_text_cache_20261001/arxiv"
        ],
        output=ROOT / "results/uq_cross_arch_20261015_arxiv_v2",
        minilm=("/home/pc/research/TAGSAM/graphclip_native_work/"
                "minilm_from_graphclip"),
        device="cuda:0",
        students=["prompt_gcn", "prompt_sage"],
        student_seeds=[150, 151, 152],
        student_steps=None,
        batch=24,
    )
    evaluator.run(args)


if __name__ == "__main__":
    main()
