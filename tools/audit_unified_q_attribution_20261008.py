"""Reconstruct archived prefixes and verify every evaluated memory hash."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"code/v2_llm_fusion"))
import run_unified_q_attribution_20261008 as A


def audit_cell(path):
    record = json.loads(path.read_text())
    p = record["protocol"]
    if record["status"] != "complete":
        raise ValueError("only completed cells can be archived-audited")
    sources = {}
    for name in ("unified_q_multimodal.py", "graphtext_prompt_student.py", "external_continual_mechanisms.py",
                 "x_conditioned_payload_routing.py", "cat_multimodal_minimal.py"):
        actual = hashlib.sha256((ROOT/"code/v2_llm_fusion"/name).read_bytes()).hexdigest()
        sources[name] = actual == p["source_hashes"][name]
        if not sources[name]:
            raise AssertionError(f"local source differs from frozen experiment: {name}")
    parents = {arm: None for arm in p["arms"]}
    verified, immutable = 0, 0
    for stage,budget in enumerate(p["budgets"]):
        for arm in p["arms"]:
            block, _ = A.load_block(path.parent, arm, stage)
            if not bool((block["stage"] == stage).all()):
                raise AssertionError("stage identity mismatch")
            if not bool(torch.isin(block["labels"], torch.tensor(p["groups"][stage])).all()):
                raise AssertionError("class outside incoming group")
            old = parents[arm]
            memory = A.UQ.append_memory(old, block) if arm == "uq_dm_edgefree" else A.append_block(old, block)
            A.V.validate_memory(memory, old, budget)
            rows = [r for r in record["rows"] if r["arm"] == arm and r["stage"] == stage]
            if not rows or any(r["memory_hash"] != A.V.memory_hash(memory) for r in rows):
                raise AssertionError("archived memory differs from evaluated memory")
            base = "cat_cctp" if arm in ("adaptive_coherent", "cat_nearest1", "cat_random5") else "uq_dm_edgefree" if arm.startswith("uq_nearest") else None
            if base:
                A.assert_fixed_content(parents[base], memory)
                immutable += 1
            verified += len(rows)
            parents[arm] = memory
    return dict(cell=path.parent.name, evaluated_memory_hashes_verified=verified,
                fixed_X_prefix_pairs_verified=immutable, source_matches=sources,
                final_rows=p["budgets"][-1], all_checks_passed=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT/"results/unified_q_attribution_20261008")
    args = p.parse_args()
    complete, pending = [], []
    for path in sorted(args.root.glob("*_o*/results.json")):
        if json.loads(path.read_text())["status"] != "complete":
            pending.append(path.parent.name)
        else:
            complete.append(audit_cell(path))
    if not complete:
        raise SystemExit("no completed archived cells")
    output = dict(complete_cells=len(complete), expected_cells=9, pending=pending,
        formal_matrix_complete=len(complete)==9 and not pending,
        cells=complete,
        scope="lossless memory, frozen prefixes, fixed X and source hashes; reencoding/source-class access were checked live before archiving")
    (args.root/"archive_audit.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(dict(complete_cells=len(complete), formal_matrix_complete=output["formal_matrix_complete"],
        evaluated_memories=sum(c["evaluated_memory_hashes_verified"] for c in complete),
        fixed_X_pairs=sum(c["fixed_X_prefix_pairs_verified"] for c in complete))))


if __name__ == "__main__":
    main()
