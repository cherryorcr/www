"""Independent structural and artifact audit for LP kernel controls."""
import argparse
import json
from pathlib import Path

import torch

import run_link_prediction_transfer as L
import run_lp_kernel_mechanism as M


def audit(root, parent):
    manifest = json.loads((root / "manifest.json").read_text())
    records = json.loads((root / "results.json").read_text())
    assert manifest["arms"] == list(M.ARMS)
    expected = len(manifest["splits"]) * len(manifest["seeds"]) * 3 * len(M.ARMS)
    assert len(records) == expected
    keys = {(r["split"], r["stage"], r["arm"], r["seed"]) for r in records}
    assert len(keys) == expected
    topology = []
    memory_count = 0
    for split in manifest["splits"]:
        parent_folder = parent / f"split_{split}"
        folder = root / f"split_{split}"
        parent_split = torch.load(parent_folder / "edge_split.pt", weights_only=False)
        L.validate_split(parent_split)
        old = {}
        for stage in range(3):
            base = torch.load(parent_folder / f"c0_s{stage}_memory.pt", weights_only=False)
            for arm in M.ARMS:
                memory = torch.load(folder / f"{arm}_s{stage}_memory.pt", weights_only=False)
                M.check_memory(memory, base, old.get(arm), arm)
                if arm != "c0_replay":
                    assert "graph_signature" not in memory
                    assert "signature_config" not in memory
                else:
                    assert memory.keys() == base.keys()
                pairs = L.pairs(memory["edge_index"], len(memory["x"]))
                topology.append({"split": split, "stage": stage, "arm": arm,
                                 "rows": len(memory["x"]), "pairs": pairs.shape[1],
                                 "isolated": int((torch.bincount(L.bidirectional(pairs)[0], minlength=len(memory["x"])) == 0).sum())})
                old[arm] = memory
                memory_count += 1
    assert memory_count == len(manifest["splits"]) * 3 * len(M.ARMS)
    summary = {"dataset": manifest["dataset"], "fits": len(records), "expected_fits": expected,
               "memories": memory_count, "expected_memories": memory_count, "passed": True,
               "parent_audit_passed": json.loads((parent / "audit.json").read_text())["passed"],
               "topology": topology}
    L.save_json(root / "audit.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "topology"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--parent", required=True)
    args = parser.parse_args()
    audit(Path(args.root), Path(args.parent))
