"""Independent post-run P2-A artifact audit, never fed back into training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

import p2a_graph_text as T
import p2_text_operators as BT
import run_small_comparison_audited as C


def audit(root, dataset, minilm):
    plan = json.loads((root / "plan.json").read_text())
    acceptance = json.loads((root / "acceptance.json").read_text())
    if not acceptance["complete"] or plan["steps_override"] is not None:
        raise AssertionError("formal run is incomplete")
    tokenizer = AutoTokenizer.from_pretrained(minilm, local_files_only=True)
    # This reread is an offline audit, outside the continual algorithm.
    raw = torch.load(C.U.DATA_CFG[dataset]["path"], map_location="cpu", weights_only=False)
    checked_memories = checked_text_rows = 0
    retained_tokens = {a: 0 for a in T.ARMS}
    for order, groups in plan["orders"].items():
        directory = root / order
        previous = {}
        for stage in range(3):
            common = json.loads((directory / f"shared_s{stage}.json").read_text())
            ids = torch.tensor(common["source_ids"])
            if any(int(i) % 10 in (1, 2, 3) for i in ids):
                raise AssertionError("holdout source in cluster audit")
            if not torch.isin(raw.y[ids], torch.tensor(groups[stage])).all():
                raise AssertionError("noncurrent raw source in cluster audit")
            for entry in common["graph_query_stats"]:
                if not all(0 <= i < len(ids) for i in entry["neighbor_local_ids"]):
                    raise AssertionError("query neighbor outside legal delta")
            expected_x = torch.stack([raw.x[torch.tensor(c)].float().mean(0)
                                      for c in common["clusters_global_ids"]])
            hashes = []
            for arm in T.ARMS:
                path = directory / f"{arm}_s{stage}_memory.pt"
                memory = torch.load(path, map_location="cpu", weights_only=False)
                evidence = json.loads((directory / f"{arm}_s{stage}_text.json").read_text())
                if C.sha(path) != evidence["file_sha256"]:
                    raise AssertionError("serialized memory changed")
                if C.tensor_hash(memory) != evidence["support_sha256"]:
                    raise AssertionError("memory tensor hash mismatch")
                keep = len(previous[arm]["x"]) if stage else 0
                if not torch.equal(memory["x"][keep:], expected_x):
                    raise AssertionError("Mean-X does not match saved source clusters")
                C.assert_frozen(previous.get(arm), memory)
                hashes.append(C.tensor_hash({k: memory[k] for k in
                    ("x", "mass", "depth", "stage", "edge_index")}))
                for local_row, (text, row) in enumerate(zip(memory["texts"][keep:], evidence["rows"])):
                    actual = BT.token_count(tokenizer, text)
                    content = sum(BT.token_count(tokenizer, p) for p in row["inputs"])
                    if (actual > T.BUDGET or content > T.BUDGET or actual != row["retained_tokens"]
                            or content != row["encoder_content_tokens"]):
                        raise AssertionError("actual text budget differs from declared audit")
                    members = common["clusters_global_ids"][local_row]
                    if not all(int(ids[i]) in members for i in row["selected_source_local_ids"]):
                        raise AssertionError("selected text source outside assigned cluster")
                    if not torch.isfinite(memory["text_emb"][keep+local_row]).all():
                        raise AssertionError("nonfinite target")
                    retained_tokens[arm] += actual
                    checked_text_rows += 1
                if len(memory["texts"][keep:]) != len(evidence["rows"]):
                    raise AssertionError("missing text audit rows")
                previous[arm] = memory
                checked_memories += 1
            if len(set(hashes)) != 1:
                raise AssertionError("between-arm nontext difference")
    result = dict(pass_audit=True, dataset=dataset, memories=checked_memories,
                  new_text_rows=checked_text_rows, retained_tokens=retained_tokens,
                  scope="independent saved memory/token/source audit; raw reread is offline audit only",
                  test_labels_evaluated=False, checkpoint_reload_in_runner=acceptance["checkpoint_reload_verified"])
    C.save_json(root / "artifact_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dataset", required=True, choices=["cora", "citeseer"])
    ap.add_argument("--minilm", required=True)
    args = ap.parse_args()
    audit(args.root, args.dataset, args.minilm)
