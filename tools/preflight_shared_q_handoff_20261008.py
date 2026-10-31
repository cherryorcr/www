"""Server-side read-only inventory of handoff checkpoints and code hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    records = []
    for path in sorted(args.manifests.glob("*_manifest.json")):
        item = json.loads(path.read_text())
        for row in item["rows"]:
            if row["stage"] != 0:
                continue
            memory_path = Path(row["memory_path"])
            info = dict(dataset=item["dataset"], arm=row["arm"], path=str(memory_path), exists=memory_path.exists())
            if memory_path.exists():
                memory = torch.load(memory_path, map_location="cpu", weights_only=False)
                info["keys"] = {k: list(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__ for k,v in memory.items()}
                info["text_chars_max"] = max(map(len, memory["texts"]))
                info["text_chars_mean"] = sum(map(len, memory["texts"])) / len(memory["texts"])
                info["self_edge_count"] = int((memory["edge_index"][0] == memory["edge_index"][1]).sum())
                info["nonself_edge_count"] = int((memory["edge_index"][0] != memory["edge_index"][1]).sum())
                info["label_values"] = sorted(set(memory["labels"].tolist())) if "labels" in memory else None
                del memory
            records.append(info)
            print(json.dumps(info), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(records=records), indent=2))


if __name__ == "__main__":
    main()
