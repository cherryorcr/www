"""Convert the local GraphCLIP Cora asset to TAGSAM's released data schema.

This is deliberately a data adapter only: it does not alter examples, labels,
texts, features, or topology.  The source path and output path are recorded in
the saved artifact so qualification runs remain auditable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cora")
    parser.add_argument(
        "--graphclip-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "graph-text-alignment" / "GraphCLIP",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    graphclip_root = args.graphclip_root.resolve()
    sys.path.insert(0, str(graphclip_root))
    from data.load import load_data  # pylint: disable=import-outside-toplevel

    previous_cwd = Path.cwd()
    os.chdir(graphclip_root)
    try:
        data, texts, class_names, class_descriptions = load_data(args.dataset, seed=0)
    finally:
        os.chdir(previous_cwd)
    if len(texts) != data.num_nodes:
        raise ValueError(f"text/node mismatch: {len(texts)} vs {data.num_nodes}")
    if data.x.ndim != 2 or data.x.size(0) != data.num_nodes:
        raise ValueError(f"invalid node features: {tuple(data.x.shape)}")

    label_list = [class_names[int(label)] for label in data.y.tolist()]
    output = args.output or (Path(__file__).resolve().parent / "data" / f"{args.dataset}.pt")
    payload = {
        "node_f": data.x.detach().cpu().float(),
        "edge_index": data.edge_index.detach().cpu().long(),
        "text_list": list(texts),
        "label_list": np.asarray(label_list),
        "all_labels": np.asarray(class_names),
        "labels_desc": np.asarray(class_descriptions),
        "audit_metadata": {
            "source": str((graphclip_root / "processed_data" / f"{args.dataset}.pt").resolve()),
            "adapter": str(Path(__file__).resolve()),
            "seed": 0,
            "num_nodes": int(data.num_nodes),
            "num_edges_directed": int(data.edge_index.size(1)),
            "feature_dim": int(data.x.size(1)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(payload["audit_metadata"])
    print(f"saved: {output.resolve()}")


if __name__ == "__main__":
    main()
