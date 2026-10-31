"""Convert Graph-COM/Text-Attributed-Graphs datasets into this project's schema.

Source schema (torch_geometric Data): x, edge_index, y, raw_texts (list[str]),
class_node_text_feat (C,384 = class-name embeddings), node_text_feat (N,384).
Our TAGSAM-style schema (like ogbn_arxiv_full_minilm_semantic.pt):
  node_f, edge_index, text_list, label_list, all_labels, labels_desc,
  original_node_ids

Design decisions (recorded for auditability):
  - node_f  <- x (raw features; the project's frozen_v1 used raw X, not text feat)
  - texts   <- raw_texts
  - labels  <- y -> class index; label names taken from class_node_text_feat order
               if no explicit name list is available, we synthesize stable names
               "class_{i}" and additionally store the class-node text (which is
               the real textual class description) in labels_desc so prompts can
               use it.
  - original_node_ids <- arange(N).  This dataset has NO external id space, so the
    project's `%10` dev/test rule is applied to these synthetic ids. That gives a
    deterministic, dataset-independent 10/10 split with the SAME rule as the other
    datasets, and it does not use labels.

The class PROMPT used at eval is built from labels_desc; we therefore set
labels_desc to the class description text (from class_node_text_feat we cannot
recover the literal string, so we use `categories.csv` when available, else
"class {i}").  For wikics/bookhis/sportsfit we use the category names from
categories.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

SNAP = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Graph-COM--Text-Attributed-Graphs/"
    "snapshots/c9fc08d286d11f164ea8cb61ded3179d23037f05"))

# file naming varies: some datasets use lowercase name.pt, WebKB uses Title.pt
LAYOUT = {
    "wikics":    ("wikics/processed_data.pt",    "wikics/raw_texts.pt",    "wikics/categories.csv"),
    "bookhis":   ("bookhis/processed_data.pt",   "bookhis/raw_texts.pt",   "bookhis/categories.csv"),
    "bookchild": ("bookchild/processed_data.pt", "bookchild/raw_texts.pt", "bookchild/categories.csv"),
    "sportsfit": ("sportsfit/processed_data.pt", "sportsfit/raw_texts.pt", "sportsfit/categories.csv"),
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LAYOUT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--snap", default=str(SNAP))
    args = ap.parse_args()

    snap = Path(args.snap)
    pf, tf, cf = LAYOUT[args.dataset]
    d = torch.load(snap / pf, map_location="cpu", weights_only=False)
    raw_texts = torch.load(snap / tf, map_location="cpu", weights_only=False)
    if len(raw_texts) != int(d.num_nodes):
        raise ValueError(f"text/node mismatch {len(raw_texts)} vs {d.num_nodes}")

    y = d.y.long()
    C = int(y.max()) + 1
    # category names if available
    names = None
    if (snap / cf).exists():
        try:
            import pandas as pd
            df = pd.read_csv(snap / cf)
            col = [c for c in df.columns if c.lower() in ("category", "name", "label", "class")]
            if col:
                names = [str(v) for v in df[col[0]].tolist()]
        except Exception:
            names = None
    if names is None or len(names) != C:
        names = [f"class_{i}" for i in range(C)]

    n = int(d.num_nodes)
    # Some datasets (bookhis/sportsfit/bookchild) ship NO raw features (x is None);
    # for those we use node_text_feat -- the 384-d text-derived node embedding --
    # as X. This is recorded in the audit metadata. wikics additionally has a
    # 300-d raw x, which we prefer when present.
    if d.x is not None:
        x_src, x_name = d.x.detach().cpu().float(), "Data.x (raw features)"
    else:
        x_src, x_name = d.node_text_feat.detach().cpu().float(), "Data.node_text_feat (384-d text embedding)"
    payload = {
        "node_f": x_src,
        "edge_index": d.edge_index.detach().cpu().long(),
        "text_list": list(raw_texts),
        "label_list": np.asarray([names[int(v)] for v in y.tolist()]),
        "all_labels": np.asarray(names),
        "labels_desc": np.asarray([str(nm) for nm in names]),
        "original_node_ids": torch.arange(n, dtype=torch.long),
        "audit_metadata": {
            "source_repo": "Graph-COM/Text-Attributed-Graphs",
            "snapshot": str(snap), "processed_data": pf, "raw_texts": tf,
            "processed_data_sha256": sha256(snap / pf),
            "raw_texts_sha256": sha256(snap / tf),
            "adapter": "prepare_tag_benchmarks.py",
            "node_f_source": x_name,
            "split_rule": "original_node_ids = arange(N); dev = %10==1, test = %10==2",
            "note": "labels used ONLY to build label_list/name mapping; never in condensation",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"{args.dataset}: N={n} edges={payload['edge_index'].shape[1]} C={C} "
          f"classes={names[:4]}... -> {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"  dev={int(((payload['original_node_ids']%10)==1).sum())} "
          f"test={int(((payload['original_node_ids']%10)==2).sum())} "
          f"pool_fraction={(payload['original_node_ids']%10>2).float().mean():.3f}", flush=True)
    with open(str(out) + ".manifest.json", "w") as f:
        json.dump({k: v for k, v in payload["audit_metadata"].items()} |
                  {"n": n, "C": C, "classes": names}, f, indent=1)


if __name__ == "__main__":
    main()
