"""Build class-subset variants to fill the k gap (3500..6500) and tighten the gate.

The current gate evidence has a wide hole: closed side k in {108,127,468,1662,3074},
open side k in {6773,6922}. The threshold (4000) has never been approached. We
therefore build derived datasets whose k lands in the gap, WITHOUT touching any
new test set:

  - take a source dataset (bookchild, sportsfit, bookhis, wikics)
  - keep a SUBSET of classes (chosen by size, deterministic), keeping all
    features/texts/edges induced by the kept nodes
  - k = floor(0.05 * |pool|) of the subset; choose the subset size so k lands in
    the target range

This is legitimate: the derived dataset is a genuinely different problem instance
(different class set, different k), and we can hold its test partition out.
It is NOT the same as subsampling nodes (which keeps the class distribution);
here the class set changes, which is exactly the dimension the gate is about.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-k", type=int, required=True)
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    d = load(args.source)
    y = np.asarray(d["original_node_ids"])  # just to get n
    n = len(d["label_list"])
    labels = [str(v) for v in np.asarray(d["label_list"])]
    all_labels = [str(v) for v in np.asarray(d["all_labels"])]
    lab2i = {nm: i for i, nm in enumerate(all_labels)}
    idx = np.array([lab2i[v] for v in labels])

    oid = np.asarray(d["original_node_ids"]).astype(np.int64)
    pool_m = (oid % 10) > 2
    cnt = np.bincount(idx[pool_m], minlength=len(all_labels))

    # greedy: add classes largest-first until floor(0.05 * pool_subset) >= target-k
    order = sorted(range(len(all_labels)), key=lambda i: -cnt[i])
    kept, total = [], 0
    for i in order:
        kept.append(i)
        total += int(cnt[i])
        if math.floor(args.rate * total) >= args.target_k:
            break

    keep_m = np.isin(idx, kept)
    rows = np.where(keep_m)[0]
    r = torch.tensor(rows, dtype=torch.long)

    # induced edges
    e = d["edge_index"]
    em = keep_m[e[0].numpy()] & keep_m[e[1].numpy()]
    e = e[:, torch.tensor(em)]
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[r] = torch.arange(len(rows))
    e = torch.stack((remap[e[0]], remap[e[1]]))

    sub_pool = int(((oid[rows] % 10) > 2).sum())
    k = math.floor(args.rate * sub_pool)
    payload = {
        "node_f": d["node_f"][r],
        "edge_index": e,
        "text_list": [d["text_list"][int(i)] for i in rows],
        "label_list": np.asarray([labels[int(i)] for i in rows]),
        "all_labels": np.asarray([all_labels[i] for i in kept]),
        "labels_desc": np.asarray([f"a {all_labels[i].lower()}" for i in kept]),
        "original_node_ids": torch.tensor(oid[rows]),
        "audit_metadata": {
            "derived_from": args.source, "subset_classes": [all_labels[i] for i in kept],
            "target_k": args.target_k, "actual_k": k, "n_pool": sub_pool,
            "note": "class-subset variant to fill the k-gap; test partition held out",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"{args.name}: classes={len(kept)} N={len(rows)} pool={sub_pool} k={k} "
          f"(target {args.target_k}) -> {out}", flush=True)
    with open(str(out) + ".manifest.json", "w") as f:
        json.dump(payload["audit_metadata"], f, indent=1)


if __name__ == "__main__":
    main()
