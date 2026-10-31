"""Generate the class-incremental arxiv split (protocol ④), in the SAME schema as
the temporal3 snapshots so existing runners work by prefix substitution.

Schema (per snapshot): node_f, edge_index, text_list, label_list, all_labels,
labels_desc, original_node_ids, arrival_stage.

Split (pre-registered): greedy size balancing, largest class first into the
currently-smallest stage -> 13/14/13 classes, pool 45,199/45,122/45,152.
dev = node_id%10==1, test = node_id%10==2 (NOT read here), pool = rest.

delta_t{k}      : pool nodes whose class is in stage k
cumulative_t{k} : pool nodes whose class is in stages 1..k
real_t{k}       : ALL nodes (incl dev/test) whose class is in stage k  (for zero-shot eval)

Edges are the induced subgraph on each snapshot's node set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


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
    ap.add_argument("--source", default="/home/pc/research/TAGSAM/data/ogbn_arxiv_full_minilm_semantic.pt")
    ap.add_argument("--out-prefix", default="/home/pc/research/TAGSAM/data/ogbn_arxiv_full_classincr3")
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()

    src = torch.load(args.source, map_location="cpu", weights_only=False)
    node_f = src["node_f"]
    edge_index = src["edge_index"]
    text_list = list(src["text_list"])
    label_list = np.asarray(src["label_list"])
    all_labels = [str(v) for v in np.asarray(src["all_labels"]).tolist()]
    labels_desc = list(src["labels_desc"])
    orig = np.asarray(src["original_node_ids"]) if "original_node_ids" in src else np.arange(len(label_list))
    n = len(label_list)

    lab2idx = {nm: i for i, nm in enumerate(all_labels)}
    idx = np.array([lab2idx[str(v)] for v in label_list])

    oid = np.asarray(orig).astype(np.int64)
    dev_m = (oid % 10) == 1
    test_m = (oid % 10) == 2
    pool_m = ~dev_m & ~test_m

    cnt = np.bincount(idx[pool_m], minlength=len(all_labels))
    stages = {0: [], 1: [], 2: []}
    tot = {0: 0, 1: 0, 2: 0}
    for i in sorted(range(len(all_labels)), key=lambda x: -cnt[x]):
        s = min(tot, key=lambda z: tot[z])
        stages[s].append(int(i))
        tot[s] += int(cnt[i])

    # class -> stage index (0-based), used to fill arrival_stage
    cls_stage = {}
    for s in range(3):
        for c in stages[s]:
            cls_stage[int(c)] = s

    def build(node_mask):
        """Induced subgraph snapshot on the given boolean node mask.

        arrival_stage is the 0-based index of the STAGE THAT INTRODUCED each
        node's class (matching temporal3's semantics: cumulative_t2 contains
        arrival_stage in {0,1}; eval_stagewise splits per-stage masks on it).
        """
        rows = np.where(node_mask)[0]
        r = torch.tensor(rows, dtype=torch.long)
        sub_oid = torch.tensor(oid[rows], dtype=torch.long)
        a_stage = torch.tensor([cls_stage[int(idx[i])] for i in rows], dtype=torch.long)
        # induced edges
        keep = node_mask[edge_index[0].numpy()] & node_mask[edge_index[1].numpy()]
        e = edge_index[:, torch.tensor(keep)]
        remap = torch.full((n,), -1, dtype=torch.long)
        remap[r] = torch.arange(len(rows))
        e = torch.stack((remap[e[0]], remap[e[1]]))
        return {
            "node_f": node_f[r],
            "edge_index": e,
            "text_list": [text_list[int(i)] for i in rows],
            "label_list": label_list[rows],
            "all_labels": np.array(all_labels),
            "labels_desc": labels_desc,
            "original_node_ids": sub_oid,
            "arrival_stage": a_stage,
        }

    prefix = args.out_prefix
    snaps, class_map = [], {}
    for s in range(3):
        cls = sorted(stages[s])
        class_map[f"stage{s + 1}"] = [
            {"index": c, "name": all_labels[c], "desc": labels_desc[c], "pool_nodes": int(cnt[c])}
            for c in cls]
        in_stage = np.isin(idx, cls)
        upto = sorted({c for k in range(s + 1) for c in stages[k]})
        in_upto = np.isin(idx, upto)

        # delta: pool only (condensation source).  cumulative: pool + dev, but
        # NOT test (matches temporal3: cumulative is the eval graph and must
        # contain dev; test stays out until its one-time read).  real: all nodes
        # of this stage's classes incl. dev/test (zero-shot eval reference).
        specs = [("delta", in_stage & pool_m),
                 ("cumulative", in_upto & ~test_m),
                 ("real", in_stage)]
        for kind, mask in specs:
            snap = build(mask)
            path = f"{prefix}_{'delta_t' if kind == 'delta' else ('cumulative_t' if kind == 'cumulative' else 'real_t')}{s + 1}.pt"
            torch.save(snap, path)
            snaps.append({"stage": s + 1, "kind": kind, "path": path,
                          "nodes": int(mask.sum()), "classes": cls})
            if kind == "delta":
                print(f"  stage{s+1} {kind:10s} nodes={int(mask.sum()):,} "
                      f"edges={snap['edge_index'].shape[1]:,}", flush=True)

    manifest = {
        "source": args.source, "source_sha256": sha256(args.source),
        "mode": "class_incremental", "protocol_version": "arxiv-classincr3-v1",
        "stages": 3, "seed": args.seed,
        "split_rule": "greedy size balancing (largest class first into smallest stage)",
        "eval_split": {"dev": "node_id % 10 == 1", "test": "node_id % 10 == 2",
                       "test_accessed": False},
        "class_map": class_map, "stage_pool_totals": tot,
        "snapshots": snaps,
        "schema": "same as temporal3: node_f/edge_index/text_list/label_list/all_labels/"
                  "labels_desc/original_node_ids/arrival_stage",
        "notes": "test NOT read by generator; delta/cumulative exclude dev+test; real_* include all",
    }
    with open(f"{prefix}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nwrote manifest; stage pool totals {tot}")


if __name__ == "__main__":
    main()
