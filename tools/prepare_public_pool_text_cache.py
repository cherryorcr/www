"""Prepare pool-only MiniLM text embeddings for public continual runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--minilm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    sys.path.insert(0, str(args.runner_dir))
    import run_unified_v3_continual as unified  # pylint: disable=import-outside-toplevel

    data = unified.load_unified(
        args.dataset,
        str(args.data),
        str(args.minilm),
        args.device,
        n_stages=5,
        split_mode="base_incremental",
        base_fraction=0.4,
        report_dev_mod=3,
    )
    pool = data["pool_mask"].detach().cpu().bool()
    texts = [str(value) for value in data["texts"]]
    pool_texts = [texts[index] for index in torch.nonzero(pool, as_tuple=False).flatten().tolist()]
    print(f"dataset={args.dataset} rows={len(texts)} pool_rows={len(pool_texts)}", flush=True)
    with torch.no_grad():
        pool_emb = data["encoder"].encode(pool_texts, batch_size=args.batch_size, max_length=256)
    if tuple(pool_emb.shape) != (int(pool.sum()), 384):
        raise ValueError(f"unexpected embedding shape: {tuple(pool_emb.shape)}")
    output = torch.zeros((len(texts), 384), dtype=torch.float32)
    output[pool] = pool_emb.float().cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"saved={args.output} shape={tuple(output.shape)}", flush=True)


if __name__ == "__main__":
    main()
