"""Evaluate the saved official TAGSAM teacher with all Cora classes."""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from dataset import GraphDataset
from epoch import epoch_test
from model import CLIP


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="cora")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--eval_time", type=int, default=1)
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    args.buffer_save_dir = os.path.join("./buffer", args.dataset_name, "gcn", "bert")
    args.sample_size = [-1, -1]
    args.batch_size_test = 2048
    args.batch_size_train = 20
    args.gnn_input_dim = 384
    args.gnn_hidden_dim = 384
    args.gnn_output_dim = 384
    args.lm_output_dim = 768
    args.graph_encoder = "gcn"
    args.text_encoder = "bert"
    seed_everything(args.seed)
    dataset = GraphDataset(args)
    model = CLIP(args).to(args.device)
    state = torch.load(os.path.join(args.buffer_save_dir, "expert_state.pt"),
                       map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    print({"classes": len(dataset.all_labels), "nodes": len(dataset),
           "accuracy_7class": epoch_test(model=model, test_dataset=dataset, args=args)})


if __name__ == "__main__":
    main()
