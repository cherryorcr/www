import os
import random
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
import wandb
from torch_geometric import seed_everything

from dataset import GraphDataset
from epoch import epoch_test, epoch_train
from model import CLIP


def main(args):
    wandb.init(
        project="TAGSAM-buffer",
        name=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        config=args,
    )

    graph_dataset = GraphDataset(args)
    expert_model = CLIP(args).to(args.device)

    optimizer = torch.optim.Adam([
        {"params": expert_model.graph_encoder.parameters(), "lr": args.graph_encoder_lr},
        {"params": expert_model.text_encoder.parameters(), "lr": args.text_encoder_lr},
    ])
    optimizer.zero_grad()

    best_acc = 0
    acc = epoch_test(model=expert_model, test_dataset=graph_dataset, args=args)
    tqdm.write(f"Init Acc: {acc}")
    for e in range(args.num_epoch_train):
        epoch_train(model=expert_model, optimizer=optimizer, train_dataset=graph_dataset, args=args)
        acc = epoch_test(model=expert_model, test_dataset=graph_dataset, args=args)
        wandb.log({f"acc": acc}, step=e)
        tqdm.write(f"Epoch {e} Acc: {acc}")

        if acc > best_acc:
            best_acc = acc
            torch.save(expert_model.state_dict(), os.path.join(args.buffer_save_dir, f"expert_state.pt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--dataset_name", type=str, default="art")
    parser.add_argument("--num_epoch_train", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--batch_size_train", type=int, default=1024)
    parser.add_argument("--batch_size_test", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=44)

    # graph encoder
    parser.add_argument("--graph_encoder", type=str, default="gcn")
    parser.add_argument("--graph_encoder_lr", type=float, default=2e-5)
    parser.add_argument("--gnn_input_dim", type=int, default=384)
    parser.add_argument("--gnn_hidden_dim", type=int, default=384)
    parser.add_argument("--gnn_output_dim", type=int, default=384)

    # text encoder
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--text_encoder_lr", type=float, default=2e-5)
    parser.add_argument("--lm_output_dim", type=int, default=768)

    args = parser.parse_args()
    args.buffer_save_dir = os.path.join("./buffer", args.dataset_name, args.graph_encoder, args.text_encoder)
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    if args.dataset_name == "art":
        args.sample_size = [10, 10]
    elif args.dataset_name == "products":
        args.sample_size = [10, 5]
    else:
        args.sample_size = [-1, -1]

    os.makedirs(args.buffer_save_dir, exist_ok=True)

    seed_everything(args.seed)
    main(args)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False