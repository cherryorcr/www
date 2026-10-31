import os

import numpy as np
import torch
from tqdm import tqdm

from buffer import seed_everything
from dataset import SynGraphDataset, GraphDataset
from epoch import epoch_train, epoch_test
from model import CLIP


def main(args):
    save_data = torch.load(os.path.join(str(args.buffer_save_dir), args.name, f"syn_data_{args.it}.pt"), map_location=args.device)
    node_f = save_data["node_f"]
    text_embeds = save_data["text_embeds"]
    graph_encoder_lr = save_data["graph_encoder_lr"]
    text_encoder_lr = save_data["text_encoder_lr"]

    syn_dataset = SynGraphDataset(node_f, text_embeds, args)
    syn_dataset.graph_encoder_lr = graph_encoder_lr
    syn_dataset.text_encoder_lr = text_encoder_lr
    syn_dataset.set_eval_model()

    graph_dataset = GraphDataset(args)

    acc_list = []
    for _ in range(args.eval_time):
        eval_model = CLIP(args).to(args.device)

        eval_optimizer = torch.optim.SGD([
            {"params": eval_model.graph_encoder.parameters(), "lr": syn_dataset.graph_encoder_lr.item(),
             "momentum": 0.9, "weight_decay": 5e-4},
            {"params": eval_model.text_encoder.parameters(), "lr": syn_dataset.text_encoder_lr.item(), "momentum": 0.9,
             "weight_decay": 5e-4},
        ])

        best_acc = 0
        for epoch in range(args.num_epoch_train):
            epoch_train(model=eval_model, optimizer=eval_optimizer, train_dataset=syn_dataset, args=args,
                        is_distill=True)
            acc = epoch_test(model=eval_model, test_dataset=graph_dataset, args=args, is_distill=True)
            if acc > best_acc:
                best_acc = acc
            if not args.is_distill:
                print(f"Epoch: {epoch}, Acc: {acc:.4f}")
        if not args.is_distill:
            print(f"Best Acc: {best_acc:.4f}")
        acc_list.append(best_acc)

    acc = np.mean(acc_list)
    print(f"Accuracy: {acc:.4f}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--dataset_name", type=str, default="products")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--it", type=int, default=5000)
    parser.add_argument("--syn_size", type=int, default=2000)
    parser.add_argument("--syn_num_summary", type=int, default=4)
    parser.add_argument("--syn_ratio_summary", type=float, default=60.0)
    parser.add_argument("--syn_lr", type=float, default=100)
    parser.add_argument("--syn_lr_lr", type=float, default=2e-6)
    parser.add_argument("--is_distill", type=bool, default=False)
    parser.add_argument("--run_name", type=str, default="")

    # eval
    parser.add_argument("--batch_size_train", type=int, default=32)
    parser.add_argument("--batch_size_test", type=int, default=4096)
    parser.add_argument("--num_epoch_train", type=int, default=15)
    parser.add_argument("--eval_time", type=int, default=1)

    # graph encoder
    parser.add_argument("--graph_encoder", type=str, default="gcn")
    parser.add_argument("--graph_encoder_lr", type=float, default=5e-3)
    parser.add_argument("--gnn_input_dim", type=int, default=384)
    parser.add_argument("--gnn_hidden_dim", type=int, default=384)
    parser.add_argument("--gnn_output_dim", type=int, default=384)

    # text encoder
    parser.add_argument("--text_encoder", type=str, default="bert")
    parser.add_argument("--text_encoder_lr", type=float, default=5e-3)
    parser.add_argument("--lm_output_dim", type=int, default=768)

    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    args.buffer_save_dir = os.path.join("./buffer", args.dataset_name, args.graph_encoder, args.text_encoder)

    if args.run_name:
        args.name = args.run_name
    else:
        args.name = f"{args.dataset_name}-{args.syn_size}-{args.seed}-{args.syn_num_summary}-{args.syn_ratio_summary}"

    if args.dataset_name == "art":
        args.sample_size = [10, 10]
    elif args.dataset_name == "products":
        args.sample_size = [10, 5]
    else:
        args.sample_size = [-1, -1]

    seed_everything(args.seed)
    main(args)
