import math

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch_geometric.loader import NeighborSampler
from torch_geometric.loader.neighbor_sampler import EdgeIndex
from tqdm import tqdm

from dataset import GraphDataset, SynGraphDataset
from model import CLIP
from reparam import ReparamModule


def epoch_train(
        model: CLIP,
        train_dataset: GraphDataset,
        optimizer: torch.optim.Optimizer,
        args,
        is_distill=False
):
    model.train()
    model=model.to(args.device)

    sampler = NeighborSampler(train_dataset.edge_index,
                                   sizes=args.sample_size, batch_size=args.batch_size_train,
                                   shuffle=True, num_workers=16)

    loss_sum, num_samples = 0, 0
    for i, (batch_size, n_id, adjs) in enumerate(tqdm(sampler, disable=is_distill)):
        node_f = train_dataset.node_f[n_id].to(args.device)
        adjs = [adj.to(args.device) for adj in adjs]
        text_embeds = train_dataset.text_embeds[n_id[:batch_size]].to(args.device)

        loss, logits = model(node_f, adjs, text_embeds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        num_samples += batch_size

        if i % 10 == 0 and not is_distill:
            tqdm.write(f'Training Loss: {loss_sum / num_samples:.4f}')

    return round(loss_sum / num_samples, 4)


@torch.no_grad()
def epoch_test(
        model: CLIP,
        test_dataset: GraphDataset,
        args,
        is_distill=False
):
    model.eval()
    model=model.to(args.device)

    label_list = test_dataset.label_list
    all_labels = test_dataset.all_labels
    all_labels_embeds = test_dataset.all_labels_embeds.to(args.device)

    acc_list = []
    for task in tqdm(test_dataset.tasks, disable=is_distill):
        acc_sum = 0.0
        num_samples = 0
        for labels_idx, batch_size, n_id, adjs in task:
            ground_truth = label_list[n_id[:batch_size]]
            if isinstance(ground_truth, str):
                ground_truth = np.array([ground_truth])
            text_input = all_labels_embeds[labels_idx]
            labels = all_labels[labels_idx]

            node_f = test_dataset.node_f[n_id].to(args.device)
            adjs = [adj.to(args.device) for adj in adjs]

            logits = model(node_f, adjs, text_input, is_eval=True)
            pred = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
            y_pred = labels[pred]
            acc = accuracy_score(ground_truth, y_pred)
            acc_sum += acc * batch_size
            num_samples += batch_size

        acc = acc_sum / num_samples
        acc_list.append(acc)

    acc = np.mean(acc_list)
    return round(acc, 4)


def epoch_train_manual(
        model: ReparamModule,
        param: torch.Tensor,
        train_dataset: SynGraphDataset,
        graph_encoder_lr: torch.Tensor,
        text_encoder_lr: torch.Tensor,
        args,
):
    model.train()
    model=model.to(args.device)

    num_nodes = len(train_dataset)
    batch_size = args.syn_batch_size_train
    num_batches = math.ceil(num_nodes // batch_size)
    node_indices = torch.arange(num_nodes, device=args.device)
    node_indices = node_indices[torch.randperm(num_nodes)]

    loss_sum, num_samples = 0, 0
    for i in range(num_batches):
        batch_idx = node_indices[i * batch_size: (i + 1) * batch_size]
        edge_index = EdgeIndex(edge_index=torch.stack([torch.arange(batch_size), torch.arange(batch_size)], dim=0),
                               e_id=None, size=(batch_size, batch_size))
        adjs = [edge_index, edge_index]

        node_f = train_dataset.node_f[batch_idx].to(args.device)
        adjs = [adj.to(args.device) for adj in adjs]
        text_embeds = train_dataset.text_embeds[batch_idx].to(args.device)

        loss, logits = model(node_f, adjs, text_embeds, flat_param=param)

        graph_mask = model.get_params_mask_for_module("graph_encoder", param)
        text_mask = model.get_params_mask_for_module("text_encoder", param)
        grad = torch.autograd.grad(loss, param, create_graph=True)[0]
        graph_grad = torch.where(graph_mask, grad, torch.tensor(0.0))
        text_grad = torch.where(text_mask, grad, torch.tensor(0.0))
        step = graph_encoder_lr * graph_grad + text_encoder_lr * text_grad
        param = param - step

        loss_sum += loss.item()
        num_samples += batch_size

    return round(loss_sum / num_samples, 4), param