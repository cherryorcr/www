import os
import pickle

import numpy as np
import torch
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm

from model import TextEncoder


class GraphDataset:
    def __init__(self, args):
        self.args = args

        data = torch.load(f'./data/{args.dataset_name}.pt', weights_only=False, map_location="cpu")
        self.node_f = data['node_f']
        self.edge_index = data['edge_index']
        self.text_list = data['text_list']
        self.label_list = data['label_list']
        self.all_labels = data['all_labels']
        self.labels_desc = data['labels_desc']

        self.text_embeds = None
        self.all_labels_embeds = None
        self.process_text2emb()

        self.test_labels = None
        self.test_samples = None
        self.test_split()

        self.tasks = None
        self.process_subgraph()

    def __len__(self):
        return len(self.node_f)

    @torch.no_grad()
    def process_text2emb(self):
        cache_file = os.path.join(self.args.buffer_save_dir, "cache.pt")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.text_embeds = cache_data.get('text_embeds')
                self.all_labels_embeds = cache_data.get('all_labels_embeds')
                return

        print("Start to process text to embeddings...")
        all_text_embeds = []
        all_labels_embeds = []
        batch_size = 64
        text_encoder = TextEncoder(self.args).to(self.args.device)

        for i in tqdm(range(0, len(self.text_list), batch_size)):
            batch_texts = self.text_list[i:i + batch_size]
            batch_embeds = text_encoder(batch_texts).detach().cpu()

            all_text_embeds.append(batch_embeds)
            del batch_embeds
            torch.cuda.empty_cache()

        labels_with_desc = [f"{label} {desc}" for label, desc in zip(self.all_labels, self.labels_desc)]
        for i in tqdm(range(0, len(self.all_labels), batch_size)):
            batch_texts = labels_with_desc[i:i + batch_size]
            batch_embeds = text_encoder(batch_texts).detach().cpu()

            all_labels_embeds.append(batch_embeds)
            del batch_embeds
            torch.cuda.empty_cache()

        self.text_embeds = torch.cat(all_text_embeds, dim=0).to("cpu")
        self.all_labels_embeds = torch.cat(all_labels_embeds, dim=0).to("cpu")

        cache_data = {
            'text_embeds': self.text_embeds,
            'all_labels_embeds': self.all_labels_embeds
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    def test_split(self):
        filtered_labels = [label for label in self.all_labels if label != "nan"]
        filtered_labels = np.array(filtered_labels)
        num_labels = len(filtered_labels)
        # Cora contains seven classes. The released evaluator groups only the
        # first five, which makes its accuracy incomparable to a seven-class
        # TAG evaluation.
        n_way = len(filtered_labels)
        num_groups = num_labels // n_way

        label_to_all_index = {label: idx for idx, label in enumerate(self.all_labels)}

        label_to_list_index = {}
        for idx, label in enumerate(self.label_list):
            if label not in label_to_list_index:
                label_to_list_index[label] = []
            label_to_list_index[label].append(idx)

        label_to_list_index = {label: np.array(indices) for label, indices in label_to_list_index.items()}

        labels = []
        samples = []
        for i in range(num_groups):
            test_labels = filtered_labels[i * n_way: (i + 1) * n_way]
            test_labels_idx = [label_to_all_index[label] for label in test_labels]
            labels.append(test_labels_idx)

            test_samples_idx = np.concatenate([label_to_list_index[label] for label in test_labels])
            samples.append(test_samples_idx)

        self.test_labels = labels
        self.test_samples = samples

    def process_subgraph(self):
        cache_file = os.path.join(self.args.buffer_save_dir, f"cache_subgraph_{self.args.batch_size_test}.pt")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.tasks = cache_data.get('tasks')
                return

        print("Start to process test subgraph...")

        tasks = []

        for labels_idx, samples_idx in tqdm(zip(self.test_labels, self.test_samples), total=len(self.test_labels)):
            samples_idx = torch.tensor(samples_idx)
            if self.args.batch_size_test == -1:
                sampler = NeighborSampler(self.edge_index, node_idx=samples_idx,
                                          sizes=self.args.sample_size, batch_size=len(samples_idx),
                                          shuffle=True, num_workers=16)
            else:
                sampler = NeighborSampler(self.edge_index, node_idx=samples_idx,
                                          sizes=self.args.sample_size, batch_size=self.args.batch_size_test,
                                          shuffle=True, num_workers=16)

            task = []
            for batch_size, n_id, adjs in sampler:
                task.append((labels_idx, batch_size, n_id, adjs))

            tasks.append(task)

        self.tasks = tasks

        cache_data = {
            'tasks': self.tasks
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)


class SynGraphDataset(GraphDataset):
    def __init__(self, syn_graph, syn_text, args):
        self.args = args

        self.node_f = syn_graph.detach().to(args.device).requires_grad_(True)
        self.edge_index = torch.stack([torch.arange(len(self)), torch.arange(len(self))], dim=0)
        self.graph_encoder_lr = torch.tensor(args.graph_encoder_lr).to(args.device).requires_grad_(True)
        self.text_encoder_lr = torch.tensor(args.text_encoder_lr).to(args.device).requires_grad_(True)
        self.text_embeds = syn_text.to(args.device).requires_grad_(False)

        self.optimizer = torch.optim.SGD([
            {'params': self.node_f, 'lr': args.syn_lr, "momentum": 0.5},
            {'params': self.graph_encoder_lr, 'lr': args.syn_lr_lr, "momentum": 0.5},
            {'params': self.text_encoder_lr, 'lr': args.syn_lr_lr, "momentum": 0.5}
        ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.95)

    def __len__(self):
        return len(self.node_f)

    def set_eval_model(self):
        self.node_f.requires_grad_(False)
        self.graph_encoder_lr.requires_grad_(False)
        self.text_encoder_lr.requires_grad_(False)

    def set_train_model(self):
        self.node_f.requires_grad_(True)
        self.graph_encoder_lr.requires_grad_(True)
        self.text_encoder_lr.requires_grad_(True)

    def compute_grad(self, loss):
        grad = torch.autograd.grad(loss, [self.node_f, self.graph_encoder_lr, self.text_encoder_lr], allow_unused=True)
        self.node_f.grad = grad[0]
        self.graph_encoder_lr.grad = grad[1]
        self.text_encoder_lr.grad = grad[2]
        torch.nn.utils.clip_grad_norm_(self.node_f, 5)

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
