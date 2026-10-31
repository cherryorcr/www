"""Native TAGSAM model, inner epoch function and response loss under access limits."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import common as C
from common import torch, R
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader.neighbor_sampler import EdgeIndex
from transformers import BertModel, BertTokenizer
from uq_high_fidelity_baselines_20261012 import pinned_functions, verified_repo
import uq_tagsam_native_selector_20261012 as N

BERT = Path('/home/pc/huggingface-cache/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594')


def components(device, dimension):
    files = ('model.py', 'selection.py', 'distill.py', 'dataset.py', 'epoch.py', 'reparam.py', 'buffer.py')
    provenance = verified_repo(N.REPO, N.COMMIT, files)
    namespace = dict(torch=torch, nn=torch.nn, F=F, np=np, GCNConv=GCNConv)
    pinned_functions(N.REPO/'model.py', {'GCN', 'GraphEncoder', 'TextProjection', 'CLIP', 'wBCELoss'}, namespace)
    spec = importlib.util.spec_from_file_location('cotag_native_tagsam_reparam', N.REPO/'reparam.py')
    reparam = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reparam)
    epochs = dict(torch=torch, math=math, EdgeIndex=EdgeIndex,
                  ReparamModule=reparam.ReparamModule, SynGraphDataset=object)
    pinned_functions(N.REPO/'epoch.py', {'epoch_train_manual'}, epochs)
    args = SimpleNamespace(device=device, graph_encoder='gcn', gnn_input_dim=dimension,
        gnn_hidden_dim=384, gnn_output_dim=384, lm_output_dim=768, syn_batch_size_train=20)
    return SimpleNamespace(model=namespace['CLIP'], loss=namespace['wBCELoss'](),
        reparam=reparam.ReparamModule, epoch=epochs['epoch_train_manual'], args=args,
        provenance=provenance)


class BertFeatures:
    def __init__(self, device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(str(BERT), local_files_only=True)
        self.model = BertModel.from_pretrained(str(BERT), local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts):
        result = []
        for start in range(0, len(texts), 16):
            tokens = self.tokenizer(list(texts[start:start+16]), return_tensors='pt',
                                    padding=True, truncation=True).to(self.device)
            result.append(self.model(**tokens).last_hidden_state[:, 0].cpu())
        return torch.cat(result)


class Synthetic:
    def __init__(self, x, text):
        self.node_f, self.text_embeds = x, text

    def __len__(self):
        return len(self.node_f)


def full_adjs(edges, n, device):
    edges = edges.to(device)
    return [EdgeIndex(edges, None, (n, n)), EdgeIndex(edges, None, (n, n))]


def train_teacher(native, raw, text, stage, device):
    torch.manual_seed(9000+stage)
    model = native.model(native.args).to(device)
    optimizer = torch.optim.Adam([
        {'params': model.graph_encoder.parameters(), 'lr': 2e-5},
        {'params': model.text_encoder.parameters(), 'lr': 2e-5}])
    x, text = raw['x'].to(device), text.to(device)
    adjs = full_adjs(raw['edge_index'], len(x), device)
    generator = torch.Generator().manual_seed(9000+stage)
    losses = []
    for epoch in range(3):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), 1024):
            idx = order[start:start+1024].to(device)
            graph, txt = model.encode_graph(x, adjs)[idx], model.encode_text(text[idx])
            logits = (1/.07)*graph@txt.T
            targets = torch.arange(len(idx), device=device)
            loss = (F.cross_entropy(logits, targets)+F.cross_entropy(logits.T, targets))/2
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite native teacher loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    return model.eval(), dict(epochs=3, minibatch=1024, lr=2e-5, updates=len(losses),
        losses=losses, selection='fixed last epoch; no external test-accuracy selection',
        message_passing='full legal source graph, not native NeighborSampler batches')


def optimize(native, initial, parent, synthetic_text, raw, raw_text, teacher,
             stage, device, steps, out):
    memory = R.L.append_edge_free(parent, dict(initial))
    keep = 0 if parent is None else len(parent['x'])
    prefix = memory['x'][:keep].to(device)
    var = torch.nn.Parameter(initial['x'].to(device).clone())
    rates = [torch.nn.Parameter(torch.tensor(.005, device=device)) for _ in range(2)]
    optimizer = torch.optim.SGD([
        {'params': [var], 'lr': 100., 'momentum': .5},
        {'params': [rates[0]], 'lr': 2e-6, 'momentum': .5},
        {'params': [rates[1]], 'lr': 2e-6, 'momentum': .5}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=.95)
    rx, rt = raw['x'].to(device), raw_text.to(device)
    text = synthetic_text.to(device)
    adjs = full_adjs(raw['edge_index'], len(rx), device)
    curve = []
    checkpoint = out/'optimizer_latest.pt'
    first = 1
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state['initial_hash'] != R.V.memory_hash(initial):
            raise AssertionError('Native checkpoint initial block differs')
        with torch.no_grad():
            var.copy_(state['x'])
            for rate, value in zip(rates, state['rates']):
                rate.copy_(value)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        curve, first = state['curve'], state['step']+1
    for update in range(first, steps+1):
        torch.manual_seed(9001+stage*1009+update)
        student = native.reparam(native.model(native.args)).to(device)
        params = torch.cat([p.detach().reshape(-1) for p in student.parameters()]).requires_grad_(True)
        support = Synthetic(torch.cat((prefix, var)), text)
        for _ in range(15):
            _, params = native.epoch(student, params, support, rates[0], rates[1], native.args)
        reference = torch.randperm(len(rx), generator=torch.Generator().manual_seed(
            700777+stage*1009+update))[:2000].to(device)
        with torch.no_grad():
            teacher_logits = (1/.07)*teacher.encode_graph(rx, adjs)[reference]@teacher.encode_text(rt[reference]).T
        logits = student(rx, adjs, rt[reference], is_eval=True, flat_param=params)[reference]
        loss = native.loss(logits, teacher_logits)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Native response loss nonfinite at {update}')
        optimizer.zero_grad(set_to_none=True)
        gradients = torch.autograd.grad(loss, [var, *rates])
        for parameter, gradient in zip((var, *rates), gradients):
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError(f'Native gradient nonfinite at {update}')
            parameter.grad = gradient
        torch.nn.utils.clip_grad_norm_([var], 5.)
        optimizer.step()
        scheduler.step()
        curve.append(dict(step=update, loss=float(loss.detach()),
            graph_lr=float(rates[0].detach()), text_lr=float(rates[1].detach()),
            inner_updates=15*(len(memory['x'])//20), feature_outer_lr=optimizer.param_groups[0]['lr']))
        if update == 1 or update % 25 == 0 or update == steps:
            C.save(out/'trace.json', curve)
            C.emit(event='native_epoch_response', stage=stage, **curve[-1])
        if update % 250 == 0 or update == steps:
            out.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix('.tmp')
            torch.save(dict(step=update, x=var.detach().cpu(), rates=[r.detach().cpu() for r in rates],
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), curve=curve,
                initial_hash=R.V.memory_hash(initial)), temporary)
            temporary.replace(checkpoint)
        del student, params, support, loss, gradients, logits
    memory['x'] = torch.cat((prefix.cpu(), var.detach().cpu()))
    R.V.validate_memory(memory, parent, len(memory['x']))
    return memory, dict(curve=curve, outer_steps=steps, inner_epochs=15, inner_batch=20,
        inner_updates_per_outer=15*(len(memory['x'])//20), omitted_tail_per_epoch=len(memory['x'])%20,
        lr_schedule='native StepLR(200, 0.95)', response_resample_cap=2000,
        final_graph_lr=float(rates[0].detach()), final_text_lr=float(rates[1].detach()))
