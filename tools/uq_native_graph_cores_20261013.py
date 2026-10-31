"""Pinned graph-only cores with explicit current-delta and common-text adapters.

Official model/objective definitions are loaded unchanged through their AST.
This avoids importing their unrelated dataset/download/test entrypoints.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch_sparse

import external_continual_mechanisms as M


PINS = {"external_GCond": "e0065cfdfd95963d988c359266a4fb0a4bdf60a1",
        "external_SFGC": "b034621b8e39eba7aa58ec009963a3071cd87079"}


@dataclass(frozen=True)
class Config:
    gcond_steps: int = 1000
    gcond_lr: float = .01
    experts: int = 5
    teacher_epochs: int = 800
    teacher_lr: float = .001
    snapshot_interval: int = 10
    start_epoch: int = 300
    expert_horizon: int = 500
    outer_steps: int = 200
    syn_steps: int = 200
    hidden: int = 256
    feature_lr: float = .001
    synthetic_lr: float = 1.
    synthetic_lr_lr: float = 1e-6


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repository_path(root, name):
    directory = root / "external_repos" / name.removeprefix("external_")
    return directory if directory.is_dir() else root / name


def registration(root):
    sources = {}
    for repository, commit in PINS.items():
        directory = repository_path(root, repository)
        actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if actual != commit:
            raise AssertionError(f"unpinned native source: {repository}: {actual}")
        relative_files = (["utils.py", "models/sgc.py", "gcond_agent_transduct.py"]
                          if repository == "external_GCond" else
                          ["models/gcn.py", "models/reparam_module.py",
                           "metagtt_transduct_adj_identity.py", "buffer_transduct.py"])
        hashes = {f: file_sha(directory / f) for f in relative_files}
        for relative, digest in hashes.items():
            original = subprocess.check_output(["git", "-C", str(directory),
                "show", f"{commit}:{relative}"])
            if hashlib.sha256(original).hexdigest() != digest:
                raise AssertionError(f"native source was edited: {repository}/{relative}")
        sources[repository] = dict(commit=actual, files=hashes)
    return dict(config=asdict(Config()), sources=sources, official_reproduction=False,
        gcond_adaptation="native SGC + native classwise gradient distance; one-step/no inner loop; identity synthetic graph; full current-delta real gradients; common nearest-5 text",
        sfgc_adaptation="native GCN + native ReparamModule + normalized multi-step trajectory objective; 5 experts x 800 epochs; 200 outer x 200 unroll; common nearest-5 text",
        sfgc_budget_difference="official examples use 200 experts and 2000 outer iterations; this is a compute-bounded core adaptation",
        access="current delta only; append-only frozen synthetic prefix; no historical raw graph",
        checkpoint_selection="fixed final construction iteration; no test/validation checkpoint search")


def native_definitions(path, names):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in names]
    if {node.name for node in body} != set(names):
        raise AssertionError("native definition missing")
    namespace = dict(torch=torch, nn=nn, F=F, math=math, np=np, torch_sparse=torch_sparse,
                     init=init, Parameter=Parameter, Module=Module)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def identity(rows, device):
    index = torch.arange(rows, device=device)
    return torch.sparse_coo_tensor(torch.stack((index, index)),
        torch.ones(rows, device=device), (rows, rows)).coalesce()


def initial_features(delta, labels, seed, device, kcenter=False):
    rng = torch.Generator().manual_seed(seed + 1009 * int(delta["stage"]))
    selected = []
    for cls in torch.unique(labels, sorted=True).tolist():
        ids = torch.where(delta["y"] == cls)[0]
        count = int((labels == cls).sum())
        if not kcenter:
            chosen = torch.randperm(len(ids), generator=rng)[:count]
        else:
            points = delta["x"][ids].float().to(device)
            available = torch.ones(len(ids), dtype=torch.bool, device=device)
            distance = torch.full((len(ids),), float("inf"), device=device)
            index = int(torch.randint(len(ids), (), generator=rng))
            choices = []
            for _ in range(count):
                choices.append(index)
                available[index] = False
                distance = torch.minimum(distance, (points - points[index]).square().sum(1))
                distance[~available] = -1
                index = int(distance.argmax())
            chosen = torch.tensor(choices)
        selected.extend(ids[chosen].tolist())
    selected = torch.tensor(selected, dtype=torch.long)
    if len(selected) != len(labels):
        raise AssertionError("initialization budget differs")
    return delta["x"][selected].float().to(device), delta["ids"][selected].tolist()


def remap_labels(real, synthetic, device):
    classes = torch.unique(synthetic, sorted=True).to(device)
    y_real = torch.searchsorted(classes, real.to(device))
    y_syn = torch.searchsorted(classes, synthetic.to(device))
    if not torch.equal(classes[y_real], real.to(device)):
        raise AssertionError("an uncovered delta class entered the native constructor")
    return classes, y_real, y_syn


def gcond_x(delta, labels, seed, device, root, cfg):
    torch.manual_seed(seed + 1009 * int(delta["stage"]))
    repository = repository_path(root, "external_GCond")
    definitions = native_definitions(repository / "models/sgc.py", ["GraphConvolution", "SGC"])
    objective = native_definitions(repository / "utils.py", ["match_loss", "distance_wb"])
    classes, real_y, syn_y = remap_labels(delta["y"], labels, device)
    initial, source_ids = initial_features(delta, labels, seed, device)
    synth = nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([synth], lr=cfg.gcond_lr)
    adj = M.normalized_adjacency(delta["edge_index"], len(real_y), device)
    real_x = delta["x"].float().to(device)
    syn_adj = identity(len(labels), device)
    counts = torch.bincount(syn_y, minlength=len(classes)).float()
    trace = []
    for step in range(cfg.gcond_steps):
        model = definitions["SGC"](nfeat=initial.shape[1], nhid=cfg.hidden,
            nclass=len(classes), nlayers=2, dropout=0., with_bn=False, device=device).to(device)
        model.initialize()
        parameters = list(model.parameters())
        real_output = model(real_x, adj)
        syn_output = model(synth, syn_adj)
        loss = torch.zeros((), device=device)
        for cls in range(len(classes)):
            real_loss = F.nll_loss(real_output[real_y == cls], real_y[real_y == cls])
            real_grad = torch.autograd.grad(real_loss, parameters, retain_graph=True)
            syn_loss = F.nll_loss(syn_output[syn_y == cls], syn_y[syn_y == cls])
            syn_grad = torch.autograd.grad(syn_loss, parameters, create_graph=True, retain_graph=True)
            loss = loss + counts[cls]/counts.max() * objective["match_loss"](
                syn_grad, [v.detach() for v in real_grad], SimpleNamespace(dis_metric="ours"), device)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("GCond-x native gradient matching diverged")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(synth.grad.norm())
        optimizer.step()
        if step == 0 or (step+1) % 100 == 0 or step+1 == cfg.gcond_steps:
            trace.append(dict(step=step+1, gradient_distance=float(loss.detach()), feature_gradient_norm=norm))
            print(f"gcond_x construction step={step+1} objective={float(loss.detach()):.6f}", flush=True)
    return synth.detach(), dict(initial_source_ids=source_ids,
        objective="unaltered official match_loss/dis_metric=ours and class-size coefficient",
        proxy="unaltered official SGC, 2 propagation steps, independent random initialization each outer step",
        initialization="class-stratified real rows; synthetic features freely optimized",
        optimizer_steps=cfg.gcond_steps, inner_steps=0, learning_rate=cfg.gcond_lr,
        trace=trace, feature_update_norm=float((synth.detach()-initial).norm()))


def flat_parameters(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()]).clone()


def sfgc(delta, labels, seed, device, root, cfg):
    torch.manual_seed(seed + 1009 * int(delta["stage"]))
    repository = repository_path(root, "external_SFGC")
    definitions = native_definitions(repository / "models/gcn.py", ["GraphConvolution", "GCN"])
    spec = importlib.util.spec_from_file_location("uq_sfgc_native_reparam", repository / "models/reparam_module.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes, real_y, syn_y = remap_labels(delta["y"], labels, device)
    real_x = delta["x"].float().to(device)
    adj = M.normalized_adjacency(delta["edge_index"], len(real_y), device)
    trajectories, expert_audit = [], []
    model_args = dict(nfeat=real_x.shape[1], nhid=cfg.hidden, nclass=len(classes),
                      dropout=0., nlayers=2, with_bn=False, device=device)
    for expert in range(cfg.experts):
        model = definitions["GCN"](**model_args).to(device)
        model.initialize()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.teacher_lr, weight_decay=5e-4)
        snapshots = [flat_parameters(model).cpu()]
        for epoch in range(1, cfg.teacher_epochs+1):
            _, output = model(real_x, adj)
            loss = F.nll_loss(output, real_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if epoch % cfg.snapshot_interval == 0:
                snapshots.append(flat_parameters(model).cpu())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("SFGC expert diverged")
        trajectories.append(snapshots)
        expert_audit.append(dict(expert=expert, training_loss=float(loss.detach()),
            training_accuracy=float((output.argmax(1)==real_y).float().mean()), snapshots=len(snapshots)))
        print(f"sfgc expert={expert} train_loss={float(loss.detach()):.6f}", flush=True)
        del model, optimizer
    initial, source_ids = initial_features(delta, labels, seed, device, kcenter=True)
    synth = nn.Parameter(initial.clone())
    synthetic_lr = torch.tensor(cfg.synthetic_lr, device=device, requires_grad=True)
    optimizer_feature = torch.optim.Adam([synth], lr=cfg.feature_lr)
    optimizer_lr = torch.optim.SGD([synthetic_lr], lr=cfg.synthetic_lr_lr, momentum=.5)
    syn_adj = identity(len(labels), device)
    rng = np.random.default_rng(seed + 1009 * int(delta["stage"]))
    trace = []
    for step in range(cfg.outer_steps):
        reparam = module.ReparamModule(definitions["GCN"](**model_args).to(device))
        expert = int(rng.integers(len(trajectories)))
        max_start = min(cfg.start_epoch, cfg.teacher_epochs-cfg.expert_horizon) // cfg.snapshot_interval
        start_index = int(rng.integers(max_start+1))
        target_index = start_index + cfg.expert_horizon // cfg.snapshot_interval
        starting = trajectories[expert][start_index].to(device)
        target = trajectories[expert][target_index].to(device)
        current = starting.detach().clone().requires_grad_(True)
        distance = (starting-target).square().sum()
        if float(distance) <= 1e-12:
            raise FloatingPointError("SFGC expert segment has zero parameter displacement")
        for _ in range(cfg.syn_steps):
            _, output = reparam(synth, syn_adj, flat_param=current)
            loss_syn = F.nll_loss(output, syn_y)
            gradient = torch.autograd.grad(loss_syn, current, create_graph=True)[0]
            current = current - synthetic_lr * gradient
        loss = (current-target).square().sum() / distance
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("SFGC native trajectory matching diverged")
        optimizer_feature.zero_grad(set_to_none=True)
        optimizer_lr.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(synth.grad.norm())
        optimizer_feature.step()
        optimizer_lr.step()
        if step == 0 or (step+1) % 20 == 0 or step+1 == cfg.outer_steps:
            trace.append(dict(step=step+1, normalized_parameter_distance=float(loss.detach()),
                feature_gradient_norm=norm, synthetic_lr=float(synthetic_lr.detach()),
                expert=expert, starting_epoch=start_index*cfg.snapshot_interval))
            print(f"sfgc construction step={step+1} objective={float(loss.detach()):.6f}", flush=True)
        del reparam, current, loss, gradient
    return synth.detach(), dict(initial_source_ids=source_ids,
        initialization="classwise K-center on raw graph features; no text used in initialization",
        objective="official normalized multi-step expert/student parameter-distance objective",
        proxy="unaltered official 2-layer GCN and ReparamModule",
        expert_audit=expert_audit, config=asdict(cfg), trace=trace,
        feature_update_norm=float((synth.detach()-initial).norm()),
        budget_limitation="5 experts and 200 outer steps, vs example 200 experts and 2000 outer steps",
        ntk_checkpoint_selection="disabled; fixed final iteration; no report/test accesses")


def build(arm, delta, parent, labels, encoder, seed, device, root, cfg=None):
    config = Config() if cfg is None else cfg
    if arm == "gcond_x_core":
        features, audit = gcond_x(delta, labels, seed, device, root, config)
    elif arm == "sfgc_core":
        features, audit = sfgc(delta, labels, seed, device, root, config)
    else:
        raise ValueError(arm)
    propagated = M.propagated_features(delta, (0., 1.), device)
    block, text_audit = M.attach_text(delta, propagated, features, labels, encoder,
        M.Config(seed=seed, text_sources=5, text_mode="nearest"))
    block["labels"] = labels.detach().cpu().clone()
    audit.update(method=arm, native_registration=registration(root), text_attachment=text_audit,
        topology="identity synthetic adjacency; implicit self loops at Student training",
        historical_raw_reads=0, prefix_updates=0, official_reproduction=False,
        identity="pinned native-core current-delta TAG adaptation")
    return block, audit
