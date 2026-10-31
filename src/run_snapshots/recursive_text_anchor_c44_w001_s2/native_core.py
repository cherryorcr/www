"""Label-free native TAGSAM model, training and differentiable ASM.

This is a deliberately small PyTorch implementation of the released TAGSAM
GCN/TextProjection model. BERT is frozen upstream: ``text_emb`` contains its
cached CLS embeddings, while the readable strings remain in the artifact.
Neither training nor condensation consumes node class labels. Graphs are
encoded in full; contrastive minibatches select roots after message passing.
This is different from the original repository's sampled-neighbor training.

``steps`` and ``inner_steps`` count optimizer/minibatch updates, not epochs.
The last minibatch is retained; a singleton tail joins the preceding batch.
Only synthetic X and optional inner learning rates are optimized by ASM.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call


DATA_KEYS = ("x", "edge_index", "text_emb", "texts", "weights", "origin_ids", "stage")
CONFIG_METADATA_KEYS = frozenset((
    "outer_steps", "inner_steps", "batch_size", "eval_every", "reference_size",
    "hidden_dim", "out_dim", "inner_lr_mode", "inner_lr_graph", "inner_lr_text",
    "inner_lr_bounds", "outer_lr", "inner_lr_outer_lr", "weighted",
    "max_grad_norm", "feature_norm_cap", "reference_policy", "reference_mass_weighted",
    "text_anchor_weight", "text_anchor_scope",
))


def normalized_adjacency(edge_index, num_nodes, *, dtype=torch.float32,
                         device=None, sparse=True):
    """Return PyG GCNConv's source-to-target D^-1/2 (A+I) D^-1/2.

    Existing self loops are replaced by one unit loop per node, matching
    add_remaining_self_loops for unweighted edges. Repeated non-loop edges
    retain their multiplicity. Matrix indices are (target, source).
    """
    if num_nodes < 1:
        raise ValueError("a graph must contain at least one node")
    edge = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    if edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("edge_index must have shape (2,E)")
    if edge.numel() and (int(edge.min()) < 0 or int(edge.max()) >= num_nodes):
        raise ValueError("edge_index has an out-of-range endpoint")
    edge = edge[:, edge[0] != edge[1]]
    loops = torch.arange(num_nodes, device=edge.device)
    source = torch.cat((edge[0], loops))
    target = torch.cat((edge[1], loops))
    degree = torch.zeros(num_nodes, dtype=dtype, device=edge.device)
    degree.index_add_(0, target, torch.ones_like(target, dtype=dtype))
    inv = degree.pow(-0.5)
    values = inv[source] * inv[target]
    adj = torch.sparse_coo_tensor(torch.stack((target, source)), values,
                                  (num_nodes, num_nodes), device=edge.device,
                                  dtype=dtype).coalesce()
    return adj if sparse else adj.to_dense()


def _mm(adj, x):
    return torch.sparse.mm(adj, x) if adj.layout != torch.strided else adj @ x


def graph_normalize(x, edge_index):
    """Aggregate X using the normalized adjacency used by both GCN layers."""
    return _mm(normalized_adjacency(edge_index, len(x), dtype=x.dtype, device=x.device), x)


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.lin = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, adj):
        # GCNConv adds bias after aggregation, not before degree normalization.
        return _mm(adj, self.lin(x)) + self.bias


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNLayer(input_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, out_dim)

    def forward(self, x, adj):
        return self.conv2(F.leaky_relu(self.conv1(x, adj)), adj)


class TextProjection(nn.Module):
    def __init__(self, text_dim, out_dim):
        super().__init__()
        self.projection = nn.Linear(text_dim, out_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, text_emb):
        projected = self.projection(text_emb)
        return self.layer_norm(projected + self.dropout(self.fc(self.gelu(projected))))


class NativeTAGModel(nn.Module):
    def __init__(self, input_dim, text_dim=768, hidden_dim=384, out_dim=384):
        super().__init__()
        self.model_config = dict(input_dim=int(input_dim), text_dim=int(text_dim),
                                 hidden_dim=int(hidden_dim), out_dim=int(out_dim))
        self.graph_encoder = GraphEncoder(input_dim, hidden_dim, out_dim)
        self.text_encoder = TextProjection(text_dim, out_dim)

    @staticmethod
    def _normalize(x):
        return x / (x.norm(dim=1, keepdim=True) + 1e-10)

    def encode_graph(self, data):
        x = data["x"]
        adj = data.get("_adj")
        if adj is None:
            adj = normalized_adjacency(data["edge_index"], len(x), dtype=x.dtype, device=x.device)
        return self._normalize(self.graph_encoder(x, adj))

    def encode_text(self, text_emb):
        return self._normalize(self.text_encoder(text_emb))

    def forward(self, data, rows=None, class_emb=None):
        graph = self.encode_graph(data)
        if rows is not None:
            graph = graph[rows]
        text = class_emb if class_emb is not None else data["text_emb"]
        if class_emb is None and rows is not None:
            text = text[rows]
        return (graph @ self.encode_text(text).T) / 0.07


def create_model(input_dim, text_dim=768, hidden_dim=384, out_dim=384):
    return NativeTAGModel(input_dim, text_dim, hidden_dim, out_dim)


def clean_data(data, device="cpu", *, clone=True, with_adjacency=False):
    """Copy only the declared data fields; never carry labels/source objects."""
    if not isinstance(data, Mapping):
        raise TypeError("data must be a mapping")
    result = {}
    for key in ("x", "text_emb"):
        value = torch.as_tensor(data[key], device=device).detach()
        if not value.is_floating_point():
            value = value.float()
        result[key] = value.clone() if clone else value
    n = len(result["x"])
    if n < 1 or result["x"].ndim != 2 or result["text_emb"].ndim != 2:
        raise ValueError("x and text_emb must be nonempty matrices")
    if len(result["text_emb"]) != n:
        raise ValueError("text_emb and x row counts differ")
    if not all(bool(torch.isfinite(result[k]).all()) for k in ("x", "text_emb")):
        raise ValueError("input features must be finite")
    result["text_emb"] = result["text_emb"].to(dtype=result["x"].dtype)
    edge = torch.as_tensor(data["edge_index"], dtype=torch.long, device=device).detach()
    result["edge_index"] = edge.clone() if clone else edge
    texts = list(data["texts"])
    if len(texts) != n or any(not isinstance(t, str) for t in texts):
        raise ValueError("texts must contain one readable string per node")
    result["texts"] = texts
    for key, default in (("origin_ids", torch.arange(n)), ("stage", torch.zeros(n))):
        value = torch.as_tensor(data.get(key, default), dtype=torch.long, device=device).detach()
        if value.shape != (n,):
            raise ValueError(f"{key} must have shape (N,)")
        result[key] = value.clone() if clone else value
    weights = torch.as_tensor(data.get("weights", torch.ones(n)),
                              dtype=result["x"].dtype, device=device).detach()
    if weights.shape != (n,) or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("weights must be finite positive numbers with shape (N,)")
    result["weights"] = weights.clone() if clone else weights
    adj = normalized_adjacency(result["edge_index"], n, dtype=result["x"].dtype, device=device)
    if with_adjacency:
        result["_adj"] = adj
    return result


def epoch_batches(num_rows, batch_size, generator):
    """One shuffled coverage pass, including a non-degenerate singleton tail."""
    if num_rows < 2 or batch_size < 2:
        raise ValueError("contrastive training needs at least two rows and batch_size>=2")
    order = torch.randperm(num_rows, generator=generator)
    batches = list(order.split(batch_size))
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = torch.cat((batches[-2], batches[-1]))
        batches.pop()
    return batches


def _batch_stream(num_rows, batch_size, generator):
    while True:
        yield from epoch_batches(num_rows, batch_size, generator)


def symmetric_infonce(logits, weights=None):
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or len(logits) < 2:
        raise ValueError("InfoNCE requires a square matrix with at least two pairs")
    target = torch.arange(len(logits), device=logits.device)
    rows = F.cross_entropy(logits, target, reduction="none")
    cols = F.cross_entropy(logits.T, target, reduction="none")
    losses = (rows + cols) / 2
    if weights is None:
        return losses.mean()
    # Query weighting only; this does not duplicate nodes in the negative pool.
    weights = weights.to(device=losses.device, dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum()


def asm_loss(student_logits, teacher_logits, query_weights=None):
    """Released row/column softmax BCE; optional query-mass weighting.

    Weighting is applied after each query's BCE is averaged over its possible
    partners. It does not change either softmax denominator, temperature, or
    the selected negative/reference pool. Uniform weights recover TAGSAM ASM.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher/student ASM reference matrices must have the same shape")
    teacher_logits = teacher_logits.detach()
    if query_weights is None:
        row = F.binary_cross_entropy(student_logits.softmax(1), teacher_logits.softmax(1))
        col = F.binary_cross_entropy(student_logits.softmax(0), teacher_logits.softmax(0))
        return (row + col) / 2
    if student_logits.ndim != 2 or student_logits.shape[0] != student_logits.shape[1]:
        raise ValueError("query-mass ASM requires a square paired reference matrix")
    weights = torch.as_tensor(query_weights, device=student_logits.device,
                              dtype=student_logits.dtype).detach()
    if weights.shape != (len(student_logits),) or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("query weights must be finite, positive and match reference rows")
    weights = weights / weights.sum()
    row_queries = F.binary_cross_entropy(student_logits.softmax(1), teacher_logits.softmax(1), reduction="none").mean(1)
    col_queries = F.binary_cross_entropy(student_logits.softmax(0), teacher_logits.softmax(0), reduction="none").mean(0)
    row, col = (row_queries * weights).sum(), (col_queries * weights).sum()
    return (row + col) / 2


def reference_summary(source, rows, *, policy, requested_count, mass_weighted=False):
    """Summarize observed representative stages, not unavailable raw lineage."""
    stage = source["stage"].detach().cpu().long()
    mass = source["weights"].detach().cpu().double()
    rows = torch.as_tensor(rows, dtype=torch.long).detach().cpu()
    if rows.ndim != 1 or len(rows) < 2 or len(rows.unique()) != len(rows):
        raise ValueError("reference rows must be unique and contain at least two nodes")
    if int(rows.min()) < 0 or int(rows.max()) >= len(stage):
        raise ValueError("reference row outside source")
    total_mass, selected_mass = float(mass.sum()), float(mass[rows].sum())
    summaries = []
    for value in stage.unique(sorted=True).tolist():
        members = stage == value
        selected = rows[stage[rows] == value]
        group_mass, chosen_mass = float(mass[members].sum()), float(mass[selected].sum())
        summaries.append({"stage": int(value), "source_rows": int(members.sum()),
                          "source_mass": group_mass, "source_mass_fraction": group_mass / total_mass,
                          "selected_rows": len(selected), "selected_mass": chosen_mass,
                          "selected_mass_fraction": chosen_mass / selected_mass,
                          "query_loss_fraction": chosen_mass / selected_mass if mass_weighted else len(selected) / len(rows)})
    return {"policy": policy, "requested_count": int(requested_count), "selected_count": len(rows),
            "source_rows": len(stage), "source_mass": total_mass, "selected_mass": selected_mass,
            "unique_rows": True, "all_source_stages_covered": all(row["selected_rows"] > 0 for row in summaries),
            "reference_mass_weighted": bool(mass_weighted), "stage_summary": summaries,
            "stage_meaning": "current source representative stage; not reconstructed raw cluster lineage",
            "within_stage_sampling": "uniform_without_replacement" if policy == "mass_stratified" else None,
            "softmax_denominator": "unchanged selected reference rows"}


def _mass_quotas(capacities, masses, count):
    """Capped proportional allocation, one per group, Hamilton rounding."""
    if count < len(capacities):
        raise ValueError("reference_size is too small to cover every retained source stage")
    if count == sum(capacities):
        return list(capacities)
    if count == len(capacities):
        return [1] * len(capacities)

    def continuous(scale):
        return [min(capacity, max(1., scale * mass)) for capacity, mass in zip(capacities, masses)]

    lo, hi = 0., 1.
    while sum(continuous(hi)) < count:
        hi *= 2
    for _ in range(80):
        middle = (lo + hi) / 2
        if sum(continuous(middle)) < count:
            lo = middle
        else:
            hi = middle
    desired = continuous((lo + hi) / 2)
    quotas = [min(cap, max(1, math.floor(value))) for cap, value in zip(capacities, desired)]
    remaining = count - sum(quotas)
    # Each remainder needs at most one seat; deterministic stage-order tie break.
    ranked = sorted(range(len(quotas)), key=lambda i: (-(desired[i] - quotas[i]), i))
    for index in ranked:
        if remaining and quotas[index] < capacities[index]:
            quotas[index] += 1
            remaining -= 1
    if remaining != 0 or sum(quotas) != count:
        raise RuntimeError("failed to allocate exact unique reference quota")
    return quotas


def select_reference_rows(source, reference_size, generator, policy="uniform", *, mass_weighted=False):
    """Choose unique label-free reference rows and return a stage/mass audit.

    mass_stratified allocates counts by each observed stage's total retained
    mass, capped by its available rows, and samples uniformly within stages.
    The reference budget must fit one row from every retained stage. Sampling
    mass and weighting the ASM query loss are independent experimental knobs.
    """
    if policy not in ("uniform", "mass_stratified"):
        raise ValueError("reference_policy must be uniform or mass_stratified")
    stage = source["stage"].detach().cpu().long()
    mass = source["weights"].detach().cpu().double()
    count = min(int(reference_size), len(stage))
    if count < 2:
        raise ValueError("reference_size must select at least two rows")
    if policy == "uniform":
        rows = torch.randperm(len(stage), generator=generator)[:count]
    else:
        groups = [torch.where(stage == value)[0] for value in stage.unique(sorted=True)]
        quotas = _mass_quotas([len(group) for group in groups],
                              [float(mass[group].sum()) for group in groups], count)
        rows = torch.cat([group[torch.randperm(len(group), generator=generator)[:quota]]
                          for group, quota in zip(groups, quotas)])
        rows = rows[torch.randperm(len(rows), generator=generator)]
    return rows, reference_summary(source, rows, policy=policy, requested_count=reference_size,
                                    mass_weighted=mass_weighted)


def _seed(seed, device):
    torch.manual_seed(int(seed))
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def train_model(data, seed, steps, lr_graph, lr_text, batch_size, device,
                hidden_dim=384, out_dim=384, callback=None, weighted=False):
    """Train a new native student; callback(model, log_row) may add log fields."""
    if int(steps) != steps or steps < 0 or min(lr_graph, lr_text) <= 0:
        raise ValueError("steps must be nonnegative and learning rates positive")
    train = clean_data(data, device, with_adjacency=True)
    _seed(seed, device)
    model = create_model(train["x"].shape[1], train["text_emb"].shape[1],
                         hidden_dim, out_dim).to(device=device, dtype=train["x"].dtype)
    optimizer = torch.optim.Adam([
        {"params": model.graph_encoder.parameters(), "lr": lr_graph},
        {"params": model.text_encoder.parameters(), "lr": lr_text},
    ])
    generator = torch.Generator().manual_seed(int(seed) + 1009)
    batches = _batch_stream(len(train["x"]), int(batch_size), generator)
    seen = torch.zeros(len(train["x"]), dtype=torch.long)
    logs = []
    if callback is not None:
        # Record the randomly initialized model before any optimizer update.
        # This makes the training curve an explicit step-0 baseline and also
        # keeps zero-step diagnostic runs well-defined.
        initial_row = {"step": 0, "loss": None, "batch_rows": [], "batch_size": 0,
                       "examples_seen": 0, "unique_rows_seen": 0,
                       "weighted": bool(weighted)}
        extra = callback(model, initial_row)
        if extra is not None:
            if not isinstance(extra, Mapping):
                raise TypeError("training callback must return a mapping or None")
            initial_row.update(extra)
        logs.append(initial_row)
    for step in range(1, int(steps) + 1):
        rows_cpu = next(batches)
        rows = rows_cpu.to(device)
        model.train()
        logits = model(train, rows)
        loss = symmetric_infonce(logits, train["weights"][rows] if weighted else None)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"nonfinite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        seen[rows_cpu] += 1
        row = {"step": step, "loss": float(loss.detach()), "batch_rows": rows_cpu.tolist(),
               "batch_size": len(rows_cpu), "examples_seen": int(seen.sum()),
               "unique_rows_seen": int((seen > 0).sum()), "weighted": bool(weighted)}
        if callback is not None:
            extra = callback(model, row)
            if extra is not None:
                if not isinstance(extra, Mapping):
                    raise TypeError("training callback must return a mapping or None")
                row.update(extra)
        logs.append(row)
    model.eval()
    return model, logs


@torch.no_grad()
def evaluate_logits(model, data, class_emb):
    """Class-independent graph/text encoding; no class labels are consumed."""
    parameter = next(model.parameters())
    evaluation = clean_data(data, parameter.device, with_adjacency=True)
    evaluation["x"] = evaluation["x"].to(parameter.dtype)
    evaluation["text_emb"] = evaluation["text_emb"].to(parameter.dtype)
    evaluation["_adj"] = evaluation["_adj"].to(dtype=parameter.dtype)
    classes = torch.as_tensor(class_emb, device=parameter.device, dtype=parameter.dtype)
    was_training = model.training
    model.eval()
    try:
        return model(evaluation, class_emb=classes)
    finally:
        model.train(was_training)


def _fingerprint(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        payload = str((tuple(value.shape), str(value.dtype))).encode() + value.numpy().tobytes()
    else:
        payload = repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _norms(x):
    norms = x.detach().double().norm(dim=1)
    return {"x_norm_mean": float(norms.mean()), "x_norm_max": float(norms.max()),
            "x_l2": float(x.detach().double().norm())}


def _safe_config(config):
    def safe(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, Mapping):
            return {str(k): safe(v) for k, v in value.items()}
        raise TypeError
    result = {}
    for key, value in config.items():
        if key not in CONFIG_METADATA_KEYS:
            continue
        try:
            result[str(key)] = safe(value)
        except TypeError:
            # No callables, source tensors, teachers, or arbitrary objects in artifacts.
            continue
    return result


def condense_asm(source, initial, teacher, config, seed, device, validator=None):
    """Optimize fixed-text/fixed-edge synthetic X through native student SGD.

    ``text_anchor_weight > 0`` adds symmetric InfoNCE from the frozen teacher
    on the synthetic graph. This keeps optimized graph features aligned with
    their fixed text embeddings. ``text_anchor_scope`` selects either every
    synthetic pair (``all``) or only rows enabled by ``optimize_rows``
    (``optimized``); message passing still uses the complete synthetic graph.

    validator(artifact, outer_step) returns a float or {"score": float,...};
    higher is better. It must do its own independent validation-only training.
    The initial artifact is a valid selection candidate. Without a validator,
    return the last finite iterate, not an adaptively selected ASM minimum.
    An optional config['artifact_callback'](artifact, step, kind) lets the
    caller save snapshots; this function performs no filesystem operations.
    """
    config = dict(config)
    outer_steps = int(config.get("outer_steps", 100))
    inner_steps = int(config.get("inner_steps", 1))
    batch_size = int(config.get("batch_size", 20))
    eval_every = int(config.get("eval_every", max(1, outer_steps)))
    text_anchor_weight = float(config.get("text_anchor_weight", 0.0))
    text_anchor_scope = str(config.get("text_anchor_scope", "all"))
    if min(outer_steps, inner_steps) < 0 or inner_steps < 1 or eval_every < 1:
        raise ValueError("outer_steps>=0, inner_steps>=1 and eval_every>=1 are required")
    if not math.isfinite(text_anchor_weight) or text_anchor_weight < 0:
        raise ValueError("text_anchor_weight must be finite and non-negative")
    if text_anchor_scope not in ("all", "optimized"):
        raise ValueError("text_anchor_scope must be all or optimized")
    src = clean_data(source, device, with_adjacency=True)
    syn = clean_data(initial, device, with_adjacency=True)
    if len(src["x"]) < 2 or len(syn["x"]) < 2 or batch_size < 2:
        raise ValueError("ASM needs at least two source/synthetic rows and batch_size>=2")
    if src["x"].shape[1] != syn["x"].shape[1] or src["text_emb"].shape[1] != syn["text_emb"].shape[1]:
        raise ValueError("source and synthetic feature spaces must agree")
    teacher_parameter = next(teacher.parameters())
    if teacher_parameter.device != src["x"].device or teacher_parameter.dtype != src["x"].dtype:
        raise ValueError("teacher must already be on source device and dtype")
    if not hasattr(teacher, "model_config"):
        raise TypeError("teacher must expose native model_config")
    architecture = dict(teacher.model_config)
    if architecture["input_dim"] != src["x"].shape[1] or architecture["text_dim"] != src["text_emb"].shape[1]:
        raise ValueError("teacher feature dimensions disagree with source")
    for key in ("hidden_dim", "out_dim"):
        if key in config and int(config[key]) != architecture[key]:
            raise ValueError(f"config {key} must agree with teacher architecture")

    generator = torch.Generator().manual_seed(int(seed) + 3011)
    reference_policy = config.get("reference_policy", "uniform")
    reference_mass_weighted = bool(config.get("reference_mass_weighted", False))
    if "reference_rows" in config:
        if reference_policy != "uniform":
            raise ValueError("explicit reference_rows cannot be combined with a sampling policy")
        reference = torch.as_tensor(config["reference_rows"], dtype=torch.long)
        reference_audit = reference_summary(src, reference, policy="explicit", requested_count=len(reference),
                                             mass_weighted=reference_mass_weighted)
    else:
        reference, reference_audit = select_reference_rows(src, int(config.get("reference_size", 256)), generator,
                                                           reference_policy, mass_weighted=reference_mass_weighted)
    if reference.ndim != 1 or len(reference) < 2 or len(reference.unique()) != len(reference):
        raise ValueError("reference rows must be unique and contain at least two nodes")
    if int(reference.min()) < 0 or int(reference.max()) >= len(src["x"]):
        raise ValueError("reference row outside source")
    reference = reference.to(device)
    reference_weights = src["weights"][reference].detach() if reference_mass_weighted else None
    teacher_was_training = teacher.training
    teacher.eval()
    try:
        with torch.no_grad():
            teacher_logits = teacher(src, reference).detach().clone()
    finally:
        teacher.train(teacher_was_training)
    if not bool(torch.isfinite(teacher_logits).all()):
        raise FloatingPointError("nonfinite teacher reference logits")

    fixed_hashes = {key: _fingerprint(syn[key]) for key in DATA_KEYS if key != "x"}
    reference_hash = _fingerprint(teacher_logits)
    variable = nn.Parameter(syn["x"].clone())
    optimize_rows = torch.as_tensor(config.get("optimize_rows", torch.ones(len(variable), dtype=torch.bool)),
                                    device=device)
    if optimize_rows.dtype != torch.bool or optimize_rows.shape != (len(variable),) or not bool(optimize_rows.any()):
        raise ValueError("optimize_rows must be a boolean row mask with at least one active row")
    frozen_x = variable.detach()[~optimize_rows].clone()
    anchor_rows = (torch.arange(len(variable), device=device) if text_anchor_scope == "all"
                   else optimize_rows.nonzero(as_tuple=False).flatten())
    if text_anchor_weight > 0 and len(anchor_rows) < 2:
        raise ValueError("text anchor scope must contain at least two synthetic pairs")
    # Detached functional parameters let the teacher propagate gradients only
    # to synthetic X. The zero-weight path deliberately skips this setup and
    # all anchor forward calls so existing ASM numerics remain unchanged.
    frozen_teacher_params = ({name: value.detach() for name, value in teacher.named_parameters()}
                             if text_anchor_weight > 0 else None)
    mode = config.get("inner_lr_mode", "fixed")
    if mode not in ("fixed", "bounded", "unbounded"):
        raise ValueError("inner_lr_mode must be fixed, bounded or unbounded")
    graph_lr = float(config.get("inner_lr_graph", 5e-3))
    text_lr = float(config.get("inner_lr_text", 5e-3))
    if min(graph_lr, text_lr) <= 0:
        raise ValueError("initial inner learning rates must be positive")
    lower, upper = config.get("inner_lr_bounds", (1e-6, 5e-2))
    if not (0 < lower < upper):
        raise ValueError("invalid inner learning-rate bounds")
    raw_lrs = None
    if mode != "fixed":
        values = torch.tensor([graph_lr, text_lr], dtype=variable.dtype, device=device)
        if mode == "bounded":
            if not bool(((values > lower) & (values < upper)).all()):
                raise ValueError("bounded learning rates must initialize strictly inside bounds")
            values = torch.logit((values - lower) / (upper - lower))
        raw_lrs = nn.Parameter(values)

    def current_lrs():
        if raw_lrs is None:
            return graph_lr, text_lr
        rates = lower + (upper - lower) * raw_lrs.sigmoid() if mode == "bounded" else raw_lrs
        return rates[0], rates[1]

    groups = [{"params": [variable], "lr": float(config.get("outer_lr", 1e-2))}]
    if raw_lrs is not None:
        groups.append({"params": [raw_lrs], "lr": float(config.get("inner_lr_outer_lr", 2e-5))})
    if any(group["lr"] <= 0 for group in groups):
        raise ValueError("outer learning rates must be positive")
    optimizer = torch.optim.Adam(groups)
    logs = []
    algorithm = ("native-tagsam-text-anchored-asm-x-only-v1" if text_anchor_weight > 0
                 else "native-tagsam-asm-x-only-v1")
    metadata = {"algorithm": algorithm, "seed": int(seed),
                 "model_config": architecture, "config": _safe_config(config),
                "optimized_fields": ["x"], "inner_lr_mode": mode,
                "reference_count": len(reference), "reference_logits_sha256": reference_hash,
                "reference_audit": reference_audit,
                "optimized_x_rows": optimize_rows.detach().cpu().tolist(),
                "frozen_x_sha256": _fingerprint(frozen_x),
                "fixed_field_sha256": fixed_hashes,
                "source_payload_embedded": False, "training_labels_used": False,
                 "text_encoder_input": "cached_frozen_BERT_CLS",
                 "graph_training": "full-graph messages; minibatch contrastive roots"}
    if text_anchor_weight > 0:
        metadata["text_anchor"] = {
            "weight": text_anchor_weight,
            "scope": text_anchor_scope,
            "pair_count": int(len(anchor_rows)),
            "teacher_frozen": True,
            "loss": "symmetric_infonce",
        }

    def artifact(step):
        payload = clean_data({**syn, "x": variable.detach()}, "cpu")
        current = current_lrs()
        rates = tuple(float(rate.detach()) if isinstance(rate, torch.Tensor) else float(rate)
                      for rate in current)
        payload["metadata"] = {**metadata, "outer_step": int(step), **_norms(variable),
                               "inner_lr_graph": rates[0], "inner_lr_text": rates[1]}
        return payload

    snapshot_callback = config.get("artifact_callback")
    best_score, best_artifact, best_step = -math.inf, None, None

    def validate(step, candidate):
        nonlocal best_score, best_artifact, best_step
        if validator is None:
            return
        result = validator(candidate, step)
        score = result["score"] if isinstance(result, Mapping) else result
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("validator score must be finite")
        logs.append({"event": "validation", "outer_step": step, "score": score})
        if score > best_score:
            best_score, best_artifact, best_step = score, candidate, step
            if snapshot_callback is not None:
                snapshot_callback(candidate, step, "best")

    initial_artifact = artifact(0)
    logs.append({"event": "initial", "outer_step": 0, **_norms(variable),
                 "reference_logits_sha256": reference_hash, "reference_audit": reference_audit})
    validate(0, initial_artifact)
    if snapshot_callback is not None:
        snapshot_callback(initial_artifact, 0, "initial")
    last_step = 0
    status = "complete"
    for outer in range(1, outer_steps + 1):
        _seed(int(seed) * 100003 + outer, device)
        student = create_model(**architecture).to(device=device, dtype=variable.dtype)
        params = dict(student.named_parameters())
        inner_generator = torch.Generator().manual_seed(int(seed) * 107 + outer)
        batches = _batch_stream(len(syn["x"]), batch_size, inner_generator)
        active_syn = {**syn, "x": variable}
        inner_graph_lr, inner_text_lr = current_lrs()
        inner_losses = []
        inner_seen = set()
        student.train()
        for inner in range(inner_steps):
            rows = next(batches).to(device)
            inner_seen.update(rows.detach().cpu().tolist())
            logits = functional_call(student, params, (active_syn, rows))
            loss = symmetric_infonce(logits, syn["weights"][rows] if config.get("weighted", False) else None)
            grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=True)
            params = {name: value - (inner_graph_lr if name.startswith("graph_encoder.") else inner_text_lr) * grad
                      for (name, value), grad in zip(params.items(), grads)}
            inner_losses.append(float(loss.detach()))
        student.eval()
        student_logits = functional_call(student, params, (src, reference))
        if not bool(torch.isfinite(student_logits).all()):
            logs.append({"event": "nonfinite", "outer_step": outer, "location": "student_reference"})
            status = "stopped_nonfinite"
            break
        asm_objective = asm_loss(student_logits, teacher_logits, reference_weights)
        anchor_objective = None
        if text_anchor_weight > 0:
            teacher_was_training = teacher.training
            teacher.eval()
            try:
                anchor_logits = functional_call(teacher, frozen_teacher_params,
                                                (active_syn, anchor_rows))
            finally:
                teacher.train(teacher_was_training)
            if not bool(torch.isfinite(anchor_logits).all()):
                logs.append({"event": "nonfinite", "outer_step": outer,
                             "location": "text_anchor_logits"})
                status = "stopped_nonfinite"
                break
            anchor_weights = (syn["weights"][anchor_rows]
                              if config.get("weighted", False) else None)
            anchor_objective = symmetric_infonce(anchor_logits, anchor_weights)
            if not bool(torch.isfinite(anchor_objective)):
                logs.append({"event": "nonfinite", "outer_step": outer,
                             "location": "text_anchor_loss"})
                status = "stopped_nonfinite"
                break
        objective = (asm_objective if anchor_objective is None else
                     asm_objective + text_anchor_weight * anchor_objective)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        if variable.grad is not None:
            variable.grad[~optimize_rows] = 0
        grad_norm = float(variable.grad.detach().double().norm()) if variable.grad is not None else 0.0
        gradient_finite = variable.grad is not None and bool(torch.isfinite(variable.grad).all())
        if raw_lrs is not None:
            gradient_finite = gradient_finite and raw_lrs.grad is not None and bool(torch.isfinite(raw_lrs.grad).all())
        if not bool(torch.isfinite(objective)) or not gradient_finite:
            logs.append({"event": "nonfinite", "outer_step": outer, "location": "outer_gradient"})
            status = "stopped_nonfinite"
            break
        max_grad = config.get("max_grad_norm", 1.0)
        if max_grad is not None:
            if float(max_grad) <= 0:
                raise ValueError("max_grad_norm must be positive or None")
            torch.nn.utils.clip_grad_norm_([variable], float(max_grad))
        previous_x = variable.detach().clone()
        previous_lrs = raw_lrs.detach().clone() if raw_lrs is not None else None
        optimizer.step()
        norm_cap = config.get("feature_norm_cap")
        if norm_cap is not None:
            if float(norm_cap) <= 0:
                raise ValueError("feature_norm_cap must be positive or None")
            with torch.no_grad():
                variable.mul_((float(norm_cap) / variable.norm(dim=1, keepdim=True).clamp_min(1e-12)).clamp(max=1))
        # Row protection also applies to norm projection and optimizer state.
        with torch.no_grad():
            variable[~optimize_rows] = frozen_x
        if not bool(torch.isfinite(variable).all()) or (raw_lrs is not None and not bool(torch.isfinite(raw_lrs).all())):
            with torch.no_grad():
                variable.copy_(previous_x)
                if raw_lrs is not None:
                    raw_lrs.copy_(previous_lrs)
            logs.append({"event": "nonfinite", "outer_step": outer, "location": "optimizer_step"})
            status = "stopped_nonfinite"
            break
        last_step = outer
        rates = current_lrs()
        rates_log = tuple(float(rate.detach()) if isinstance(rate, torch.Tensor) else float(rate)
                          for rate in rates)
        row = {"event": "outer", "outer_step": outer,
               "asm_loss": float(asm_objective.detach()),
               "text_anchor_loss": (None if anchor_objective is None
                                    else float(anchor_objective.detach())),
               "total_loss": float(objective.detach()),
               "inner_loss_mean": sum(inner_losses) / len(inner_losses),
               "inner_steps": inner_steps, "inner_unique_rows_seen": len(inner_seen),
               "x_grad_l2": grad_norm, "inner_lr_graph": rates_log[0],
               "inner_lr_text": rates_log[1], **_norms(variable)}
        logs.append(row)
        if outer % eval_every == 0 or outer == outer_steps:
            candidate = artifact(outer)
            validate(outer, candidate)
            if snapshot_callback is not None:
                snapshot_callback(candidate, outer, "last")

    last_artifact = artifact(last_step)
    if validator is not None and (last_step % eval_every != 0 and last_step != outer_steps):
        validate(last_step, last_artifact)
    if snapshot_callback is not None:
        snapshot_callback(last_artifact, last_step, "last")
    if any(_fingerprint(syn[key]) != digest for key, digest in fixed_hashes.items()):
        raise RuntimeError("ASM modified a fixed synthetic data field")
    if _fingerprint(teacher_logits) != reference_hash:
        raise RuntimeError("ASM modified the teacher reference target")
    if not torch.equal(variable.detach()[~optimize_rows], frozen_x):
        raise RuntimeError("ASM modified protected historical features")
    selected = best_artifact if validator is not None else last_artifact
    selected_step = best_step if validator is not None else last_step
    selected["metadata"].update({"selection": "validation_max" if validator is not None else "last_finite",
                                 "selected_outer_step": selected_step, "last_outer_step": last_step,
                                 "best_validation_score": best_score if validator is not None else None,
                                 "status": status, "last_norms": _norms(variable)})
    logs.append({"event": "selected", "outer_step": selected_step, "last_outer_step": last_step,
                 "validation_score": best_score if validator is not None else None, "status": status})
    return selected, logs
