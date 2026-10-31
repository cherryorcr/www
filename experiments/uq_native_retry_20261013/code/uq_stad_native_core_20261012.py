"""Pinned STAD core with sparse coarsening and a local, auditable generator."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import string
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

REPO = Path("/home/pc/research/STAD_official")
COMMIT = "5aaf1a06e44638499373b1358c85d1546187edae"
MODEL = Path("/home/pc/research/TAG_condensation/experiments/uq_final_evidence_20261012/models/qwen2.5-1.5b")
INTERNAL_SPLIT_POLICY = "classwise-10pct-sparse-v1"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def official_components(root=REPO):
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT:
        raise ValueError("STAD source revision differs")
    files = ("models/GNN.py", "utils_gdd.py", "deep_robust_utils.py", "tagcond.py", "prompt.py")
    hashes = {}
    for name in files:
        expected = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{name}"])
        actual = (root / name).read_bytes()
        if expected.replace(b"\r\n", b"\n") != actual.replace(b"\r\n", b"\n"):
            raise ValueError(f"modified STAD source: {name}")
        hashes[name] = dict(actual_sha256=hashlib.sha256(actual).hexdigest(),
                            canonical_sha256=hashlib.sha256(expected.replace(b"\r\n", b"\n")).hexdigest())

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    utils = load("stad_native_deep_utils", root / "deep_robust_utils.py")
    sys.modules["deep_robust_utils"] = utils
    sketch = load("stad_native_sketch", root / "utils_gdd.py")
    prompts = load("stad_native_prompts", root / "prompt.py")
    tree = ast.parse((root / "models/GNN.py").read_text(encoding="utf-8-sig"))
    definitions = [n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name in {"GraphConvolution", "GCN", "SelfTrainingGNN"}]
    for cls in definitions:
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef) or method.name != "self_train":
                continue
            for i, node in enumerate(method.body):
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                        and t.id == "final_test_acc" for t in node.targets):
                    method.body[i] = ast.Assign(targets=node.targets, value=ast.Constant(None))
            method.body = [n for n in method.body if not
                           (isinstance(n, ast.Expr) and "Final test Acc" in ast.unparse(n))]
    namespace = dict(torch=torch, nn=nn, F=F, math=math, np=np, Parameter=nn.Parameter,
                     Module=nn.Module, utils=utils, deepcopy=deepcopy, optim=torch.optim)
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 str(root / "models/GNN.py"), "exec"), namespace)
    tree = ast.parse((root / "tagcond.py").read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TAGCond")
    keywords = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {"clean", "tfidf_topk", "sbert_topk", "extract_keywords"}]
    kw_namespace = dict(re=re, string=string, np=np, torch=torch, TfidfVectorizer=TfidfVectorizer)
    exec(compile(ast.Module(body=keywords, type_ignores=[]), str(root / "tagcond.py"), "exec"), kw_namespace)
    return (SimpleNamespace(**namespace), sketch, prompts, kw_namespace,
            dict(commit=head, file_hashes=hashes, official_reproduction=False,
                 ast_change="disable unused final test scoring; never import keys or official entrypoints"))


def sparse_coarsen(adj, clusters, count, normalize):
    """Exactly P.T @ A @ P with P[i,c]=1/size(c), without an N-by-K array."""
    adj = adj.coalesce()
    clusters = torch.as_tensor(clusters, device=adj.device, dtype=torch.long)
    mass = torch.bincount(clusters, minlength=count).to(adj.dtype)
    if len(mass) != count or bool((mass == 0).any()):
        raise ValueError("STAD requires every cluster to be occupied")
    row, col = adj.indices()
    a, b = clusters[row], clusters[col]
    keep = a != b
    weights = adj.values()[keep] / (mass[a[keep]] * mass[b[keep]])
    pooled = torch.sparse_coo_tensor(torch.stack((a[keep], b[keep])), weights,
                                    (count, count), device=adj.device).coalesce()
    return normalize(pooled, sparse=True).coalesce()


def audit_sparse():
    models, sketch, _, _, provenance = official_components()
    edges = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    adj = torch.sparse_coo_tensor(edges, torch.arange(1., 9.), (6, 6)).coalesce()
    normalized = models.utils.normalize_adj_tensor(adj, sparse=True)
    clusters = np.array([0, 0, 1, 1, 2, 2])
    native = sketch.clst_condense_adj(normalized, 3, 6, clusters, "cpu")[0]
    actual = sparse_coarsen(normalized, clusters, 3, models.utils.normalize_adj_tensor)
    error = float((native.to_dense() - actual.to_dense()).abs().max())
    torch.manual_seed(81)
    values = torch.randn(3, 5)
    smooth_error = float((sketch.smooth_twice(native, values) - sketch.smooth_twice(actual, values)).abs().max())
    if error > 1e-6 or smooth_error > 1e-6:
        raise AssertionError((error, smooth_error))
    return dict(passed=True, coarsening_max_error=error, double_smoothing_max_error=smooth_error,
                source=provenance, dense_projection_avoided=True)


class TextEncoder:
    def __init__(self, encoder):
        self.encoder = encoder

    @torch.no_grad()
    def encode(self, texts, convert_to_tensor=True, device=None, normalize_embeddings=False):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        if not texts:
            return torch.empty((0, 384), device=self.encoder.device)
        parts = []
        for start in range(0, len(texts), 64):
            tokens = self.encoder.tokenizer(texts[start:start + 64], padding=True, truncation=True,
                max_length=256, return_tensors="pt").to(self.encoder.device)
            hidden = self.encoder.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            parts.append((hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9))
        result = torch.cat(parts).float()
        if normalize_embeddings:
            result = F.normalize(result, dim=1)
        return result[0] if single else result


def internal_split(y, seed):
    rng = np.random.RandomState(seed)
    train, val = torch.zeros_like(y, dtype=torch.bool), torch.zeros_like(y, dtype=torch.bool)
    for cls in torch.unique(y):
        ids = torch.where(y.cpu() == cls.cpu())[0].numpy()
        ids = rng.permutation(ids)
        n = max(1, round(.1 * len(ids)))
        train[torch.as_tensor(ids[:n], device=y.device)] = True
        val[torch.as_tensor(ids[n:2 * n], device=y.device)] = True
    unlabeled = ~(train | val)
    if bool((train & val).any()):
        raise AssertionError("invalid STAD internal masks")
    if not bool(val.any()):
        raise ValueError("STAD needs a nonempty global internal validation set")
    if not bool(unlabeled.any()):
        raise ValueError("STAD needs a nonempty global pseudo-label pool")
    return train, val, unlabeled


def candidate_validation(models, candidate, syn_adj, labels, x, adj, y, val_mask, seed):
    torch.manual_seed(seed)
    model = models.GCN(384, 64, int(y.max()) + 1, device=str(x.device)).to(x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
    best, best_epoch = -1., -1
    for epoch in range(600):
        model.train()
        loss = F.cross_entropy(model(candidate, syn_adj), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, adj).argmax(1)
            score = float((pred[val_mask] == y[val_mask]).float().mean())
        if score > best:
            best, best_epoch = score, epoch + 1
    return dict(internal_val_accuracy=best, best_epoch=best_epoch, steps=600, seed=seed)


def generation_requests(delta, clusters, labels, centers, encoder, official, class_names, dataset):
    _, _, prompts, methods, _ = official
    agent = SimpleNamespace(sbert=encoder, device=str(centers.device))
    for name in ("clean", "tfidf_topk", "sbert_topk", "extract_keywords"):
        setattr(agent, name, MethodType(methods[name], agent))
    backgrounds = dict(arxiv="A computer science citation network. Nodes are papers with titles and abstracts; edges represent citations.",
                       sportsfit="An Amazon sports and fitness product network. Nodes are products; edges represent product relationships.",
                       bookhis=prompts.SYS_BOOKHIS)
    documents = [[] for _ in labels]
    for cluster, text in zip(clusters, delta["texts"]):
        documents[int(cluster)].append(str(text))
    requests, repairs = [], 0
    for cluster in range(len(labels)):
        try:
            words, _ = agent.extract_keywords(documents[cluster], centers[cluster], kk=256)
        except ValueError as error:
            if "empty vocabulary" not in str(error):
                raise
            words, repairs = [], repairs + 1
        label = class_names[int(labels[cluster])]
        requests.append(dict(cluster=cluster, messages=[
            dict(role="system", content=prompts.system_prompt_template_text.format(dataset_name=dataset, sys_dataset=backgrounds[dataset])),
            dict(role="user", content=prompts.user_prompt_template_text.format(center_key=", ".join(words), center_label1=label, tok_lim=256))]))
    return requests, repairs


def generate(requests, cache, seed, device, batch=8, model_path=MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    provenance = json.loads((model_path / "generator_provenance.json").read_text())
    if provenance["repository"] != "Qwen/Qwen2.5-1.5B-Instruct":
        raise ValueError("unregistered STAD generator")
    settings = dict(temperature=1.2, top_p=.85, max_new_tokens=256, candidates=3,
                    seed=seed, batch=batch, model_revision=provenance["revision"],
                    schema_enforced=False)
    cache.mkdir(parents=True, exist_ok=True)
    records, missing = {}, []
    for request in requests:
        path = cache / f"node{request['cluster']:05d}.json"
        key = fingerprint(dict(request=request, settings=settings))
        if path.exists():
            record = json.loads(path.read_text())
            if record["request_sha256"] != key:
                raise ValueError("STAD cached generation has a different protocol")
            records[request["cluster"]] = record
        else:
            missing.append((request, path, key))
    if missing:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            torch_dtype=torch.float16).to(device).eval()
        for start in range(0, len(missing), batch):
            pending = missing[start:start + batch]
            prompts = [tokenizer.apply_chat_template(item[0]["messages"], tokenize=False,
                                                      add_generation_prompt=True) for item in pending]
            tokens = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to(device)
            torch.manual_seed(seed + start * 101)
            started = time.perf_counter()
            with torch.no_grad():
                output = model.generate(**tokens, do_sample=True, temperature=1.2, top_p=.85,
                    max_new_tokens=256, num_return_sequences=3, pad_token_id=tokenizer.pad_token_id)
            decoded = tokenizer.batch_decode(output[:, tokens["input_ids"].shape[1]:], skip_special_tokens=True)
            seconds = (time.perf_counter() - started) / len(pending)
            for number, (request, path, key) in enumerate(pending):
                texts = [text.strip() for text in decoded[number * 3:(number + 1) * 3]]
                if any(not text for text in texts):
                    raise RuntimeError("STAD local generator produced empty candidate")
                record = dict(request_sha256=key, request=request, texts=texts,
                              settings=settings, retry=0, seconds=seconds,
                              schema_valid=["[[LABEL]]" in text and "[[SUMMARY]]" in text
                                            for text in texts],
                              text_cap_for_common_student=320, keyword_fallback=False)
                save(path, record)
                records[request["cluster"]] = record
            print(json.dumps(dict(event="stad_generated", completed=start + len(missing[start:start + batch]),
                                  target=len(missing))), flush=True)
        del model, tokenizer
        torch.cuda.empty_cache()
    return [records[request["cluster"]] for request in requests], provenance


def construct(delta, rows, encoder, seed, device, out, dataset, class_names, official):
    models, sketch, _, _, provenance = official
    out.mkdir(parents=True, exist_ok=True)
    stage_seed = seed + 1009 * int(delta["stage"])
    random.seed(stage_seed)
    np.random.seed(stage_seed)
    torch.manual_seed(stage_seed)
    raw_encoder = TextEncoder(encoder)
    # STAD's native semantic space is MiniLM; common readout features are pooled separately.
    x = raw_encoder.encode(delta["texts"]).detach()
    classes = sorted(torch.unique(delta["y"]).tolist())
    mapping = {int(c): i for i, c in enumerate(classes)}
    y = torch.tensor([mapping[int(c)] for c in delta["y"]], device=device)
    train, val, unlabeled = internal_split(y, stage_seed)
    split_audit = dict(policy=INTERNAL_SPLIT_POLICY, seed=stage_seed,
        mask_sha256={name: hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
                     for name, mask in (("train", train), ("validation", val), ("unlabeled", unlabeled))},
        classes=[dict(label=int(c), source_rows=int((y == i).sum()),
            train_rows=int((train & (y == i)).sum()), validation_rows=int((val & (y == i)).sum()),
            unlabeled_rows=int((unlabeled & (y == i)).sum())) for i, c in enumerate(classes)],
        disjoint=True, external_nodes_used=False)
    edges = delta["edge_index"].to(device)
    raw_adj = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1], device=device),
                                     (len(x), len(x))).coalesce()
    adj = models.utils.normalize_adj_tensor(raw_adj, sparse=True).coalesce()
    core_path = out / "core.pt"
    if core_path.exists():
        core = torch.load(core_path, map_location=device, weights_only=False)
    else:
        started = time.perf_counter()
        model = models.SelfTrainingGNN(384, 64, len(classes), alpha=.5, k=10, mode="gcnmlp").to(device)
        logits, best_adj, _, val_acc, _ = model.self_train(x, adj, y.clone(), train, val,
            torch.zeros_like(train), unlabeled, num_stages=10, pretrain_stages=0, epochs_per_stage=600)
        logits = logits.detach()
        clusters = KMeans(n_clusters=rows, n_init=10, random_state=stage_seed).fit_predict(logits.cpu().numpy())
        clusters, count = sketch.regenerate_cluster_labels(clusters, logits, rows)
        if count != rows:
            raise AssertionError("STAD changed the registered memory budget")
        centers = sketch.clst_condense_feat(clusters, rows, x.cpu().numpy()).to(device)
        syn_adj = sparse_coarsen(best_adj, clusters, rows, models.utils.normalize_adj_tensor)
        labels = sketch.clst_condense_label(rows, logits, clusters, device)
        core = dict(clusters=clusters, centers=centers.cpu(), adj=syn_adj.cpu(), labels=labels.cpu(),
                    internal_val_accuracy=float(val_acc), seconds=time.perf_counter() - started)
        torch.save(core, core_path)
        del model, logits
        torch.cuda.empty_cache()
    clusters = core["clusters"]
    labels, syn_adj = core["labels"].to(device), core["adj"].to(device).coalesce()
    requests, repairs = generation_requests(delta, clusters, labels, core["centers"].to(device),
        raw_encoder, official, [class_names[c] for c in classes], dataset)
    generated, generator = generate(requests, out / "llm", stage_seed, device)
    texts = [[" ".join(text.split())[:320] for text in record["texts"]] for record in generated]
    embeddings = raw_encoder.encode([text for group in texts for text in group]).reshape(rows, 3, 384).detach()
    rng = random.Random(stage_seed)
    choices = [[0] * rows] + [[rng.randrange(3) for _ in range(rows)] for _ in range(99)]
    source = F.normalize(sketch.smooth_twice(raw_adj, x), dim=1)
    scores = []
    for number, choice in enumerate(choices):
        candidate = embeddings[torch.arange(rows, device=device), choice]
        target = F.normalize(sketch.smooth_twice(syn_adj, candidate), dim=1)
        torch.manual_seed(stage_seed * 1000 + number)
        score = sketch.subsample_sinkhorn(source, target, sub=2000, eps=.05)
        if not math.isfinite(score):
            raise FloatingPointError("nonfinite STAD Wasserstein score")
        scores.append(score)
    shortlist = sorted(range(100), key=lambda number: (scores[number], number))[:5]
    selections = []
    for number in shortlist:
        candidate = embeddings[torch.arange(rows, device=device), choices[number]]
        selections.append(dict(candidate=number, wasserstein=scores[number], **candidate_validation(
            models, candidate, syn_adj, labels, x, adj, y, val, stage_seed + 7000)))
    winner = max(selections, key=lambda r: (r["internal_val_accuracy"], -r["candidate"]))["candidate"]
    selected_texts = [texts[i][j] for i, j in enumerate(choices[winner])]
    cluster_ids = torch.as_tensor(clusters, dtype=torch.long)
    mass = torch.bincount(cluster_ids, minlength=rows).float()
    pooled_x = torch.zeros((rows, delta["x"].shape[1]))
    pooled_x.index_add_(0, cluster_ids, delta["x"].cpu())
    pooled_x /= mass[:, None]
    weighted = syn_adj.cpu().coalesce()
    support = weighted.indices()[:, weighted.indices()[0] != weighted.indices()[1]]
    block = dict(x=pooled_x, text_emb=encoder.encode(selected_texts).cpu(), texts=selected_texts,
        labels=torch.tensor(classes)[labels.cpu()], edge_index=support,
        mass=mass, depth=torch.zeros(rows, dtype=torch.long),
        stage=torch.full((rows,), int(delta["stage"]), dtype=torch.long))
    audit = dict(native_source=provenance, generator=generator, core_seconds=core["seconds"],
        internal_split=split_audit,
        internal_val_accuracy=core["internal_val_accuracy"], supervised_rows=int(train.sum()),
        internal_validation_rows=int(val.sum()), pseudo_pool_rows=int(unlabeled.sum()),
        source_rows=len(x), slots=rows, final_edges=support.shape[1],
        core_weighted_edges=len(weighted.values()), keyword_empty_vocabulary_repairs=repairs,
        generation_candidates=3 * rows, generated_schema_retries=sum(r["retry"] for r in generated),
        generated_schema_valid_fraction=(sum(sum(r["schema_valid"]) for r in generated) / (3 * rows)),
        wasserstein_scores=scores, selection=selections, winner=winner,
        label_rule="native mean predicted logits, mapped to current global class ids",
        historical_raw_reads=0, report_dev_used_for_selection=False, test_used_for_selection=False,
        adaptations=["current delta only; frozen block diagonal prefix",
            "classwise 10% labeled / 10% internal val; singleton train-only; pairs train/val; remainder pseudo pool",
            "common MiniLM semantic core; original X pooled for common Student",
            "local Qwen2.5-1.5B rather than native qwen3-max API",
            "native raw text candidate selection; prompt schema compliance audited, not enforced",
            "three samples; native 100 combinations, WSD top-five, internal GCN validation",
            "common 320-character cap before candidate scoring and final storage",
            "sparse equivalent of native dense coarsening", "binary synthetic support in common Student; weighted native candidate evaluation"],
        official_reproduction=False)
    return block, audit


def append(parent, block):
    if parent is None:
        return block
    result = {name: torch.cat((parent[name], block[name]))
              for name in ("x", "text_emb", "labels", "mass", "depth", "stage")}
    result["texts"] = list(parent["texts"]) + list(block["texts"])
    result["edge_index"] = torch.cat((parent["edge_index"], block["edge_index"] + len(parent["x"])), dim=1)
    return result


if __name__ == "__main__":
    torch.set_num_threads(2)
    print(json.dumps(audit_sparse(), indent=2))
