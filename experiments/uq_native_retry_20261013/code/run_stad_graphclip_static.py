"""Pinned STAD construction with local LLM text and canonical GraphCLIP transfer.

This is STAD-local-canonical, not an official score reproduction. See the
frozen protocol for input access, internal split, encoder and LLM adaptations.
Official code is loaded from a separate MIT-licensed checkout; keys.py and
the official train/test entrypoints are never imported or executed.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import random
import re
import string
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace, MethodType

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import run_tagsam_graphclip_static as S
from small_comparison_metrics import evaluate

C, G, P, U = S.C, S.G, S.P, S.U
COMMIT = "5aaf1a06e44638499373b1358c85d1546187edae"
ARM = "stad_local_canonical"
PRE_METRIC_KEY_FIX_SHA = "fa350342847138e76392219e333bc896057133443144e43522f083c16abe7f5d"


def normalized_metrics(value):
    return json.loads(json.dumps(value))


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_components(root):
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip()
    if head != COMMIT:
        raise ValueError("STAD checkout is not pinned")
    files = ["models/GNN.py", "utils_gdd.py", "deep_robust_utils.py", "tagcond.py", "prompt.py"]
    for path in files:
        pinned = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{path}"])
        actual = (root / path).read_bytes()
        if pinned.replace(b"\r\n", b"\n") != actual.replace(b"\r\n", b"\n"):
            raise ValueError(f"official STAD source was modified: {path}")
    utils = module_at("stad_deep_utils", root / "deep_robust_utils.py")
    sys.modules["deep_robust_utils"] = utils
    sketch = module_at("stad_sketch", root / "utils_gdd.py")
    prompts = module_at("stad_prompts", root / "prompt.py")
    tree = ast.parse((root / "models/GNN.py").read_text(encoding="utf-8-sig"))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name in {"GraphConvolution", "GCN", "SelfTrainingGNN"}]
    # The only official-model edit removes test scoring; no test mask is exposed.
    for cls in classes:
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "self_train":
                for i, node in enumerate(fn.body):
                    if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                            and t.id == "final_test_acc" for t in node.targets)):
                        fn.body[i] = ast.Assign(targets=node.targets, value=ast.Constant(None))
                fn.body = [n for n in fn.body if not (isinstance(n, ast.Expr)
                    and "Final test Acc" in ast.unparse(n))]
    namespace = dict(torch=torch, nn=nn, F=F, math=math, np=np,
                     Parameter=nn.Parameter, Module=nn.Module, utils=utils,
                     deepcopy=deepcopy, optim=torch.optim)
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])),
                 str(root / "models/GNN.py"), "exec"), namespace)
    tree = ast.parse((root / "tagcond.py").read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TAGCond")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in {"clean", "tfidf_topk", "sbert_topk", "extract_keywords"}]
    kw_namespace = dict(re=re, string=string, np=np, torch=torch,
                        TfidfVectorizer=TfidfVectorizer)
    exec(compile(ast.Module(body=methods, type_ignores=[]),
                 str(root / "tagcond.py"), "exec"), kw_namespace)
    meta = {"commit": head, "file_sha256": {p: C.sha(root / p) for p in files},
            "license": "MIT; Copyright (c) 2026 laiyurui",
            "ast_change": "self_train final test score and its print disabled"}
    return SimpleNamespace(**namespace), sketch, prompts, kw_namespace, meta


class TextEncoder:
    """Official unnormalized mean pooling, using the canonical frozen weights."""
    def __init__(self, encoder):
        self.encoder = encoder

    @torch.no_grad()
    def encode(self, texts, convert_to_tensor=True, device=None):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        parts = []
        for start in range(0, len(texts), 64):
            tokens = self.encoder.tokenizer(texts[start:start + 64], padding=True,
                truncation=True, max_length=256, return_tensors="pt").to(self.encoder.device)
            hidden = self.encoder.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            parts.append(((hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)).float())
        result = torch.cat(parts)
        return result[0] if single else result


def validate_sources(data, ids, stage):
    seen = [c for group in data["stage_classes"][:stage + 1] for c in group]
    if (not len(ids) or len(ids.unique()) != len(ids)
            or not data["pool_mask"][ids].all()
            or not torch.isin(data["y"][ids], torch.tensor(seen)).all()):
        raise ValueError("STAD construction source is not legal seen pool")


def llm_candidates(request, cache, endpoint):
    fingerprint = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    if cache.exists():
        record = json.loads(cache.read_text())
        if record["request_sha256"] != fingerprint:
            raise ValueError("stale LLM response cache")
        return record
    started = time.perf_counter()
    response = requests.post(endpoint + "/chat/completions", json=request, timeout=240)
    response.raise_for_status()
    payload = response.json()
    choices = sorted(payload["choices"], key=lambda c: c["index"])
    texts = [c["message"]["content"].strip() for c in choices]
    if len(texts) != 3 or any(not t or "[[SUMMARY]]" not in t for t in texts):
        raise ValueError("incomplete STAD generation; refusing keyword fallback")
    record = {"request_sha256": fingerprint, "request": request, "texts": texts,
              "usage": payload.get("usage", {}), "response_model": payload.get("model"),
              "finish_reasons": [c["finish_reason"] for c in choices],
              "seconds": time.perf_counter() - started}
    C.save_json(cache, record)
    return record


def candidate_validation(models, x, adj, labels, data, seen, seed, device):
    torch.manual_seed(seed)
    model = models.GCN(x.shape[1], 64, len(seen), device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.0005)
    seen_tensor = torch.tensor(seen)
    mask = data["selector_dev_mask"] & torch.isin(data["y"], seen_tensor)
    ids = torch.where(mask)[0].to(device)
    target = data["y"].to(device)[ids]
    best, best_epoch = -1., -1
    for epoch in range(600):
        model.train()
        loss = F.cross_entropy(model(x, adj), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            model.eval()
            pred = seen_tensor.to(device)[model(data["x_device"], data["real_adj"])[ids].argmax(1)]
            score = float((pred == target).float().mean())
        if score > best:
            best, best_epoch = score, epoch + 1
    return {"selector_accuracy": best, "epoch": best_epoch, "seed": seed, "steps": 600}


def construct(data, ids, budget, stage, out, args, official):
    validate_sources(data, ids, stage)
    stage_dir = out / f"s{stage}"
    stage_dir.mkdir(exist_ok=True)
    memory_path, audit_path = stage_dir / "memory.pt", stage_dir / "construction.json"
    if memory_path.exists() and audit_path.exists():
        memory = torch.load(memory_path, map_location="cpu", weights_only=False)
        audit = json.loads(audit_path.read_text())
        if C.tensor_hash(memory) != audit["tensor_sha256"]:
            raise ValueError("STAD memory cache corruption")
        return memory, audit
    if args.audit_only:
        raise ValueError("audit-only cannot construct missing memory")
    models, sketch, prompts, kw_methods, _ = official
    device = args.device
    started = time.perf_counter()
    seed = 15 + stage
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats(device)
    raw = C.raw_support(data, ids, stage)
    x = raw["x"].to(device)
    if x.shape[1] != 384:
        raise ValueError("STAD MiniLM transfer requires canonical 384-dimensional attributes")
    edges = raw["edge_index"].to(device)
    raw_adj = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1], device=device),
                                      (len(ids), len(ids))).coalesce()
    adj = models.utils.normalize_adj_tensor(raw_adj, sparse=True).to(device)
    seen = [c for group in data["stage_classes"][:stage + 1] for c in group]
    mapping = {c: i for i, c in enumerate(seen)}
    y = torch.tensor([mapping[int(v)] for v in data["y"][ids]], device=device)
    tr, va = train_test_split(np.arange(len(ids)), test_size=.2,
                             stratify=y.cpu().numpy(), random_state=seed)
    train_mask = torch.zeros(len(ids), dtype=torch.bool, device=device)
    val_mask = torch.zeros_like(train_mask)
    train_mask[tr], val_mask[va] = True, True
    core_path = stage_dir / "core.pt"
    if core_path.exists():
        core = torch.load(core_path, map_location=device, weights_only=False)
    else:
        core_started = time.perf_counter()
        model = models.SelfTrainingGNN(384, 64, len(seen)).to(device)
        logits, best_adj, _, val_acc, _ = model.self_train(
            x, adj, y.clone(), train_mask, val_mask,
            torch.zeros_like(train_mask), val_mask.clone(),
            num_stages=10, pretrain_stages=0, epochs_per_stage=600)
        logits = logits.detach()
        clusters = KMeans(n_clusters=budget, n_init=10, random_state=seed).fit_predict(
            logits.cpu().numpy())
        clusters, count = sketch.regenerate_cluster_labels(clusters, logits, budget)
        if count != budget:
            raise AssertionError("STAD cluster budget mismatch")
        centers = sketch.clst_condense_feat(clusters, budget, x.cpu().numpy()).to(device)
        syn_adj, _, _ = sketch.clst_condense_adj(best_adj, budget, len(ids), clusters, device)
        labels = sketch.clst_condense_label(budget, logits, clusters, device)
        core = {"clusters": clusters, "centers": centers.cpu(),
                "adj": syn_adj.cpu(), "labels": labels.cpu(),
                "internal_val_accuracy": val_acc,
                "core_seconds": time.perf_counter() - core_started}
        torch.save(core, core_path)
        del model, logits
    clusters, syn_adj, labels = core["clusters"], core["adj"].to(device), core["labels"].to(device)
    encoder = TextEncoder(data["encoder"])
    keyword_agent = SimpleNamespace(sbert=encoder, device=device)
    for name in ("clean", "tfidf_topk", "sbert_topk", "extract_keywords"):
        setattr(keyword_agent, name, MethodType(kw_methods[name], keyword_agent))
    generated = []
    llm_dir = stage_dir / "llm"
    llm_dir.mkdir(exist_ok=True)
    for cluster in range(budget):
        cache = llm_dir / f"node{cluster:04d}.json"
        docs = [raw["texts"][i] for i in np.where(clusters == cluster)[0]]
        words, _ = keyword_agent.extract_keywords(docs, core["centers"][cluster].to(device), kk=256)
        label = U.DATA_CFG[data["dataset"]]["names"][seen[int(labels[cluster])]]
        background = prompts.SYS_CORA if data["dataset"] == "cora" else prompts.SYS_CIESEER
        request = {"model": args.llm_model, "messages": [
            {"role": "system", "content": prompts.system_prompt_template_text.format(
                dataset_name=data["dataset"], sys_dataset=background)},
            {"role": "user", "content": prompts.user_prompt_template_text.format(
                center_key=", ".join(words), center_label1=label, tok_lim=256)}],
            "temperature": 1.2, "top_p": .85, "max_tokens": 256, "n": 3,
            "seed": 150000 + stage * 1000 + cluster,
            "chat_template_kwargs": {"enable_thinking": False}}
        generated.append(llm_candidates(request, cache, args.endpoint))
        if cluster % 10 == 0 or cluster + 1 == budget:
            print(json.dumps({"stage": stage, "generated_nodes": cluster + 1,
                              "target": budget}), flush=True)
    texts = [r["texts"] for r in generated]
    embeds = encoder.encode([t for row in texts for t in row]).reshape(budget, 3, 384)
    # Cache embeddings once; the official routine re-encodes identical candidates.
    rng = random.Random(seed)
    choices = [[0] * budget] + [[rng.randrange(3) for _ in range(budget)] for _ in range(99)]
    source_smooth = F.normalize(sketch.smooth_twice(raw_adj, x), dim=1)
    alignment = []
    for number, choice in enumerate(choices):
        candidate = embeds[torch.arange(budget, device=device), choice]
        target_smooth = F.normalize(sketch.smooth_twice(syn_adj, candidate), dim=1)
        torch.manual_seed(seed * 1000 + number)
        score = sketch.subsample_sinkhorn(source_smooth, target_smooth, sub=2000, eps=.05)
        if not math.isfinite(score):
            raise FloatingPointError("STAD Sinkhorn score is nonfinite")
        alignment.append(score)
    shortlist = sorted(range(100), key=lambda n: (alignment[n], n))[:5]
    selection = []
    for number in shortlist:
        candidate = embeds[torch.arange(budget, device=device), choices[number]]
        result = candidate_validation(models, candidate, syn_adj, labels,
                                      data, seen, 7000 + stage, device)
        selection.append({"candidate": number, "sinkhorn": alignment[number], **result})
    winner = max(selection, key=lambda r: (r["selector_accuracy"], -r["candidate"]))["candidate"]
    synth_x = embeds[torch.arange(budget, device=device), choices[winner]].detach().cpu()
    selected_texts = [texts[i][j] for i, j in enumerate(choices[winner])]
    syn_adj = syn_adj.coalesce().cpu()
    memory = {"x": synth_x, "text_emb": F.normalize(synth_x, dim=1),
              "texts": selected_texts, "edge_index": syn_adj.indices(),
              "adj_values": syn_adj.values(), "adj_normalized": True,
              "mass": torch.tensor(np.bincount(clusters, minlength=budget)).float(),
              "stage": torch.full((budget,), stage, dtype=torch.long),
              "depth": torch.ones(budget, dtype=torch.long)}
    if not torch.isfinite(synth_x).all() or len(synth_x) != budget:
        raise AssertionError("invalid STAD synthetic artifact")
    audit = {"stage": stage, "rows": budget, "source_rows": len(ids),
             "source_ids": ids.tolist(), "source_ids_sha256": C.tensor_hash({"source_ids": ids}),
             "source_support_sha256": C.tensor_hash(raw), "constructor_seed": seed,
             "initial_labeled_rows": len(tr), "internal_val_pseudo_rows": len(va),
             "internal_val_accuracy": float(core["internal_val_accuracy"]),
             "core_seconds": core["core_seconds"], "edges": len(syn_adj.values()),
             "weighted_topology": True, "llm_candidates": budget * 3,
             "llm_usage": {key: sum(r["usage"].get(key, 0) for r in generated)
                           for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
             "llm_seconds": sum(r["seconds"] for r in generated),
             "sinkhorn_scores": alignment, "selection": selection, "winner": winner,
             "text_characters": sum(len(t) for t in selected_texts),
             "minilm_tokens": sum(len(data["encoder"].tokenizer.encode(t)) for t in selected_texts),
             "build_seconds_this_attempt": time.perf_counter() - started,
             "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
             "tensor_sha256": C.tensor_hash(memory), "test_labels_evaluated": False,
             "report_dev_used_for_selection": False}
    torch.save(memory, memory_path)
    audit["memory_file_sha256"] = C.sha(memory_path)
    C.save_json(audit_path, audit)
    return memory, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cora", "citeseer"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--official", default="/home/pc/research/STAD_official")
    ap.add_argument("--minilm", default=str(Path(U.__file__).parent / "minilm_from_graphclip"))
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--llm-model", default="Qwen3-32B-AWQ-vLLM")
    ap.add_argument("--stages", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    official = official_components(Path(args.official))
    data = S.load_data(args.dataset, args.minilm, args.device)
    stages, budgets = S.stage_inputs(data)
    manifest = {"version": "stad-local-canonical-v1", "dataset": args.dataset,
                "official": official[-1], "budgets": budgets, "stage_classes": data["stage_classes"],
                "seeds": list(P.STUDENT_SEEDS), "rate": .05, "llm_model": args.llm_model,
                "source_sha256": C.sha(U.DATA_CFG[args.dataset]["path"]),
                "runner_sha256": C.sha(__file__), "student_runner_sha256": C.sha(C.__file__),
                "dependencies": {Path(m.__file__).name: C.sha(m.__file__) for m in (S, U, G, P)},
                "protocol": "EXTERNAL_TAG_BASELINE_PROTOCOL_ZH_2026-09-17.md",
                "test_labels_evaluated": False, "report_dev_used_for_selection": False,
                "access": "full-history seen legal pool only; selector-dev for text candidate validation",
                "notice": "local Qwen/canonical student adaptation, not official score reproduction"}
    target = out / "manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        previous = json.loads(target.read_text())
        expected = {**previous, "runner_sha256": manifest["runner_sha256"]}
        if previous["runner_sha256"] != PRE_METRIC_KEY_FIX_SHA or expected != manifest:
            raise ValueError("STAD manifest mismatch")
        C.save_json(out / "metric_key_fix_provenance.json", {
            "previous_runner_sha256": previous["runner_sha256"],
            "current_runner_sha256": manifest["runner_sha256"],
            "reason": "normalize integer dictionary keys on both sides of JSON metric audit",
            "construction_or_training_changed": False})
    C.save_json(target, manifest)
    results, audits, checked = [], [], 0
    for stage in args.stages:
        ids = torch.cat(stages[:stage + 1])
        memory, audit = construct(data, ids, budgets[stage], stage, out, args, official)
        rows = []
        for seed in P.STUDENT_SEEDS:
            key = out / f"s{stage}" / f"seed{seed}"
            checkpoint, report = key.with_suffix(".pt"), key.with_suffix(".json")
            steps = P.student_steps(budgets[stage])
            if report.exists() and checkpoint.exists():
                row = json.loads(report.read_text())
                if row["support_sha256"] != C.tensor_hash(memory) or row["steps"] != steps:
                    raise ValueError("stale STAD student")
            elif args.audit_only:
                raise ValueError("audit-only cannot train missing fits")
            else:
                row = C.fit(memory, seed, steps, args.device, data, stage, checkpoint=checkpoint)
                C.save_json(report, row)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = G.CLIPStudent(memory["x"].shape[1]).to(args.device)
            model.load_state_dict(saved["state"])
            actual = evaluate(model, data, stage)
            if normalized_metrics(actual) != normalized_metrics(row["metrics"]):
                raise AssertionError("saved student metric mismatch")
            torch.manual_seed(seed)
            initial = G.CLIPStudent(memory["x"].shape[1]).to(args.device)
            if C.tensor_hash(initial.state_dict()) != row["start_state_sha256"]:
                raise AssertionError("student initialization mismatch")
            if C.tensor_hash(saved["state"]) != row["end_state_sha256"]:
                raise AssertionError("student checkpoint hash mismatch")
            del model, initial
            rows.append(row)
            checked += 1
        results.append({"arm": ARM, "stage": stage, "rows": budgets[stage],
                        "metrics": C.aggregate(rows), "per_seed": rows})
        audits.append(audit)
        C.save_json(out / "results.json", {"manifest": manifest, "results": results, "audits": audits})
        print(json.dumps({"stage": stage, "all": results[-1]["metrics"]["all"]}), flush=True)
    C.save_json(out / "acceptance.json", {"recomputed_reporting_fits": checked,
                "stages": args.stages, "errors": [], "test_labels_evaluated": False})
    print("STAD_LOCAL_CANONICAL_DONE", flush=True)


if __name__ == "__main__":
    main()
