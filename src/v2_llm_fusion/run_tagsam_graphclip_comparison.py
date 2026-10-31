"""TAGSAM response-matching adapters, not an official end-to-end reproduction.

Uses the pinned official wBCELoss directly. Backbone, text initialization,
continual access, topology and optimization budget are declared adaptations.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import run_small_comparison_audited as C
from run_multiscale_asm import _func_graph, _func_text, _asm_loss

U, G, SCA, P = C.U, C.G, C.SCA, C.P
OFFICIAL_COMMIT = "d812f06fd16db3cd4d497804a8e504d2c738faa8"
ARMS = ["init_selfloop", "tagsam_selfloop", "tagsam_semantic", "asm_rowcol_semantic"]


def official_loss(repo):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise ValueError("unexpected TAGSAM source revision")
    path = repo / "model.py"
    canonical = subprocess.check_output(["git", "-C", str(repo), "show", "HEAD:model.py"])
    if canonical != path.read_bytes():
        raise ValueError("official model.py has local modifications")
    module = ast.parse(path.read_text())
    node = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "wBCELoss")
    # Isolate the original loss from unrelated GPT/BERT download imports.
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["wBCELoss"](), {"repository": "https://github.com/SundayVHan/TAGSAM",
        "commit": commit, "model_sha256": C.sha(path), "class": "model.py:wBCELoss",
        "other_source_sha256": {n: C.sha(repo / n) for n in ("distill.py", "dataset.py", "epoch.py", "selection.py")}}


def optimize(initial, old, raw, teacher, stage, arm, device, loss_fn, steps):
    keep = 0 if old is None else len(old["x"])
    semantic = arm.endswith("semantic")
    initial = dict(initial)
    if not semantic:
        initial["edge_index"] = torch.empty((2, 0), dtype=torch.long)
    memory = SCA.append_memory(old, initial, keep, cross_degree=int(semantic), device=device)
    if arm == "init_selfloop":
        return memory, {"optimizer_steps": 0, "inner_steps_total": 0}
    adj = G.norm_adj(memory["edge_index"], len(memory["x"]), device)
    real_adj = G.norm_adj(raw["edge_index"], len(raw["x"]), device)
    rx, rt = raw["x"].to(device), raw["text_emb"].to(device)
    text = memory["text_emb"].to(device)
    var = torch.nn.Parameter(initial["x"].to(device).clone())
    prefix = memory["x"][:keep].to(device)
    rate_g = torch.nn.Parameter(torch.tensor(.005, device=device))
    rate_t = torch.nn.Parameter(torch.tensor(.005, device=device))
    tagsam = arm.startswith("tagsam")
    if tagsam:
        outer = torch.optim.SGD([{"params": [var], "lr": 100., "momentum": .5},
                                 {"params": [rate_g, rate_t], "lr": 2e-6, "momentum": .5}])
    else:
        outer = torch.optim.Adam([var], lr=.01)
    reference = torch.randperm(len(rx), generator=torch.Generator().manual_seed(777 + stage))[:256].to(device)
    with torch.no_grad():
        teacher_response = teacher.logits(rx, real_adj, rt[reference])[reference].detach()
    curve, started = [], time.perf_counter()
    for outer_step in range(1, steps + 1):
        torch.manual_seed(9001 + stage * 1009 + outer_step)
        student = G.CLIPStudent(var.shape[1]).to(device)
        params = dict(student.named_parameters())
        synthetic = torch.cat((prefix, var))
        gen = torch.Generator().manual_seed(777 + stage * 1009 + outer_step)
        for _ in range(10):
            rows = torch.randperm(len(synthetic), generator=gen)[:min(16, len(synthetic))].to(device)
            graph = _func_graph(params, synthetic, adj)[rows]
            text_features = _func_text(params, text[rows])
            logits = params["logit_scale"].exp().clamp(max=100) * graph @ text_features.T
            target = torch.arange(len(rows), device=device)
            inner_loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
            gradients = torch.autograd.grad(inner_loss, tuple(params.values()), create_graph=True)
            params = {name: value - ((rate_t if name.startswith("text_proj") else rate_g)
                       if tagsam else .005) * grad
                       for (name, value), grad in zip(params.items(), gradients)}
        graph, text_features = _func_graph(params, rx, real_adj)[reference], _func_text(params, rt[reference])
        response = params["logit_scale"].exp().clamp(max=100) * graph @ text_features.T
        loss = loss_fn(response, teacher_response) if tagsam else _asm_loss(response, teacher_response)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{arm} nonfinite objective at {outer_step}")
        outer.zero_grad(set_to_none=True)
        variables = [var, rate_g, rate_t] if tagsam else [var]
        gradients = torch.autograd.grad(loss, variables)
        for parameter, gradient in zip(variables, gradients):
            parameter.grad = gradient
        torch.nn.utils.clip_grad_norm_([var], 5.)
        outer.step()
        curve.append({"step": outer_step, "loss": float(loss.detach()),
                      "graph_inner_lr": float(rate_g.detach()), "text_inner_lr": float(rate_t.detach()),
                      "x_shift_rms": float((var.detach() - initial["x"].to(device)).square().mean().sqrt())})
        if outer_step % 25 == 0:
            print(json.dumps({"arm": arm, "stage": stage, **curve[-1]}), flush=True)
    memory["x"] = torch.cat((prefix.cpu(), var.detach().cpu()))
    C.assert_frozen(old, memory)
    return memory, {"optimizer_steps": steps, "inner_steps_total": 10 * steps,
        "reference_rows": reference.cpu().tolist(), "source_rows": len(rx),
        "seconds": time.perf_counter() - started, "curve": curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 106)))
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    ap.add_argument("--official", default="/home/pc/research/TAGSAM_official")
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    loss_fn, provenance = official_loss(Path(args.official))
    data_path = U.DATA_CFG[args.dataset]["path"]
    data = U.load_unified(args.dataset, data_path, str(Path(U.__file__).parent / "minilm_from_graphclip"), args.device)
    cache = C.artifact_dir(args.dataset, "v4") / "text_emb_pool_only.pt"
    data["text_emb"] = torch.load(cache, map_location="cpu", weights_only=False)
    if data["text_emb"][~data["pool_mask"]].abs().max() != 0:
        raise ValueError("holdout embeddings in source cache")
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = G.norm_adj(data["edge_index"], len(data["x"]), args.device)
    data["x_device"] = data["x"].to(args.device)
    stages = [torch.where(data["pool_mask"] & torch.isin(data["y"], torch.tensor(cs)))[0]
              for cs in data["stage_classes"]]
    budgets = [math.floor(.05 * sum(len(ids) for ids in stages[:s + 1])) for s in range(3)]
    manifest = {"version": "tagsam-graphclip-adapter-v1", "dataset": args.dataset,
        "source_sha256": C.sha(data_path), "runner_sha256": C.sha(__file__),
        "trainer_sha256": C.sha(C.__file__), "official": provenance,
        "arms": args.arms, "seeds": args.seeds, "budgets": budgets, "outer_steps": args.steps,
        "inner_steps": 10, "batch": 16, "reference_cap": 256,
        "teacher_steps": 4096, "teacher_seeds": [9000, 9001, 9002],
        "data_access": "current delta induced training graph only; old synthetic rows frozen",
        "student": "same canonical GCN/MiniLM/Adam/steps as audited core matrix",
        "adaptations": ["canonical GCN and MiniLM instead of official encoders",
            "dispersion synthetic mean features and center concat instead of official text selector",
            "fixed-rate continual blocks instead of static full-history condensation",
            "fixed canonical downstream Adam, not distilled learning rates",
            "100 outer x 10 inner default vs official 5000 x 15 epochs",
            "fixed delta reference up to 256 vs official sampling batch up to 2000"],
        "test_labels_evaluated": False, "report_dev_used_for_selection": False}
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()) != manifest:
        raise ValueError("stale adapter configuration")
    C.save_json(out / "manifest.json", manifest)
    raw_supports, teachers = [], []
    for stage, ids in enumerate(stages):
        raw = C.raw_support(data, ids, stage)
        checkpoint = out / f"teacher_s{stage}.pt"
        teacher = G.CLIPStudent(data["x"].shape[1]).to(args.device)
        started = time.perf_counter()
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            teacher.load_state_dict(saved["state"])
        else:
            teacher = G.train_student(raw, 9000 + stage, args.device, 4096)
            torch.save({"state": {k: v.cpu() for k, v in teacher.state_dict().items()},
                        "source_ids": ids, "seconds": time.perf_counter() - started,
                        "steps": 4096, "support_sha256": C.tensor_hash(raw)}, checkpoint)
        raw_supports.append(raw)
        teachers.append(teacher.eval())
    results, audits = [], []
    for arm in args.arms:
        old = None
        for stage, ids in enumerate(stages):
            memory_path = out / f"{arm}_s{stage}_memory.pt"
            audit_path = out / f"{arm}_s{stage}_construction.json"
            keep = 0 if old is None else len(old["x"])
            if memory_path.exists() and audit_path.exists():
                memory = torch.load(memory_path, map_location="cpu", weights_only=False)
                audit = json.loads(audit_path.read_text())
                if C.tensor_hash(memory) != audit["tensor_sha256"]:
                    raise ValueError("memory hash mismatch")
            else:
                block = U.build_stage_block(data, ids, budgets[stage] - keep, data["encoder"],
                    stage, "joint_g5_c320", "concat", args.device, "dispersion", data["stage_classes"][stage], .5, .5)
                memory, audit = optimize(block, old, raw_supports[stage], teachers[stage],
                    stage, arm, args.device, loss_fn, args.steps)
                audit.update(arm=arm, stage=stage, tensor_sha256=C.tensor_hash(memory))
                torch.save(memory, memory_path)
                C.save_json(audit_path, audit)
            C.assert_frozen(old, memory)
            if len(memory["x"]) != budgets[stage]:
                raise AssertionError("memory budget differs")
            audits.append(audit)
            rows, steps = [], P.student_steps(budgets[stage])
            for seed in args.seeds:
                key = f"{arm}_s{stage}_seed{seed}"
                rowpath, checkpoint = out / f"{key}.json", out / f"{key}.pt"
                if rowpath.exists() and checkpoint.exists():
                    row = json.loads(rowpath.read_text())
                    if row["support_sha256"] != C.tensor_hash(memory):
                        raise ValueError("stale fit support")
                else:
                    row = C.fit(memory, seed, steps, args.device, data, stage, checkpoint=checkpoint)
                    C.save_json(rowpath, row)
                rows.append(row)
            result = {"arm": arm, "stage": stage, "steps": steps, "rows": len(memory["x"]),
                      "per_seed": rows, "metrics": C.aggregate(rows)}
            results.append(result)
            C.save_json(out / "progress.json", dict(manifest=manifest, results=results, audits=audits))
            print(json.dumps({"arm": arm, "stage": stage, "all": result["metrics"]["all"]}), flush=True)
            old = memory
    C.save_json(out / "results.json", dict(manifest=manifest, results=results, audits=audits))
    print("TAGSAM_ADAPTER_DONE", flush=True)


if __name__ == "__main__":
    main()
