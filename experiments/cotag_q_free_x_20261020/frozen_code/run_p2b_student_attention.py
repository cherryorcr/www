"""Controlled small-graph P2-B experiment; never alter the frozen student runner."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

import p2b_student_attention as T
import run_p2a_graph_conditioned_text as A

ARMS = ("center_concat", "fixed_anchor", "semantic_700", "semantic_701", "student_700", "student_701")


def validate_seeds(seeds):
    if len(set(seeds)) != len(seeds) or set(seeds) & {300, 301, 302, 310, 700, 701, 702}:
        raise ValueError("duplicate or training/report seed overlap")


def run_order(data, name, groups, args, plan_hash):
    out = args.output / name
    out.mkdir(parents=True, exist_ok=True)
    stages, budgets = A.O.set_stages(data, groups)
    A.O.lock_json(out / "manifest.json", dict(plan_sha256=plan_hash, stage_classes=groups,
        budgets=budgets, source_ids=[ids.tolist() for ids in stages]))
    old, reference_old, results = {}, None, []
    for stage, ids in enumerate(stages[:args.stages]):
        keep = budgets[stage - 1] if stage else 0
        source = A.legal_delta(data, ids)
        encoder = A.MeteredEncoder(data["encoder"], args.device)
        started = time.perf_counter()
        block, joint, clusters, allocation = A.shared_clusters(
            source, budgets[stage] - keep, stage, encoder, args.device)
        reference = A.SCA.append_memory(reference_old, block, keep, device=args.device)
        A.B.sync(args.device)
        shared_seconds = time.perf_counter() - started
        archive = A.C.RESULTS / "p1x_x_operator_20260918" / data["dataset"] / name / f"mean_x_s{stage}_memory.pt"
        previous = torch.load(archive, map_location="cpu", weights_only=False)
        if A.C.tensor_hash(previous) != A.C.tensor_hash(reference):
            raise AssertionError("frozen Mean-X/topology regression")
        evidence_path, shared_path = out / f"evidence_s{stage}.pt", out / f"shared_s{stage}.json"
        if evidence_path.exists() and shared_path.exists():
            shared = json.loads(shared_path.read_text())
            if shared["reference_sha256"] != A.C.tensor_hash(reference) or shared["evidence_sha256"] != A.C.sha(evidence_path):
                raise AssertionError("cached reference/evidence changed")
            saved = torch.load(evidence_path, map_location="cpu", weights_only=False)
            texts = saved["texts"]
            evidence = {k: v.to(args.device) for k, v in saved["tensors"].items()}
        else:
            encoder.seconds, encoder.sequences = 0., 0
            texts, evidence, evidence_audit = T.prepare_evidence(source, joint, clusters,
                                                               encoder, args.device, stage)
            torch.save(dict(texts=texts, tensors={k: v.cpu() for k, v in evidence.items()}), evidence_path)
            shared = dict(stage=stage, cluster_topology_seconds=shared_seconds,
                evidence=evidence_audit, evidence_sha256=A.C.sha(evidence_path),
                allocation=allocation, clusters_global_ids=[ids[c].tolist() for c in clusters],
                reference_sha256=A.C.tensor_hash(reference), source_ids=ids.tolist())
            A.C.save_json(shared_path, shared)
        memories, audits = {}, {}
        for arm in ARMS:
            memory_path, audit_path = out / f"{arm}_s{stage}_memory.pt", out / f"{arm}_s{stage}_construction.json"
            if memory_path.exists() and audit_path.exists():
                memory = torch.load(memory_path, map_location="cpu", weights_only=False)
                audit = json.loads(audit_path.read_text())
                if A.C.sha(memory_path) != audit["file_sha256"]:
                    raise AssertionError("memory file changed")
                checked = A.append_fixed_edges(old.get(arm), {k: memory[k][keep:] for k in A.FIELDS}, reference)
                if A.C.tensor_hash(memory) != A.C.tensor_hash(checked):
                    raise AssertionError("resumed control mismatch")
            else:
                if arm == "center_concat":
                    target, meta = evidence["anchor"].cpu(), dict(seconds=0., selected_step=0)
                elif arm == "fixed_anchor":
                    A.B.sync(args.device)
                    fixed_started = time.perf_counter()
                    with torch.no_grad():
                        target, weights = T.AnchoredAttention().to(args.device)(**evidence)
                    target = target.cpu()
                    A.B.sync(args.device)
                    meta = dict(seconds=time.perf_counter() - fixed_started, selected_step=0,
                                weights=weights.cpu().tolist())
                else:
                    mode, seed = arm.split("_")
                    initial = A.append_fixed_edges(old.get(arm), dict(block,
                        texts=texts, text_emb=evidence["anchor"].cpu()), reference)
                    target, state, meta = T.train_attention(mode, int(seed), stage, initial, keep,
                        source, clusters, evidence, args.device, args.outer_steps,
                        args.inner_steps, args.validation_steps)
                    torch.save(state, out / f"{arm}_s{stage}_attention.pt")
                    if meta["nonzero_gradient_steps"] == 0:
                        raise AssertionError("learning arm had no gradients")
                    if set(meta["train_local_ids"]) & set(meta["validation_local_ids"]):
                        raise AssertionError("outer split overlaps")
                memory = A.append_fixed_edges(old.get(arm), dict(block, texts=texts, text_emb=target), reference)
                if arm == "center_concat":
                    prior_path = A.C.RESULTS / "p2a_graph_conditioned_text_20260918" / data["dataset"] / name / f"center_concat_s{stage}_memory.pt"
                    prior = torch.load(prior_path, map_location="cpu", weights_only=False)
                    if A.C.tensor_hash(prior) != A.C.tensor_hash(memory) or prior["texts"] != memory["texts"]:
                        raise AssertionError("same-token concat control changed vs P2-A")
                torch.save(memory, memory_path)
                audit = dict(arm=arm, stage=stage, meta=meta,
                    file_sha256=A.C.sha(memory_path), support_sha256=A.C.tensor_hash(memory),
                    control_sha256=A.C.tensor_hash({k: memory[k] for k in
                        ("x", "mass", "depth", "stage", "edge_index")}),
                    payload=A.O.payload_stats(memory))
                A.C.save_json(audit_path, audit)
            memories[arm], audits[arm] = memory, audit
        if len({a["control_sha256"] for a in audits.values()}) != 1:
            raise AssertionError("non-text inputs differ")
        stage_results = []
        for arm in ARMS:
            memory, audit = memories[arm], audits[arm]
            rows = [A.fit_or_load(memory, data, stage, seed, arm, out, args.device, args.steps)
                    for seed in args.seeds]
            result = dict(arm=arm, stage=stage, rows=len(memory["x"]), per_seed=rows,
                metrics=A.C.aggregate(rows), meta_seconds=audit["meta"]["seconds"],
                selected_step=audit["meta"]["selected_step"], shared=shared)
            results.append(result)
            stage_results.append(result)
            A.C.save_json(out / "progress.json", dict(results=results))
            print(json.dumps(dict(event="stage_done", dataset=data["dataset"], order=name,
                stage=stage, arm=arm, selected_step=result["selected_step"],
                f1=result["metrics"]["all"]["cumulative_f1"]["mean"])), flush=True)
        for i in range(len(args.seeds)):
            if len({r["per_seed"][i]["start_state_sha256"] for r in stage_results}) != 1:
                raise AssertionError("student initializations differ")
        old, reference_old = memories, reference
        del source, evidence
    A.C.save_json(out / "results.json", dict(results=results))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--orders", nargs="+", default=["canonical", "order1701", "order1702"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 106)))
    ap.add_argument("--minilm", default=str(Path(A.U.__file__).parent / "minilm_from_graphclip"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--outer-steps", type=int, default=40)
    ap.add_argument("--inner-steps", type=int, default=16)
    ap.add_argument("--validation-steps", type=int, default=32)
    ap.add_argument("--stages", type=int, default=3)
    args = ap.parse_args()
    validate_seeds(args.seeds)
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    orders = A.stage_orders(args.dataset, args.orders)
    if not set(orders) <= {"canonical", "order1701", "order1702"}:
        raise ValueError("independent confirmation requires a separate registered plan")
    smoke = args.steps is not None or (args.outer_steps, args.inner_steps, args.validation_steps, args.stages) != (40, 16, 32, 3)
    code = (__file__, T.__file__, A.__file__, A.T.__file__, A.BT.__file__, A.C.__file__,
            A.O.__file__, A.B.__file__, A.G.__file__, A.U.__file__, A.SCA.__file__, A.P.__file__,
            Path(A.C.__file__).with_name("small_comparison_metrics.py"))
    plan = dict(version="p2b-student-attention-v1", classification="smoke" if smoke else "known-order screen",
        dataset=args.dataset, orders=orders, arms=list(ARMS), seeds=args.seeds,
        outer_steps=args.outer_steps, inner_steps=args.inner_steps, validation_steps=args.validation_steps,
        stages=args.stages, report_steps_override=args.steps, construction_seeds=[700, 701],
        probe_seeds=[300, 301, 302], checkpoint_seed=310,
        rate=.05, allocation="dispersion", x="mean", topology="fixed legacy frozen-old cross-degree1",
        token_cap=64, snippets=5, rank=16, residual_fraction=.25, temperature=.2,
        outer_lr=.01, anchor_penalty=.1, old_penalty=.25, outer_batch=16,
        student=dict(batch=16, lr=.001, steps="max(50,min(2048,16*rows))"),
        report="all-prompt report-dev seen nodes; full transductive graph; no test",
        outer_validation="source loss held out only; may occur in fixed construction",
        screening_gate=dict(final_gain_each_dataset=.5, stage_gain_each_dataset=0.,
            worst_order_gain=-2., old_new_gain=-2., controls=["center_concat", "fixed_anchor", "semantic"],
            each_method_seed_positive_vs_concat=True),
        code_sha256={Path(p).name: A.C.sha(p) for p in code},
        source_sha256=A.C.sha(A.U.DATA_CFG[args.dataset]["path"]),
        text_cache_sha256=A.C.sha(A.C.artifact_dir(args.dataset, "v4") / "text_emb_pool_only.pt"),
        minilm_sha256={p.name: A.C.sha(p) for p in Path(args.minilm).glob("*") if p.is_file()},
        python=platform.python_version(), torch=torch.__version__, gpu=torch.cuda.get_device_name(args.device))
    A.O.lock_json(args.output / "plan.json", plan)
    data = A.B.prepare(args.dataset, args.minilm, args.device, args.output)
    for name, groups in orders.items():
        run_order(data, name, groups, args, A.C.sha(args.output / "plan.json"))
    A.C.save_json(args.output / "acceptance.json", dict(complete=True, classification=plan["classification"],
        fits=len(orders)*len(ARMS)*args.stages*len(args.seeds), fixed_nontext_inputs=True,
        old_rows_text_edges_frozen=True, report_checkpoint_reload_verified=True,
        report_used_for_training=False, test_labels_evaluated=False))
    print("P2B_DONE", flush=True)


if __name__ == "__main__":
    main()
