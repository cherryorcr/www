"""Registered exact-5% ablations; reuse sealed scores only after hash checks."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ARMS = ("no_gm", "no_dm", "fixed_q", "random_init_learned", "random_init_fixed",
        "payload_controls")
PAYLOADS = ("uq_dm_edgefree", "uq_class_shared_payload", "uq_random5_payload")
SIGNATURE = ("groups", "budgets", "stage_raw_rows", "memory_rate", "actual_final_rate",
             "student", "student_seeds", "student_steps", "student_batch", "pool_mask_hash",
             "report_mask_hash", "test_mask_hash", "stage_id_hashes")
COMMON_SOURCES = ("unified_q_multimodal.py", "graphtext_prompt_student.py",
                  "external_continual_mechanisms.py", "run_unified_q_revision_20261007.py",
                  "run_unified_q_large_smoke_20261007.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_helper(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slot_diagnostics(state, C, U):
    q, q0, delta, block = (state[k] for k in ("q", "q0", "delta", "block"))
    raw = U._norm(delta["text_emb"].float().to(q.device))
    labels, raw_labels = block["labels"], delta["y"].to(q.device)
    out = C.diagnostics(q, q, q0, raw, raw_labels, labels, block["texts"])
    column = q / q.sum(0).clamp_min(1e-6)[None, :]
    entropy = -(column.clamp_min(1e-12) * column.clamp_min(1e-12).log()).sum(0)
    out["column_support_entropy_mean"] = float(entropy.mean())
    out["effective_support_size_mean"] = float(entropy.exp().mean())
    out["per_class"] = []
    for cls in torch.unique(labels, sorted=True):
        ids = torch.where(labels == cls)[0]
        texts = [block["texts"][int(i)] for i in ids]
        emb = U._norm(block["text_emb"][ids].float().to(q.device))[:512]
        pair = ((emb @ emb.T).sum() - len(emb)) / max(1, len(emb) * (len(emb) - 1))
        out["per_class"].append(dict(cls=int(cls), slots=len(ids), unique_texts=len(set(texts)),
            unique_text_ratio=len(set(texts)) / len(texts), text_pair_cosine=float(pair),
            cosine_scope="first up to 512 class slots; off-diagonal pairs"))
    return out


def payload_block(block, delta, encoder, seed, mode, R):
    altered = dict(block)
    labels = block["labels"]
    if mode == "uq_random5_payload":
        candidate, audit = R.M.attach_text(delta, delta["x"], block["x"], labels,
            encoder, R.M.Config(seed=seed, text_sources=5, text_mode="random", text_chars=320))
        altered.update(texts=candidate["texts"], text_emb=candidate["text_emb"].float().cpu(),
                       text_source_ids=audit["text_source_ids"])
    else:
        raw_texts = [R.UQ._append_text(t, 320) for t in delta["texts"]]
        emb = R.UQ._norm(torch.as_tensor(encoder.encode(raw_texts)).float().cpu())
        texts, sources = [], []
        medoids = {}
        for cls in torch.unique(labels, sorted=True).tolist():
            candidates = torch.where(delta["y"] == cls)[0]
            center = R.UQ._norm(emb[candidates].mean(0, keepdim=True))[0]
            medoids[cls] = int(candidates[(emb[candidates] @ center).argmax()])
        for cls in labels.tolist():
            index = medoids[cls]
            texts.append(raw_texts[index])
            sources.append([int(delta["ids"][index])])
        altered.update(texts=texts, text_emb=torch.as_tensor(encoder.encode(texts)).float().cpu(),
                       text_source_ids=sources)
        audit = dict(rule="one real text medoid per current class; same payload on all class slots",
                     text_chars=320, text_source_ids=sources)
    torch.testing.assert_close(altered["x"], block["x"], rtol=0, atol=0)
    unique = len(set(altered["texts"])) / len(labels)
    audit.update(intervention="fixed-X storage-only; no change to primary construction feedback",
        primary_feature_hash=R.V.hash_tensor(block["x"]), historical_raw_reads=0,
        text_unique_ratio=unique, training_on_stored_payload=True,
        construction_forward_uses_altered_payload=False)
    return altered, audit


def checkpoint_write(path, value):
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as stream:
        torch.save(value, stream)
    temporary.replace(path)


def run(args):
    frozen = args.root / "experiments/uq_final_evidence_20261012/code"
    sys.path.insert(0, str(frozen))
    sys.path.insert(1, str(args.root / "code/v2_llm_fusion"))
    import run_unified_q_matrix_20261008 as R
    import run_uq_final_confirmation_20261012 as E
    if Path(R.__file__).resolve().parent != frozen.resolve():
        raise AssertionError("immutable constructor import failed")
    C = import_helper("uq_ablation_assignment_20261014")
    B = import_helper("run_uq_baseline_expansion_20261013")
    reference = B.read_reference(args.reference, args.dataset)
    data = B.load_data(args, reference, R)
    ids = R.L.stage_ids(data)
    sizes = list(map(len, ids))
    budgets, floors = R.V.choose_budgets(reference, sizes, data["stage_classes"], "fixed_rate", .05)
    base = R.q_config("uq_dm_edgefree", args.construction_seed, 30, 4096)
    cfg = (R.q_config("uq_" + args.arm, args.construction_seed, 30, 4096)
           if args.arm in ("no_gm", "no_dm", "fixed_q") else base)
    if args.arm == "random_init_fixed":
        cfg = replace(base, train_q=False)
    arms = list(PAYLOADS) if args.arm == "payload_controls" else ["uq_" + args.arm]
    protocol = dict(name="uq-component-ablation-20261014-v1", dataset=args.dataset,
        groups=data["stage_classes"], budgets=budgets, stage_raw_rows=sizes,
        order_seed=args.order_seed, construction_seed=args.construction_seed,
        student_seeds=[150, 151, 152], student_steps=2048, student_batch=24,
        student=R.P.model_config(), encoder=args.minilm, memory_rate=.05,
        actual_final_rate=budgets[-1] / sum(sizes), class_floor_adjustments=floors,
        pool_mask_hash=R.V.hash_tensor(data["pool_mask"]),
        report_mask_hash=R.V.hash_tensor(data["report_dev_mask"]),
        test_mask_hash=R.V.hash_tensor(data["test_mask"]),
        stage_id_hashes=[R.V.hash_tensor(v) for v in ids],
        budget_mode="fixed_rate_current_legal_pool", arms=arms, q_config=asdict(cfg),
        changes={k: [asdict(base)[k], asdict(cfg)[k]] for k in asdict(base)
                 if asdict(base)[k] != asdict(cfg)[k]},
        randomized_initial=args.arm.startswith("random_init"),
        randomization="within-class row permutation; identical logit scale/capacities",
        access="current-delta-only; own frozen synthetic prefix; zero historical raw reads",
        synthetic_topology="none; common raw transductive inference graph",
        method_selection_from_test=False, official_reproduction=False,
        primary_split="sealed_test; report_dev stored separately; post-hoc diagnosis, no tuning",
        source_hashes={Path(m.__file__).name: sha(m.__file__) for m in (R, R.L, R.M, R.V, R.P, R.UQ, E)},
        derived_hashes={Path(m.__file__).name: sha(m.__file__) for m in (C, B)},
        runner_hash=sha(__file__),
        payload_control_scope="fixed-X storage-only, not an end-to-end text construction ablation")
    source = args.root / f"results/uq_final_evidence_20261012/confirmation/{args.dataset}_o{args.order_seed}_c{args.construction_seed}/results.json"
    primary = json.loads(source.read_text())
    mismatches = [k for k in SIGNATURE if primary["protocol"].get(k) != protocol[k]]
    mismatches += [k for k in COMMON_SOURCES if primary["protocol"]["source_hashes"].get(k)
                   != protocol["source_hashes"].get(k)]
    if mismatches:
        raise AssertionError(f"primary protocol differs: {mismatches}")
    primary_rows = sorted([r for r in primary["rows"] if r["arm"] == "uq_dm_edgefree"], key=lambda r: r["stage"])
    if len(primary_rows) != len(ids):
        raise AssertionError("the existing primary is incomplete")
    protocol.update(primary_source=str(source.relative_to(args.root)), primary_source_hash=sha(source))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    result = json.loads(path.read_text()) if path.exists() else dict(protocol=protocol, rows=[], status="running")
    if result["protocol"] != protocol:
        raise ValueError("protocol changed; do not overwrite existing results")
    R.V.write_json(args.output / "protocol.json", protocol)
    R.V.write_json(path, result)
    counts = [len([r for r in result["rows"] if r["arm"] == arm]) for arm in arms]
    if len(set(counts)) != 1:
        raise AssertionError("partial stage commit")
    done = counts[0]
    if done == len(ids):
        print(json.dumps(dict(event="already_complete", output=str(args.output))), flush=True)
        return
    checkpoint = args.output / "latest.pt.gz"
    saved, parents = None, {arm: None for arm in arms}
    if checkpoint.exists():
        with gzip.open(checkpoint, "rb") as stream:
            saved = torch.load(stream, map_location="cpu", weights_only=False)
        if saved["stage"] not in (done - 1, done):
            raise AssertionError("checkpoint/metrics stages disagree")
        if saved["stage"] == done - 1:
            parents = saved["memories"]
            for arm in arms:
                row = next(r for r in result["rows"] if r["arm"] == arm and r["stage"] == done-1)
                if row["memory_hash"] != R.V.memory_hash(parents[arm]):
                    raise AssertionError("checkpoint memory hash differs")
    elif done:
        raise AssertionError("resume memory missing")
    for stage in range(done, len(ids)):
        if shutil.disk_usage(args.root).free < 8 * 2**30:
            raise OSError("fewer than 8 GiB free; pause new construction without deleting files")
        if saved is not None and saved["stage"] == stage:
            memories, audits, costs, checks = (saved[k] for k in ("memories", "audits", "costs", "checks"))
        else:
            delta = R.L.delta_view(data, ids[stage], stage)
            parent = parents[arms[0]]
            new_rows = budgets[stage] - (0 if parent is None else len(parent["x"]))
            labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_rows))
            R.V.sync(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            diagnostics = {}
            if args.arm.startswith("random_init"):
                block, audit = C.optimize(delta, parent, labels, data["encoder"], data["prompt_emb"],
                    cfg, args.device, randomized_initial=True)
            else:
                block, audit = R.UQ.optimize_stage(delta, parent, labels, data["encoder"],
                    data["prompt_emb"], cfg, args.device, use_boundary=False, use_topology=False,
                    diagnostic_callback=lambda state: diagnostics.update(slot_diagnostics(state, C, R.UQ)))
                audit["diagnostics"] = diagnostics
            R.V.sync(args.device)
            cost = dict(construction_seconds=time.perf_counter() - started,
                        construction_peak_mb=torch.cuda.max_memory_allocated(args.device) / 2**20)
            memories, audits, costs, checks = {}, {}, {}, {}
            for arm in arms:
                altered, detail = (payload_block(block, delta, data["encoder"], args.construction_seed, arm, R)
                                   if arm != arms[0] else (block, audit))
                memory = R.L.append_edge_free(parents[arm], altered)
                check = R.V.validate_memory(memory, parents[arm], budgets[stage])
                if memory["edge_index"].numel():
                    raise AssertionError("edge-free memory contains stored edges")
                if "q_row_sum_max_error" in detail:
                    check.update(R.assignment_checks(detail))
                    if not check["sinkhorn_feasible"]:
                        raise AssertionError("Sinkhorn infeasible")
                if args.arm == "payload_controls":
                    if arm == arms[0] and R.V.memory_hash(memory) != primary_rows[stage]["memory_hash"]:
                        raise AssertionError("diagnostic replay changed the frozen primary memory")
                    if arm != arms[0]:
                        torch.testing.assert_close(memory["x"], memories[arms[0]]["x"], rtol=0, atol=0)
                memories[arm], audits[arm], checks[arm] = memory, R.compact_audit(detail), check
                costs[arm] = dict(cost, primary_reconstruction_for_missing_saved_memory=args.arm == "payload_controls")
            checkpoint_write(checkpoint, dict(stage=stage, memories=memories, audits=audits, costs=costs, checks=checks))
            print(json.dumps(dict(event="constructed", dataset=args.dataset, arm=args.arm,
                stage=stage, **cost)), flush=True)
        new = []
        for arm in arms:
            reused = args.arm == "payload_controls" and arm == arms[0]
            started = time.perf_counter()
            scores = primary_rows[stage]["scores"] if reused else E.evaluate(memories[arm], data, stage, [150,151,152], args.device)
            R.V.sync(args.device)
            new.append(dict(arm=arm, stage=stage, scores=scores, score=scores["report_dev"],
                raw_rows=sizes[stage], memory_rows=budgets[stage],
                new_rows=budgets[stage]-(budgets[stage-1] if stage else 0),
                memory_hash=R.V.memory_hash(memories[arm]), audit=audits[arm], checks=checks[arm],
                student_seconds=0. if reused else time.perf_counter()-started,
                reused_evaluation=reused, **costs[arm]))
            print(json.dumps(dict(event="evaluated", dataset=args.dataset, arm=arm, stage=stage,
                reused=reused, test_f1=scores["sealed_test"]["final_f1"], test_acc=scores["sealed_test"]["accuracy"])), flush=True)
        result["rows"].extend(new)
        R.V.write_json(path, result)
        parents, saved = memories, None
        gc.collect()
        torch.cuda.empty_cache()
    result["status"] = "complete"
    R.V.write_json(path, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    p.add_argument("--dataset", choices=("arxiv", "sportsfit", "bookhis"), required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--order-seed", type=int, default=9101)
    p.add_argument("--construction-seed", type=int, default=271006)
    args = p.parse_args()
    torch.set_num_threads(2)
    run(args)
