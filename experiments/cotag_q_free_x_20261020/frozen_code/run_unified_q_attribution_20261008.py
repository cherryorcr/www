"""P0/P1 frozen-memory attribution, without changing the primary constructor.

All variants consume the same stage-local delta. Feature constructors run
once; text controls inherit their exact X, mass, labels and topology. Only
incremental blocks/text sidecars are archived to avoid cumulative duplication.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

import run_unified_q_matrix_20261008 as R
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cat_multimodal_minimal as MM
import x_conditioned_payload_routing as XCR
import run_unified_q_student_ab_20261007 as AB

L, V, M, P, UQ = R.L, R.V, R.M, R.P, R.UQ
PRIMARY = ("cat_cctp", "adaptive_coherent", "uq_dm_edgefree")
CONTROLS = ("cat_nearest1", "cat_random5", "uq_nearest1", "uq_nearest5")
TEXT_KEYS = ("text_emb", "texts", "text_source_ids")
FIXED_KEYS = ("x", "labels", "mass", "depth", "stage", "edge_index")


@contextmanager
def preserve_rng():
    py, np_state = random.getstate(), np.random.get_state()
    cpu = torch.random.get_rng_state()
    gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.random.set_rng_state(cpu)
        if gpu is not None:
            torch.cuda.set_rng_state_all(gpu)


def assert_fixed_content(base: dict, variant: dict) -> dict:
    for key in FIXED_KEYS:
        if not torch.equal(base[key], variant[key]):
            raise AssertionError(f"text-only control changed {key}")
    return {"fixed_content_exact": True,
            "fixed_hashes": {key: R.digest(base[key]) for key in FIXED_KEYS}}


def replace_payload(base: dict, attached: dict, source_ids: list) -> dict:
    result = dict(base)
    result.update(text_emb=attached["text_emb"].detach().cpu().clone(),
                  texts=list(attached["texts"]),
                  text_source_ids=[list(v) for v in source_ids])
    assert_fixed_content(base, result)
    return result


def text_control(delta, features, base, encoder, mode, top_k, seed,
                 uq_materialization=False):
    cfg = M.Config(seed=seed, text_mode=mode, text_sources=top_k, text_chars=320)
    if uq_materialization:
        delta = dict(delta, texts=[UQ._append_text(t, 320) for t in delta["texts"]])
    with preserve_rng():
        attached, audit = M.attach_text(
            delta, features, base["x"], base["labels"], encoder, cfg)
    result = replace_payload(base, attached, audit["text_source_ids"])
    audit["fixed_content"] = assert_fixed_content(base, result)
    audit["uq_candidate_preprocessing"] = uq_materialization
    return result, audit


def adaptive_payload(delta, features, base, encoder, seed, device):
    # These are the historical large-runner settings, not newly tuned values.
    cfg = MM.MMConfig(
        text_sources=5, text_chars=320, support_pool_size=32,
        chunk_chars=120, payload_max_chunks=8, payload_max_sources=5,
        payload_graph_weight=0.35, payload_text_weight=0.45,
        payload_prompt_weight=0.20, payload_redundancy=0.20,
        payload_require_lead=True, payload_contrastive_weight=0.35,
        payload_source_repeat_penalty=0.10, seed=seed + 76)
    with preserve_rng():
        attached, audit = XCR.attach_x_conditioned_multi_payload(
            delta=delta, source_features=features, x0=base["x"],
            text0=base["text_emb"], base_texts=base["texts"],
            base_source_ids=base["text_source_ids"], labels=base["labels"],
            encoder=encoder, cfg=cfg, mass=base["mass"], device=device,
            max_slots=3, routing_steps=30, routing_temperature=0.15,
            load_weight=0.03, anchor_weight=0.10,
            candidate_mode="coherent_raw_medoid", adaptive_slots=True,
            balanced_materialization=False, cctp_anchor_swap=False)
    result = replace_payload(base, attached, audit["text_source_ids"])
    audit.update(fixed_content=assert_fixed_content(base, result),
                 payload_config=asdict(cfg), reused_exact_cat_x=True)
    return result, audit


def validate_payload(block, delta, encoder):
    mapping = dict(zip(delta["ids"].tolist(), delta["y"].tolist()))
    if len(block["text_source_ids"]) != len(block["x"]):
        raise AssertionError("text provenance length mismatch")
    for cls, sources in zip(block["labels"].tolist(), block["text_source_ids"]):
        if not sources or any(source not in mapping or mapping[source] != cls
                              for source in sources):
            raise AssertionError("text provenance escaped current class/delta")
    if max(map(len, block["texts"])) > 320:
        raise AssertionError("text character budget exceeded")
    encoded = torch.as_tensor(encoder.encode(block["texts"])).float().cpu()
    # The real encoder returns normalized embeddings; UQ and coherent normalize
    # again, so tolerate only floating point re-encoding differences.
    error = float((encoded - block["text_emb"].float().cpu()).abs().max())
    if error > 1e-4:
        raise AssertionError(f"stored text is not its string embedding: {error}")
    return {"current_delta_text_sources": True, "same_class_text_sources": True,
            "text_chars_max": max(map(len, block["texts"])),
            "text_reencode_max_error": error,
            "unique_texts": len(set(block["texts"]))}


def save_archive(path: Path, value: dict):
    if shutil.disk_usage(path.parent).free < 0.75 * 2**30:
        raise RuntimeError("disk guard: nothing deleted; archive not written")
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def archive_block(output, arm, stage, block, audit, base_arm=None):
    path = output / f"block_{arm}_s{stage}.pt"
    value = {"arm": arm, "stage": stage, "base_arm": base_arm,
             "payload": ({key: block[key] for key in TEXT_KEYS}
                         if base_arm else block),
             "fixed_hashes": {key: R.digest(block[key]) for key in FIXED_KEYS},
             "block_hash": V.memory_hash(block), "audit": audit}
    save_archive(path, value)
    return value


def load_block(output, arm, stage):
    saved = torch.load(output / f"block_{arm}_s{stage}.pt",
                       map_location="cpu", weights_only=False)
    if saved["arm"] != arm or saved["stage"] != stage:
        raise AssertionError("archive identity mismatch")
    if saved["base_arm"]:
        base, _ = load_block(output, saved["base_arm"], stage)
        block = replace_payload(base, saved["payload"],
                                saved["payload"]["text_source_ids"])
    else:
        block = saved["payload"]
    if V.memory_hash(block) != saved["block_hash"]:
        raise AssertionError("archive block hash mismatch")
    if saved["fixed_hashes"] != {k: R.digest(block[k]) for k in FIXED_KEYS}:
        raise AssertionError("archive immutable content mismatch")
    return block, saved["audit"]


def append_block(parent, block):
    memory = L.append_edge_free(parent, block)
    memory["text_source_ids"] = (list(parent["text_source_ids"]) if parent else []) + list(block["text_source_ids"])
    if "support_ids" in block:
        memory["support_ids"] = (list(parent.get("support_ids", [])) if parent else []) + list(block["support_ids"])
    return memory


def legacy_score(memory, data, stage, seeds, device, steps):
    """Historical loss/trainer, but the SAME current inference graph/operator."""
    if memory["edge_index"].numel():
        raise ValueError("legacy loss control is defined only for edge-free memory")
    scores = []
    with preserve_rng():
        for seed in seeds:
            model = L.U.G.train_student(memory, seed, device, steps, lr=.001, batch=24)
            with torch.no_grad():
                pred = model.logits(data["x_device"], data["real_adj"],
                                    data["prompt_emb"].to(device)).argmax(1).cpu()
            scores.append(dict(seed=seed, **AB.score_prediction(pred, data, stage)))
            del model
    result = {key: (None if scores[0][key] is None else
                   float(np.mean([v[key] for v in scores])))
              for key in ("final_f1", "accuracy", "old_f1", "new_f1")}
    result.update(per_student=scores, student_steps=steps, student_batch=24,
                  student_seeds=seeds,
                  final_f1_std=float(np.std([v["final_f1"] for v in scores], ddof=1)) if len(scores) > 1 else 0.0,
                  accuracy_std=float(np.std([v["accuracy"] for v in scores], ddof=1)) if len(scores) > 1 else 0.0)
    return result


def run(args):
    reference = L.read(args.reference)
    groups = V.choose_groups(reference, args.order_seed)
    manifest = dict(groups=groups, report_dev_mod=reference.get("report_dev_mod", 3))
    data = L.load_large(args.dataset, args.minilm, args.text_cache, args.device, manifest)
    ids = L.stage_ids(data)
    budgets, _ = V.choose_budgets(reference, list(map(len, ids)), groups, "fixed_schedule", .05)
    arms = list(PRIMARY) + (list(CONTROLS) if args.text_controls else [])
    cfg = R.q_config("uq_dm_edgefree", args.construction_seed, args.q_steps, 2048)
    source_modules = (R, M, P, UQ, L, V, MM, XCR, AB, R.I, R.I.D, L.U.G)
    protocol = dict(
        name="unified-q-frozen-attribution-20261008", dataset=args.dataset,
        groups=groups, budgets=budgets, budget_mode="fixed_schedule",
        order_seed=args.order_seed, construction_seed=args.construction_seed,
        student=P.model_config(), student_seeds=args.student_seeds,
        student_steps=args.student_steps, student_batch=24, arms=arms,
        legacy_final=args.legacy_final, max_stages=args.max_stages,
        report_mask_hash=V.hash_tensor(data["report_dev_mask"]),
        pool_mask_hash=V.hash_tensor(data["pool_mask"]),
        stage_id_hashes=[V.hash_tensor(v) for v in ids],
        report_dev_mod=manifest["report_dev_mod"],
        inference_operator="shared symmetric normalized full transductive graph",
        inference_candidates="all class prompts, same for both Student losses",
        legacy_control="original row-wise InfoNCE trainer; matched batch/steps/seeds and current inference operator, not full historical-protocol replay",
        controls="post-construction fixed-X attribution; no reoptimization from altered parent text",
        access="each constructor/text adapter sees current delta and own frozen synthetic prefix only",
        historical_raw_reads_by_constructor=0, boundary_enabled=False,
        evaluation_split="report-dev; not sealed test", q_config=asdict(cfg),
        cat_config=asdict(M.Config(seed=args.construction_seed)),
        memory_persistence="lossless incremental blocks and text sidecars; cumulative prefixes reconstructible",
        source_hashes={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                       for m in source_modules},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    protocol = json.loads(json.dumps(protocol))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    result = json.loads(path.read_text()) if path.exists() else dict(protocol=protocol, rows=[], status="running")
    if result["protocol"] != protocol:
        raise ValueError("existing output protocol/source changed; use a fresh directory")
    V.write_json(args.output / "protocol.json", protocol)
    V.write_json(path, result)
    parents = {arm: None for arm in arms}
    for stage, raw_ids in enumerate(ids[:args.max_stages]):
        delta = L.delta_view(data, raw_ids, stage)
        # Historical coherent routing consumes the same public class prompts.
        delta["prompt_emb"] = data["prompt_emb"]
        new_n = budgets[stage] - (budgets[stage - 1] if stage else 0)
        features = None
        blocks = {}
        for arm in arms:
            base_arm = ("cat_cctp" if arm == "adaptive_coherent" or arm.startswith("cat_nearest") or arm == "cat_random5"
                        else "uq_dm_edgefree" if arm.startswith("uq_nearest") else None)
            archive = args.output / f"block_{arm}_s{stage}.pt"
            V.sync(args.device)
            start = time.perf_counter()
            if archive.exists():
                block, audit = load_block(args.output, arm, stage)
            else:
                if arm == "cat_cctp":
                    block, audit = R.make_baseline(delta, new_n, arm, data["encoder"], args.construction_seed, args.device)
                    block["text_source_ids"] = audit["text_source_ids"]
                elif arm == "uq_dm_edgefree":
                    labels = M.row_labels(M.ratio_allocation(delta["y"], new_n))
                    block, audit = UQ.optimize_stage(delta, parents[arm], labels,
                        data["encoder"], data["prompt_emb"], cfg, args.device,
                        use_boundary=False, use_topology=False)
                    checks = R.assignment_checks(audit)
                    if not checks["sinkhorn_feasible"] or audit["text_hard_forward_max_error"] > 1e-4:
                        raise AssertionError(f"primary assignment invariant failed: {checks}")
                    audit["assignment_checks"] = checks
                else:
                    if features is None:
                        features = M.propagated_features(delta, (0.0, 1.0), args.device).cpu()
                    if arm == "adaptive_coherent":
                        block, audit = adaptive_payload(delta, features, blocks[base_arm], data["encoder"], args.construction_seed, args.device)
                    else:
                        block, audit = text_control(delta, features, blocks[base_arm], data["encoder"],
                            "random" if arm == "cat_random5" else "nearest",
                            1 if arm.endswith("1") else 5, args.construction_seed,
                            uq_materialization=arm.startswith("uq_"))
                V.sync(args.device)
                audit["attribution_build_seconds"] = time.perf_counter() - start
                with preserve_rng():
                    audit["payload_validation"] = validate_payload(block, delta, data["encoder"])
                archive_block(args.output, arm, stage, block, audit, base_arm)
            blocks[arm] = block
            parent = parents[arm]
            # Preserve the primary's exact parent representation/API.
            memory = UQ.append_memory(parent, block) if arm == "uq_dm_edgefree" else append_block(parent, block)
            checks = V.validate_memory(memory, parent, budgets[stage])
            if memory["edge_index"].numel():
                raise AssertionError("edge-free memory acquired edges")
            if base_arm:
                checks.update(assert_fixed_content(parents[base_arm], memory))
            families = ["prompt"]
            if args.legacy_final and arm in PRIMARY and stage == len(ids) - 1:
                families.append("legacy_loss_matched")
            for family in families:
                saved = [r for r in result["rows"] if (r["arm"], r["stage"], r["student_family"]) == (arm, stage, family)]
                if saved:
                    if len(saved) != 1 or saved[0]["memory_hash"] != V.memory_hash(memory):
                        raise AssertionError("resumed memory differs from evaluated memory")
                    continue
                V.sync(args.device)
                started = time.perf_counter()
                with preserve_rng():
                    score = (L.score(memory, data, stage, args.student_seeds, args.device, args.student_steps, 24)
                             if family == "prompt" else legacy_score(memory, data, stage, args.student_seeds, args.device, args.student_steps))
                V.sync(args.device)
                row = dict(arm=arm, stage=stage, student_family=family, score=score,
                    memory_rows=budgets[stage], new_rows=new_n, raw_rows=len(raw_ids),
                    memory_hash=V.memory_hash(memory), x_hash=R.digest(memory["x"]),
                    text_hash=R.digest(memory["text_emb"]), edge_hash=R.digest(memory["edge_index"]),
                    checks=checks, audit=R.compact_audit(audit),
                    construction_seconds=audit["attribution_build_seconds"],
                    student_seconds=time.perf_counter() - started,
                    method_class="internal_historical_candidate" if arm == "adaptive_coherent" else
                                 "external_adapter" if arm == "cat_cctp" else
                                 "primary" if arm == "uq_dm_edgefree" else "text_attribution_control")
                result["rows"].append(row)
                V.write_json(path, result)
                print(json.dumps(dict(event="evaluated", dataset=args.dataset, order=args.order_seed,
                    arm=arm, stage=stage, student=family, f1=score["final_f1"], accuracy=score["accuracy"])), flush=True)
            parents[arm] = memory
    result["status"] = "complete" if args.max_stages >= len(ids) else "smoke_complete"
    V.write_json(path, result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("wikics", "sportsfit", "bookhis"), required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--order-seed", type=int, default=9101)
    p.add_argument("--construction-seed", type=int, default=271006)
    p.add_argument("--student-seeds", nargs="+", type=int, default=[150,151,152])
    p.add_argument("--student-steps", type=int, default=2048)
    p.add_argument("--q-steps", type=int, default=30)
    p.add_argument("--text-controls", action="store_true")
    p.add_argument("--legacy-final", action="store_true")
    p.add_argument("--max-stages", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
