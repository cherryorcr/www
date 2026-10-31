"""P1 teacher-response variance decomposition for TGC-R.

This diagnostic keeps the canonical student and the C0 budget fixed while
changing one construction factor at a time.  It is intentionally separate
from the frozen R8 runner: its report is evidence for the next gate design,
not a new main-method claim.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

import tgc_r6_experiment as E
import run_tgc_r5_joint as V
import tgc_r7_stable as Q

R, T, D, B = E.R, E.T, E.D, E.B

DATASETS = ("cora", "citeseer")
ORDERS = ("order19600", "order19609", "order19610")
STAGES = (1, 2)
REPORT_SEEDS = (310, 311, 312)
TEACHER_G0 = (8800, 8801)
TEACHER_G1 = (8810, 8811)
ARMS = ("c0", "g0", "g1", "init_swap", "batch_swap", "p_initial",
        "text_c0", "topology_c0", "recursive_g0")
TEACHER_SPECS = {
    "g0": (TEACHER_G0, TEACHER_G0),
    "g1": (TEACHER_G1, TEACHER_G1),
    "init_swap": (TEACHER_G1, TEACHER_G0),
    "batch_swap": (TEACHER_G0, TEACHER_G1),
}


def clone_memory(memory):
    return T.clone_memory(memory)


def append_parent(parent, block, device):
    return R.SCA.append_memory(parent, block, len(parent["x"]),
                               topology_mode="frozen_old", cross_degree=1,
                               refresh_mode="frozen_old", device=device)


def _teacher_context(old, delta, base, q, init_seeds, batch_seeds, cfg, device):
    """Build H while decoupling model initialization from batch trajectory."""
    if len(init_seeds) != len(batch_seeds):
        raise ValueError("teacher seed groups must have equal length")
    started = time.perf_counter()
    support = Q.M.lifted_support(old, delta, base, q)
    x = support["x"].to(device)
    text = support["text_emb"].to(device)
    adjacency = Q.M.adjacency(support, "cpu").to_dense().to(device)
    parts, audits = [], []
    old_n = len(old["x"])
    for init_seed, batch_seed in zip(init_seeds, batch_seeds):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(init_seed))
            model = R.G.CLIPStudent(x.shape[1], text_dim=text.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(int(batch_seed) * 7 + 1)
        sequence, losses = [], []
        for _ in range(cfg.teacher_steps):
            ids = Q.M.sample_rows(len(x), old_n, cfg.teacher_batch, generator,
                                  "compact", old_n / len(base["x"]))
            sequence.append(ids)
            loss = Q.T._info_nce(model, x, text, adjacency, ids.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            parts.extend((model.encode_graph(x, adjacency), model.encode_text(text)))
        audits.append({
            "init_seed": int(init_seed),
            "batch_seed": int(batch_seed),
            "steps": int(cfg.teacher_steps),
            "batch_sequence_hash": Q.N.fingerprint({"ids": torch.stack(sequence)}),
            "state_sha256": Q.N.fingerprint(model.state_dict()),
            "loss_first": losses[0],
            "loss_last": losses[-1],
        })
        del model, optimizer
    h = torch.cat(parts, 1).detach() / (len(parts) ** 0.5)
    return {
        "support": support,
        "h": h,
        "adjacency": Q.J.raw_adjacency(support, device).to_dense(),
        "audit": {
            "teachers": audits,
            "h_hash": Q.N.fingerprint(h),
            "support_hash": T.memory_hash(support),
            "parent_hash": T.memory_hash(old),
            "current_delta_ids": delta["current_delta_ids"].tolist(),
            "seconds": time.perf_counter() - started,
            "historical_raw_reads": 0,
            "report_labels_used": False,
        },
    }


def _coarse(ctx, p, old_n):
    u = Q.J.full_assignment(p.to(ctx["h"].device), old_n)
    return (u.T @ (ctx["adjacency"] @ u)).detach().cpu()


def _materialize(ctx, base, delta, p, encoder, cfg):
    coarse = _coarse(ctx, p, len(base["x"]) - p.shape[1])
    memory, audit = Q.K.materialize(base, delta, p, coarse, encoder, cfg)
    return memory, {"coarse_hash": Q.N.fingerprint(coarse), **audit}


def _replace_text(memory, c0, old_n):
    result = clone_memory(memory)
    result["texts"][old_n:] = list(c0["texts"][old_n:])
    result["text_emb"][old_n:] = c0["text_emb"][old_n:].clone()
    return result


def _replace_topology(memory, c0):
    result = clone_memory(memory)
    result["edge_index"] = c0["edge_index"].clone()
    return result


def _component_diff(reference, candidate, old_n):
    return {
        "x_max_abs": float((reference["x"] - candidate["x"]).abs().max()),
        "x_mean_l2": float((reference["x"] - candidate["x"]).norm(dim=1).mean()),
        "text_equal": reference["texts"] == candidate["texts"],
        "text_emb_max_abs": float((reference["text_emb"] - candidate["text_emb"]).abs().max()),
        "edges_equal": bool(torch.equal(reference["edge_index"], candidate["edge_index"])),
        "old_x_equal": bool(torch.equal(reference["x"][:old_n], candidate["x"][:old_n])),
        "old_text_equal": reference["texts"][:old_n] == candidate["texts"][:old_n],
    }


def _build_factor(data, order, stage, ids, budgets, c0, old, arm, args, out):
    old_n = len(old["x"])
    block = E.load(out / f"c0_s{stage}_block.pt")
    delta = T.raw_delta_block(data, ids[stage], stage)
    q = D.current_assignment(data, ids[stage], c0, old_n, stage, args.device)
    base = append_parent(old, block, args.device)
    cfg = V.CFG
    if arm == "recursive_g0":
        init_seeds, batch_seeds = TEACHER_SPECS["g0"]
    else:
        init_seeds, batch_seeds = TEACHER_SPECS.get(arm, TEACHER_SPECS["g0"])
    teacher_cfg = replace(V.TEACHER, teacher_seeds=TEACHER_G0)
    context = _teacher_context(old, delta, base, q, init_seeds, batch_seeds,
                               teacher_cfg, args.device)
    return _finalize_state(data, stage, ids, c0, old, arm, context,
                           q, base, delta, args, out)


def _finalize_state(data, stage, ids, c0, old, arm, context, q, base,
                    delta, args, out):
    """Run P and materialization from an already-fixed teacher context."""
    old_n = len(old["x"])
    cfg = V.CFG
    init_seeds = [item["init_seed"] for item in context["audit"]["teachers"]]
    batch_seeds = [item["batch_seed"] for item in context["audit"]["teachers"]]
    learned, initial, assignment = Q.learn(
        context, q, data["y"][ids[stage]], old_n, cfg)
    p = initial if arm == "p_initial" else learned
    memory, material = _materialize(context, base, delta, p, data["encoder"], cfg)
    if arm == "text_c0":
        memory = _replace_text(memory, c0, old_n)
    elif arm == "topology_c0":
        memory = _replace_topology(memory, c0)
    info = {
        "arm": arm,
        "stage": stage,
        "parent_role": "recursive_g0" if arm == "recursive_g0" else "c0",
        "parent_hash": T.memory_hash(old),
        "base_hash": T.memory_hash(base),
        "delta_hash": T.memory_hash(delta),
        "teacher_init_seeds": list(init_seeds),
        "teacher_batch_seeds": list(batch_seeds),
        "context": context["audit"],
        "assignment": assignment,
        "assignment_sha256": Q.N.fingerprint({"p": p}),
        "p_initial_sha256": Q.N.fingerprint({"p": initial}),
        "p_optimized_sha256": Q.N.fingerprint({"p": learned}),
        "p_max_abs_from_initial": float((learned - initial).abs().max()),
        "materialization": material,
        "memory_sha256": T.memory_hash(memory),
        "historical_raw_reads": 0,
        "report_labels_used": False,
    }
    return memory, info, {"context": context, "q": q, "p": p, "learned": learned,
                          "initial": initial, "base": base, "delta": delta}


def _build_cell(data, order, args, spec):
    ids, budgets = D.set_stream(data, order)
    out = args.output / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    E.R.stable_lock(out / "manifest.json", E.manifest(data, order, budgets, ids))
    all_metrics = []
    for stage in range(3):
        c0, c0_audit = B.build_c0(data, ids[stage], budgets, stage, out, args.device)
        c0_path = out / f"c0_s{stage}_memory.pt"
        if not c0_path.exists():
            torch.save(c0, c0_path)
        if stage == 0:
            for arm in ARMS:
                torch.save(c0, out / f"{arm}_s0_memory.pt")
            continue
        parent_c0 = E.load(out / f"c0_s{stage-1}_memory.pt")
        arm_memories = {"c0": c0}
        arm_infos = {"c0": {"arm": "c0", "stage": stage,
                             "memory_sha256": T.memory_hash(c0),
                             "c0_audit": c0_audit,
                             "historical_raw_reads": 0,
                             "report_labels_used": False}}
        built = {}
        # Build each teacher condition exactly once.  All downstream component
        # controls below reuse the corresponding context and P endpoint.
        for arm in ("g0", "g1", "init_swap", "batch_swap"):
            memory, info, state = _build_factor(
                data, order, stage, ids, budgets, c0, parent_c0, arm,
                args, out)
            arm_memories[arm], arm_infos[arm], built[arm] = memory, info, state

        g0 = built["g0"]
        for arm in ("p_initial", "text_c0", "topology_c0"):
            derived = arm
            memory, info, state = _finalize_state(
                data, stage, ids, c0, parent_c0, derived, g0["context"],
                g0["q"], g0["base"], g0["delta"], args, out)
            arm_memories[arm], arm_infos[arm], built[arm] = memory, info, state
        recursive_parent = (E.load(out / f"recursive_g0_s{stage-1}_memory.pt")
                            if stage > 1 else parent_c0)
        memory, info, state = _build_factor(
            data, order, stage, ids, budgets, c0, recursive_parent,
            "recursive_g0", args, out)
        arm_memories["recursive_g0"], arm_infos["recursive_g0"] = memory, info
        built["recursive_g0"] = state
        reference = built["g0"]
        old_n = len(parent_c0["x"])
        for arm, state in built.items():
            arm_infos[arm]["relative_to_g0"] = {
                "h_hash_equal": state["context"]["audit"]["h_hash"]
                == reference["context"]["audit"]["h_hash"],
                "h_max_abs": float((state["context"]["h"]
                                     - reference["context"]["h"]).abs().max()),
                "p_max_abs": float((state["p"] - reference["p"]).abs().max()),
                **_component_diff(arm_memories["g0"], arm_memories[arm], old_n),
            }
        for arm, memory in arm_memories.items():
            torch.save(memory, out / f"{arm}_s{stage}_memory.pt")
            E.save(out / f"{arm}_s{stage}_build_audit.json", arm_infos[arm])
        E.save(out / f"s{stage}_decision.json", {
            "stage": stage, "report_labels_used": False,
            "historical_raw_reads": 0,
            "arms": {arm: {"memory_sha256": T.memory_hash(memory),
                           "parent_hash": arm_infos[arm].get("parent_hash")}
                     for arm, memory in arm_memories.items()},
        })
        print(json.dumps({"event": "constructed", "dataset": data["dataset"],
                          "order": order, "stage": stage,
                          "arms": list(arm_memories)}), flush=True)
    return out


def _evaluate_cell(data, out, args, spec):
    rows = []
    for stage in STAGES:
        for arm in ARMS:
            memory = E.load(out / f"{arm}_s{stage}_memory.pt")
            per_seed = []
            for seed in REPORT_SEEDS:
                row = E.R.fit(memory, data, stage, seed, arm, out,
                              args.device, None)
                per_seed.append(row)
            aggregate = E.R.C.aggregate(per_seed)
            record = {"dataset": data["dataset"], "order": out.name,
                      "stage": stage, "arm": arm,
                      "memory_sha256": T.memory_hash(memory),
                      "metrics": aggregate, "per_seed": per_seed}
            rows.append(record)
            print(json.dumps({"event": "evaluated", "dataset": data["dataset"],
                              "order": out.name, "stage": stage, "arm": arm,
                              "f1": aggregate["all"]["cumulative_f1"]["mean"]}), flush=True)
    E.save(out / "results.json", {"results": rows})
    return rows


def _write_summary(root, records):
    json_records = []
    for record in records:
        metrics = record["metrics"]["all"]
        json_records.append({"dataset": record["dataset"], "order": record["order"],
                             "stage": record["stage"], "arm": record["arm"],
                             "memory_sha256": record["memory_sha256"],
                             "f1": metrics["cumulative_f1"]["mean"],
                             "f1_std": metrics["cumulative_f1"]["std"],
                             "old_f1": metrics["old_f1"]["mean"],
                             "new_f1": metrics["new_f1"]["mean"]})
    E.save(root / "variance_metrics.json", {"rows": json_records})
    with (root / "variance_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(json_records[0]))
        writer.writeheader()
        writer.writerows(json_records)
    summary = {}
    for row in json_records:
        key = (row["dataset"], row["arm"])
        summary.setdefault(key, []).append(row)
    compact = []
    for (dataset, arm), values in sorted(summary.items()):
        mean_f1 = sum(v["f1"] for v in values) / len(values)
        compact.append({"dataset": dataset, "arm": arm,
                        "cells": len(values), "mean_f1": mean_f1,
                        "mean_old_f1": sum(v["old_f1"] for v in values) / len(values),
                        "mean_new_f1": sum(v["new_f1"] for v in values) / len(values)})
    E.save(root / "variance_summary.json", {"rows": compact})


def protocol():
    return json.loads(json.dumps({
        "name": "tgc-r9-p1-teacher-response-variance-v1",
        "datasets": DATASETS, "orders": ORDERS, "stages": list(STAGES),
        "arms": ARMS, "report_seeds": REPORT_SEEDS,
        "teacher_groups": {"g0": TEACHER_G0, "g1": TEACHER_G1},
        "factorial": {
            "g0": "same init and batch group 0",
            "g1": "same init and batch group 1",
            "init_swap": "init group 1 with group 0 batch trajectory",
            "batch_swap": "init group 0 with group 1 batch trajectory",
            "p_initial": "same g0 H, skip P optimization",
            "text_c0": "same g0 X/topology/P, replace incoming text with C0",
            "topology_c0": "same g0 X/text/P, replace topology with C0",
            "recursive_g0": "same g0 mechanism but stage-2 parent is own stage-1 synthetic memory",
        },
        "budget_rate": .05,
        "student": "canonical two-layer GCN + frozen MiniLM, bidirectional InfoNCE, batch16, Adam .001, scratch",
        "construction_access": "previous synthetic memory + current raw delta only; no historical raw replay",
        "selection": "no arm selected from report F1; all registered arms reported",
        "primary_diagnostics": ["H/P/materialization variance", "student F1 regret relative to g0",
                                "old/new trade-off", "parent recursion effect"],
        "test_labels_evaluated": False,
        "source_hashes": R.source_hashes(),
        "assignment_config": asdict(V.CFG),
        "teacher_config": asdict(V.TEACHER),
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--text-cache-root", type=Path,
                        default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    E.R.stable_lock(args.output / "protocol.json", protocol())
    cache = args.text_cache_root / f"{args.dataset}_r005" / "text_emb_pool_only.pt"
    data = R.R.load_data(args.dataset, args.minilm, args.device, cache)
    all_records = []
    for order in ORDERS:
        out = _build_cell(data, order, args, protocol())
        result_path = out / "results.json"
        if result_path.exists():
            all_records.extend(E.read(result_path)["results"])
        else:
            all_records.extend(_evaluate_cell(data, out, args, protocol()))
    _write_summary(args.output, all_records)
    E.save(args.output / "completion.json", {"status": "complete",
             "dataset": args.dataset, "orders": list(ORDERS),
             "report_seeds": list(REPORT_SEEDS), "test_labels_evaluated": False})
    print(json.dumps({"event": "complete", "dataset": args.dataset,
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
