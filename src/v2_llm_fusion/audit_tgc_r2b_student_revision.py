"""Reload R2B artifacts, check access contracts and independently score predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_tgc_r2b_student_revision as B
import tgc_student_revision as S

R, T = B.R, B.T
ARMS = ("c0", "teacher") + tuple(
    f"{kind}_k{k}" for k in B.HORIZONS
    for kind in ("student", "stop", "random", "guarded"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def independent_scores(model, data, stage):
    model.eval()
    logits = model.logits(data["x_device"], data["real_adj"],
                          data["prompt_emb"].to(data["x_device"].device))
    logits = logits.cpu()[data["report_dev_mask"]].numpy()
    y = data["y"][data["report_dev_mask"]].numpy()
    groups = data["stage_classes"][:stage + 1]
    classes = [int(c) for group in groups for c in group]
    allowed = np.array(sorted(classes))
    active = np.isin(y, allowed)
    y, logits = y[active], logits[active]
    result = {}
    for scope in ("all", "seen"):
        pred = (logits.argmax(1) if scope == "all"
                else allowed[logits[:, allowed].argmax(1)])
        scores = {}
        for cls in classes:
            tp = np.sum((y == cls) & (pred == cls))
            fp = np.sum((y != cls) & (pred == cls))
            fn = np.sum((y == cls) & (pred != cls))
            scores[cls] = float(200 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.
        old = [scores[int(c)] for g in groups[:-1] for c in g]
        result[scope] = {
            "cumulative_f1": float(np.mean(list(scores.values()))),
            "accuracy": float(np.mean(y == pred) * 100),
            "old_f1": float(np.mean(old)) if old else None,
            "new_f1": float(np.mean([scores[int(c)] for c in groups[-1]])),
        }
    return result


def audit_cell(data, order, args, memo):
    cell = args.root / data["dataset"] / order
    reference = args.reference_root / data["dataset"] / order
    manifest, protocol = read(reference / "manifest.json"), read(cell / "protocol.json")
    require(read(cell / "completion.json")["fit_slots"] == 168, "incomplete cell")
    require(protocol["report_seeds"] == list(B.REPORT_SEEDS), "report seeds changed")
    require(protocol["outer"] == json.loads(json.dumps(S.Config().__dict__)), "outer config changed")
    require(not protocol["labels_used_for_selection"], "label-based selection")
    for name, sha in protocol["source_hashes"].items():
        require(R.C.sha(Path(B.__file__).with_name(name)) == sha, f"source changed: {name}")
    code_root = Path(R.__file__).resolve().parent.parent
    for module in (R.C, R.G, R.P, R.U, R.SCA):
        path = Path(module.__file__).resolve()
        key = path.relative_to(code_root).as_posix()
        require(manifest["protocol"]["source_hashes"][key] == R.C.sha(path),
                f"canonical source differs: {key}")
    require(manifest["source_dataset_sha256"] == R.C.sha(R.U.DATA_CFG[data["dataset"]]["path"]),
            "source dataset changed")
    require(manifest["pool_sha256"] == R.C.tensor_hash({"pool": data["pool_mask"]}), "pool changed")
    require(manifest["report_mask_sha256"] == R.C.tensor_hash({"report": data["report_dev_mask"]}),
            "report mask changed")
    ids, budgets = R.A.set_stream(data, manifest["stage_classes"], .05)
    require([i.tolist() for i in ids] == manifest["stage_source_ids"], "stage source changed")
    require(list(budgets) == manifest["budgets"], "budget changed")
    results = read(cell / "results.json")["results"]
    require(len(results) == 28, "missing results")
    result_map = {(r["stage"], r["arm"]): r for r in results}
    require(len(result_map) == 28, "duplicate results")
    count, executed, maximum, same_memory_repeats = 0, 0, 0., []
    invariants, numerical, gate_records = [], [], []
    for stage in (1, 2):
        base, old = load(reference / f"c0_s{stage}_memory.pt"), load(reference / f"c0_s{stage-1}_memory.pt")
        delta = T.raw_delta_block(data, ids[stage], stage)
        support = T._concat_support(old, delta)
        decision = read(cell / f"s{stage}_decisions.json")
        arms = {a["arm"]: a for a in decision["arms"]}
        require(set(arms) == set(ARMS), "wrong arm matrix")
        require(decision["base_hash"] == T.memory_hash(base), "base changed")
        require(decision["support_hash"] == T.memory_hash(support), "probe access mismatch")
        require(not decision["report_labels_used"], "report leakage flag")
        memories = {"c0": base, "teacher": load(cell / f"teacher_s{stage}_memory.pt")}
        for k in B.HORIZONS:
            opt = arms[f"student_k{k}"]["audit"]["optimization"]
            require(opt["input_memory_sha256"] == T.memory_hash(base), "optimization parent changed")
            require(opt["probe_sha256"] == T.memory_hash(support), "wrong probe")
            require(opt["historical_raw_reads"] == 0 and not opt["report_labels_used"], "access audit failed")
            require(opt["config"] == protocol["outer"] and opt["horizon"] == k, "optimization config changed")
            require(opt["nonzero_gradient_steps"] == 16, "missing hypergradient")
            expected_splits, expected_batches = S.probe_plan(len(old["x"]), len(support["x"]), stage, S.Config())
            expected_splits = {key: {part: value.tolist() for part, value in split.items()}
                               for key, split in expected_splits.items()}
            expected_batches = {part: {key: [value.tolist() for value in batch]
                                      for key, batch in splits.items()}
                                for part, splits in expected_batches.items()}
            require(opt["splits"] == expected_splits and opt["probe_batches"] == expected_batches,
                    "probe split changed")
            selected = min((r for r in opt["curve"] if "validation_objective" in r),
                           key=lambda r: r["validation_objective"])
            require(selected["step"] == opt["selected_outer_step"], "wrong checkpoint selection")
            require(selected["validation_probe"] == opt["selected_validation"], "selected scores differ")
            accepted = S.acceptance(opt["selected_validation"], opt["baseline_probe"]["validation"], S.Config())
            guard = read(cell / f"guarded_k{k}_s{stage}_decision.json")
            require(guard["accepted"] == accepted == opt["accepted"], "gate not reproduced")
            require(guard == arms[f"guarded_k{k}"]["audit"], "guard decision changed")
            stop = arms[f"stop_k{k}"]["audit"]["optimization"]
            require(stop["nonzero_gradient_steps"] == 0, "stop gradient changed X")
            memories[f"student_k{k}"] = load(cell / f"student_k{k}_s{stage}_memory.pt")
            memories[f"random_k{k}"] = load(cell / f"random_k{k}_s{stage}_memory.pt")
            memories[f"stop_k{k}"] = base
            memories[f"guarded_k{k}"] = memories[f"student_k{k}"] if accepted else base
            require(guard["candidate_hash"] == T.memory_hash(memories[f"student_k{k}"]), "guard candidate mismatch")
            require(guard["selected_hash"] == T.memory_hash(memories[f"guarded_k{k}"]), "guard support mismatch")
            torch.testing.assert_close((memories[f"student_k{k}"]["x"] - base["x"]).norm(dim=1),
                                       (memories[f"random_k{k}"]["x"] - base["x"]).norm(dim=1),
                                       atol=1e-5, rtol=1e-5)
            gate_records.append({"stage": stage, "horizon": k, "accepted": accepted})
        for arm in ARMS:
            memory = memories[arm]
            inv = S.invariants(base, memory, decision["editable_rows"], len(old["x"]),
                               decision["protected_rows"], .1)
            require(T.memory_hash(memory) == arms[arm]["memory_hash"], "arm memory changed")
            require(result_map[stage, arm]["memory_sha256"] == T.memory_hash(memory), "result memory mismatch")
            invariants.append({"stage": stage, "arm": arm, **inv})
            actual_rows = []
            for seed in B.REPORT_SEEDS:
                path = cell / f"{arm}_s{stage}_seed{seed}.json"
                row, checkpoint = read(path), path.with_suffix(".pt")
                require(row["support_sha256"] == R.C.tensor_hash(memory), "fit support mismatch")
                require(row["support_memory_sha256"] == T.memory_hash(memory), "fit full memory mismatch")
                require(row["seed"] == seed and row["stage"] == stage, "fit seed/stage mismatch")
                require(row["steps"] == R.P.student_steps(len(memory["x"])), "fit steps changed")
                require(row["effective_batch"] == R.P.STUDENT.batch, "fit batch changed")
                require(row["checkpoint_sha256"] == R.C.sha(checkpoint), "checkpoint damaged")
                saved = load(checkpoint)
                require(row["end_state_sha256"] == R.C.tensor_hash(saved["state"]), "end state hash differs")
                require(saved["result"]["start_state_sha256"] == row["start_state_sha256"], "start state differs")
                c0_row = read(cell / f"c0_s{stage}_seed{seed}.json")
                require(row["start_state_sha256"] == c0_row["start_state_sha256"], "not paired scratch training")
                if row["executed_here"]:
                    require(checkpoint.stat().st_mtime >= (cell / f"s{stage}_decisions.json").stat().st_mtime,
                            "decision saved after report fit")
                else:
                    source = reference / f"c0_s{stage}_seed{seed}.json"
                    require(row["reused_source_sha256"] == R.C.sha(source), "reused record changed")
                    require(row["checkpoint_sha256"] == read(source)["checkpoint_sha256"], "reused checkpoint changed")
                key = (data["dataset"], order, stage, row["end_state_sha256"])
                if key not in memo:
                    model = R.G.CLIPStudent(data["x"].shape[1]).to(args.device)
                    model.load_state_dict(saved["state"])
                    memo[key] = independent_scores(model, data, stage)
                    del model
                for scope, metrics in memo[key].items():
                    for metric, value in metrics.items():
                        expected = row["metrics"][scope][metric]
                        diff = 0. if value is None and expected is None else abs(value - expected)
                        maximum = max(maximum, diff)
                        require(diff <= 1e-7, f"prediction mismatch: {path.name}/{scope}/{metric}: {diff}")
                if arm.startswith(("stop_", "guarded_")):
                    src = "c0" if arm.startswith("stop_") else (
                        "student_" + arm.split("_", 1)[1] if arms[arm]["audit"]["accepted"] else "c0")
                    original = read(cell / f"{src}_s{stage}_seed{seed}.json")
                    require(row["support_sha256"] == original["support_sha256"], "policy support differs")
                    same_memory_repeats.append({"stage": stage, "arm": arm, "seed": seed, "source": src,
                        "equal_end_state": row["end_state_sha256"] == original["end_state_sha256"],
                        "delta_f1_pp": row["metrics"]["all"]["cumulative_f1"] - original["metrics"]["all"]["cumulative_f1"]})
                actual_rows.append(row)
                count += 1
                executed += int(row["executed_here"])
            require(actual_rows == result_map[stage, arm]["per_seed"], "summary per_seed mismatch")
            require(R.C.aggregate(actual_rows) == result_map[stage, arm]["metrics"], "summary aggregate mismatch")
        if order == "canonical" and stage == 1:
            numerical = S.numerical_check(base, 7500, args.device)
    return {"dataset": data["dataset"], "order": order, "records_checked": count,
            "executed_fits": executed, "reused_records": count - executed,
            "max_metric_difference_pp": maximum, "invariants": invariants,
            "gate_records": gate_records, "same_memory_repeats": same_memory_repeats,
            "numerical_adam_checks": numerical}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dataset", nargs="+", default=["cora", "citeseer"])
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--text-cache-root", type=Path,
                    default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    cells, errors, memo = [], [], {}
    for dataset in args.dataset:
        data = R.R.load_data(dataset, args.minilm, args.device,
                            args.text_cache_root / f"{dataset}_r005" / "text_emb_pool_only.pt")
        for order in B.ORDERS:
            try:
                result = audit_cell(data, order, args, memo)
                cells.append(result)
                print(json.dumps({"dataset": dataset, "order": order,
                                  "records": result["records_checked"]}), flush=True)
            except Exception as error:
                errors.append({"dataset": dataset, "order": order, "error": repr(error)})
    count = sum(c["records_checked"] for c in cells)
    result = {"passed": not errors and count == len(args.dataset) * 336,
              "records_checked": count, "unique_state_evaluations": len(memo), "cells": cells,
              "errors": errors, "test_labels_evaluated": False,
              "scoring": "independent per-class TP/FP/FN; report-dev only",
              "metric_tolerance_pp": 1e-7}
    R.save_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
