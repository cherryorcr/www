"""P-only Student-guided text assignment experiment.

The runner constructs one shared CaT X block per stage and evaluates the same
fixed-X CCTP baseline, a graph-conditioned hard assignment, and graph-
conditioned assignments optimized with one or three differentiable Student
unroll steps.  No old synthetic row, old edge, or historical raw node is
modified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import torch

import external_continual_mechanisms as M
import run_semantic_recursive_unified_baselines_20260926 as BASE
import run_external_continual_adapters_20260927 as E
import student_guided_text_assignment as P


ARMS = ("cctp_top3", "p_graph", "p_graph_unroll1", "p_graph_unroll3",
        "p_projection_aware", "random_text")
P_UNROLLS = {"p_graph": 0, "p_graph_unroll1": 1, "p_graph_unroll3": 3,
             "p_projection_aware": 0}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                         encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol(args, cat_cfg: M.Config, p_cfg: P.PConfig) -> dict:
    paths = {Path(__file__).name: __file__,
             Path(M.__file__).name: M.__file__,
             Path(P.__file__).name: P.__file__,
             "student.py": BASE.G.__file__,
             "evaluation.py": BASE.__file__}
    return {
        "name": "student-guided-graph-conditioned-p-only-20261001",
        "status": "P-only selector; fixed CaT X, fixed original Student, hard top-k provenance",
        "datasets": args.datasets,
        "orders": args.orders,
        "arms": list(args.arms),
        "report_seeds": args.report_seeds,
        "config": {"cat": asdict(cat_cfg), "p": asdict(p_cfg)},
        "student_steps": "max(50,min(2048,16*memory_rows))",
        "student_batch": 16,
        "student": "same original 2-layer GCN/text projection, bidirectional InfoNCE, scratch",
        "memory": "same 5% cumulative legal-pool rows; old prefix append-only",
        "construction_access": "only current delta enters X, P candidates, and text provenance",
        "selector": "row-wise masked P over same-class current-delta text sources",
        "graph_condition": "current-delta propagation feature + fixed synthetic X + class prompt",
        "unroll": "1 or 3 differentiable SGD steps on original Student InfoNCE",
        "projection": "hard top-k current-delta sources, fixed characters, frozen MiniLM re-encoding",
        "topology": "fixed inner topology from initial soft assignment; final Student uses normal memory materialization",
        "evaluation": "all-prompt seen-node report-dev Macro-F1, Accuracy, old/new F1",
        "report_driven_selection": False,
        "source_hashes": {k: file_hash(Path(v)) for k, v in paths.items()},
    }


def prepare_cell(data: dict, order: str, args, cat_cfg: M.Config,
                 p_cfg: P.PConfig) -> None:
    root = args.output_root / data["dataset"] / order
    root.mkdir(parents=True, exist_ok=True)
    if (root / "seal.json").exists():
        verify_cell(root, data, order, args)
        return

    ids, budgets = BASE.set_stream(data, order)
    groups = data["stage_classes"]
    parents = {arm: None for arm in args.arms}
    records = []
    shared_x_hashes = {}

    for stage in range(3):
        E.sync(args.device)
        stage_start = time.perf_counter()
        delta = E.delta_view(data, ids[stage], stage)
        old_rows = 0 if stage == 0 else len(next(iter(parents.values()))["x"])
        new_rows = budgets[stage] - old_rows
        if new_rows < 1:
            raise ValueError(f"stage {stage} has no new memory rows")

        x_start = time.perf_counter()
        x_block, x_audit = M.cat_block(
            delta, new_rows, data["encoder"],
            replace(cat_cfg, text_mode="nearest"), args.device)
        E.sync(args.device)
        x_seconds = time.perf_counter() - x_start
        shared_x = x_block["x"].detach().clone()
        shared_x_hash = hashlib.sha256(
            shared_x.contiguous().numpy().tobytes()).hexdigest()
        shared_x_hashes[str(stage)] = shared_x_hash
        source_features = M.propagated_features(
            delta, (0.0, 1.0), args.device)
        labels = torch.tensor(x_audit["row_classes"], dtype=torch.long)

        for arm in args.arms:
            start = time.perf_counter()
            if arm == "cctp_top3":
                arm_cfg = replace(cat_cfg, text_mode="text_center",
                                  text_sources=3)
                block, audit = M.attach_text(
                    delta, source_features, shared_x, labels,
                    data["encoder"], arm_cfg)
            elif arm == "random_text":
                arm_cfg = replace(cat_cfg, text_mode="random",
                                  text_sources=3)
                block, audit = M.attach_text(
                    delta, source_features, shared_x, labels,
                    data["encoder"], arm_cfg)
            else:
                arm_p_cfg = replace(p_cfg, unroll_steps=P_UNROLLS[arm])
                if arm == "p_projection_aware":
                    block, audit = P.build_projection_aware_block(
                        delta, source_features, shared_x, labels,
                        parents[arm], data["encoder"], arm_p_cfg, BASE.G,
                        args.device)
                else:
                    block, audit = P.build_p_only_block(
                        delta, source_features, shared_x, labels,
                        parents[arm], data["encoder"], arm_p_cfg, BASE.G,
                        args.device)
            memory = M.append_edge_free(parents[arm], block)
            E.check_memory(memory, parents[arm], budgets[stage], edge_free=True)
            elapsed = time.perf_counter() - start
            audit.update(
                shared_x_hash=shared_x_hash,
                shared_x_source="one CaT X optimization shared by all P arms",
                x_optimizer_seconds=float(x_seconds),
                construction_seconds_accounted=float(elapsed + x_seconds),
                source_x_audit=x_audit,
            )
            path = root / f"{arm}_s{stage}_memory.pt"
            torch.save(memory, path)
            records.append({
                "arm": arm,
                "stage": stage,
                "rows": int(budgets[stage]),
                "memory_file": path.name,
                "memory_sha256": E.memory_hash(memory),
                "file_sha256": file_hash(path),
                "delta_ids": ids[stage].tolist(),
                "parent_sha256": E.memory_hash(parents[arm])
                if parents[arm] else None,
                "construction_seconds": float(elapsed + x_seconds),
                "artifact_load_or_construct_seconds": float(
                    time.perf_counter() - stage_start),
                "audit": audit,
                "resources": E.resource_stats(memory, data["encoder"]),
            })
            parents[arm] = memory
            print(json.dumps({"event": "constructed",
                              "dataset": data["dataset"],
                              "order": order, "stage": stage,
                              "arm": arm, "seconds": elapsed}), flush=True)

    save(root / "construction.json", {
        "stage_classes": groups,
        "budgets": budgets,
        "records": records,
        "shared_x_hashes": shared_x_hashes,
    })
    save(root / "seal.json", {
        "construction_sha256": file_hash(root / "construction.json"),
        "protocol_sha256": file_hash(args.output_root / "protocol.json"),
    })


def verify_cell(root: Path, data: dict, order: str, args) -> dict:
    seal = read(root / "seal.json")
    if seal["construction_sha256"] != file_hash(root / "construction.json"):
        raise AssertionError("construction seal changed")
    if seal["protocol_sha256"] != file_hash(args.output_root / "protocol.json"):
        raise AssertionError("protocol seal changed")
    record = read(root / "construction.json")
    previous = {arm: None for arm in args.arms}
    for item in record["records"]:
        path = root / item["memory_file"]
        memory = BASE.load(path)
        if file_hash(path) != item["file_sha256"] or E.memory_hash(memory) != item["memory_sha256"]:
            raise AssertionError("memory artifact changed after seal")
        parent = previous[item["arm"]]
        E.check_memory(memory, parent, item["rows"], edge_free=True)
        delta_ids = set(item["delta_ids"])
        for source_ids in item["audit"].get("text_source_ids", []):
            if not set(source_ids).issubset(delta_ids):
                raise AssertionError("P/text provenance left current delta")
        previous[item["arm"]] = memory
    return record


def evaluate_cell(data: dict, order: str, args) -> None:
    root = args.output_root / data["dataset"] / order
    construction = verify_cell(root, data, order, args)
    BASE.set_stream(data, order)
    results = []
    for item in construction["records"]:
        memory = BASE.load(root / item["memory_file"])
        steps = BASE.protocol_v3.student_steps(
            len(memory["x"]), override=args.student_steps)
        scores = []
        for seed in args.report_seeds:
            path = root / f"{item['arm']}_s{item['stage']}_seed{seed}.json"
            if path.exists():
                result = read(path)
                if result["memory_sha256"] != item["memory_sha256"]:
                    raise AssertionError("resumed evaluation memory mismatch")
            else:
                E.sync(args.device)
                started = time.perf_counter()
                model = BASE.G.train_student(
                    memory, int(seed), args.device, steps, lr=0.001, batch=16)
                E.sync(args.device)
                seconds = time.perf_counter() - started
                result = {
                    "seed": int(seed),
                    "steps": int(steps),
                    **BASE.score_model(model, data, data["stage_classes"],
                                       item["stage"]),
                    "student_seconds": float(seconds),
                    "memory_sha256": item["memory_sha256"],
                }
                checkpoint = path.with_suffix(".pt")
                torch.save({"state": {k: v.detach().cpu()
                                       for k, v in model.state_dict().items()}},
                           checkpoint)
                result["checkpoint_sha256"] = file_hash(checkpoint)
                save(path, result)
                del model
            scores.append(result)
        aggregate = BASE.aggregate_stage_scores(scores)
        results.append({
            "dataset": data["dataset"],
            "order": order,
            "arm": item["arm"],
            "stage": item["stage"],
            "memory_rows": item["rows"],
            "aggregate": aggregate,
            "per_student": scores,
            "student_wall_seconds": float(sum(
                s["student_seconds"] for s in scores)),
            "construction_seconds": item["construction_seconds"],
            "resources": item["resources"],
        })
        save(root / "progress.json", {"rows": results})
        print(json.dumps({"event": "evaluated", "dataset": data["dataset"],
                          "order": order, "arm": item["arm"],
                          "stage": item["stage"],
                          "f1": aggregate["mean_f1"]}), flush=True)
    save(root / "results.json", {
        "protocol": read(args.output_root / "protocol.json"),
        "rows": results,
    })


def audit_cell(data: dict, order: str, args) -> None:
    root = args.output_root / data["dataset"] / order
    construction = verify_cell(root, data, order, args)
    ids, budgets = BASE.set_stream(data, order)
    fit_count, maximum = 0, 0.0
    for item in construction["records"]:
        if item["delta_ids"] != ids[item["stage"]].tolist():
            raise AssertionError("delta source IDs changed")
        if item["rows"] != budgets[item["stage"]]:
            raise AssertionError("memory budget changed")
        memory = BASE.load(root / item["memory_file"])
        for seed in args.report_seeds:
            path = root / f"{item['arm']}_s{item['stage']}_seed{seed}.json"
            checkpoint = path.with_suffix(".pt")
            result = read(path)
            if result["checkpoint_sha256"] != file_hash(checkpoint):
                raise AssertionError("checkpoint changed")
            model = BASE.G.CLIPStudent(memory["x"].shape[1]).to(args.device)
            model.load_state_dict(BASE.load(checkpoint)["state"])
            model.eval()
            score = BASE.score_model(model, data, data["stage_classes"],
                                     item["stage"])
            for key in ("cumulative_f1", "accuracy", "old_f1", "new_f1"):
                if score[key] is not None and result[key] is not None:
                    maximum = max(maximum, abs(score[key] - result[key]))
            fit_count += 1
            del model
    if maximum > 1e-7:
        raise AssertionError(f"checkpoint metric mismatch: {maximum}")
    save(root / "audit.json", {
        "passed": True,
        "fit_count": fit_count,
        "maximum_metric_error_pp": maximum,
        "scope": "P provenance, fixed-X hash, append-only prefix, budget, checkpoint reload",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--orders", nargs="+",
                        default=["order19611", "order19612"])
    parser.add_argument("--arms", nargs="+", choices=ARMS,
                        default=list(ARMS))
    parser.add_argument("--report-seeds", nargs="+", type=int,
                        default=list(range(150, 156)))
    parser.add_argument("--seed", type=int, default=270927)
    parser.add_argument("--cat-steps", type=int, default=3000)
    parser.add_argument("--p-outer-steps", type=int, default=16)
    parser.add_argument("--p-outer-lr", type=float, default=0.03)
    parser.add_argument("--p-attention-temperature", type=float, default=0.15)
    parser.add_argument("--p-text-sources", type=int, default=3)
    parser.add_argument("--p-unroll-lr", type=float, default=1e-3)
    parser.add_argument("--p-student-seed", type=int, default=90401)
    parser.add_argument("--p-anchor-blend", type=float, default=0.95)
    parser.add_argument("--p-anchor-strength", type=float, default=8.0)
    parser.add_argument("--p-max-anchor-shift", type=float, default=0.20)
    parser.add_argument("--p-lambda-anchor", type=float, default=0.50)
    parser.add_argument("--p-projection-candidate-pool", type=int, default=8)
    parser.add_argument("--p-projection-unroll-steps", type=int, default=-1)
    parser.add_argument("--p-projection-min-gain", type=float, default=0.0)
    parser.add_argument("--p-projection-lambda-prompt", type=float, default=0.50)
    parser.add_argument("--p-projection-min-cosine", type=float, default=0.98)
    parser.add_argument("--p-projection-max-accepts", type=int, default=1)
    parser.add_argument("--p-projection-disable-prompt-gate", action="store_true")
    parser.add_argument("--student-steps", type=int)
    parser.add_argument("--phase", choices=("all", "prepare", "evaluate", "audit"),
                        default="all")
    args = parser.parse_args()
    if set(args.arms) - set(ARMS):
        raise ValueError("unknown P selector arm")
    torch.set_num_threads(2)
    cat_cfg = M.Config(seed=args.seed, cat_steps=args.cat_steps)
    p_cfg = P.PConfig(text_sources=args.p_text_sources,
                      attention_temperature=args.p_attention_temperature,
                      outer_steps=args.p_outer_steps,
                      outer_lr=args.p_outer_lr,
                      student_lr=args.p_unroll_lr,
                      student_seed=args.p_student_seed,
                      anchor_blend=args.p_anchor_blend,
                      anchor_strength=args.p_anchor_strength,
                      max_anchor_shift=args.p_max_anchor_shift,
                      lambda_anchor=args.p_lambda_anchor,
                      projection_candidate_pool=args.p_projection_candidate_pool,
                      projection_unroll_steps=args.p_projection_unroll_steps,
                      projection_min_gain=args.p_projection_min_gain,
                      projection_lambda_prompt=args.p_projection_lambda_prompt,
                      projection_min_cosine=args.p_projection_min_cosine,
                      projection_max_accepts=args.p_projection_max_accepts,
                      projection_require_prompt_gain=not args.p_projection_disable_prompt_gate)
    manifest = protocol(args, cat_cfg, p_cfg)
    protocol_path = args.output_root / "protocol.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    if protocol_path.exists() and read(protocol_path) != manifest:
        raise ValueError("existing output root has a different protocol")
    if not protocol_path.exists():
        save(protocol_path, manifest)

    data = {name: BASE.load_data(name, args.minilm, args.device, args.cache_root)
            for name in args.datasets}
    for value in data.values():
        value["prompt_emb"] = value["encoder"].encode(value["prompts"])

    phases = ("prepare", "evaluate", "audit") if args.phase == "all" else (args.phase,)
    for phase in phases:
        for name in args.datasets:
            for order in args.orders:
                if phase == "prepare":
                    prepare_cell(data[name], order, args, cat_cfg, p_cfg)
                elif phase == "evaluate":
                    evaluate_cell(data[name], order, args)
                else:
                    audit_cell(data[name], order, args)
    if args.phase == "all":
        rows = [
            row
            for name in args.datasets
            for order in args.orders
            for row in read(args.output_root / name / order / "results.json")["rows"]
        ]
        save(args.output_root / "results.json", {
            "protocol": manifest,
            "rows": rows,
        })
    print("STUDENT_GUIDED_P_SELECTOR_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

