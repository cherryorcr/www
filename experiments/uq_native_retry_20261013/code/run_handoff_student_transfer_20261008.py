"""Same archived memory under two Student protocols, paired within Student.

Construction is never tuned to either downstream score. External memory stays
unchanged; labels for the prompt trainer are recovered from stage-local saved
provenance or the original TAGSAM class allocation rule, not report labels.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_unified_q_attribution_20261008 as A
import run_unified_q_student_ab_20261007 as AB
import run_public_continual_baselines_large_20261001 as OLD

R, L, V, M, P, U = A.R, A.L, A.V, A.M, A.P, A.UQ


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_dict_hash(memory):
    h = hashlib.sha256()
    for key, value in sorted(memory.items()):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous()
            h.update(str((key, str(value.dtype), tuple(value.shape))).encode())
            h.update(value.numpy().tobytes())
    return h.hexdigest()


def original_hash(memory, kind):
    return OLD.memory_sha256(memory) if kind == "large_baseline_memory" else tensor_dict_hash(memory)


def recover_labels(memory, row, delta, parent_labels):
    count = len(memory["x"]) - len(parent_labels)
    if count <= 0 or count > len(delta["x"]):
        raise AssertionError("invalid new-memory budget for label recovery")
    new_x = memory["x"][-count:]
    ids = delta["ids"].tolist()
    mapping = {int(i): pos for pos, i in enumerate(ids)}
    if row.get("source_ids") is not None:
        source = [int(i) for i in row["source_ids"]]
        if len(source) != count or any(i not in mapping for i in source):
            raise AssertionError("real coreset provenance escapes current delta")
        local = torch.tensor([mapping[i] for i in source])
        if not torch.equal(new_x, delta["x"][local]):
            raise AssertionError("real coreset source order differs from stored X")
        labels = delta["y"][local].long()
        rule = "exact saved source IDs and unchanged raw X"
    elif row.get("row_classes") is not None:
        labels = torch.tensor(row["row_classes"], dtype=torch.long)
        if len(labels) != count:
            raise AssertionError("saved class allocation length mismatch")
        sources = row.get("text_source_ids")
        if sources is None or len(sources) != count:
            raise AssertionError("missing synthetic text provenance")
        for cls, values in zip(labels.tolist(), sources):
            values = [int(i) for i in values]
            if not values or any(i not in mapping or int(delta["y"][mapping[i]]) != cls for i in values):
                raise AssertionError("synthetic label is inconsistent with its source texts")
        rule = "saved constructor row_classes, verified against same-class text sources"
    elif row["arm"] == "tagsam_selfloop":
        # The original wrapper builds sorted class blocks with dispersion
        # allocation. It drops allocations when appending; recover that rule
        # on the legal current delta, never by predicting report/test labels.
        allocation = L.U.allocate_rows_by_class(delta, torch.arange(len(ids)),
            sorted(set(delta["y"].tolist())), count, "dispersion")
        if len(parent_labels) == 0:
            saved = {int(k): int(v) for k, v in memory["allocation"].items()}
            if saved != allocation:
                raise AssertionError("TAGSAM stored base allocation differs from recovery")
        labels = M.row_labels(allocation)
        rule = "original sorted dispersion class allocation, recalculated on current delta; base checked against saved allocation"
    else:
        raise ValueError("cannot recover lawful memory labels")
    if len(labels) != count or not torch.isin(labels, torch.unique(delta["y"])).all():
        raise AssertionError("label recovery violates current stage classes/budget")
    return torch.cat((parent_labels, labels)), dict(rule=rule, new_labels=labels.tolist())


def score(memory, data, stage, student, device, seeds, steps):
    batch = 16 if student == "legacy" else 24
    adjacency = (L.U.G.norm_adj(data["edge_index"], len(data["x"]), device)
                 if student == "legacy" else data["real_adj"])
    records = []
    for seed in seeds:
        model = (L.U.G.train_student(memory, seed, device, steps, lr=.001, batch=batch)
                 if student == "legacy" else P.train_prompt_student(memory, data["prompt_emb"],
                    seed, device, steps, lr=.001, batch=batch))
        with torch.no_grad():
            pred = model.logits(data["x_device"], adjacency, data["prompt_emb"].to(device)).argmax(1).cpu()
        records.append(dict(seed=seed, **AB.score_prediction(pred, data, stage)))
        del model
    value = {key: float(np.mean([r[key] for r in records])) if records[0][key] is not None else None
             for key in ("final_f1", "accuracy", "old_f1", "new_f1")}
    value.update(final_f1_std=float(np.std([r["final_f1"] for r in records], ddof=1)),
                 accuracy_std=float(np.std([r["accuracy"] for r in records], ddof=1)),
                 per_student=records, student_batch=batch, student_steps=steps, student_seeds=seeds)
    return value


def legacy_difference(score_value, source):
    if source is None or "student_seeds" not in source:
        return None
    stored = dict(zip(source["student_seeds"], source["per_student"]))
    return {str(r["seed"]): r["final_f1"] - stored[r["seed"]]["cumulative_macro_f1"]
            for r in score_value["per_student"] if r["seed"] in stored}


def run(args):
    manifest = json.loads(args.manifest.read_text())
    data = L.load_large(args.dataset, args.minilm, args.text_cache, args.device, manifest)
    ids = L.stage_ids(data)
    id_hashes = [V.hash_tensor(v) for v in ids]
    if manifest["expected_stage_ids"] is not None and id_hashes != manifest["expected_stage_ids"]:
        raise AssertionError("handoff stage source IDs differ")
    if manifest["expected_report_mask_kind"] != "small_comparison_tensor_mask":
        raise ValueError("unknown handoff mask hash convention")
    if tensor_dict_hash({"mask": data["report_dev_mask"]}) != manifest["expected_report_mask"]:
        raise AssertionError("handoff report-dev mask changed")
    if sha(args.text_cache / "text_emb_pool_only.pt") != manifest["text_cache_sha256"]:
        raise AssertionError("handoff input text cache changed")
    rows_by_key = {(r["arm"], r["stage"]): r for r in manifest["rows"]}
    arms = ["uq_dm_edgefree"] + list(dict.fromkeys(r["arm"] for r in manifest["rows"]))
    cfg = R.q_config("uq_dm_edgefree", 271006, args.q_steps, 2048)
    protocol = dict(name="handoff-order-two-Student-paired-20261008", dataset=args.dataset,
        groups=manifest["groups"], budgets=manifest["budgets"], arms=arms,
        student_seeds=[150,151,152], student_steps=args.student_steps, students=manifest["students"],
        student_config=P.model_config(), main_construction_seed=271006,
        pool_mask_hash=V.hash_tensor(data["pool_mask"]), report_mask_hash=V.hash_tensor(data["report_dev_mask"]),
        stage_id_hashes=id_hashes, main_config=asdict(cfg), source_manifest_sha256=sha(args.manifest),
        evaluation="report-dev; complete raw graph for inference; all-class prompts",
        budget_scope="equal rows; original text lengths and synthetic topology retained, not equal bytes",
        puma_scope="reconstructed, not reused: original runner saved only scores/hash",
        labels_scope="recover from construction provenance/current-delta allocation only",
        student_scope="GCN loss/training/operator protocols; not a cross-architecture test",
        sources={Path(m.__file__).name:sha(m.__file__) for m in (R,L,V,M,P,U,OLD,L.U.G,AB)},
        runner_sha256=sha(__file__), max_stages=args.max_stages)
    protocol = json.loads(json.dumps(protocol))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    result = json.loads(path.read_text()) if path.exists() else dict(protocol=protocol, rows=[], status="running")
    if result["protocol"] != protocol:
        raise ValueError("protocol changed on resume")
    V.write_json(path, result)
    parents = {a:None for a in arms}
    for stage, raw_ids in enumerate(ids[:args.max_stages]):
        delta = dict(L.delta_view(data, raw_ids, stage), prompt_emb=data["prompt_emb"])
        budget = manifest["budgets"][stage]
        new_n = budget - (manifest["budgets"][stage-1] if stage else 0)
        for arm in arms:
            parent = parents[arm]
            source_row = rows_by_key.get((arm, stage))
            label_audit = None
            if arm in ("uq_dm_edgefree", "puma_full"):
                archive = args.output / f"block_{arm}_s{stage}.pt"
                if archive.exists():
                    block, audit = A.load_block(args.output, arm, stage)
                else:
                    V.sync(args.device)
                    start = time.perf_counter()
                    with A.preserve_rng():
                        if arm == "uq_dm_edgefree":
                            labels = M.row_labels(M.ratio_allocation(delta["y"], new_n))
                            block, audit = U.optimize_stage(delta, parent, labels, data["encoder"],
                                data["prompt_emb"], cfg, args.device, use_boundary=False, use_topology=False)
                        else:
                            block, audit = R.make_baseline(delta, new_n, "puma_tag", data["encoder"], 271006, args.device)
                            block["text_source_ids"] = audit["cat_distribution_matching"]["text_source_ids"]
                    V.sync(args.device)
                    audit["build_seconds"] = time.perf_counter() - start
                    A.archive_block(args.output, arm, stage, block, audit)
                memory = A.append_block(parent, block)
                provenance = dict(kind="new_construction", block_hash=V.memory_hash(block), audit=R.compact_audit(audit))
            else:
                memory_path = Path(source_row["memory_path"])
                memory = torch.load(memory_path, map_location="cpu", weights_only=False)
                digest = original_hash(memory, source_row["hash_kind"])
                if digest != source_row["memory_hash"]:
                    raise AssertionError(f"{arm}/s{stage}: archived memory hash mismatch")
                previous_labels = torch.empty(0,dtype=torch.long) if parent is None else parent["labels"]
                labels, label_audit = recover_labels(memory, source_row, delta, previous_labels)
                memory = dict(memory, labels=labels)
                provenance = dict(kind="reused_sealed_memory", path=str(memory_path), hash=digest,
                    source_json_sha256=source_row["source_sha256"], label_recovery=label_audit)
                audit = dict(build_seconds=source_row.get("construction_seconds"), reused=True)
            checks = V.validate_memory(memory, parent, budget)
            hash_value = V.memory_hash(memory)
            for student in ("legacy", "prompt"):
                saved = [r for r in result["rows"] if (r["arm"],r["stage"],r["student"]) == (arm,stage,student)]
                if saved:
                    if len(saved) != 1 or saved[0]["memory_hash"] != hash_value:
                        raise AssertionError("resume memory changed")
                    continue
                V.sync(args.device)
                start = time.perf_counter()
                with A.preserve_rng():
                    evaluated = score(memory, data, stage, student, args.device, [150,151,152], args.student_steps)
                V.sync(args.device)
                record = dict(arm=arm, stage=stage, student=student, score=evaluated,
                    checks=checks, memory_rows=budget, memory_hash=hash_value,
                    construction_seconds=audit.get("build_seconds"), student_seconds=time.perf_counter()-start,
                    provenance=provenance, labels_hash=V.hash_tensor(memory["labels"]),
                    x_hash=V.hash_tensor(memory["x"]), text_hash=V.hash_tensor(memory["text_emb"]),
                    edge_hash=V.hash_tensor(memory["edge_index"]), text_chars_max=max(map(len,memory["texts"])),
                    text_chars_total=sum(map(len,memory["texts"])), nonself_edges=int((memory["edge_index"][0] != memory["edge_index"][1]).sum()))
                if student == "legacy" and source_row and arm != "puma_full":
                    record["legacy_replay_f1_difference"] = legacy_difference(evaluated, source_row.get("legacy_score"))
                result["rows"].append(record)
                V.write_json(path, result)
                print(json.dumps(dict(event="evaluated", dataset=args.dataset, arm=arm, stage=stage,
                    student=student, f1=evaluated["final_f1"], accuracy=evaluated["accuracy"])), flush=True)
            parents[arm] = memory
    result["status"] = "complete" if args.max_stages == 5 else "smoke_complete"
    V.write_json(path, result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-stages", type=int, default=5)
    p.add_argument("--q-steps", type=int, default=30)
    p.add_argument("--student-steps", type=int, default=2048)
    args = p.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
