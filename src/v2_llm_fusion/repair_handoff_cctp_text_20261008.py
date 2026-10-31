"""Correct the mislabeled SportsFit CCTP payload at exactly fixed archived X.

The erroneous text-center results remain untouched in the original run.
This separate construction writes nearest-5 text using current-delta raw only.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

import run_handoff_student_transfer_20261008 as H

ARM = "cat_cctp_repaired"
SOURCE_ARM = "cat_mmd_tag_cctp_top5"


def new_fixed_block(memory, labels, start):
    if memory["edge_index"].numel():
        raise AssertionError("this fixed-X repair expects edge-free CaT")
    block = {key: memory[key][start:].detach().cpu().clone()
             for key in ("x", "mass", "depth", "stage")}
    block.update(labels=labels[start:].clone(), edge_index=memory["edge_index"].clone())
    return block


def run(args):
    manifest = json.loads(args.manifest.read_text())
    if manifest["dataset"] != "sportsfit":
        raise ValueError("repair only the audited SportsFit mislabel")
    source_rows = [r for r in manifest["rows"] if r["arm"] == SOURCE_ARM]
    if len(source_rows) != 5 or any(r.get("text_mode") != "text_center" for r in source_rows):
        raise AssertionError("original text mode was not the audited text_center")
    data = H.L.load_large("sportsfit", args.minilm, args.text_cache, args.device, manifest)
    if H.tensor_dict_hash({"mask": data["report_dev_mask"]}) != manifest["expected_report_mask"]:
        raise AssertionError("report mask differs")
    if H.sha(args.text_cache / "text_emb_pool_only.pt") != manifest["text_cache_sha256"]:
        raise AssertionError("text cache differs")
    cfg = H.M.Config(seed=271006, text_mode="nearest", text_sources=5, text_chars=320)
    ids = H.L.stage_ids(data)
    protocol = dict(name="handoff-cctp-fixed-X-text-repair-20261008", dataset="sportsfit",
        arms=[ARM], groups=manifest["groups"], budgets=manifest["budgets"],
        student_steps=2048, student_seeds=[150, 151, 152], max_stages=5,
        pool_mask_hash=H.V.hash_tensor(data["pool_mask"]),
        report_mask_hash=H.V.hash_tensor(data["report_dev_mask"]),
        stage_id_hashes=[H.V.hash_tensor(v) for v in ids], config=asdict(cfg),
        source_manifest_sha256=H.sha(args.manifest), runner_sha256=H.sha(__file__),
        sources={Path(m.__file__).name: H.sha(m.__file__) for m in (H, H.M, H.P, H.L.U.G)},
        change="nearest-5 text only; X/mass/depth/stage/labels/topology exactly fixed",
        selection="implementation repair, not chosen using report/test outcomes",
        evaluation="report-dev; two frozen Student protocols; no parameter search")
    protocol = json.loads(json.dumps(protocol))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    result = json.loads(path.read_text()) if path.exists() else dict(protocol=protocol, rows=[], status="running")
    if result["protocol"] != protocol:
        raise ValueError("repair protocol changed on resume")
    H.V.write_json(path, result)
    parent, original_parent = None, None
    for stage, raw_ids in enumerate(ids):
        source = next(r for r in source_rows if r["stage"] == stage)
        delta = dict(H.L.delta_view(data, raw_ids, stage), prompt_emb=data["prompt_emb"])
        original = torch.load(source["memory_path"], map_location="cpu", weights_only=False)
        if H.original_hash(original, source["hash_kind"]) != source["memory_hash"]:
            raise AssertionError("original CaT archive changed")
        old_labels = original_parent["labels"] if original_parent else torch.empty(0, dtype=torch.long)
        labels, label_audit = H.recover_labels(original, source, delta, old_labels)
        original = dict(original, labels=labels)
        H.V.validate_memory(original, original_parent, manifest["budgets"][stage])
        base = new_fixed_block(original, labels, len(old_labels))
        if (args.output / f"block_{ARM}_s{stage}.pt").exists():
            block, audit = H.A.load_block(args.output, ARM, stage)
            H.A.assert_fixed_content(base, block)
        else:
            H.V.sync(args.device)
            start = time.perf_counter()
            with H.A.preserve_rng():
                features = H.M.propagated_features(delta, (0.0, 1.0), args.device)
                attached, audit = H.M.attach_text(delta, features, base["x"], base["labels"], data["encoder"], cfg)
                block = H.A.replace_payload(base, attached, audit["text_source_ids"])
            H.V.sync(args.device)
            audit.update(repair_seconds=time.perf_counter()-start,
                original_memory_hash=source["memory_hash"], original_text_mode="text_center",
                original_build_seconds=source.get("construction_seconds"), historical_raw_reads=0,
                fixed_content=H.A.assert_fixed_content(base, block),
                payload_validation=H.A.validate_payload(block, delta, data["encoder"]), label_recovery=label_audit)
            H.A.archive_block(args.output, ARM, stage, block, audit)
        memory = H.A.append_block(parent, block)
        checks = H.V.validate_memory(memory, parent, manifest["budgets"][stage])
        checks.update(H.A.assert_fixed_content(original, memory))
        digest = H.V.memory_hash(memory)
        for student in ("legacy", "prompt"):
            saved = [r for r in result["rows"] if (r["stage"], r["student"]) == (stage, student)]
            if saved:
                if len(saved) != 1 or saved[0]["memory_hash"] != digest:
                    raise AssertionError("repaired memory changed on resume")
                continue
            H.V.sync(args.device)
            start = time.perf_counter()
            with H.A.preserve_rng():
                score = H.score(memory, data, stage, student, args.device, [150, 151, 152], 2048)
            H.V.sync(args.device)
            source_time = source.get("construction_seconds")
            result["rows"].append(dict(arm=ARM, stage=stage, student=student, score=score,
                memory_hash=digest, memory_rows=len(memory["x"]), checks=checks,
                construction_seconds=None if source_time is None else source_time + audit["repair_seconds"],
                student_seconds=time.perf_counter()-start, repair_seconds=audit["repair_seconds"],
                provenance=dict(kind="fixed-X-payload-repair", source_memory_hash=source["memory_hash"], audit=audit),
                labels_hash=H.V.hash_tensor(memory["labels"]), x_hash=H.V.hash_tensor(memory["x"]),
                text_hash=H.V.hash_tensor(memory["text_emb"]), edge_hash=H.V.hash_tensor(memory["edge_index"]),
                text_chars_max=max(map(len, memory["texts"])), text_chars_total=sum(map(len, memory["texts"])),
                nonself_edges=0))
            H.V.write_json(path, result)
            print(json.dumps(dict(stage=stage, student=student, f1=score["final_f1"], accuracy=score["accuracy"])), flush=True)
        parent, original_parent = memory, original
    result["status"] = "complete"
    H.V.write_json(path, result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:1")
    args = p.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
