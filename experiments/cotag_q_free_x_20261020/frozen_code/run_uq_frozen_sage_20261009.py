"""Evaluate archived confirmation memories without invoking a constructor."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

import run_unified_q_text_confirmation_20261008 as C
sys.path.insert(0, str(Path(__file__).resolve().parent))
import graphtext_sage_transfer as S

L, V = C.L, C.V


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sources(protocol):
    actual = {Path(m.__file__).name: sha256(m.__file__)
              for m in (C.R, C.L, C.V, C.M, C.P, C.U, C.A, C.S, C.H)}
    if actual != protocol["source_hashes"]:
        raise AssertionError("frozen constructor/evaluator dependencies changed")
    if sha256(C.__file__) != protocol["runner_sha256"]:
        raise AssertionError("confirmation runner changed")
    return actual


def check_data(data, protocol):
    if V.hash_tensor(data["pool_mask"]) != protocol["pool_mask_hash"]:
        raise AssertionError("pool mask differs from constructor")
    if V.hash_tensor(data["report_dev_mask"]) != protocol["report_mask_hash"]:
        raise AssertionError("report-dev mask differs from previous readouts")
    if [V.hash_tensor(v) for v in L.stage_ids(data)] != protocol["stage_id_hashes"]:
        raise AssertionError("arrival nodes differ from constructor")


def run(args):
    source_path = args.source / "results.json"
    original = json.loads(source_path.read_text())
    p = original["protocol"]
    if original["status"] != "complete" or len(original["stages"]) != 20 or len(original["rows"]) != 8:
        raise AssertionError("only complete five-stage confirmation cells are accepted")
    check_sources(p)
    protocol = dict(name="uq-frozen-sage-transfer-v1", dataset=p["dataset"],
        order_seed=p["order_seed"], construction_seed=p["construction_seed"],
        source=str(args.source.resolve()), source_result_sha256=sha256(source_path),
        constructor_protocol=p, student=S.model_config(),
        student_steps=p["student_steps"], student_batch=p["student_batch"],
        student_seeds=p["student_seeds"], arms=p["arms"],
        source_hashes={Path(m.__file__).name: sha256(m.__file__) for m in (S, C.P, L, V)},
        runner_sha256=sha256(__file__), construction_repeated=False,
        evaluation="final-memory report-dev, full transductive graph; not sealed test")
    path = args.output / "results.json"
    result = json.loads(path.read_text()) if path.exists() else dict(protocol=protocol, rows=[], status="running")
    if result["protocol"] != protocol:
        raise AssertionError("refusing to resume a different source or Student protocol")
    V.write_json(args.output / "protocol.json", protocol)
    V.write_json(path, result)
    data = L.load_large(p["dataset"], args.minilm, args.text_cache, args.device,
                        dict(groups=p["groups"], report_dev_mod=3))
    check_data(data, p)
    for arm in p["arms"]:
        archive = args.source / f"final_{arm}.pt.gz"
        archive_hash = sha256(archive)
        memory = C.load_memory(archive)
        before = V.memory_hash(memory)
        expected = [r for r in original["rows"] if r["arm"] == arm]
        if len(expected) != 2 or any(r["memory_hash"] != before for r in expected):
            raise AssertionError("archive does not match GCN/SGC memory")
        V.validate_memory(memory, None, p["budgets"][-1])
        if memory["edge_index"].numel():
            raise AssertionError("expected edge-free memory")
        previous = [r for r in result["rows"] if r["arm"] == arm]
        if previous:
            if len(previous) != 1 or previous[0]["archive_sha256"] != archive_hash or previous[0]["memory_hash"] != before:
                raise AssertionError("resume archive mismatch")
            continue
        V.sync(args.device)
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        with C.A.preserve_rng():
            scores = L.score(memory, data, len(p["groups"]) - 1, p["student_seeds"],
                args.device, p["student_steps"], p["student_batch"], trainer=S.train_sage_student)
        V.sync(args.device)
        if V.memory_hash(memory) != before or sha256(archive) != archive_hash:
            raise AssertionError("readout modified input memory")
        result["rows"].append(dict(arm=arm, student="prompt_sage", memory_hash=before,
            archive_sha256=archive_hash, memory_rows=len(memory["x"]),
            archive_bytes=archive.stat().st_size, text_chars=sum(map(len, memory["texts"])),
            unique_texts=len(set(memory["texts"])), score=scores,
            student_seconds=time.perf_counter() - started,
            student_peak_mb=torch.cuda.max_memory_allocated(args.device) / 2**20 if args.device.startswith("cuda") else None,
            checks=dict(same_source_memory=True, data_split_matches=True, memory_unchanged=True)))
        V.write_json(path, result)
        print(json.dumps(dict(event="evaluated", dataset=p["dataset"], order=p["order_seed"],
            seed=p["construction_seed"], arm=arm, f1=scores["final_f1"], accuracy=scores["accuracy"])), flush=True)
    result["status"] = "complete"
    V.write_json(path, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    torch.set_num_threads(2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
