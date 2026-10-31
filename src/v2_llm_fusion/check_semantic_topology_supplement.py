"""Post-result degree/storage checks and internal-operator microbenchmarks.

No student fitting or selection. Frozen source and result files are read-only.
Timing includes internal kernel/edge selection, not the full condensation run.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from statistics import median

import torch

import run_semantic_topology_controls as R
import summarize_topology_causal_ablation as S

T = R.R


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def degrees(memory):
    return torch.bincount(memory["edge_index"][0], minlength=len(memory["x"]))


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    root, baseline = Path(args.root), Path(args.baseline_root)
    repo = Path(R.__file__).resolve().parents[2]
    manifest = S.read_json(root / "manifest.json")
    for name, digest in manifest["sources"].items():
        assert T.sha(repo / name) == digest, name
    structural, timings = [], []
    for order, groups in manifest["orders"].items():
        folder, source = root / f"order_{order}", baseline / f"order_{order}"
        meta = S.read_json(folder / "construction.json")
        base_meta = S.read_json(source / "construction.json")
        for stage in range(len(groups)):
            memories = {}
            for arm in ("c0", "internal_only", "t_degree"):
                p = source / base_meta[stage]["arms"][arm]["path"]
                assert T.sha(p) == base_meta[stage]["arms"][arm]["sha256"]
                memories[arm] = load(p)
            base, keep = memories["c0"], meta[stage]["keep"]
            block = {key: base[key][keep:] for key in R.FIELDS}
            block["texts"] = base["texts"][keep:]
            n = len(block["x"])
            for arm in R.ARMS:
                p = folder / meta[stage]["arms"][arm]["path"]
                assert T.sha(p) == meta[stage]["arms"][arm]["sha256"]
                mem = load(p)
                memories[arm] = mem
                match = "t_degree" if arm.endswith("capacity") else "internal_only"
                internal = T.components(mem["edge_index"], keep, len(mem["x"]))[1] - keep
                internal_degree = torch.bincount(internal.flatten(), minlength=n)
                same_degree = torch.equal(degrees(mem), degrees(memories[match]))
                same_cross = torch.equal(T.components(mem["edge_index"], keep, len(mem["x"]))[2],
                                         T.components(memories[match]["edge_index"], keep, len(mem["x"]))[2])
                assert same_cross
                if arm.startswith("cycles"):
                    assert same_degree
                    assert bool((internal_degree == (2 if n >= 3 else n - 1)).all())
                structural.append(dict(dataset=manifest["dataset"], order=int(order), stage=stage,
                                       arm=arm, matched_baseline=match, same_full_degree=same_degree,
                                       same_current_cross=same_cross,
                                       max_new_internal_degree=int(internal_degree.max()),
                                       memory_file_bytes=p.stat().st_size,
                                       tensor_bytes=sum(v.numel() * v.element_size() for v in mem.values()
                                                        if isinstance(v, torch.Tensor)),
                                       text_utf8_bytes=sum(len(s.encode("utf-8")) for s in mem["texts"])))
            operations = {
                "c0": lambda: R.U.SCA.memory_edges(block, device=args.device),
                "single_ring": lambda: R.B.select_internal(block, T.T_WEIGHTS, "degree_balanced"),
                "knn": lambda: T.undirected(R.semantic_internal(block, "knn")[0]),
                "cycles": lambda: T.undirected(R.semantic_internal(block, "cycles")[0]),
            }
            matches = dict(c0="c0", single_ring="t_degree", knn="knn_nearest", cycles="cycles_nearest")
            values = {name: [] for name in operations}
            # One warmup plus rotated measured runs on exactly the same block.
            for repeat in range(args.repeats + 1):
                names = list(operations)
                shift = (stage + repeat) % len(names)
                for name in names[shift:] + names[:shift]:
                    R.L.sync(args.device)
                    start = time.perf_counter()
                    edge = operations[name]()
                    R.L.sync(args.device)
                    elapsed = time.perf_counter() - start
                    expected = T.components(memories[matches[name]]["edge_index"], keep, len(base["x"]))[1] - keep
                    assert torch.equal(T.unique_pairs(edge, n), expected), (stage, name)
                    if repeat:
                        values[name].append(elapsed)
            for name, samples in values.items():
                timings.append(dict(dataset=manifest["dataset"], order=int(order), stage=stage,
                                    operator=name, new_rows=n, median_seconds=median(samples),
                                    seconds=samples, repeats=len(samples)))
            print(json.dumps(dict(event="stage_checked", dataset=manifest["dataset"], order=order,
                                  stage=stage, new_rows=n)), flush=True)
    minilm = Path(manifest["minilm"])
    result = dict(passed=True, dataset=manifest["dataset"], memory_checks=len(structural),
                  cycle_degree_checks=sum(r["arm"].startswith("cycles") for r in structural),
                  structural=structural, internal_microbenchmark=timings,
                  source_sha256=T.sha(__file__), frozen_sources_checked=manifest["sources"],
                  hardware=dict(cpu=platform.processor(), gpu=torch.cuda.get_device_name(args.device),
                                torch=str(torch.__version__), cpu_threads=4),
                  minilm_files_sha256_after_run={str(p.relative_to(minilm)): T.sha(p)
                                                for p in minilm.rglob("*") if p.is_file()},
                  limits=["Post-result diagnostic, not an additional student experiment.",
                          "Internal topology only: excludes cross edges, X/text construction, I/O and student.",
                          "MiniLM hashes recorded after training, not retrospectively claimed pre-registered."])
    R.save_json(root / "supplemental_topology_checks.json", result)
    S.write_csv(root / "internal_topology_microbenchmark.csv", timings)
    S.write_csv(root / "memory_storage_checks.csv", structural)
    print(json.dumps(dict(event="DONE", dataset=manifest["dataset"], passed=True,
                          memories=len(structural))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--baseline-root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    if a.repeats < 1:
        p.error("repeats must be positive")
    main(a)
