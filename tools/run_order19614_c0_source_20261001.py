"""Build the sealed three-stage C0 source for order19614.

This runner is deliberately limited to the common C0 constructor.  It does
not run X+P, response selection, semantic-recursive construction, or any
evaluation.  The resulting source is consumed by the frozen small-graph
evaluation runners.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
CODE_ROOT = HERE.parents[1] / "code" / "v2_llm_fusion"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# Several historical snapshots contain a module with the same name.  Pin the
# dependency before importing the runners so all append/topology helpers use
# the current v2 implementation.
_gate_spec = importlib.util.spec_from_file_location(
    "run_gate1_continual", CODE_ROOT / "run_gate1_continual.py")
if _gate_spec is None or _gate_spec.loader is None:
    raise ImportError("cannot load current run_gate1_continual")
_gate_module = importlib.util.module_from_spec(_gate_spec)
sys.modules["run_gate1_continual"] = _gate_module
_gate_spec.loader.exec_module(_gate_module)

import run_semantic_recursive_unified_baselines_20260926 as B
import run_student_compatible_append as SCA
import run_unified_v3_continual as U


ORDER = "order19614"
DATASETS = ("cora", "citeseer")


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True),
                    encoding="utf-8")


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_hash(memory: dict) -> str:
    digest = hashlib.sha256()
    for key in ("x", "text_emb", "mass", "depth", "stage", "edge_index"):
        value = memory[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    for text in memory["texts"]:
        encoded = str(text).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def append_c0(previous: dict | None, block: dict) -> dict:
    """Materialize the frozen-old C0 append without legacy shrink helpers."""
    if previous is None:
        return {
            key: (value.clone() if isinstance(value, torch.Tensor)
                  else list(value) if isinstance(value, list) else value)
            for key, value in block.items()
        }
    out = {}
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        out[key] = torch.cat((previous[key], block[key]))
    out["texts"] = list(previous["texts"]) + list(block["texts"])
    out["edge_index"] = SCA.append_topology(
        previous, block, cross_degree=1, device="cpu")
    return out


def build_dataset(dataset: str, args: argparse.Namespace) -> dict:
    data = B.load_data(dataset, args.minilm, args.device, args.cache_root)
    ids_by_stage, budgets = B.set_stream(data, ORDER)
    out = args.output_root / dataset / ORDER
    if out.exists() and (out / "manifest.json").exists():
        existing = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("stage_classes") != data["stage_classes"]:
            raise AssertionError("existing source has different class groups")
        for stage in range(3):
            if not (out / f"c0_s{stage}_memory.pt").exists():
                raise FileNotFoundError(out / f"c0_s{stage}_memory.pt")
        return existing
    if out.exists():
        raise FileExistsError(f"partial source exists: {out}")
    out.mkdir(parents=True, exist_ok=False)

    previous = None
    records = []
    for stage, (ids, budget) in enumerate(zip(ids_by_stage, budgets)):
        old_rows = 0 if previous is None else len(previous["x"])
        new_rows = int(budget - old_rows)
        started = time.perf_counter()
        block = U.build_stage_block(
            data, ids, new_rows, data["encoder"], stage,
            "joint_g5_c320", "concat", args.device, "dispersion",
            data["stage_classes"][stage],
        )
        # Materialize the C0 text topology before the legacy append helper.
        # This avoids its version-dependent implicit fallback while keeping
        # the registered text-kNN edge construction unchanged.
        block["edge_index"] = SCA.text_knn_edges(
            block["text_emb"], U.protocol_v3.DEGREE)
        memory = append_c0(previous, block)
        if len(memory["x"]) != int(budget):
            raise AssertionError("C0 budget mismatch")
        if previous is not None:
            for key in ("x", "text_emb", "mass", "depth", "stage"):
                if not torch.equal(memory[key][:old_rows], previous[key]):
                    raise AssertionError(f"old prefix changed: {key}")
            if memory["texts"][:old_rows] != previous["texts"]:
                raise AssertionError("old text prefix changed")
            old_edges = memory["edge_index"][:,
                (memory["edge_index"] < old_rows).all(0)]
            if not torch.equal(old_edges, previous["edge_index"]):
                raise AssertionError("old topology changed")
        memory_path = out / f"c0_s{stage}_memory.pt"
        torch.save(memory, memory_path)
        block_path = out / f"c0_s{stage}_block.pt"
        torch.save(block, block_path)
        records.append({
            "stage": stage,
            "classes": [int(v) for v in data["stage_classes"][stage]],
            "source_ids": [int(v) for v in ids.tolist()],
            "old_rows": old_rows,
            "new_rows": new_rows,
            "budget": int(budget),
            "memory_file": memory_path.name,
            "block_file": block_path.name,
            "memory_sha256": sha256_file(memory_path),
            "memory_hash": memory_hash(memory),
            "construction_seconds": time.perf_counter() - started,
            "historical_raw_reads": 0,
            "report_test_labels_used_for_construction": False,
        })
        previous = memory

    manifest = {
        "name": "c0-order19614-source-v1",
        "dataset": dataset,
        "order": ORDER,
        "stage_classes": [[int(v) for v in group]
                          for group in data["stage_classes"]],
        "stage_source_ids": [ids.tolist() for ids in ids_by_stage],
        "stage_train_counts": [int(len(ids)) for ids in ids_by_stage],
        "budgets": [int(v) for v in budgets],
        "rate": 0.05,
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "topology": "C0 edge-aware append; old-old topology frozen",
        "construction": "current pool delta only; no historical raw reads",
        "historical_raw_reads": 0,
        "report_test_labels_used_for_construction": False,
        "independent_from_development_orders": True,
        "source_order_seed": 19614,
        "records": records,
    }
    save_json(out / "manifest.json", manifest)
    save_json(out / "audit.json", {
        "manifest_sha256": sha256_file(out / "manifest.json"),
        "memory_sha256": [row["memory_sha256"] for row in records],
        "source_order_seed": 19614,
        "historical_raw_reads": 0,
        "report_test_labels_used_for_construction": False,
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = [build_dataset(dataset, args) for dataset in args.datasets]
    save_json(args.output_root / "protocol.json", {
        "name": "c0-order19614-source-v1",
        "datasets": args.datasets,
        "order": ORDER,
        "source_only": True,
        "constructor": "C0 dispersion + mean-X + concat/re-encode text + edge-aware append",
        "rate": 0.05,
        "historical_raw_reads": 0,
        "report_test_labels_used_for_construction": False,
    })
    print(json.dumps({"event": "complete", "order": ORDER,
                      "datasets": [m["dataset"] for m in summaries],
                      "output_root": str(args.output_root)}, ensure_ascii=True),
          flush=True)


if __name__ == "__main__":
    main()
