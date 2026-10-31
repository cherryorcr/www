"""Audited current-delta TAG adaptations; SR is loaded without modification."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import torch

import run_semantic_recursive_unified_baselines_20260926 as BASE
import run_unified_v3_continual as U
import external_continual_mechanisms as M

CAT_ARMS = ("cat_mmd_tag", "cat_mmd_tag_tagsam", "cat_mmd_tag_graph_text",
            "cat_mmd_tag_graph_text_aligned", "cat_mmd_tag_text_center")
EDGE_FREE_ARMS = CAT_ARMS + ("gecc_hard_tag",)
CAT_TEXT_MODES = {
    "cat_mmd_tag": "nearest",
    "cat_mmd_tag_tagsam": "tagsam_style",
    "cat_mmd_tag_graph_text": "graph_conditioned",
    "cat_mmd_tag_graph_text_aligned": "graph_text_aligned",
    "cat_mmd_tag_text_center": "text_center",
}
ARMS = ("semantic_recursive", *CAT_ARMS, "gecc_hard_tag",
        "k_center_native", "medoid_native")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory_hash(memory):
    digest = hashlib.sha256(json.dumps(memory["texts"], ensure_ascii=True).encode())
    for key in ("x", "text_emb", "mass", "depth", "stage", "edge_index"):
        tensor = memory[key].detach().cpu().contiguous()
        digest.update(f"{key}:{tensor.dtype}:{list(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def delta_view(data, ids, stage):
    ids = ids.cpu().long()
    if len(ids.unique()) != len(ids) or not bool(data["pool_mask"][ids].all()):
        raise ValueError("construction source is not unique pool-only delta")
    prompt_emb = data.get("prompt_emb")
    if prompt_emb is None and data.get("prompts") is not None:
        prompt_emb = data["encoder"].encode(data["prompts"])
    return {"ids": ids, "stage": stage, "x": data["x"][ids].clone(),
            "y": data["y"][ids].clone(), "text_emb": data["text_emb"][ids].clone(),
            "texts": [data["texts"][i] for i in ids.tolist()],
            "prompt_emb": prompt_emb,
            "edge_index": BASE.induced_edges(data["edge_index"], ids, len(data["x"]))}


def check_memory(memory, parent, budget, edge_free=False):
    n = len(memory["x"])
    if n != budget or len(memory["texts"]) != n:
        raise AssertionError("memory budget or text count mismatch")
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if len(memory[key]) != n or not bool(torch.isfinite(memory[key]).all()):
            raise AssertionError(f"invalid memory field {key}")
    e = memory["edge_index"]
    if e.shape[0] != 2 or (e.numel() and (int(e.min()) < 0 or int(e.max()) >= n)):
        raise AssertionError("invalid topology endpoint")
    if edge_free and e.numel():
        raise AssertionError("native edge-free mechanism gained edges")
    if parent:
        keep = len(parent["x"])
        for key in ("x", "text_emb", "mass", "depth", "stage"):
            if not torch.equal(memory[key][:keep], parent[key]):
                raise AssertionError(f"parent changed: {key}")
        if memory["texts"][:keep] != parent["texts"]:
            raise AssertionError("parent text changed")
        old = e[:, (e < keep).all(0)]
        if not torch.equal(old, parent["edge_index"]):
            raise AssertionError("parent edges changed")


def resource_stats(memory, encoder):
    tensors = sum(memory[k].numel() * memory[k].element_size()
                  for k in ("x", "text_emb", "mass", "depth", "stage", "edge_index"))
    chars = [len(t) for t in memory["texts"]]
    tokens = encoder.tokenizer(memory["texts"], add_special_tokens=False,
                               truncation=False)["input_ids"]
    return {"tensor_bytes": tensors,
            "text_utf8_bytes": sum(len(t.encode("utf-8")) for t in memory["texts"]),
            "text_chars": sum(chars), "text_chars_max": max(chars),
            "text_tokens": sum(map(len, tokens)),
            "edge_entries": int(memory["edge_index"].shape[1])}


def protocol(args, cfg):
    paths = {Path(__file__).name: __file__, Path(M.__file__).name: M.__file__,
             "student.py": BASE.G.__file__, "evaluation.py": BASE.__file__,
             "protocol_v3.py": U.protocol_v3.__file__, "SCA.py": U.SCA.__file__,
             "unified.py": U.__file__}
    return {"name": "external-continual-tag-adapters-v2-20260927",
            "status": "source-checked mechanism adaptations; not official end-to-end reproduction",
            "datasets": args.datasets, "orders": args.orders, "arms": args.arms,
            "report_seeds": args.report_seeds,
            # ``cfg.text_mode`` is only the constructor default.  Each CaT
            # arm receives its mode through CAT_TEXT_MODES below, so record
            # that the protocol is per-arm instead of implying that every
            # arm used the default nearest selector.
            "config": {**asdict(cfg), "text_mode": "per-arm via cat_text_modes"},
            "student_steps_override": args.student_steps,
            "student_steps": "max(50,min(2048,16*memory_rows)) unless smoke override",
            "student_batch": 16, "student_lr": 0.001,
            "student": "same original 2-layer GCN/text projection, bidirectional InfoNCE, scratch",
            "budget": "same 5% cumulative legal pool rows within each dataset/order",
            "evaluation": "all-prompt, seen-node report-dev; transductive full real graph",
            "construction": "class labels of current legal pool allowed; sliced delta passed to constructors",
            "test_labels_evaluated": False, "report_driven_selection": False,
            "topology": "CaT/GECC: no edges; coreset: own text append; SR: sealed artifact",
            "cat_text_modes": CAT_TEXT_MODES,
            "base_stage": "each external arm constructs stage0 itself; SR keeps its sealed base",
            "source_root": str(args.source_root), "recursive_root": str(args.recursive_root),
            "cat_upstream": "superallen13/CaT-CGL@43d6fcfd7b1754fe53b9f911492d7cb0c113a2eb",
            "gecc_upstream": "rockcor/GECC@d731be00cc3beb7585e8fa7b60ca21a6c3cb4e3f",
            "cat_changes": "TAG text attachment; own 5% stream; shared scratch InfoNCE replaces supervised TiM",
            "gecc_changes": "hard variant/default hop weights; sklearn Lloyd; current delta only, no cumulative raw; TAG text attachment",
            "physical_loader": "full offline fixture for slicing/evaluation; no OS-level access isolation",
            "source_hashes": {k: file_hash(v) for k, v in paths.items()},
            "student_module": str(BASE.G.__file__), "torch": torch.__version__}


def prepare_cell(data, order, args, cfg):
    root = args.output_root / data["dataset"] / order
    root.mkdir(parents=True, exist_ok=True)
    if (root / "seal.json").exists():
        verify_cell(root)
        return
    ids, budgets = BASE.set_stream(data, order)
    groups = data["stage_classes"]
    flattened = [c for g in groups for c in g]
    if len(flattened) != len(set(flattened)):
        raise ValueError("this restricted adapter requires disjoint class increments")
    records = []
    for arm in args.arms:
        parent = None
        for stage in range(3):
            sync(args.device)
            start = time.perf_counter()
            delta = delta_view(data, ids[stage], stage)
            rows = budgets[stage] - (len(parent["x"]) if parent else 0)
            if arm == "semantic_recursive":
                src = (args.source_root / data["dataset"] / order / "c0_s0_memory.pt"
                       if stage == 0 else args.recursive_root /
                       f"{data['dataset']}_{order}_s{stage}" / "semantic_perm_memory.pt")
                memory = BASE.load(src)
                audit = {"sealed_source": str(src), "source_sha256": file_hash(src),
                         "construction_seconds": None, "loaded_without_reconstruction": True}
            elif arm in EDGE_FREE_ARMS:
                if arm in CAT_ARMS:
                    builder = M.cat_block
                    arm_cfg = replace(cfg, text_mode=CAT_TEXT_MODES[arm])
                else:
                    builder = M.gecc_block
                    arm_cfg = cfg
                block, audit = builder(delta, rows, data["encoder"], arm_cfg, args.device)
                memory = M.append_edge_free(parent, block)
            else:
                method = "k_center" if arm == "k_center_native" else "kmeans_medoid"
                block, audit = U.build_real_selection_block(
                    delta, torch.arange(len(delta["x"])), rows, method, stage, args.device)
                # BASE intentionally loads the original Student module, whose
                # legacy memory helpers predate stored topology.  Materialize
                # own edges explicitly; never invoke that legacy shrink path.
                block["edge_index"] = U.SCA.text_knn_edges(block["text_emb"])
                if parent is None:
                    memory = block
                else:
                    memory = {key: torch.cat((parent[key], block[key]))
                              for key in ("x", "text_emb", "mass", "depth", "stage")}
                    memory["texts"] = list(parent["texts"]) + list(block["texts"])
                    memory["edge_index"] = U.SCA.append_topology(parent, block, 1, device=args.device)
                audit["topology"] = "own text-based append, not C0 slot topology"
            check_memory(memory, parent, budgets[stage], arm in EDGE_FREE_ARMS)
            sync(args.device)
            elapsed = time.perf_counter() - start
            path = root / f"{arm}_s{stage}_memory.pt"
            torch.save(memory, path)
            record = {"arm": arm, "stage": stage, "rows": budgets[stage],
                      "memory_file": path.name, "memory_sha256": memory_hash(memory),
                      "file_sha256": file_hash(path), "delta_ids": ids[stage].tolist(),
                      "parent_sha256": memory_hash(parent) if parent else None,
                      "construction_seconds": None if arm == "semantic_recursive" else elapsed,
                      "artifact_load_or_construct_seconds": elapsed,
                      "audit": audit, "resources": resource_stats(memory, data["encoder"])}
            records.append(record)
            parent = memory
            print(json.dumps({"event": "constructed", "dataset": data["dataset"], "order": order,
                              "arm": arm, "stage": stage, "seconds": elapsed}), flush=True)
    save(root / "construction.json", {"stage_classes": groups, "budgets": budgets,
                                      "records": records})
    save(root / "seal.json", {"construction_sha256": file_hash(root / "construction.json"),
                             "protocol_sha256": file_hash(args.output_root / "protocol.json")})


def verify_cell(root):
    seal = read(root / "seal.json")
    if seal["construction_sha256"] != file_hash(root / "construction.json"):
        raise AssertionError("construction seal changed")
    if seal["protocol_sha256"] != file_hash(root.parents[1] / "protocol.json"):
        raise AssertionError("protocol changed")
    record = read(root / "construction.json")
    previous = {}
    for item in record["records"]:
        memory = BASE.load(root / item["memory_file"])
        if file_hash(root / item["memory_file"]) != item["file_sha256"] or memory_hash(memory) != item["memory_sha256"]:
            raise AssertionError("memory changed after seal")
        check_memory(memory, previous.get(item["arm"]), item["rows"],
                     item["arm"] in EDGE_FREE_ARMS)
        previous[item["arm"]] = memory
    return record


def evaluate_cell(data, order, args):
    root = args.output_root / data["dataset"] / order
    construction = verify_cell(root)
    BASE.set_stream(data, order)
    results = []
    for item in construction["records"]:
        memory = BASE.load(root / item["memory_file"])
        arm, stage = item["arm"], item["stage"]
        steps = U.protocol_v3.student_steps(len(memory["x"]), override=args.student_steps)
        scores = []
        for seed in args.report_seeds:
            path = root / f"{arm}_s{stage}_seed{seed}.json"
            if path.exists():
                result = read(path)
                if result["memory_sha256"] != item["memory_sha256"] or result["steps"] != steps:
                    raise AssertionError("resumed evaluation mismatch")
            else:
                sync(args.device)
                start = time.perf_counter()
                model = BASE.G.train_student(memory, seed, args.device, steps, lr=0.001, batch=16)
                sync(args.device)
                seconds = time.perf_counter() - start
                result = {"seed": seed, "steps": steps,
                          **BASE.score_model(model, data, data["stage_classes"], stage),
                          "student_seconds": seconds, "memory_sha256": item["memory_sha256"]}
                checkpoint = path.with_suffix(".pt")
                torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, checkpoint)
                result["checkpoint_sha256"] = file_hash(checkpoint)
                save(path, result)
                del model
            scores.append(result)
        results.append({"dataset": data["dataset"], "order": order,
                        "arm": arm, "stage": stage, "memory_rows": item["rows"],
                        "aggregate": BASE.aggregate_stage_scores(scores), "per_student": scores,
                        "student_wall_seconds": sum(s["student_seconds"] for s in scores),
                        "construction_seconds": item["construction_seconds"],
                        "resources": item["resources"]})
        save(root / "progress.json", {"rows": results})
        print(json.dumps({"event": "evaluated", "dataset": data["dataset"], "order": order,
                          "arm": arm, "stage": stage,
                          "f1": results[-1]["aggregate"]["mean_f1"]}), flush=True)
    save(root / "results.json", {"protocol": read(args.output_root / "protocol.json"), "rows": results})


def audit_cell(data, order, args):
    root = args.output_root / data["dataset"] / order
    construction = verify_cell(root)
    ids, budgets = BASE.set_stream(data, order)
    fits, maximum = 0, 0.0
    for item in construction["records"]:
        arm, stage = item["arm"], item["stage"]
        if item["delta_ids"] != ids[stage].tolist() or item["rows"] != budgets[stage]:
            raise AssertionError("source or budget changed")
        sources = item["audit"].get("text_source_ids", [])
        if any(not set(s).issubset(set(item["delta_ids"])) for s in sources):
            raise AssertionError("text used non-delta source")
        memory = BASE.load(root / item["memory_file"])
        for seed in args.report_seeds:
            path = root / f"{arm}_s{stage}_seed{seed}.json"
            result = read(path)
            if result["checkpoint_sha256"] != file_hash(path.with_suffix(".pt")):
                raise AssertionError("checkpoint changed")
            model = BASE.G.CLIPStudent(memory["x"].shape[1]).to(args.device)
            model.load_state_dict(BASE.load(path.with_suffix(".pt"))["state"])
            model.eval()
            score = BASE.score_model(model, data, data["stage_classes"], stage)
            for key in ("cumulative_f1", "accuracy", "old_f1", "new_f1"):
                if score[key] is not None:
                    maximum = max(maximum, abs(score[key] - result[key]))
            fits += 1
            del model
    if maximum > 1e-7:
        raise AssertionError(f"checkpoint metric mismatch: {maximum}")
    save(root / "audit.json", {"passed": True, "fit_count": fits,
                              "maximum_metric_error_pp": maximum,
                              "scope": "sealed memory/prefix/edges/budget/source IDs/checkpoint reload metrics"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--recursive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--orders", nargs="+", default=["order19611", "order19612"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--report-seeds", nargs="+", type=int, default=list(range(150, 156)))
    parser.add_argument("--cat-steps", type=int, default=3000)
    parser.add_argument("--gecc-restarts", type=int, default=20)
    parser.add_argument("--student-steps", type=int)
    parser.add_argument("--phase", choices=("all", "prepare", "evaluate", "audit"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = M.Config(cat_steps=args.cat_steps, gecc_restarts=args.gecc_restarts)
    manifest = protocol(args, cfg)
    path = args.output_root / "protocol.json"
    if path.exists() and read(path) != json.loads(json.dumps(manifest)):
        raise ValueError("existing protocol differs; use a new output root")
    if not path.exists():
        save(path, manifest)
    data = {name: BASE.load_data(name, args.minilm, args.device, args.cache_root)
            for name in args.datasets}
    for value in data.values():
        value["prompt_emb"] = value["encoder"].encode(value["prompts"])
    phases = ("prepare", "evaluate", "audit") if args.phase == "all" else (args.phase,)
    for phase in phases:
        for name in args.datasets:
            for order in args.orders:
                if phase == "prepare":
                    prepare_cell(data[name], order, args, cfg)
                else:
                    {"evaluate": evaluate_cell, "audit": audit_cell}[phase](data[name], order, args)
    if args.phase == "all":
        rows = [row for name in args.datasets for order in args.orders
                for row in read(args.output_root / name / order / "results.json")["rows"]]
        save(args.output_root / "results.json", {"protocol": manifest, "rows": rows})
    print("EXTERNAL_CONTINUAL_ADAPTERS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
