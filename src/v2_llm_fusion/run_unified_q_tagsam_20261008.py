"""TAGSAM response-loss adapter under the frozen continual outer protocol.

Uses the pinned official loss and the existing, explicit 10-step response
unroll adapter. This is not an official TAGSAM end-to-end reproduction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

import run_unified_q_matrix_20261008 as R
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tagsam_graphclip_comparison as T


def initial_block(delta, rows, encoder, seed, device):
    allocation = R.M.ratio_allocation(delta["y"], rows)
    labels = R.M.row_labels(allocation)
    generator = torch.Generator().manual_seed(seed + 1009 * int(delta["stage"]))
    selected = []
    for cls, count in allocation.items():
        ids = torch.where(delta["y"] == cls)[0]
        selected.extend(ids[torch.randperm(len(ids), generator=generator)[:count]].tolist())
    x = delta["x"][torch.tensor(selected)].float().to(device)
    source = R.M.propagated_features(delta, (0.0, 1.0), device)
    cfg = R.M.Config(seed=seed, text_sources=5, text_mode="nearest")
    block, audit = R.M.attach_text(delta, source, x, labels, encoder, cfg)
    block = {key: value.cpu() if isinstance(value, torch.Tensor) else value
             for key, value in block.items()}
    block["labels"] = labels
    return block, audit


def reference_memory(delta, parent):
    n = 0 if parent is None else len(parent["x"])
    current = dict(x=delta["x"], text_emb=delta["text_emb"], texts=delta["texts"],
                   labels=delta["y"], edge_index=delta["edge_index"],
                   mass=torch.ones(len(delta["x"])),
                   depth=torch.zeros(len(delta["x"]), dtype=torch.long),
                   stage=torch.full((len(delta["x"]),), delta["stage"], dtype=torch.long))
    if parent is None:
        return current
    output = {key: torch.cat((parent[key], current[key]))
              for key in ("x", "text_emb", "labels", "mass", "depth", "stage")}
    output["texts"] = parent["texts"] + current["texts"]
    output["edge_index"] = torch.cat((parent["edge_index"], current["edge_index"] + n), 1)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--official", type=Path, default=Path("/home/pc/research/TAGSAM_official"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--order-seed", type=int, default=9101)
    parser.add_argument("--construction-seed", type=int, default=271006)
    parser.add_argument("--teacher-steps", type=int, default=2048)
    parser.add_argument("--outer-steps", type=int, default=100)
    parser.add_argument("--student-steps", type=int, default=2048)
    parser.add_argument("--student-seeds", type=int, nargs="+", default=[150,151,152])
    parser.add_argument("--budget-mode", default="fixed_schedule", choices=["fixed_schedule"])
    parser.add_argument("--q-steps", type=int, default=30)
    parser.add_argument("--sinkhorn-max-steps", type=int, default=2048)
    args = parser.parse_args()
    torch.set_num_threads(2)
    loss_fn, provenance = T.official_loss(args.official)
    ref = R.L.read(args.reference)
    groups = R.V.choose_groups(ref, args.order_seed)
    data = R.L.load_large(args.dataset, args.minilm, args.text_cache, args.device,
                         dict(ref, groups=groups))
    ids = R.L.stage_ids(data)
    budgets, _ = R.V.choose_budgets(ref, list(map(len, ids)), groups, "fixed_schedule", .05)
    protocol = R.protocol(args, data, groups, ids, budgets, ["tagsam_response_adapter"])
    protocol.update(name="tagsam-response-adapter-matched-20261008", official_source=provenance,
        constructor_proxy="legacy CLIPStudent/row-wise InfoNCE; NOT changed final Prompt Student",
        teacher_steps=args.teacher_steps, outer_steps=args.outer_steps, inner_steps=10,
        reference_cap=256, constructor_text="fixed graph-nearest top-5, 320 chars",
        initialization="class-ratio stratified real-node X, no CaT pre-optimization",
        adaptation="current delta+own synthetic parent; freeze prefix; original wBCELoss; reduced unroll budget; common final Student",
        official_reproduction=False,
        response_runner_sha256=hashlib.sha256(Path(T.__file__).read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    if path.exists():
        raise FileExistsError("preserve previous result; use a fresh output")
    result = dict(protocol=protocol, rows=[], status="running")
    parent = None
    for stage, current_ids in enumerate(ids):
        delta = R.L.delta_view(data, current_ids, stage)
        old_n = 0 if parent is None else len(parent["x"])
        new_n = budgets[stage] - old_n
        R.V.sync(args.device)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(args.device)
        initial, attachment_audit = initial_block(delta, new_n, data["encoder"],
                                                  args.construction_seed, args.device)
        raw = reference_memory(delta, parent)
        teacher = T.G.train_student(raw, 9000 + stage, args.device, args.teacher_steps,
                                    lr=.001, batch=24)
        memory, audit = T.optimize(initial, parent, raw, teacher, stage,
                                   "tagsam_selfloop", args.device, loss_fn, args.outer_steps)
        memory["labels"] = (initial["labels"] if parent is None else
                            torch.cat((parent["labels"], initial["labels"])))
        R.V.sync(args.device)
        build = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(args.device) / 2**20
        checks = R.V.validate_memory(memory, parent, budgets[stage])
        if memory["edge_index"].numel():
            raise AssertionError("self-loop adapter unexpectedly added edges")
        started = time.perf_counter()
        score = R.L.score(memory, data, stage, args.student_seeds,
                          args.device, args.student_steps, 24)
        R.V.sync(args.device)
        result["rows"].append(dict(arm="tagsam_response_adapter", stage=stage,
            raw_rows=len(current_ids), new_rows=new_n, memory_rows=budgets[stage],
            score=score, construction_seconds=build,
            student_seconds=time.perf_counter()-started, construction_peak_mb=peak,
            memory_hash=R.V.memory_hash(memory), checks=checks,
            audit=dict(optimization=audit, attachment=attachment_audit,
                       historical_raw_reads=0), method_class="external_adapter"))
        R.V.write_json(path, result)
        print(json.dumps(dict(event="evaluated", dataset=args.dataset, stage=stage,
                              f1=score["final_f1"], accuracy=score["accuracy"])), flush=True)
        parent = memory
        del teacher, raw
    result["status"] = "complete"
    R.V.write_json(path, result)


if __name__ == "__main__":
    main()
