"""Independent completeness, support, checkpoint and prediction audit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F

import run_frozen_baseline_joint_kernel as E

C, M, J, R, U, G, P = E.C, E.M, E.J, E.R, E.U, E.G, E.P


def normal(value):
    return json.loads(json.dumps(value))


def expected_groups(dataset, count, order):
    n = {"cora": 7, "citeseer": 6}[dataset]
    classes = list(range(n))
    if order != "canonical":
        random.Random(int(order[5:])).shuffle(classes)
    widths = [2, 2, n - 4] if count == 3 else [n - 4, 1, 1, 1, 1]
    offsets = np.cumsum([0] + widths).tolist()
    return [sorted(classes[offsets[i]:offsets[i+1]]) for i in range(count)]


def edge_set(memory):
    return set(map(tuple, memory["edge_index"].T.tolist()))


@torch.no_grad()
def inspect_cell(path, data, device, reevaluate, errors, records):
    def check(value, detail):
        if not bool(value):
            errors.append(f"{path.name}: {detail}")
    payload = json.loads((path / "results.json").read_text())
    manifest = payload["manifest"]
    count = int(path.name.split("_")[0][1:])
    order = path.name.split("_")[1]
    groups = expected_groups(data["dataset"], count, order)
    check(groups == manifest["stage_classes"], "class stream")
    data["stage_classes"] = groups
    G.STAGE_CLASSES = groups
    ids = [torch.where(data["pool_mask"] & torch.isin(data["y"], torch.tensor(g)))[0]
           for g in groups]
    budgets = [math.floor(manifest["rate"] * sum(map(len, ids[:s+1]))) for s in range(count)]
    check(budgets == manifest["budgets"], "budgets")
    check(manifest["source_ids_sha256"] == U.sha256_json([v.tolist() for v in ids]), "source IDs")
    check(manifest["seeds"] == list(range(100, 106)), "six frozen seeds")
    expected = {(a, s) for a in manifest["arms"] for s in range(count)}
    check({(r["arm"], r["stage"]) for r in payload["results"]} == expected, "complete arm-stage matrix")
    check(len(payload["results"]) == len(expected), "duplicate result")
    fits = 0
    for arm in manifest["arms"]:
        old = None
        for stage in range(count):
            key = f"{arm}_s{stage}"
            memory = torch.load(path / f"{key}_memory.pt", map_location="cpu", weights_only=False)
            metadata = json.loads((path / f"{key}_construction.json").read_text())
            check(C.tensor_hash(memory) == metadata["tensor_sha256"], f"{key}: support hash")
            check(len(memory["x"]) == budgets[stage], f"{key}: row count")
            check(torch.isfinite(memory["x"]).all() and torch.isfinite(memory["text_emb"]).all(), f"{key}: finite rows")
            edges = memory["edge_index"]
            check(edges.shape[0] == 2 and (edges >= 0).all() and (edges < budgets[stage]).all(), f"{key}: edges")
            check(not (edges[0] == edges[1]).any(), f"{key}: implicit self loops only")
            check(edge_set(memory) == {(v, u) for u, v in edge_set(memory)}, f"{key}: symmetric edges")
            if arm != "recluster":
                C.assert_frozen(old, memory)
                if old is not None:
                    check(torch.equal(old["depth"], memory["depth"][:len(old["x"])]), f"{key}: old depth")
            if arm == "k_center":
                sources = memory["source_ids"]
                check(torch.isin(sources, torch.cat(ids[:stage+1])).all(), f"{key}: legal coreset sources")
                check(torch.equal(memory["x"], data["x"][sources].float()), f"{key}: coreset X")
                check(torch.equal(memory["text_emb"], data["text_emb"][sources]), f"{key}: coreset text embedding")
                check(memory["texts"] == [data["texts"][int(i)] for i in sources], f"{key}: coreset text")
            diagnostic = {"cell": path.name, "dataset": data["dataset"], "arm": arm, "stage": stage,
                          "payload": E.payload(memory), "normalized_old_block_relative_drift": E.adjacency_drift(old, memory)
                          if arm != "recluster" else None}
            if arm == "joint_kernel":
                control = torch.load(path / f"c0_s{stage}_memory.pt", map_location="cpu", weights_only=False)
                for field in ("x", "text_emb", "mass", "depth", "stage"):
                    check(torch.equal(memory[field], control[field]), f"{key}: same C0 {field}")
                check(memory["texts"] == control["texts"], f"{key}: same C0 text")
                check(memory["edge_index"].shape == control["edge_index"].shape, f"{key}: equal edge budget")
                keep = 0 if old is None else len(old["x"])
                sources = json.loads((path / f"{key}_sources.json").read_text())
                partitions = sources["source_groups"]
                check(sorted(i for g in partitions for i in g) == ids[stage].tolist(), f"{key}: source partition")
                check(sources["current_delta"] == ids[stage].tolist(), f"{key}: source delta")
                source_joint = torch.cat((F.normalize(data["x"][ids[stage]].float(), dim=1),
                                          data["text_emb"][ids[stage]]), dim=1)
                graph = J.source_local_graph_summary(data, ids[stage], source_joint)
                positions = {int(i): j for j, i in enumerate(ids[stage])}
                for i, group in enumerate(partitions):
                    check(torch.equal(data["x"][group].float().mean(0), memory["x"][keep+i]), f"{key}: synthetic mean {i}")
                    signature = F.normalize(graph[[positions[v] for v in group]].mean(0)[None], dim=1)[0]
                    check(torch.equal(signature, memory["graph_signature"][keep+i]), f"{key}: signature {i}")
                block = {k: memory[k][keep:] for k in ("x", "text_emb", "graph_signature")}
                expected_edges = J.kernel_edges(block) if old is None else J.append_kernel_topology(old, block)
                check(torch.equal(expected_edges, memory["edge_index"]), f"{key}: deterministic topology")
                if old is not None:
                    check(torch.equal(old["graph_signature"], memory["graph_signature"][:keep]), f"{key}: frozen signature")
                first, second = edge_set(control), edge_set(memory)
                diagnostic["edge_jaccard_c0"] = len(first & second) / max(1, len(first | second))
                check(metadata["shared_c0_seconds"] == json.loads((path / f"c0_s{stage}_construction.json").read_text())["build_seconds"],
                      f"{key}: construction charged")
            summary = next(r for r in payload["results"] if (r["arm"], r["stage"]) == (arm, stage))
            check([r["seed"] for r in summary["per_seed"]] == list(range(100,106)), f"{key}: complete seed list")
            for row in summary["per_seed"]:
                fit_key = f"{key}_seed{row['seed']}"
                fit_path = path / f"{fit_key}.pt"
                check(json.loads((path / f"{fit_key}.json").read_text()) == row, f"{fit_key}: row consistency")
                check(C.sha(fit_path) == row["checkpoint_sha256"], f"{fit_key}: checkpoint file hash")
                saved = torch.load(fit_path, map_location="cpu", weights_only=False)
                check(C.tensor_hash(saved["state"]) == row["end_state_sha256"], f"{fit_key}: model hash")
                check(row["support_sha256"] == C.tensor_hash(memory), f"{fit_key}: training support")
                check(row["steps"] == P.student_steps(budgets[stage]), f"{fit_key}: student steps")
                check(row["effective_batch"] == min(16, budgets[stage]), f"{fit_key}: batch")
                prediction = saved["prediction"]
                mask = data["report_dev_mask"]
                check(prediction["stage_classes"] == groups and prediction["stage"] == stage, f"{fit_key}: prediction task")
                check(torch.equal(prediction["target"], data["y"][mask]), f"{fit_key}: report targets")
                check(torch.equal(prediction["original_ids"], data["original_ids"][mask]), f"{fit_key}: report IDs")
                scored = normal(M.score_logits(prediction["logits"], prediction["target"].numpy(), groups, stage))
                check(scored == row["metrics"], f"{fit_key}: prediction metrics")
                if reevaluate:
                    model = G.CLIPStudent(data["x"].shape[1]).to(device)
                    model.load_state_dict(saved["state"])
                    check(normal(M.evaluate(model, data, stage)) == row["metrics"], f"{fit_key}: independent model evaluation")
                    del model
                fits += 1
            check(C.aggregate(summary["per_seed"]) == summary["metrics"], f"{key}: aggregate")
            records.append(diagnostic)
            old = memory
    return fits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reevaluate", action="store_true")
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    archive = args.root.parent
    cache = archive / "p5_rate_curve_small_20260919" / f"{args.dataset}_r005/text_emb_pool_only.pt"
    data = R.load_data(args.dataset, "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", args.device, cache)
    errors, records, checked, cells = [], [], 0, 0
    for phase in ("baseline", "matrix"):
        path = args.root / phase / args.dataset
        frozen = json.loads((path / "freeze.json").read_text())
        if frozen["source_hashes"] != E.source_hashes():
            errors.append(f"{phase}: frozen source hash mismatch")
        if not args.partial and not (path / "completion.json").exists():
            errors.append(f"{phase}: not completed")
        for cell in sorted(path.glob("s*_*_r*")):
            if not (cell / "results.json").exists():
                if not args.partial:
                    errors.append(f"{cell.name}: incomplete")
                continue
            try:
                checked += inspect_cell(cell, data, args.device, args.reevaluate, errors, records)
                cells += 1
            except Exception as ex:
                errors.append(f"{cell.name}: {type(ex).__name__}: {ex}")
        print(json.dumps({"phase": phase, "checked_fits": checked, "errors": len(errors)}), flush=True)
    if not args.partial and (checked != 1746 or cells != 19):
        errors.append(f"incomplete matrix: fits={checked}/1746 cells={cells}/19")
    result = {"dataset": args.dataset, "checked_fits": checked, "checked_cells_including_baseline": cells,
              "independent_model_reevaluation": args.reevaluate, "partial": args.partial,
              "errors": errors, "error_count": len(errors), "auditor_sha256": C.sha(__file__),
              "records": records, "test_labels_evaluated": False}
    C.save_json(args.root / f"audit_{args.dataset}{'_partial' if args.partial else ''}.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
