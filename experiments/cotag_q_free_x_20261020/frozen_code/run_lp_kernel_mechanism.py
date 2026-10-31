"""Paired no-graph-signature controls for the frozen LP transfer experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

import run_link_prediction_transfer as L
import safe_joint_kernel_append as S

SPECS = {
    "t_degree": ((0., 1., 0.), "degree_balanced"),
    "xt_degree": ((4 / 7, 3 / 7, 0.), "degree_balanced"),
    "t_coverage": ((0., 1., 0.), "coverage"),
    "xt_coverage": ((4 / 7, 3 / 7, 0.), "coverage"),
}
ARMS = ("c0_replay", *SPECS)
ROW_KEYS = ("x", "text_emb", "mass", "depth", "stage")
PROTOCOL = "LP_KERNEL_MECHANISM_PROTOCOL_ZH_2026-09-20.md"


def append_control(old, block, arm):
    weights, variant = SPECS[arm]
    assert weights[2] == 0
    assert set(block) == set(ROW_KEYS) | {"texts"}, "control block must not carry source IDs or signature"
    internal = S.select_internal(block, weights, variant)
    keep = len(old["x"]) if old is not None else 0
    if old is None:
        out = {k: S.copy_value(v) for k, v in block.items()}
        out["edge_index"] = internal
    else:
        assert set(old) == set(block) | {"edge_index"}
        out = {k: torch.cat((old[k], block[k])) for k in ROW_KEYS}
        out["texts"] = old["texts"] + block["texts"]
        cross, _ = S.cross_pairs(old, block, weights, variant)
        out["edge_index"] = torch.cat((old["edge_index"], internal + keep, cross), dim=1)
        for key in ROW_KEYS:
            assert torch.equal(old[key], out[key][:keep])
        assert torch.equal(old["edge_index"], out["edge_index"][:, (out["edge_index"] < keep).all(0)])
    return out


def block_from_c0(memory, keep):
    block = {k: memory[k][keep:].clone() for k in ROW_KEYS}
    block["texts"] = list(memory["texts"][keep:])
    return block


def check_memory(memory, base, old, arm):
    for key in ROW_KEYS:
        assert torch.equal(memory[key], base[key]), key
    assert memory["texts"] == base["texts"]
    n = len(memory["x"])
    edge = L.pairs(memory["edge_index"], n)
    assert edge.shape == L.pairs(base["edge_index"], n).shape
    assert memory["edge_index"].shape[1] == edge.shape[1] * 2
    if old is not None:
        keep = len(old["x"])
        for key in ROW_KEYS:
            assert torch.equal(old[key], memory[key][:keep])
        assert old["texts"] == memory["texts"][:keep]
        assert torch.equal(old["edge_index"], memory["edge_index"][:, (memory["edge_index"] < keep).all(0)])
    else:
        keep = 0
    if arm != "c0_replay":
        assert set(memory) == set(ROW_KEYS) | {"texts", "edge_index"}
        internal = edge[:, (edge >= keep).all(0)] - keep
        counts = torch.bincount(internal.flatten(), minlength=n - keep)
        assert int(counts.max()) <= 2
        assert internal.shape[1] == min(n - keep, (n - keep) * (n - keep - 1) // 2)
        if keep:
            cross = edge[:, (edge[0] < keep) & (edge[1] >= keep)]
            assert cross.shape[1] == n - keep
            assert torch.equal(torch.bincount(cross[1] - keep, minlength=n - keep), torch.ones(n - keep).long())
            capacity = (n - keep + keep - 1) // keep
            assert int(torch.bincount(cross[0], minlength=keep).max()) <= capacity


def parent_evidence(parent):
    manifest = json.loads((parent / "manifest.json").read_text())
    audit = json.loads((parent / "audit.json").read_text())
    assert manifest["steps"] == 256 and manifest["rate"] == .05
    assert manifest["split_seeds"] == list(L.SPLITS) and manifest["student_seeds"] == list(L.SEEDS)
    assert audit["passed"] and audit["fits"] == 270
    here = Path(__file__).parent
    for name, expected in manifest["source_hashes"].items():
        path = here / name if name.endswith(".py") else here.parents[1] / "docs/current" / name
        assert L.sha(path) == expected, f"parent source changed: {name}"
    files = [parent / n for n in ("manifest.json", "results.json", "audit.json")]
    for split in L.SPLITS:
        folder = parent / f"split_{split}"
        files.extend(folder / n for n in ("edge_split.pt", "construction.json"))
        for stage in range(3):
            files.extend(folder / n for n in (f"evaluation_s{stage}.pt", f"c0_s{stage}_memory.pt"))
    return {str(p.relative_to(parent)): L.sha(p) for p in files}


def configure(device):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def run(args):
    import run_safe_joint_kernel_experiments as E
    configure(args.device)
    parent = Path(args.parent)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    reference = json.loads((parent / "manifest.json").read_text())
    assert reference["dataset"] == args.dataset
    assert reference["torch_version"] == torch.__version__
    assert reference["device"] == args.device
    source_files = [Path(__file__), Path(L.__file__), Path(S.__file__), Path(E.__file__), Path(E.F.__file__),
                    Path(E.C.__file__), Path(__file__).resolve().parents[2] / "docs/current" / PROTOCOL]
    config = {"dataset": args.dataset, "parent": str(parent), "parent_hashes": parent_evidence(parent),
              "source_hashes": {p.name: L.sha(p) for p in source_files}, "arms": list(ARMS),
              "specs": {k: [list(w), v] for k, (w, v) in SPECS.items()}, "steps": 256,
              "splits": args.splits, "seeds": args.seeds, "device": args.device,
              "torch_version": torch.__version__, "purpose": "paired development mechanism controls"}
    path = out / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == config, "follow-up protocol changed"
    L.save_json(path, config)
    records = []
    for split in args.splits:
        source = parent / f"split_{split}"
        folder = out / f"split_{split}"
        folder.mkdir(exist_ok=True)
        edge_split = torch.load(source / "edge_split.pt", map_location="cpu", weights_only=False)
        L.validate_split(edge_split)
        construction = json.loads((source / "construction.json").read_text())
        old, builds = {}, []
        for stage in range(3):
            base = torch.load(source / f"c0_s{stage}_memory.pt", map_location="cpu", weights_only=False)
            assert E.F.J_hash(base) == construction["stage_records"][stage]["arms"]["c0"]["content_sha256"]
            keep = len(old["c0_replay"]["x"]) if stage else 0
            block = block_from_c0(base, keep)
            evaluation = source / f"evaluation_s{stage}.pt"
            ev = torch.load(evaluation, map_location="cpu", weights_only=False)
            eval_hash = L.sha(evaluation)
            for arm in ARMS:
                started = time.perf_counter()
                memory = base if arm == "c0_replay" else append_control(old.get(arm), block, arm)
                construction_seconds = time.perf_counter() - started
                check_memory(memory, base, old.get(arm), arm)
                torch.save(memory, folder / f"{arm}_s{stage}_memory.pt")
                content_hash = E.F.J_hash(memory)
                builds.append({"arm": arm, "stage": stage, "content_sha256": content_hash,
                               "payload": E.payload(memory), "topology_construction_seconds": construction_seconds,
                               "reused_c0": arm == "c0_replay"})
                old[arm] = memory
                tx = L.features(memory["x"], memory["text_emb"])
                tp = L.pairs(memory["edge_index"], len(tx))
                tn = L.synthetic_negatives(tp, len(tx))
                for seed in args.seeds:
                    stem = folder / f"{arm}_s{stage}_seed{seed}"
                    output = stem.with_suffix(".json")
                    if output.exists():
                        record = json.loads(output.read_text())
                        assert record["memory_sha256"] == content_hash
                        assert record["evaluation_sha256"] == eval_hash
                        assert L.sha(stem.with_suffix(".pt")) == record["checkpoint_sha256"]
                    else:
                        model, ps, ns, costs = L.fit(tx, tp, tn, ev["x"], ev["edge_index"],
                            ev["test_pos"], ev["test_neg"], seed, 256, args.device)
                        record = {"dataset": args.dataset, "split": split, "stage": stage, "arm": arm, "seed": seed,
                                  "cost": costs, "memory_sha256": content_hash, "evaluation_sha256": eval_hash,
                                  "metrics": L.metrics(ps, ns, ev["test_pos"], ev["test_neg"], ev["arrival"], stage)}
                        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                                    "pos_scores": ps, "neg_scores": ns, "metadata": record}, stem.with_suffix(".pt"))
                        record["checkpoint_sha256"] = L.sha(stem.with_suffix(".pt"))
                        L.save_json(output, record)
                    records.append(record)
                    print(json.dumps({k: record[k] for k in ("dataset", "split", "stage", "arm", "seed")}
                                     | record["metrics"]["all"]), flush=True)
        L.save_json(folder / "construction.json", builds)
    L.save_json(out / "results.json", records)
    L.save_json(out / "complete.json", {"fits": len(records), "expected": len(args.splits) * len(args.seeds) * 3 * len(ARMS)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("cora", "citeseer"))
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", type=int, nargs="+", default=list(L.SPLITS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(L.SEEDS))
    run(parser.parse_args())
