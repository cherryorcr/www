"""Large-graph LP validation from the frozen base+4 C0 memories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

import protocol_v3 as P
import run_link_prediction_transfer as L
import large_lp_support as B

ARMS = ("c0", "t_degree", "xt_degree", "t_coverage", "xt_coverage", "full_raw_equal")
WEIGHTS = {"t": (0., 1., 0.), "xt": (4 / 7, 3 / 7, 0.)}
VARIANT = {"t_degree": ("t", "degree_balanced"), "xt_degree": ("xt", "degree_balanced"),
           "t_coverage": ("t", "coverage"), "xt_coverage": ("xt", "coverage")}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def memory_block(base, keep):
    keys = ("x", "text_emb", "mass", "depth", "stage")
    return {**{k: base[k][keep:].clone() for k in keys},
            "texts": list(base["texts"][keep:])}


def build_memories(c0_root, output):
    memories = {arm: [] for arm in ARMS if arm != "full_raw_equal"}
    old = {arm: None for arm in VARIANT}
    construction = []
    (output / "memories").mkdir(parents=True, exist_ok=True)
    for stage in range(5):
        base = torch.load(c0_root / f"append_fixed_s{stage}_memory.pt",
                          map_location="cpu", weights_only=False)
        keep = 0 if stage == 0 else len(memories["c0"][stage - 1]["x"])
        memories["c0"].append(base)
        block = memory_block(base, keep)
        stage_record = {"stage": stage, "keep": keep, "rows": len(base["x"]), "arms": {}}
        for arm in VARIANT:
            key, variant = VARIANT[arm]
            start = time.perf_counter()
            candidate = B.append(old[arm], block, WEIGHTS[key], variant)
            for field in ("x", "text_emb", "mass", "depth", "stage"):
                assert torch.equal(candidate[field], base[field]), (arm, stage, field)
            assert candidate["texts"] == base["texts"]
            assert L.pairs(candidate["edge_index"], len(candidate["x"])).shape == \
                L.pairs(base["edge_index"], len(base["x"])).shape
            old[arm] = candidate
            memories[arm].append(candidate)
            path = output / "memories" / f"{arm}_s{stage}.pt"
            torch.save(candidate, path)
            stage_record["arms"][arm] = {
                "build_seconds": time.perf_counter() - start,
                "rows": len(candidate["x"]),
                "pairs": int(L.pairs(candidate["edge_index"], len(candidate["x"])).shape[1]),
                "sha256": sha(path)}
        path = output / "memories" / f"c0_s{stage}.pt"
        torch.save(base, path)
        stage_record["arms"]["c0"] = {
            "rows": len(base["x"]),
            "pairs": int(L.pairs(base["edge_index"], len(base["x"])).shape[1]),
            "sha256": sha(path)}
        construction.append(stage_record)
    save(output / "construction.json", construction)
    return memories


def topology(memory):
    edge = L.pairs(memory["edge_index"], len(memory["x"]))
    deg = torch.bincount(L.bidirectional(edge)[0], minlength=len(memory["x"]))
    return {"rows": len(memory["x"]), "pairs": int(edge.shape[1]),
            "isolated": int((deg == 0).sum()), "max_degree": int(deg.max())}


def load_data(args, config):
    raw = torch.load(args.data, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError("large validation expects the TAG dictionary format")
    labels = [str(x) for x in raw["all_labels"].tolist()]
    mapping = {name: i for i, name in enumerate(labels)}
    y = torch.tensor([mapping[str(x)] for x in raw["label_list"].tolist()], dtype=torch.long)
    original = raw["original_node_ids"].long()
    pool = P.split_masks_v4(original)[-1]
    text = torch.load(args.text, map_location="cpu", weights_only=False).float()
    assert len(text) == len(raw["node_f"])
    assert int(text[~pool].abs().sum()) == 0
    arrival = torch.full((len(y),), -1, dtype=torch.long)
    for stage, classes in enumerate(config["stage_classes"]):
        ids = pool & torch.isin(y, torch.tensor(classes))
        arrival[ids] = stage
    assert int((arrival >= 0).sum()) == int(config["pool"])
    return {"x": raw["node_f"].float(), "text": text, "y": y,
            "edge": raw["edge_index"].long(), "pool": pool, "arrival": arrival}


def run(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    p5 = json.loads((Path(args.c0_root) / "results.json").read_text())
    config = {
        "dataset": args.dataset, "data": str(args.data), "text": str(args.text),
        "c0_root": str(args.c0_root), "stage_classes": p5["stage_classes"],
        "pool": p5["pool"], "rate": p5["rate"], "arms": list(ARMS),
        "split_seeds": args.splits, "student_seeds": args.seeds, "steps": args.steps,
        "construction": "frozen C0 rows with T/XT kernel and degree/coverage topology",
        "evaluation": "seen pool nodes; held-out real edges; no test/dev raw nodes",
        "data_sha256": sha(args.data), "text_sha256": sha(args.text),
        "p5_config_sha256": sha(Path(args.c0_root) / "results.json")}
    manifest = out / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == config, "large LP protocol changed"
    save(manifest, config)
    data = load_data(args, config)
    memories = build_memories(Path(args.c0_root), out)
    save(out / "topology.json", {arm: [topology(m) for m in vals]
                                 for arm, vals in memories.items()})
    records = []
    for split_seed in args.splits:
        folder = out / f"split_{split_seed}"
        folder.mkdir(exist_ok=True)
        split = B.split_edges(data["edge"], data["arrival"], split_seed)
        torch.save(split, folder / "edge_split.pt")
        save(folder / "split_counts.json", {
            k: int(v.shape[1]) for k, v in split.items()
            if isinstance(v, torch.Tensor) and v.ndim == 2})
        for stage in range(5):
            ids = torch.where((data["arrival"] >= 0) & (data["arrival"] <= stage))[0]
            local = {k: L.localize(split[k], ids, len(data["arrival"]))
                     for k in ("train_pos", "train_neg", "test_pos", "test_neg")}
            real_x = L.features(data["x"][ids], data["text"][ids])
            real_edge = L.bidirectional(local["train_pos"])
            evaluation = {"ids": ids, "arrival": data["arrival"][ids], "x": real_x,
                          "edge_index": real_edge, "test_pos": local["test_pos"],
                          "test_neg": local["test_neg"]}
            torch.save(evaluation, folder / f"evaluation_s{stage}.pt")
            for arm in ARMS:
                if arm == "full_raw_equal":
                    train_x, train_pos, train_neg = real_x, local["train_pos"], local["train_neg"]
                else:
                    memory = memories[arm][stage]
                    train_x = L.features(memory["x"], memory["text_emb"])
                    train_pos = L.pairs(memory["edge_index"], len(train_x))
                    train_neg = B.training_negatives(train_pos, len(train_x), split_seed + stage)
                for seed in args.seeds:
                    stem = folder / f"{arm}_s{stage}_seed{seed}"
                    result_path = stem.with_suffix(".json")
                    if result_path.exists():
                        record = json.loads(result_path.read_text())
                    else:
                        model, ps, ns, costs = L.fit(
                            train_x, train_pos, train_neg, real_x, real_edge,
                            local["test_pos"], local["test_neg"], seed,
                            args.steps, args.device)
                        record = {
                            "dataset": args.dataset, "split": split_seed, "stage": stage,
                            "arm": arm, "seed": seed, "cost": costs,
                            "metrics": L.metrics(
                                ps, ns, local["test_pos"], local["test_neg"],
                                evaluation["arrival"], stage)}
                        torch.save({
                            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                            "pos_scores": ps, "neg_scores": ns, "metadata": record},
                            stem.with_suffix(".pt"))
                        record["checkpoint_sha256"] = sha(stem.with_suffix(".pt"))
                        save(result_path, record)
                    records.append(record)
                    print(json.dumps({
                        k: record[k] for k in ("dataset", "split", "stage", "arm", "seed")}
                        | record["metrics"]["all"]), flush=True)
    save(out / "results.json", records)
    save(out / "complete.json", {
        "fits": len(records),
        "expected": len(args.splits) * 5 * len(ARMS) * len(args.seeds)})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--c0-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--splits", type=int, nargs="+", default=[39001, 39002])
    p.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103])
    run(p.parse_args())
