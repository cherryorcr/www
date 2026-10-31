"""Canonical confirmation for terminal paths emitted by the SCA beam runner."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v2_llm_fusion"))
sys.path.insert(1, str(Path(__file__).parent))
import protocol_v3
import run_gate1_continual as G


def configure_dataset(dataset: str) -> tuple[str, str]:
    if dataset == "cora":
        return ("/home/pc/research/GraphCLIP/processed_data/cora.pt",
                "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    if dataset == "citeseer":
        G.CLASS_NAMES = ["agents", "machine learning", "information retrieval",
                         "databases", "human computer interaction",
                         "artificial intelligence"]
        G.PROMPTS = [f"a research paper about {name}" for name in G.CLASS_NAMES]
        G.STAGE_CLASSES = [[0, 1], [2, 3], [4, 5]]
        return ("/home/pc/research/GraphCLIP/processed_data/citeseer.pt",
                "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    raise ValueError(dataset)


def evaluate(memory: dict, stage: int, seeds: list[int], batch: int,
             device: str, xg, real_adj, prompt_emb, y, dev_mask) -> dict:
    syn = G.memory_to_trainable(memory)
    steps = protocol_v3.student_steps(len(syn["x"]))
    rows = []
    for seed in seeds:
        model = G.train_student(syn, seed, device, steps, batch=batch)
        score = G.eval_stagewise(model, xg, real_adj, prompt_emb, y,
                                 dev_mask, stage)
        rows.append({
            "seed": int(seed),
            "current_f1": float(score["current_f1"]),
            "cumulative_f1": float(score["cumulative_macro_f1"]),
            "old_mean_f1": (float(score["old_mean_f1"])
                            if stage > 0 else None),
            "per_stage": score["per_stage"],
        })
        del model
    current = np.asarray([r["current_f1"] for r in rows], dtype=float)
    cumulative = np.asarray([r["cumulative_f1"] for r in rows], dtype=float)
    old = [r["old_mean_f1"] for r in rows if r["old_mean_f1"] is not None]
    return {
        "steps": int(steps),
        "batch": int(batch),
        "seeds": list(seeds),
        "current_f1": {"mean": float(current.mean()), "std": float(current.std())},
        "cumulative_f1": {"mean": float(cumulative.mean()), "std": float(cumulative.std())},
        "old_mean_f1": ({"mean": float(np.mean(old)), "std": float(np.std(old))}
                        if old else None),
        "per_student": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--minilm", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--student-seeds", type=int, nargs="+",
                        default=list(protocol_v3.STUDENT_SEEDS))
    parser.add_argument("--student-batch", type=int,
                        default=protocol_v3.STUDENT.batch)
    args = parser.parse_args()
    default_data, default_minilm = configure_dataset(args.dataset)
    args.data = args.data or default_data
    args.minilm = args.minilm or default_minilm

    artifact_dir = Path(args.artifact_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("existing runs are immutable")
    source = json.loads((artifact_dir / "results.json").read_text())
    final = source["results"][-1]
    terminal = final.get("beam_paths", [])
    if not terminal:
        raise ValueError("artifact does not contain beam terminal paths")

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    x, y = data.x, data.y
    n = len(x)
    dev_mask, _, _ = protocol_v3.split_masks(torch.arange(n))
    encoder = G.FrozenMiniLM(args.minilm, args.device)
    prompt_emb = encoder.encode(G.PROMPTS)
    real_adj = G.norm_adj(data.edge_index, n, args.device)
    xg = x.float().to(args.device)
    rows = []
    for item in terminal:
        path = item["path"]
        lineage = "_".join(path)
        matches = sorted(artifact_dir.glob(f"beam_s2_*_{lineage}_memory.pt"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one artifact for path {path}, got {matches}")
        memory = torch.load(matches[0], map_location="cpu", weights_only=False)
        score = evaluate(memory, 2, args.student_seeds, args.student_batch,
                         args.device, xg, real_adj, prompt_emb, y, dev_mask)
        rows.append({"path": path, "artifact": matches[0].name, "score": score,
                     "beam_score": float(item["score"])})
    ranking = sorted(rows,
                     key=lambda r: (r["score"]["cumulative_f1"]["mean"],
                                    r["score"]["current_f1"]["mean"]),
                     reverse=True)
    result = {
        **protocol_v3.config_dict(dataset=args.dataset,
                                  student_batch=args.student_batch,
                                  student_seeds=args.student_seeds),
        "method": "student_compatible_append_beam_terminal_confirmation",
        "source_artifact_dir": str(artifact_dir),
        "student_batch": args.student_batch,
        "student_seeds": args.student_seeds,
        "test_accessed": False,
        "test_source_sealed": True,
        "ranking": ranking,
    }
    json.dump(result, (out / "results.json").open("w"), indent=1)
    print(json.dumps({"ranking": [r["path"] for r in ranking],
                      "cumulative": [r["score"]["cumulative_f1"]["mean"]
                                      for r in ranking]}), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
