"""Canonical confirmation sweep for Student-Compatible Append candidates.

This runner does not select or rebuild anything.  It loads every candidate
memory emitted by ``run_student_compatible_append.py`` and evaluates each one
with the canonical v3 student seeds, batch, and step rule.  The purpose is to
check whether the short seed x batch screen predicts the final protocol.
"""
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
    recipes = source["candidate_recipes"]
    selected = {str(row["stage"]): row["selected_candidate"]
                for row in source["results"]}

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    x, y, texts = data.x, data.y, list(data.raw_texts)
    n = len(x)
    dev_mask, _, pool_mask = protocol_v3.split_masks(torch.arange(n))
    pool = torch.where(pool_mask)[0]
    encoder = G.FrozenMiniLM(args.minilm, args.device)
    prompt_emb = encoder.encode(G.PROMPTS)
    real_adj = G.norm_adj(data.edge_index, n, args.device)
    xg = x.float().to(args.device)

    results = []
    for stage in range(len(G.STAGE_CLASSES)):
        candidates = []
        for recipe in recipes:
            path = artifact_dir / f"sca_s{stage}_{recipe}_memory.pt"
            memory = torch.load(path, map_location="cpu", weights_only=False)
            score = evaluate(memory, stage, args.student_seeds,
                              args.student_batch, args.device, xg, real_adj,
                              prompt_emb, y, dev_mask)
            candidates.append({"candidate": recipe, "score": score})
        ranking = sorted(candidates,
                         key=lambda row: row["score"]["cumulative_f1"]["mean"],
                         reverse=True)
        results.append({
            "stage": stage,
            "selected_candidate": selected[str(stage)],
            "canonical_ranking_by_cumulative": [r["candidate"] for r in ranking],
            "selected_rank": next(i for i, r in enumerate(ranking)
                                   if r["candidate"] == selected[str(stage)]) + 1,
            "candidates": candidates,
        })
        print(json.dumps({
            "stage": stage,
            "selected": selected[str(stage)],
            "ranking": [r["candidate"] for r in ranking],
            "selected_rank": results[-1]["selected_rank"],
        }), flush=True)

    json.dump({
        **protocol_v3.config_dict(
            dataset=args.dataset,
            student_batch=args.student_batch,
            student_seeds=args.student_seeds,
        ),
        "method": "student_compatible_append_canonical_sweep",
        "source_artifact_dir": str(artifact_dir),
        "student_batch": args.student_batch,
        "student_seeds": args.student_seeds,
        "test_accessed": False,
        "test_source_sealed": True,
        "results": results,
    }, (out / "results.json").open("w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
