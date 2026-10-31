"""Isolate text compression on immutable, matched-row P1 synthetic memories."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import run_unified_v3_continual as U
import p2_text_operators as T


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def specifications() -> list[dict]:
    specs = [{"operator": "concat"}]
    specs += [{"operator": "diverse_concat", "relevance_weight": v} for v in (0.25, 0.5, 0.75)]
    specs += [{"operator": "dual_proto", "alpha": v} for v in (0.25, 0.5, 0.75)]
    specs += [{"operator": op} for op in ("mean_pool", "weighted_pool", "set_pool",
                                        "no_reencode", "no_fuse", "no_text")]
    specs += [{"operator": "concat", "g": g} for g in (3, 8)]
    specs += [{"operator": "concat", "budget": b} for b in (32, 128)]
    return [dict(operator="concat", budget=64, g=5, relevance_weight=0.5, alpha=0.5) | s
            for s in specs]


def spec_name(s: dict) -> str:
    return (f'{s["operator"]}_t{s["budget"]}_g{s["g"]}'
            f'_r{s["relevance_weight"]}_a{s["alpha"]}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--minilm", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--student-seeds", nargs="+", type=int, default=list(U.protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    root, out = Path(args.reference), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    reference = json.loads((root / "results.json").read_text())
    if reference["budget_mode"] != "fixed-rate" or reference["allocation_mode"] != "dispersion":
        raise ValueError("P2 requires the fixed-rate dispersion reference")
    if reference["test_accessed"] or reference["response_marginal"]:
        raise ValueError("text isolation requires a sealed-test non-selector reference")
    if sha(Path(reference["data"])) != reference["source_hash_sha256"]:
        raise ValueError("source file changed after reference construction")
    D = U.load_unified(reference["dataset"], reference["data"], args.minilm, args.device,
                       n_stages=reference["n_stages"], split_mode=reference["split_mode"],
                       base_fraction=reference["base_fraction"],
                       report_dev_mod=reference["report_dev_mod"])
    D["text_emb"] = torch.load(root / "text_emb_pool_only.pt", map_location="cpu", weights_only=False)
    if (D["text_emb"].shape != (len(D["x"]), 384)
            or D["text_emb"][~D["pool_mask"]].abs().max() != 0):
        raise ValueError("invalid pool-only embedding cache")
    if D["stage_classes"] != reference["stage_classes"]:
        raise ValueError("stage partition differs from the frozen reference")
    D["prompt_emb"] = D["encoder"].encode(D["prompts"])
    D["real_adj"] = U.G.norm_adj(D["edge_index"], len(D["x"]), args.device)
    D["x_device"] = D["x"].to(args.device)
    joint = torch.cat((F.normalize(D["x"].float(), dim=1), D["text_emb"]), dim=1)
    refs = [r for r in reference["results"] if r["arm"] == "append_fixed"]
    stage_data = []
    for r in refs:
        stage = r["stage"]
        mempath = root / f"append_fixed_s{stage}_memory.pt"
        memory = torch.load(mempath, map_location="cpu", weights_only=False)
        clusters = []
        for cls in D["stage_classes"][stage]:
            ids = torch.where(D["pool_mask"] & (D["y"] == cls))[0]
            count = int(r["allocated_new_rows_by_class"][str(cls)])
            if count <= 0:
                continue
            assign = U.SCA.cluster_assign(joint[ids], torch.ones(len(ids)), count, device=args.device)
            clusters.extend(ids[assign == j] for j in range(int(assign.max()) + 1))
        keep = int(r["keep_old_rows"])
        reconstructed = torch.stack([D["x"][members].float().mean(0) for members in clusters])
        if not torch.allclose(reconstructed, memory["x"][keep:], atol=1e-6, rtol=0):
            raise AssertionError("synthetic feature/cluster reconstruction differs from reference")
        if not torch.equal(torch.tensor([len(m) for m in clusters]).float(), memory["mass"][keep:]):
            raise AssertionError("cluster source masses differ from reference")
        stage_data.append((r, memory, clusters, sha(mempath)))
    for spec in specifications():
        name = spec_name(spec)
        if args.only and name not in args.only and spec["operator"] not in args.only:
            continue
        target = out / name
        target.mkdir(exist_ok=True)
        final = target / "results.json"
        if final.exists():
            existing = json.loads(final.read_text())
            if (existing["student_seeds"] != args.student_seeds
                    or existing["spec"] != spec
                    or existing["code_sha256"] != sha(Path(T.__file__))):
                raise ValueError(f"refusing stale completed artifact: {target}")
            print(f"SKIP {name}", flush=True)
            continue
        old_texts, old_embeddings, old_audits = [], [], []
        results = []
        print(f"RUN {reference['dataset']} {name}", flush=True)
        for r, reference_memory, clusters, reference_hash in stage_data:
            start = time.perf_counter()
            if len(old_texts) != r["keep_old_rows"]:
                raise AssertionError("P2 fixed-rate control must preserve every old row")
            for members in clusters:
                text, embedding, audit = T.construct(joint, D["text_emb"], members,
                    D["texts"], D["encoder"], **spec)
                old_texts.append(text)
                old_embeddings.append(embedding)
                old_audits.append(audit)
            memory = dict(reference_memory)
            memory.update(texts=list(old_texts), text_emb=torch.stack(old_embeddings),
                          text_input_audit=list(old_audits))
            build_seconds = time.perf_counter() - start
            torch.save(memory, target / f"append_fixed_s{r['stage']}_memory.pt")
            student_args = SimpleNamespace(student_seeds=args.student_seeds,
                student_steps=None, student_batch=reference["student_batch"])
            metrics = U.run_students(memory, r["stage"], D, student_args, args.device)
            row = {"stage": r["stage"], "arm": "append_fixed",
                "keep_old_rows": r["keep_old_rows"], "new_rows": len(clusters),
                "text_audit": T.aggregate_audit(old_audits), "build_seconds": build_seconds,
                "reference_memory_sha256": reference_hash,
                "topology_and_features_fixed": True, **metrics}
            results.append(row)
            (target / "progress.json").write_text(json.dumps(results, indent=2))
            print(json.dumps({"name": name, "stage": r["stage"],
                              "f1": metrics["cum_dev_f1_mean"]}), flush=True)
        manifest = {"protocol": "P2-token-control-v1", "dataset": reference["dataset"],
            "reference": str(root), "reference_results_sha256": sha(root / "results.json"),
            "spec": spec, "student_seeds": args.student_seeds,
            "student_batch": reference["student_batch"], "test_accessed": False,
            "selector_used": False, "source_cache": "sealed_pool_common_preprocessing",
            "content_budget_policy": "per_row_token_cap_excluding_special_tokens",
            "topology_policy": "immutable_reference_concat_g5_c320",
            "code_sha256": sha(Path(T.__file__)), "runner_sha256": sha(Path(__file__)),
            "stage_classes": D["stage_classes"], "results": results}
        final.write_text(json.dumps(manifest, indent=2))
    print(f"P2_TOKEN_MATRIX_DONE {reference['dataset']}", flush=True)


if __name__ == "__main__":
    main()
