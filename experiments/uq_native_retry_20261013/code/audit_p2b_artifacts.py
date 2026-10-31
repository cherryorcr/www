"""Offline P2-B provenance, immutability and learned-target reconstruction audit."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

import p2b_student_attention as T
import run_p2b_student_attention as R

A = T.A


def read(path):
    return json.loads(Path(path).read_text())


def audit(root, dataset, minilm):
    plan, acceptance = read(root / "plan.json"), read(root / "acceptance.json")
    if not acceptance["complete"] or plan["classification"] != "known-order screen":
        raise AssertionError("not a complete formal screen")
    code = (R.__file__, T.__file__, A.__file__, A.T.__file__, A.BT.__file__, A.C.__file__,
            A.O.__file__, A.B.__file__, A.G.__file__, A.U.__file__, A.SCA.__file__, A.P.__file__,
            Path(A.C.__file__).with_name("small_comparison_metrics.py"))
    for file in code:
        if A.C.sha(file) != plan["code_sha256"][Path(file).name]:
            raise AssertionError(f"frozen source changed: {file}")
    if A.C.sha(A.U.DATA_CFG[dataset]["path"]) != plan["source_sha256"]:
        raise AssertionError("raw source changed")
    if A.C.sha(A.C.artifact_dir(dataset, "v4") / "text_emb_pool_only.pt") != plan["text_cache_sha256"]:
        raise AssertionError("frozen source text cache changed")
    if {p.name: A.C.sha(p) for p in Path(minilm).glob("*") if p.is_file()} != plan["minilm_sha256"]:
        raise AssertionError("text encoder/tokenizer changed")
    tokenizer = AutoTokenizer.from_pretrained(minilm, local_files_only=True)
    raw = torch.load(A.U.DATA_CFG[dataset]["path"], map_location="cpu", weights_only=False)
    nmem = ntext = nfit = nlearn = 0
    for order, groups in plan["orders"].items():
        directory = root / order
        previous = {}
        for stage in range(3):
            shared = read(directory / f"shared_s{stage}.json")
            ids = torch.tensor(shared["source_ids"])
            if any(int(i) % 10 in (1, 2, 3) for i in ids):
                raise AssertionError("holdout construction source")
            if not torch.isin(raw.y[ids], torch.tensor(groups[stage])).all():
                raise AssertionError("source outside current stage")
            clusters = shared["clusters_global_ids"]
            if sorted(sum(clusters, [])) != sorted(ids.tolist()):
                raise AssertionError("invalid source partition")
            expected_x = torch.stack([raw.x[torch.tensor(c)].float().mean(0) for c in clusters])
            path = directory / f"evidence_s{stage}.pt"
            if A.C.sha(path) != shared["evidence_sha256"]:
                raise AssertionError("evidence file changed")
            evidence = torch.load(path, map_location="cpu", weights_only=False)
            controls, starts = [], {seed: set() for seed in plan["seeds"]}
            for entry in shared["evidence"]["query_stats"]:
                if not all(0 <= i < len(ids) for i in entry["neighbor_local_ids"]):
                    raise AssertionError("noncurrent neighbor query")
            for arm in R.ARMS:
                path = directory / f"{arm}_s{stage}_memory.pt"
                memory = torch.load(path, map_location="cpu", weights_only=False)
                rec = read(directory / f"{arm}_s{stage}_construction.json")
                if A.C.sha(path) != rec["file_sha256"] or A.C.tensor_hash(memory) != rec["support_sha256"]:
                    raise AssertionError("memory hash mismatch")
                keep = len(previous[arm]["x"]) if stage else 0
                A.C.assert_frozen(previous.get(arm), memory)
                if not torch.equal(memory["x"][keep:], expected_x):
                    raise AssertionError("Mean-X/source mismatch")
                controls.append(A.C.tensor_hash({k: memory[k] for k in
                    ("x", "mass", "depth", "stage", "edge_index")}))
                if memory["texts"][keep:] != evidence["texts"]:
                    raise AssertionError("readable evidence differs across arms")
                if arm == "center_concat":
                    expected_t = evidence["tensors"]["anchor"]
                else:
                    model = T.AnchoredAttention(seed=int(arm.rsplit("_", 1)[1]) if arm.startswith(("student_", "semantic_")) else 700)
                    if arm.startswith(("student_", "semantic_")):
                        state = torch.load(directory / f"{arm}_s{stage}_attention.pt", map_location="cpu", weights_only=False)
                        model.load_state_dict(state)
                        meta = rec["meta"]
                        if A.C.tensor_hash(state) != meta["selected_parameter_sha256"]:
                            raise AssertionError("attention parameters changed")
                        train, val = meta["train_local_ids"], meta["validation_local_ids"]
                        if set(train) & set(val) or sorted(train + val) != list(range(len(ids))):
                            raise AssertionError("invalid source-loss split")
                        seed = int(arm.rsplit("_", 1)[1])
                        expected_split = T.source_split(len(ids), seed, stage)
                        if [train, val] != [x.tolist() for x in expected_split]:
                            raise AssertionError("split seed drift")
                        losses = [r for r in meta["curve"] if "validation" in r]
                        best = min(losses, key=lambda r: r["validation"])
                        if best["step"] != meta["selected_step"] or best["validation"] != meta["selected_validation"]:
                            raise AssertionError("wrong selected checkpoint")
                        if meta["nonzero_gradient_steps"] != plan["outer_steps"] or meta["report_labels_used"]:
                            raise AssertionError("gradient/data boundary failure")
                        nlearn += 1
                    with torch.no_grad():
                        expected_t, weights = model(**evidence["tensors"])
                    torch.testing.assert_close(weights, torch.tensor(rec["meta"]["weights"]), atol=2e-5, rtol=2e-4)
                torch.testing.assert_close(memory["text_emb"][keep:], expected_t, atol=2e-5, rtol=2e-4)
                for row, text in zip(shared["evidence"]["rows"], memory["texts"][keep:]):
                    retained = A.BT.token_count(tokenizer, text)
                    parts = sum(A.BT.token_count(tokenizer, p) for p in row["inputs"])
                    if max(retained, parts) > 64 or retained != row["retained_tokens"] or parts != row["part_tokens"]:
                        raise AssertionError("token audit mismatch")
                    ntext += 1
                for i, row in enumerate(shared["evidence"]["rows"]):
                    if not all(int(ids[j]) in clusters[i] for j in row["selected_local_ids"]):
                        raise AssertionError("snippet from another cluster")
                for seed in plan["seeds"]:
                    fit_path = directory / f"{arm}_s{stage}_seed{seed}.json"
                    fit = read(fit_path)
                    ckpt = torch.load(fit_path.with_suffix(".pt"), map_location="cpu", weights_only=False)
                    if fit["support_sha256"] != rec["support_sha256"] or A.C.tensor_hash(ckpt["state"]) != fit["end_state_sha256"]:
                        raise AssertionError("student checkpoint mismatch")
                    starts[seed].add(fit["start_state_sha256"])
                    nfit += 1
                previous[arm] = memory
                nmem += 1
            if len(set(controls)) != 1 or any(len(x) != 1 for x in starts.values()):
                raise AssertionError("unpaired comparison")
    result = dict(pass_audit=True, dataset=dataset, memories=nmem, new_text_rows=ntext,
        fits=nfit, learned_checkpoints=nlearn, frozen_code_files=len(code), budget_violations=0,
        source_and_token_audit=True, parameters_reproduce_targets=True,
        old_memory_frozen=True, scope="offline raw-data reread for audit only",
        report_reload_verified_by_runner=acceptance["report_checkpoint_reload_verified"])
    A.C.save_json(root / "artifact_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--minilm", required=True)
    args = ap.parse_args()
    audit(args.root, args.dataset, args.minilm)
