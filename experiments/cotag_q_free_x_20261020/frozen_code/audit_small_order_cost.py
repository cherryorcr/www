"""Independent artifact/source/selector audit; never selects candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import run_small_order_cost_confirmation as R

C, U, SCA, G = R.C, R.U, R.SCA, R.G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)
    audit = {"datasets": [], "errors": [], "test_labels_evaluated": False}
    for dataset in ("cora", "citeseer"):
        base = args.root / dataset
        plan = json.loads((base / "plan.json").read_text())
        device = args.device or plan["device"]
        amendment = json.loads((base / "validation_amendment.json").read_text())
        if (amendment["current_runner_sha256"] != C.sha(R.__file__)
                or amendment["original_plan_sha256"] != C.sha(base / "plan.json")):
            raise AssertionError("validation amendment provenance mismatch")
        for filename, expected in plan["code_sha256"].items():
            source = (args.root / "runner_v1_snapshot.py"
                      if Path(filename).name == Path(R.__file__).name else Path(filename))
            if C.sha(source) != expected:
                raise AssertionError(f"frozen code provenance mismatch: {filename}")
        diagnostic = args.root / "cora/raw_regression_diagnostic.json"
        if amendment["diagnostic_sha256"] != C.sha(diagnostic):
            raise AssertionError("numerical diagnostic provenance mismatch")
        preprocessing = json.loads((base / "preprocessing.json").read_text())
        if preprocessing["source_hash"] != C.sha(U.DATA_CFG[dataset]["path"]):
            raise AssertionError("dataset provenance mismatch")
        if plan["orders"] != R.stage_orders(dataset):
            raise AssertionError("order preregistration mismatch")
        data = U.load_unified(dataset, U.DATA_CFG[dataset]["path"],
                             str(Path(U.__file__).parent / "minilm_from_graphclip"), device)
        data["text_emb"] = torch.load(C.artifact_dir(dataset, "v4") / "text_emb_pool_only.pt",
                                     map_location="cpu", weights_only=False)
        checked, rebuilt, selectors, student_fits = 0, 0, 0, 0
        allocations, parameter_bytes = [], set()
        for order, groups in plan["orders"].items():
            out = base / order
            stages, budgets = R.set_stages(data, groups)
            payload = json.loads((out / "results.json").read_text())
            for arm in plan["arms"]:
                old = None
                for stage in range(3):
                    if arm.endswith("finetune"):
                        continue
                    memory_path = out / f"{arm}_s{stage}_memory.pt"
                    memory = torch.load(memory_path, map_location="cpu", weights_only=False)
                    saved = json.loads((out / f"{arm}_s{stage}_construction.json").read_text())
                    if C.tensor_hash(memory) != saved["tensor_sha256"]:
                        raise AssertionError("construction digest mismatch")
                    if "source_ids" in memory:
                        ids = memory["source_ids"]
                        if not data["pool_mask"][ids].all():
                            raise AssertionError("holdout source")
                        seen = torch.cat(stages[:stage+1])
                        if not torch.isin(ids, seen).all():
                            raise AssertionError("future source")
                        if not torch.equal(memory["x"], data["x"][ids].float()):
                            raise AssertionError("real-source feature mismatch")
                    if arm not in ("full_history_syn", "recluster", "raw_equal", "raw_long"):
                        C.assert_frozen(old, memory)
                    if arm in ("v4", "dispersion"):
                        keep = len(old["x"]) if old else 0
                        n = budgets[stage] - keep
                        if arm == "v4":
                            selection = saved["selection"]
                            alloc = {int(k): v for k, v in selection["selected_class_rows"].items()}
                            if set(alloc) != set(groups[stage]) or sum(alloc.values()) != n:
                                raise AssertionError("class allocation mismatch")
                            block = U.build_stage_block_from_class_rows(data, stages[stage], alloc,
                                data["encoder"], stage, "joint_g5_c320", "concat", device,
                                inner_mode="dispersion", diversity_lambda=.5, dual_alpha=.5)
                            counts = selection["response_evaluations"]
                            if selection["selection_training_runs"] != counts * 3 * 3 * len(selection["response_steps"]):
                                raise AssertionError("selector fit accounting mismatch")
                            if selection["selection_optimizer_steps_total"] != counts * 3 * 3 * sum(selection["response_steps"]):
                                raise AssertionError("selector step accounting mismatch")
                            selectors += selection["selection_training_runs"]
                            prior = U._base_class_allocation(data, stages[stage], groups[stage], n, "dispersion")
                            allocations.append({"order": order, "stage": stage,
                                "prior": prior, "selected": alloc, "changed": prior != alloc,
                                "net_rows_moved": sum(abs(prior[k] - alloc[k]) for k in prior) // 2,
                                "dispersion_v4_class_enumeration_differs": groups[stage] != sorted(groups[stage])})
                        else:
                            block = U.build_stage_block(data, stages[stage], n, data["encoder"],
                                stage, "joint_g5_c320", "concat", device, "dispersion",
                                groups[stage], .5, .5)
                        reproduced = SCA.append_memory(old, block, keep, topology_mode="frozen_old",
                            cross_degree=1, refresh_mode="frozen_old", device=device)
                        if C.tensor_hash(reproduced) != C.tensor_hash(memory) or reproduced["texts"] != memory["texts"]:
                            raise AssertionError("synthetic rebuild mismatch")
                        rebuilt += 1
                    old = memory
                    checked += 1
            for result in payload["results"]:
                for row in result["per_seed"]:
                    key = f"{result['arm']}_s{result['stage']}_seed{row['seed']}"
                    saved_row = json.loads((out / f"{key}.json").read_text())
                    if row != saved_row:
                        raise AssertionError("summary does not match per-fit evidence")
                    state = torch.load(out / f"{key}.pt", map_location="cpu", weights_only=False)
                    if C.tensor_hash(state["state"]) != row["end_state_sha256"]:
                        raise AssertionError("model digest mismatch")
                    parameter_bytes.add(sum(v.numel() * v.element_size() for v in state["state"].values()))
                    student_fits += 1
        regression = json.loads((base / "canonical/canonical_regression.json").read_text())
        if len(regression) != 234 or (checked, rebuilt, student_fits) != (99, 18, 702):
            raise AssertionError("independent audit coverage incomplete")
        audit["datasets"].append({"dataset": dataset, "memory_artifacts_checked": checked,
            "frozen_code_files_checked": len(plan["code_sha256"]),
            "source_and_amendment_provenance_checked": True,
            "synthetic_artifacts_rebuilt": rebuilt, "selector_fits": selectors,
            "report_fit_digests_checked": student_fits,
            "canonical_endpoint_drift_fits": sum(not r["end_state_bitwise_equal"] for r in regression),
            "student_state_tensor_bytes": sorted(parameter_bytes),
            "v4_allocation_decisions": allocations,
            "errors": []})
        print(json.dumps(audit["datasets"][-1]), flush=True)
    C.save_json(args.root / "independent_acceptance.json", audit)


if __name__ == "__main__":
    main()
