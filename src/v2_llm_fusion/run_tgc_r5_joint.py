"""Shared-assignment factorial, recursive rollout, maintenance and capacity."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

import run_tgc_r5_old_confirmation as O
import tgc_shared_assignment as J

D, R, T, M, B = O.D, O.R, O.T, O.M, O.B
read, load, save = O.read, O.load, O.save
DATASETS = ("cora", "citeseer")
ORDERS = ("order19341", "order19342")
SEEDS = tuple(range(100, 106))
ARMS = ("c0", "content", "topology", "joint", "recursive", "maintenance", "capacity")
CFG = J.Config()
TEACHER = replace(M.Config(), teacher_seeds=(8400, 8401), train_seeds=(8500, 8501),
                  validation_seeds=(8510, 8511), probe_seed=8490)


def configure():
    D.ARMS, D.ORDERS, D.SEEDS = ARMS, ORDERS, SEEDS
    O.ARMS, O.ORDERS, O.SEEDS, O.DATASETS = ARMS, ORDERS, SEEDS, DATASETS


def protocol():
    return json.loads(json.dumps({
        "name": "tgc-r5-shared-assignment-v1", "datasets": DATASETS, "orders": ORDERS,
        "class_groups": {d: {o: R.A.stage_groups(d, 3, o) for o in ORDERS} for d in DATASETS},
        "rate": .05, "stages": 3, "report_seeds": SEEDS, "arms": ARMS,
        "assignment": asdict(CFG), "teacher_and_maintenance": asdict(TEACHER),
        "objective": "balanced teacher graph/text reconstruction + random-walk message compression + anchored KL + log-mass penalty",
        "teacher": "two canonical students on lifted legal support; 128 steps, batch64, compact old fraction",
        "assignment_scope": "class-masked current delta soft allocation; old rows identity; no report labels",
        "materialized_content": "soft source-weighted synthetic X; top-5 membership text snippets, 320 chars then MiniLM re-encode",
        "materialized_topology": "P^T A P normalized pair scores; exact C0 new-edge count and one old link per new row; binary symmetric edges",
        "factorial": "content/topology/joint share a single assignment and C0 parent, no individual-win gate",
        "recursive": "each recursive/maintenance/capacity arm reads its own prior output; s0 shared C0",
        "maintenance": "joint then R4 old-X relational response, <=10% old rows, relative displacement <=.1, train old KL<=.001",
        "capacity": "at most one class-row transfer per stage via donor merge/receiver split; fixed total new rows; choose by reconstruction+message",
        "student": {"batch": 16, "lr": .001, "steps": "max(50,min(2048,16*rows))", "scratch": True},
        "primary_metric": "all-prompt pooled report-dev Macro-F1; final and three-stage mean",
        "interaction": "joint - content - topology + C0, paired same cell and seed",
        "development_only": True, "promotion": "no main-method promotion from these reused development conditions alone",
        "test_labels_used": False, "report_before_seal": False,
        "access": "physical offline fixture loader; construction receives only old synthetic/current pool slices; no historical raw replay",
        "source_hashes": R.source_hashes(),
    }))


def append_parent(parent, block, device):
    return R.SCA.append_memory(parent, block, len(parent["x"]), topology_mode="frozen_old",
                               cross_degree=1, refresh_mode="frozen_old", device=device)


def prepare(data, order, args):
    ids, budgets = D.set_stream(data, order)
    out = args.output / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    R.stable_lock(out / "manifest.json", {
        "dataset": data["dataset"], "order": order, "stage_classes": data["stage_classes"],
        "budgets": budgets, "stage_source_ids": [i.tolist() for i in ids],
        "source_dataset_sha256": R.C.sha(R.U.DATA_CFG[data["dataset"]]["path"]),
        "pool_sha256": R.C.tensor_hash({"pool": data["pool_mask"]}),
        "report_mask_sha256": R.C.tensor_hash({"report": data["report_dev_mask"]})})
    for stage in range(3):
        if (out / f"s{stage}_decision.json").exists():
            D.verify_decision(out, stage)
            continue
        if (args.output / "selection_seal.json").exists():
            raise ValueError("cannot construct after seal")
        c0, c0_audit = B.build_c0(data, ids[stage], budgets, stage, out, args.device)
        memories = {a: c0 for a in ARMS}
        decision = {"stage": stage, "construction": c0_audit, "arms": {}, "report_labels_used": False}
        if stage:
            old = load(out / f"c0_s{stage-1}_memory.pt")
            block = load(out / f"c0_s{stage}_block.pt")
            delta = T.raw_delta_block(data, ids[stage], stage)
            labels = data["y"][ids[stage]].cpu()
            q = D.current_assignment(data, ids[stage], c0, len(old["x"]), stage, args.device)
            shared, p, info = J.construct(old, delta, c0, q, labels, data["encoder"], TEACHER, args.device, CFG)
            memories.update(shared)
            save(out / f"shared_s{stage}_construction.json", info)
            torch.save(p, out / f"shared_s{stage}_assignment.pt")
            for arm in ("content", "topology", "joint"):
                decision["arms"][arm] = {"parent_arm": "c0", "parent_hash": T.memory_hash(old),
                                          "construction": f"shared_s{stage}_construction.json"}
            for arm in ("recursive", "maintenance", "capacity"):
                parent = load(out / f"{arm}_s{stage-1}_memory.pt")
                base = append_parent(parent, block, args.device)
                if T.memory_hash(base) == T.memory_hash(c0) and arm != "capacity":
                    candidates, branch_p, audit = shared, p, info
                else:
                    candidates, branch_p, audit = J.construct(parent, delta, base, q, labels, data["encoder"],
                                                              TEACHER, args.device, CFG, capacity=arm == "capacity")
                memory = candidates["joint"]
                D.save_memory(out, f"{arm}_pre_revision", stage, memory)
                if arm == "maintenance":
                    support = J.lift_soft(parent, delta, memory, branch_p)
                    contexts, context_audit = M.build_context(memory, parent, delta, support, stage,
                                                              TEACHER, args.device, diagnostics=False)
                    memory, opt = M.optimize(memory, contexts["lift_compact"], context_audit["editable_old"],
                                             stage, "old", "relation", TEACHER, args.device)
                    save(out / f"maintenance_s{stage}_optimization.json", opt)
                    save(out / f"maintenance_s{stage}_context_audit.json", context_audit)
                    del contexts
                memories[arm] = memory
                save(out / f"{arm}_s{stage}_construction.json", audit)
                torch.save(branch_p, out / f"{arm}_s{stage}_assignment.pt")
                decision["arms"][arm] = {"parent_arm": arm, "parent_hash": T.memory_hash(parent),
                                          "construction": f"{arm}_s{stage}_construction.json"}
        decision["memories"] = {a: D.save_memory(out, a, stage, memories[a]) for a in ARMS}
        decision["locked_at"] = datetime.now(timezone.utc).isoformat()
        save(out / f"s{stage}_decision.json", decision)
        print(json.dumps({"event": "prepared", "dataset": data["dataset"], "order": order, "stage": stage}), flush=True)
    save(out / "prepared.json", {"status": "prepared"})


def edge_set(memory):
    return set(map(tuple, memory["edge_index"].T.tolist()))


def validate(out, stage, budgets):
    c0 = load(out / f"c0_s{stage}_memory.pt")
    decision = read(out / f"s{stage}_decision.json")
    for arm in ARMS:
        memory = load(out / f"{arm}_s{stage}_memory.pt")
        edges = edge_set(memory)
        assert len(memory["x"]) == budgets[stage]
        assert memory["edge_index"].shape == c0["edge_index"].shape
        assert len(edges) == memory["edge_index"].shape[1]
        assert all(i != j and 0 <= i < budgets[stage] and 0 <= j < budgets[stage] for i, j in edges)
        assert all((j, i) in edges for i, j in edges)
        assert all(torch.isfinite(memory[k]).all() for k in ("x", "text_emb", "mass"))
        if not stage:
            assert T.memory_hash(memory) == T.memory_hash(c0)
            continue
        if arm == "c0":
            continue
        parent_arm = decision["arms"][arm]["parent_arm"]
        parent = load(out / f"{parent_arm}_s{stage-1}_memory.pt")
        assert T.memory_hash(parent) == decision["arms"][arm]["parent_hash"]
        old_n = len(parent["x"])
        historical = memory["edge_index"][:, (memory["edge_index"] < old_n).all(0)]
        assert torch.equal(historical, parent["edge_index"])
        for key in ("text_emb", "mass", "depth", "stage"):
            assert torch.equal(memory[key][:old_n], parent[key])
        assert memory["texts"][:old_n] == parent["texts"]
        if arm == "maintenance":
            before = load(out / f"maintenance_pre_revision_s{stage}_memory.pt")
            opt = read(out / f"maintenance_s{stage}_optimization.json")
            M.validate_memory(before, memory, opt["rows"], TEACHER.radius)
            assert opt["final"]["train"]["old_kl"] <= TEACHER.old_epsilon + 1e-7
            assert opt["nonzero_gradient_steps"] == TEACHER.outer_steps
        else:
            assert torch.equal(memory["x"][:old_n], parent["x"])
    if stage:
        content, topology, joint = (load(out / f"{a}_s{stage}_memory.pt") for a in ("content", "topology", "joint"))
        assert torch.equal(content["edge_index"], c0["edge_index"])
        assert torch.equal(topology["x"], c0["x"]) and torch.equal(topology["text_emb"], c0["text_emb"])
        assert torch.equal(joint["x"], content["x"]) and torch.equal(joint["text_emb"], content["text_emb"])
        assert torch.equal(joint["edge_index"], topology["edge_index"])
        cap = read(out / f"capacity_s{stage}_construction.json")["capacity"]
        assert sum(cap["selected_allocation"].values()) == budgets[stage] - budgets[stage-1]
        assert cap["rows_transferred"] <= 1


def main():
    configure()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--phase", choices=("prepare", "seal", "evaluate", "audit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--text-cache-root", type=Path,
                        default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    R.stable_lock(args.output / "protocol.json", protocol())
    if args.phase == "seal":
        O.seal(args.output)
    else:
        if not args.dataset:
            parser.error("dataset required")
        if args.phase in ("evaluate", "audit"):
            D.verify_seal(args.output)
        data = R.R.load_data(args.dataset, args.minilm, args.device,
                             args.text_cache_root / f"{args.dataset}_r005" / "text_emb_pool_only.pt")
        for order in ORDERS:
            if args.phase == "prepare":
                prepare(data, order, args)
            elif args.phase == "evaluate":
                D.evaluate(data, order, args)
            else:
                O.audit_scores(data, order, args, validate)
    print(f"TGC_R5_JOINT_{args.phase.upper()}_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
