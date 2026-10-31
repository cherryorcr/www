"""R13: bounded student-response gate for the R11 anchor candidate.

Only C0 and the sealed R11 anchor025 memory are compared.  A scratch student
trajectory evaluates unlabeled old/new/mixed InfoNCE response; the candidate
is deployed only when a preregistered response rule accepts it, otherwise the
decision is exactly C0.  Report F1 is never read during construction.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import math

import torch

import tgc_r6_experiment as E
import tgc_selector_confirmation as Selector
import tgc_student_revision as StudentRevision
import run_tgc_r11_c0_anchor as R11

R, T, D, B = E.R, E.T, E.D, E.B
SOURCE = Path("/home/pc/research/TAG_condensation/results/tgc_r12_classfloor_20260923a")
DATASETS = ("cora", "citeseer")
ORDERS = ("order19601", "order19602", "order19603")
SEEDS = tuple(range(150, 156))
SELECTOR_SEEDS = (13000, 13001, 13002)
CHECKPOINTS = (32, 64, 128)
GAIN_MIN = 1e-4
OLD_TOLERANCE = 0.01
ARMS = ("c0", "anchor025", "selected_curve", "selected_new64",
        "selected_conservative")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save(path, value):
    E.save(Path(path), value)


def source_files(source):
    names = ["protocol.json", "selection_seal.json"]
    for dataset in DATASETS:
        for order in ORDERS:
            for stage in range(3):
                names.extend([f"{dataset}/{order}/c0_s{stage}_memory.pt",
                              f"{dataset}/{order}/anchor025_s{stage}_memory.pt"])
    return {name: R.C.sha(source / name) for name in names}


def protocol():
    return json.loads(json.dumps({
        "name": "tgc-r13-bounded-plasticity-gate-v1",
        "datasets": DATASETS,
        "orders": ORDERS,
        "evaluation_stages": [0, 1, 2],
        "report_seeds": list(SEEDS),
        "selector_seeds": list(SELECTOR_SEEDS),
        "selector_checkpoints": list(CHECKPOINTS),
        "arms": ARMS,
        "candidate": "sealed R11 anchor025",
        "source_root": str(SOURCE),
        "source_artifact_root": str(SOURCE),
        "response_rule": {
            "gain_min": GAIN_MIN,
            "old_tolerance": OLD_TOLERANCE,
            "min_positive_seeds": 2,
            "selected_curve": "mean mixed InfoNCE gain at 32/64/128; old safe at all checkpoints",
            "selected_new64": "mean new InfoNCE gain at 64; old safe at 64",
            "selected_conservative": "all selector seeds positive at 64; old safe at all checkpoints",
        },
        "probe": "old synthetic rows plus current raw delta; unlabeled old/new/mixed InfoNCE",
        "history_access": "sealed synthetic parent + current raw delta only; no historical raw graph",
        "selection_by_report": False,
        "report_labels_used": False,
        "development_only": True,
        "independent_confirmation": False,
        "student": {"batch": 16, "lr": .001, "steps": "max(50,min(2048,16*rows))"},
        "source_protocol": "R12 sealed C0/anchor025 memories; R12 class-floor arms are not reused",
        "source_hashes": R.source_hashes(),
    }))


def _gain_records(records):
    details = []
    for record in records:
        curves = {name: {int(point["step"]): point for point in record[name]}
                  for name in ("c0", "candidate")}
        if any(set(curves[name]) != set(CHECKPOINTS) for name in curves):
            raise ValueError("incomplete selector curve")
        gains = {str(step): {
            key: curves["c0"][step][key] - curves["candidate"][step][key]
            for key in ("old", "new", "mixed")}
            for step in CHECKPOINTS}
        if not all(math.isfinite(value) for point in gains.values()
                   for value in point.values()):
            raise FloatingPointError("nonfinite selector gain")
        details.append({
            "seed": record["seed"],
            "gains": gains,
            "curve_score": sum(gains[str(step)]["mixed"] for step in CHECKPOINTS) / len(CHECKPOINTS),
        })
    return details


def _policy(details, name):
    if name == "selected_curve":
        scores = [item["curve_score"] for item in details]
        safe_steps = CHECKPOINTS
    elif name == "selected_new64":
        scores = [item["gains"]["64"]["new"] for item in details]
        safe_steps = (64,)
    elif name == "selected_conservative":
        scores = [item["gains"]["64"]["new"] for item in details]
        safe_steps = CHECKPOINTS
    else:
        raise ValueError(name)
    old_safe = all(item["gains"][str(step)]["old"] >= -OLD_TOLERANCE
                   for item in details for step in safe_steps)
    positive = sum(value >= GAIN_MIN for value in scores)
    accepted = (sum(scores) / len(scores) >= GAIN_MIN
                and positive >= (len(details) if name == "selected_conservative" else 2)
                and old_safe)
    return {"mean_gain": sum(scores) / len(scores),
            "positive_seeds": positive,
            "old_safe": old_safe,
            "accepted": accepted,
            "selected_arm": "anchor025" if accepted else "c0"}


def response_scores(base, candidate, support, stage, device, out):
    batches = Selector.probe_batches(support["old_rows"], len(support["x"]), stage)
    records = []
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    for seed in SELECTOR_SEEDS:
        trajectories = {}
        for name, memory in (("c0", base), ("candidate", candidate)):
            trajectories[name] = Selector.scratch_trajectory(
                memory, support, batches, seed, device, checkpoints=CHECKPOINTS)
        left, right = trajectories["c0"], trajectories["candidate"]
        if left["start_hash"] != right["start_hash"]:
            raise AssertionError("candidate and C0 start hashes differ")
        if left["batch_sequence"] != right["batch_sequence"]:
            raise AssertionError("candidate and C0 selector batches differ")
        path = out / f"r13_s{stage}_selector_seed{seed}.pt"
        torch.save(trajectories, path)
        records.append({"seed": seed, "start_hash": left["start_hash"],
                        "c0": left["curve"], "candidate": right["curve"],
                        "checkpoint_file": path.name,
                        "checkpoint_sha256": R.C.sha(path)})
    details = _gain_records(records)
    policies = {name: _policy(details, name)
                for name in ("selected_curve", "selected_new64", "selected_conservative")}
    return {
        "selector_seeds": list(SELECTOR_SEEDS),
        "checkpoints": list(CHECKPOINTS),
        "per_seed": records,
        "per_seed_gains": details,
        "policies": policies,
        "probe_batches": {key: [value.tolist() for value in values]
                          for key, values in batches.items()},
        "historical_raw_reads": 0,
        "report_labels_used": False,
        "probe_scope": "old synthetic plus current raw delta; unlabeled InfoNCE",
    }


def prepare(data, order, args, spec):
    ids, budgets = D.set_stream(data, order)
    out = args.output / data["dataset"] / order
    source = SOURCE / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    manifest = E.manifest(data, order, budgets, ids)
    manifest["source_artifact_hashes"] = source_files(SOURCE)
    R.stable_lock(out / "manifest.json", manifest)
    if (out / "prepared.json").exists():
        return
    if (args.output / "selection_seal.json").exists():
        raise ValueError("cannot prepare after seal")
    for stage in range(3):
        c0 = load(source / f"c0_s{stage}_memory.pt")
        anchor = load(source / f"anchor025_s{stage}_memory.pt")
        memories = {"c0": c0, "anchor025": anchor,
                    "selected_curve": c0, "selected_new64": c0,
                    "selected_conservative": c0}
        decision = {
            "stage": stage,
            "source_c0_hash": T.memory_hash(c0),
            "source_anchor_hash": T.memory_hash(anchor),
            "selection_by_report": False,
            "report_labels_used": False,
            "historical_raw_reads": 0,
            "policies": {},
        }
        if stage == 0:
            response = None
        else:
            old = load(source / f"c0_s{stage-1}_memory.pt")
            delta = T.raw_delta_block(data, ids[stage], stage)
            support = T._concat_support(old, delta)
            response = response_scores(c0, anchor, support, stage, args.device, out)
            decision["response"] = response
            for policy, info in response["policies"].items():
                decision["policies"][policy] = info
                memories[policy] = anchor if info["accepted"] else c0
        if response is None:
            for policy in ("selected_curve", "selected_new64", "selected_conservative"):
                decision["policies"][policy] = {
                    "accepted": False, "selected_arm": "c0",
                    "mean_gain": 0.0, "positive_seeds": 0,
                    "old_safe": True,
                }
        for arm in ARMS:
            audit = {
                "arm": arm,
                "source": str(source / f"{('anchor025' if arm == 'anchor025' else 'c0')}_s{stage}_memory.pt"),
                "memory_sha256": T.memory_hash(memories[arm]),
                "report_labels_used": False,
                "historical_raw_reads": 0,
                "selected_from": ("anchor025" if memories[arm] is anchor else "c0"),
            }
            if response is not None and arm.startswith("selected_"):
                audit["policy"] = response["policies"][arm]
            save(out / f"{arm}_s{stage}_build_audit.json", audit)
        decision["memories"] = {arm: D.save_memory(out, arm, stage, memories[arm])
                                 for arm in ARMS}
        decision["locked_at"] = datetime.now(timezone.utc).isoformat()
        save(out / f"s{stage}_decision.json", decision)
        print(json.dumps({"event": "prepared", "dataset": data["dataset"],
                          "order": order, "stage": stage,
                          "policies": decision["policies"]}), flush=True)
    save(out / "prepared.json", {"status": "prepared"})


def validate(data, out, stage, ids, budgets, args):
    manifest = E.read(out / "manifest.json")
    assert manifest["source_artifact_hashes"] == source_files(SOURCE)
    source = SOURCE / data["dataset"] / out.name
    decision = E.read(out / f"s{stage}_decision.json")
    assert decision["selection_by_report"] is False
    assert decision["report_labels_used"] is False
    for arm in ARMS:
        memory = E.load(out / f"{arm}_s{stage}_memory.pt")
        E.memory_checks(memory, budgets[stage])
        expected = decision["memories"][arm]
        assert expected["memory_sha256"] == T.memory_hash(memory)
        if stage == 0:
            assert T.memory_hash(memory) == T.memory_hash(E.load(source / "c0_s0_memory.pt"))
    if stage > 0:
        old = E.load(source / f"c0_s{stage-1}_memory.pt")
        for arm in ARMS:
            memory = E.load(out / f"{arm}_s{stage}_memory.pt")
            for key in ("x", "text_emb", "mass", "depth", "stage"):
                assert torch.equal(memory[key][:len(old["x"])], old[key])
            historical = memory["edge_index"][:, (memory["edge_index"] < len(old["x"])).all(0)]
            assert torch.equal(historical, old["edge_index"])
        response = decision["response"]
        assert response["historical_raw_reads"] == 0
        assert response["report_labels_used"] is False
        for policy in ("selected_curve", "selected_new64", "selected_conservative"):
            selected = decision["policies"][policy]["selected_arm"]
            assert selected in ("c0", "anchor025")
            assert T.memory_hash(E.load(out / f"{policy}_s{stage}_memory.pt")) == T.memory_hash(E.load(out / f"{selected}_s{stage}_memory.pt"))


if __name__ == "__main__":
    E.main(protocol, prepare, validate)
