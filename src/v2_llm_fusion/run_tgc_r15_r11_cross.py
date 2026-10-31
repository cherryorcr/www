"""R15: crossed content/topology controls built from sealed R11 memories.

The experiment keeps R11 materialization intact and only crosses the already
sealed R11 content with C0 topology, and C0 content with R11 topology.  It is
an attribution experiment for the new-class plasticity gap, not a new teacher
optimization.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import torch

import tgc_r6_experiment as E
import run_tgc_r11_c0_anchor as R11

R, T, D, B = E.R, E.T, E.D, E.B
SOURCE = Path("/home/pc/research/TAG_condensation/results/tgc_r11_c0_anchor_20260923b")
DATASETS = ("cora", "citeseer")
ORDERS = ("order19601", "order19602", "order19603")
SEEDS = tuple(range(150, 156))
ARMS = ("c0", "anchor025", "anchor_content_c0_topology", "c0_content_anchor_topology")


def protocol():
    return json.loads(json.dumps({
        "name": "tgc-r15-r11-cross-content-topology-v1",
        "datasets": DATASETS, "orders": ORDERS,
        "evaluation_stages": [0, 1, 2], "report_seeds": list(SEEDS),
        "arms": ARMS,
        "source_root": str(SOURCE),
        "intervention": (
            "cross sealed R11 anchor025 content (X/text/mass) with C0 topology, "
            "and C0 content with sealed R11 topology"
        ),
        "history_access": "sealed R11/C0 synthetic memories only; no raw graph",
        "historical_raw_reads": 0, "report_labels_used": False,
        "selection_by_report": False, "development_only": True,
        "independent_confirmation": False,
        "student": {"batch": 16, "lr": .001,
                    "steps": "max(50,min(2048,16*rows))"},
        "promotion_rule": {"mean_final_min_pp": .5, "stage_mean_min_pp": 0.0,
                            "worst_final_min_pp": -2.0, "mean_new_min_pp": 0.0},
        "source_hashes": R.source_hashes(),
    }))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save(path, value):
    E.save(Path(path), value)


def source_files(source):
    names = ["protocol.json", "selection_seal.json"]
    for stage in range(3):
        names.extend([f"c0_s{stage}_memory.pt", f"anchor025_s{stage}_memory.pt"])
    return {name: R.C.sha(source / name) for name in names
            if (source / name).exists()}


def cross_memories(c0, anchor):
    content_c0_topology = T.clone_memory(anchor)
    content_c0_topology["edge_index"] = c0["edge_index"].clone()
    c0_content_anchor_topology = T.clone_memory(c0)
    c0_content_anchor_topology["edge_index"] = anchor["edge_index"].clone()
    return {
        "anchor_content_c0_topology": content_c0_topology,
        "c0_content_anchor_topology": c0_content_anchor_topology,
    }


def prepare(data, order, args, spec):
    ids, budgets = D.set_stream(data, order)
    out = args.output / data["dataset"] / order
    source = SOURCE / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    manifest = E.manifest(data, order, budgets, ids)
    manifest["source_artifact_hashes"] = source_files(source)
    R.stable_lock(out / "manifest.json", manifest)
    if (out / "prepared.json").exists():
        return
    assert not (args.output / "selection_seal.json").exists()
    for stage in range(3):
        c0 = load(source / f"c0_s{stage}_memory.pt")
        anchor = load(source / f"anchor025_s{stage}_memory.pt")
        memories = {"c0": c0, "anchor025": anchor}
        audits = {
            "c0": {"source": str(source / f"c0_s{stage}_memory.pt"),
                   "memory_sha256": T.memory_hash(c0), "historical_raw_reads": 0,
                   "report_labels_used": False},
            "anchor025": {"source": str(source / f"anchor025_s{stage}_memory.pt"),
                          "memory_sha256": T.memory_hash(anchor), "historical_raw_reads": 0,
                          "report_labels_used": False},
        }
        if stage == 0:
            for arm in ARMS[2:]:
                memories[arm] = c0
                audits[arm] = {"alias": "c0", "memory_sha256": T.memory_hash(c0),
                               "historical_raw_reads": 0, "report_labels_used": False}
        else:
            memories.update(cross_memories(c0, anchor))
            audits.update({
                "anchor_content_c0_topology": {
                    "content_source": "anchor025", "topology_source": "c0",
                    "memory_sha256": T.memory_hash(memories["anchor_content_c0_topology"]),
                    "historical_raw_reads": 0, "report_labels_used": False,
                },
                "c0_content_anchor_topology": {
                    "content_source": "c0", "topology_source": "anchor025",
                    "memory_sha256": T.memory_hash(memories["c0_content_anchor_topology"]),
                    "historical_raw_reads": 0, "report_labels_used": False,
                },
            })
        for arm in ARMS:
            audits[arm]["payload"] = R.payload(memories[arm])
            audits[arm]["text"] = R.SCA.text_audit(memories[arm]["texts"], data["encoder"].tokenizer)
            save(out / f"{arm}_s{stage}_build_audit.json", audits[arm])
        decision = {
            "stage": stage, "report_labels_used": False, "historical_raw_reads": 0,
            "memories": {arm: D.save_memory(out, arm, stage, memories[arm]) for arm in ARMS},
            "locked_at": datetime.now(timezone.utc).isoformat(),
        }
        save(out / f"s{stage}_decision.json", decision)
        print(json.dumps({"event": "prepared", "dataset": data["dataset"],
                          "order": order, "stage": stage}), flush=True)
    save(out / "prepared.json", {"status": "prepared"})


def validate(data, out, stage, ids, budgets, args):
    source = SOURCE / data["dataset"] / out.name
    manifest = E.read(out / "manifest.json")
    assert manifest["source_artifact_hashes"] == source_files(source)
    c0 = load(source / f"c0_s{stage}_memory.pt")
    anchor = load(source / f"anchor025_s{stage}_memory.pt")
    for arm in ARMS:
        memory = E.load(out / f"{arm}_s{stage}_memory.pt")
        E.memory_checks(memory, budgets[stage])
        if stage == 0:
            assert T.memory_hash(memory) == T.memory_hash(c0)
            continue
        old = load(SOURCE / data["dataset"] / out.name / f"c0_s{stage-1}_memory.pt")
        for key in ("x", "text_emb", "mass", "depth", "stage"):
            assert torch.equal(memory[key][:len(old["x"])], old[key])
        historical = memory["edge_index"][:, (memory["edge_index"] < len(old["x"])).all(0)]
        assert torch.equal(historical, old["edge_index"])
        if arm == "anchor_content_c0_topology":
            assert torch.equal(memory["edge_index"], c0["edge_index"])
            for key in ("x", "text_emb", "mass"):
                assert torch.equal(memory[key], anchor[key])
            assert memory["texts"] == anchor["texts"]
        elif arm == "c0_content_anchor_topology":
            assert torch.equal(memory["edge_index"], anchor["edge_index"])
            for key in ("x", "text_emb", "mass"):
                assert torch.equal(memory[key], c0[key])
            assert memory["texts"] == c0["texts"]
        elif arm == "c0":
            assert T.memory_hash(memory) == T.memory_hash(c0)
        elif arm == "anchor025":
            assert T.memory_hash(memory) == T.memory_hash(anchor)


if __name__ == "__main__":
    E.main(protocol, prepare, validate)
