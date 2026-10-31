"""R16: topology-safe materialization for the R11 content candidate.

R15 showed that the R11 assignment/X/text signal can help when its coarse
topology is replaced by C0 topology.  R16 keeps the R11 ``anchor025``
content completely fixed and changes only the new-block topology:

* ``degree_safe`` preserves C0 old-old/cross edges, the C0 number of new-new
  edges, and every new node's internal degree, then applies deterministic
  content-scored 2-switches inside the new block;
* ``cross_guard`` starts from C0 topology and only accepts an R11 cross edge
  when its content score clears a fixed margin and the old endpoint remains
  within a C0-derived capacity.  Otherwise it falls back to C0.

No historical raw graph is read and old synthetic rows/old-old edges are not
modified.  This is a topology diagnosis, not a new student selector.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import tgc_r6_experiment as E
import run_tgc_r11_c0_anchor as R11

R, T, D, B = E.R, E.T, E.D, E.B
SOURCE = Path("/home/pc/research/TAG_condensation/results/tgc_r11_c0_anchor_20260923b")
DATASETS = ("cora", "citeseer")
ORDERS = ("order19601", "order19602", "order19603")
SEEDS = tuple(range(150, 156))
ARMS = ("c0", "anchor025", "anchor_content_c0_topology",
        "degree_safe", "cross_guard")
SWITCH_MARGIN = 0.05


def protocol():
    return json.loads(json.dumps({
        "name": "tgc-r16-topology-safe-v1",
        "datasets": DATASETS, "orders": ORDERS,
        "evaluation_stages": [0, 1, 2], "report_seeds": list(SEEDS),
        "arms": ARMS, "source_root": str(SOURCE),
        "intervention": (
            "freeze R11 anchor025 X/text/mass; keep C0 scaffold; apply "
            "degree-preserving new-new content rewiring and guarded cross "
            "edge replacement"
        ),
        "content_source": "sealed R11 anchor025",
        "topology_controls": {
            "degree_safe": "C0 old-old/cross; C0 new-new count and per-new internal degree; content-scored 2-switches",
            "cross_guard": "C0 topology fallback; R11 cross candidate only above margin and C0-derived old endpoint capacity",
            "switch_margin": SWITCH_MARGIN,
        },
        "text": "unchanged sealed R11 top-5/320-char concat-reencode",
        "history_access": "sealed C0/R11 synthetic memories only; no historical raw graph",
        "historical_raw_reads": 0, "report_labels_used": False,
        "selection_by_report": False, "development_only": True,
        "independent_confirmation": False,
        "student": {"batch": 16, "lr": .001,
                    "steps": "max(50,min(2048,16*rows))"},
        "promotion_rule": {"mean_final_min_pp": .5, "stage_mean_min_pp": 0.0,
                            "worst_final_min_pp": -2.0, "mean_new_min_pp": 0.0,
                            "positive_final_cells": 4},
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
    return {name: R.C.sha(source / name) for name in names if (source / name).exists()}


def undirected_pairs(edges, lower, upper):
    """Return sorted i<j pairs from a symmetric edge tensor."""
    edges = edges.cpu()
    mask = (edges[0] >= lower) & (edges[0] < upper)
    mask &= (edges[1] >= lower) & (edges[1] < upper)
    selected = edges[:, mask]
    if selected.numel() == 0:
        return []
    result = set()
    for left, right in selected.T.tolist():
        if left == right:
            continue
        result.add(tuple(sorted((int(left), int(right)))))
    return sorted(result)


def cross_pairs(edges, old_n, new_n):
    edges = edges.cpu()
    mask = ((edges[0] < old_n) & (edges[1] >= old_n) &
            (edges[1] < new_n))
    return sorted(set((int(a), int(b)) for a, b in edges[:, mask].T.tolist()))


def content_scores(memory):
    x = F.normalize(memory["x"].float(), dim=1)
    text = F.normalize(memory["text_emb"].float(), dim=1)
    return 0.5 * (x @ x.T) + 0.5 * (text @ text.T)


def score_pair(scores, pair):
    return float(scores[pair[0], pair[1]])


def degree_safe_edges(c0, anchor, old_n):
    """C0 scaffold plus degree-preserving, content-scored 2-switches."""
    total = len(c0["x"])
    c0_internal = undirected_pairs(c0["edge_index"], old_n, total)
    current = set(c0_internal)
    target_degree = {node: 0 for node in range(old_n, total)}
    for left, right in current:
        target_degree[left] += 1
        target_degree[right] += 1
    scores = content_scores(anchor)

    # A 2-switch preserves both edge count and every endpoint degree.  Search
    # the best positive move repeatedly; the small synthetic blocks make this
    # deterministic local improvement inexpensive and auditable.
    max_rounds = max(1, 4 * len(current))
    switches = []
    for _ in range(max_rounds):
        edges = sorted(current)
        best = None
        for index, first in enumerate(edges):
            a, b = first
            for second in edges[index + 1:]:
                c, d = second
                if len({a, b, c, d}) < 4:
                    continue
                old_score = score_pair(scores, first) + score_pair(scores, second)
                for candidate in (((a, c), (b, d)), ((a, d), (b, c))):
                    normalized = tuple(sorted(tuple(sorted(edge)) for edge in candidate))
                    if normalized[0] == normalized[1] or any(left == right for left, right in normalized):
                        continue
                    if any(edge in current for edge in normalized):
                        continue
                    gain = (score_pair(scores, normalized[0]) +
                            score_pair(scores, normalized[1]) - old_score)
                    if gain <= 1e-8:
                        continue
                    key = (gain, tuple(-v for edge in normalized for v in edge))
                    if best is None or key > best[0]:
                        best = (key, first, second, normalized, gain)
        if best is None:
            break
        _, first, second, normalized, gain = best
        current.remove(first)
        current.remove(second)
        current.update(normalized)
        switches.append({"removed": [list(first), list(second)],
                         "added": [list(edge) for edge in normalized],
                         "gain": gain})

    old_edges = c0["edge_index"][:, (c0["edge_index"] < old_n).all(0)].cpu()
    cross = c0["edge_index"][:,
             (c0["edge_index"][0] < old_n) ^ (c0["edge_index"][1] < old_n)].cpu()
    internal = torch.tensor(sorted(current), dtype=torch.long).T if current else torch.empty((2, 0), dtype=torch.long)
    internal = torch.cat((internal, internal.flip(0)), dim=1)
    result = torch.cat((old_edges, internal, cross), dim=1)
    return result, {"mode": "degree_safe", "switches": switches,
                    "switch_count": len(switches),
                    "target_internal_edges": len(c0_internal),
                    "target_internal_degree": target_degree}


def guarded_cross_edges(c0, anchor, old_n):
    """Use R11 cross candidates only when content gain is safely positive."""
    total = len(c0["x"])
    c0_cross = cross_pairs(c0["edge_index"], old_n, total)
    anchor_cross = cross_pairs(anchor["edge_index"], old_n, total)
    c0_by_new = {new: old for old, new in c0_cross}
    anchor_by_new = {new: old for old, new in anchor_cross}
    scores = content_scores(anchor)
    old_load = {}
    for old, _ in c0_cross:
        old_load[old] = old_load.get(old, 0) + 1
    capacity = max(1, max(old_load.values(), default=1))
    chosen = dict(c0_by_new)
    changed = []
    for new in sorted(c0_by_new):
        candidate = anchor_by_new.get(new)
        baseline = c0_by_new[new]
        if candidate is None or candidate == baseline:
            continue
        gain = score_pair(scores, (candidate, new)) - score_pair(scores, (baseline, new))
        if gain > SWITCH_MARGIN and old_load.get(candidate, 0) < capacity:
            chosen[new] = candidate
            old_load[baseline] = max(0, old_load.get(baseline, 0) - 1)
            old_load[candidate] = old_load.get(candidate, 0) + 1
            changed.append({"new": new, "from_old": baseline, "to_old": candidate,
                            "content_gain": gain})

    old_edges = c0["edge_index"][:, (c0["edge_index"] < old_n).all(0)].cpu()
    internal = c0["edge_index"][:,
               (c0["edge_index"][0] >= old_n) &
               (c0["edge_index"][1] >= old_n)].cpu()
    # Keep the exact C0 internal block and reconstruct only the cross block.
    cross = torch.tensor([[old, new] for new, old in sorted(chosen.items())], dtype=torch.long).T
    if cross.numel():
        cross = torch.cat((cross, cross.flip(0)), dim=1)
    else:
        cross = torch.empty((2, 0), dtype=torch.long)
    result = torch.cat((old_edges, internal, cross), dim=1)
    return result, {"mode": "cross_guard", "switch_margin": SWITCH_MARGIN,
                    "old_endpoint_capacity": capacity, "changed": changed,
                    "changed_count": len(changed), "cross_edges": len(chosen)}


def make_memories(c0, anchor, old_n):
    content_c0 = T.clone_memory(anchor)
    content_c0["edge_index"] = c0["edge_index"].clone()
    degree_edges, degree_audit = degree_safe_edges(c0, anchor, old_n)
    cross_edges_value, cross_audit = guarded_cross_edges(c0, anchor, old_n)
    degree = T.clone_memory(anchor)
    degree["edge_index"] = degree_edges
    guarded = T.clone_memory(anchor)
    guarded["edge_index"] = cross_edges_value
    return {
        "anchor_content_c0_topology": content_c0,
        "degree_safe": degree,
        "cross_guard": guarded,
    }, {"degree_safe": degree_audit, "cross_guard": cross_audit}


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
            old = load(source / f"c0_s{stage-1}_memory.pt")
            derived, topology_audit = make_memories(c0, anchor, len(old["x"]))
            memories.update(derived)
            audits.update({arm: {"content_source": "anchor025", "topology_source": "c0-constrained",
                                 "topology_audit": topology_audit["degree_safe" if arm == "degree_safe" else "cross_guard"],
                                 "memory_sha256": T.memory_hash(memory),
                                 "historical_raw_reads": 0, "report_labels_used": False}
                           for arm, memory in derived.items() if arm != "anchor_content_c0_topology"})
            audits["anchor_content_c0_topology"] = {
                "content_source": "anchor025", "topology_source": "c0",
                "memory_sha256": T.memory_hash(memories["anchor_content_c0_topology"]),
                "historical_raw_reads": 0, "report_labels_used": False,
            }
        for arm, memory in memories.items():
            audits[arm]["payload"] = R.payload(memory)
            audits[arm]["text"] = R.SCA.text_audit(memory["texts"], data["encoder"].tokenizer)
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
        old = load(source / f"c0_s{stage-1}_memory.pt")
        for key in ("x", "text_emb", "mass", "depth", "stage"):
            assert torch.equal(memory[key][:len(old["x"])], old[key])
        assert memory["texts"][:len(old["x"])] == old["texts"]
        historical = memory["edge_index"][:, (memory["edge_index"] < len(old["x"])).all(0)]
        assert torch.equal(historical, old["edge_index"])
        if arm == "c0":
            assert T.memory_hash(memory) == T.memory_hash(c0)
        elif arm == "anchor025":
            assert T.memory_hash(memory) == T.memory_hash(anchor)
        else:
            for key in ("x", "text_emb", "mass"):
                assert torch.equal(memory[key], anchor[key])
            assert memory["texts"] == anchor["texts"]
            audit = E.read(out / f"{arm}_s{stage}_build_audit.json")
            if arm == "anchor_content_c0_topology":
                assert torch.equal(memory["edge_index"], c0["edge_index"])
            else:
                old_n = len(old["x"])
                old_edges = memory["edge_index"][:, (memory["edge_index"] < old_n).all(0)]
                assert torch.equal(old_edges, old["edge_index"])
                assert audit["topology_audit"]["mode"] in ("degree_safe", "cross_guard")
    return None


if __name__ == "__main__":
    E.main(protocol, prepare, validate)
