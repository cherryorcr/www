"""Local synthetic split/merge proposals and paired-prefix selection for v6-A."""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

import run_small_comparison_audited as C

U, SCA, G, P = C.U, C.SCA, C.G, C.P
SELECTION_GROUPS = ((6200, 6201), (6300, 6301))
REPORT_SEEDS = tuple(range(9200, 9206))
FAMILIES = ("local", "recluster")
FIELDS = ("x", "text_emb", "texts", "mass", "depth", "stage")


def horizons(steps):
    values = [max(1, int(steps) // d) for d in (8, 4, 2)] + [int(steps)]
    if len(set(values)) != 4:
        raise ValueError("v6-A requires four distinct horizons")
    return values


def joint_points(data, ids):
    return torch.cat((F.normalize(data["x"][ids].float(), dim=1),
                      data["text_emb"][ids].float()), dim=1)


def source_clusters(data, ids, allocation, device):
    records = []
    for cls, count in sorted(allocation.items()):
        members = ids[data["y"][ids] == int(cls)]
        assign = SCA.cluster_assign(joint_points(data, members),
                                   torch.ones(len(members)), int(count), device)
        if int(assign.max()) + 1 != int(count):
            raise AssertionError("clustering returned fewer rows than allocated")
        records.extend({"class": int(cls), "ids": members[assign == j]}
                       for j in range(int(count)))
    return records


def synthesize_records(data, records, stage):
    xs, texts, masses = [], [], []
    for record in records:
        ids = record["ids"]
        if len(ids) == 0 or not data["pool_mask"][ids].all():
            raise AssertionError("invalid synthetic source")
        points = joint_points(data, ids)
        text = G.fuse_new_only(points, torch.arange(len(ids)),
                               [data["texts"][int(i)] for i in ids],
                               budget_chars=320, g=5)
        xs.append(data["x"][ids].float().mean(0))
        texts.append(text)
        masses.append(float(len(ids)))
    return {"x": torch.stack(xs), "texts": texts,
            "text_emb": data["encoder"].encode(texts),
            "mass": torch.tensor(masses),
            "stage": torch.full((len(records),), stage, dtype=torch.long),
            "depth": torch.zeros(len(records), dtype=torch.long)}


def validate_partition(data, ids, records, stage_classes):
    flat = torch.cat([r["ids"] for r in records])
    if not torch.equal(flat.sort().values, ids.sort().values):
        raise AssertionError("source partition duplicated, dropped or introduced rows")
    if not data["pool_mask"][flat].all():
        raise AssertionError("holdout source access")
    for record in records:
        if record["class"] not in stage_classes:
            raise AssertionError("historical or future class source")
        if not (data["y"][record["ids"]] == record["class"]).all():
            raise AssertionError("mixed class synthetic source")


def validate_base(data, ids, records, block, stage, device):
    validate_partition(data, ids, records, data["stage_classes"][stage])
    for row, record in enumerate(records):
        members = record["ids"]
        if not torch.equal(block["x"][row], data["x"][members].float().mean(0)):
            raise AssertionError("C0 cluster reconstruction mismatch")
        text = G.fuse_new_only(joint_points(data, members), torch.arange(len(members)),
                               [data["texts"][int(i)] for i in members], 320, 5)
        if text != block["texts"][row] or block["mass"][row] != len(members):
            raise AssertionError("C0 text or mass reconstruction mismatch")


def local_edit(data, records, donor, recipient, device):
    """Merge the least-cost donor pair; split the highest-SSE recipient row."""
    donor_slots = [i for i, r in enumerate(records) if r["class"] == donor]
    recipient_slots = [i for i, r in enumerate(records)
                       if r["class"] == recipient and len(r["ids"]) >= 2]
    if len(donor_slots) < 2 or not recipient_slots:
        return None
    centers = {i: joint_points(data, records[i]["ids"]).mean(0)
               for i in donor_slots}
    pairs = []
    for offset, left in enumerate(donor_slots):
        for right in donor_slots[offset + 1:]:
            nl, nr = len(records[left]["ids"]), len(records[right]["ids"])
            ward = nl * nr / (nl + nr) * float((centers[left] - centers[right]).square().sum())
            pairs.append((ward, left, right))
    _, left, free = min(pairs)
    def sse(row):
        points = joint_points(data, records[row]["ids"])
        return float((points - points.mean(0)).square().sum())
    split = max(recipient_slots, key=lambda i: (sse(i), -i))
    members = records[split]["ids"]
    assign = SCA.cluster_assign(joint_points(data, members), torch.ones(len(members)), 2, device)
    if len(torch.unique(assign)) != 2:
        return None
    proposal = deepcopy(records)
    proposal[left] = {"class": donor, "ids": torch.cat((records[left]["ids"],
                                                          records[free]["ids"])).sort().values}
    proposal[split] = {"class": recipient, "ids": members[assign == 0]}
    proposal[free] = {"class": recipient, "ids": members[assign == 1]}
    return proposal, {"donor": donor, "recipient": recipient, "merge_keep": left,
                      "merge_free": free, "split": split, "changed_slots": sorted([left, free, split])}


def aligned_recluster(data, ids, base_records, local_records, edit, device):
    """Match a full class recluster to the same slot labels using source geometry."""
    records = deepcopy(base_records)
    for cls in (edit["donor"], edit["recipient"]):
        slots = [i for i, r in enumerate(local_records) if r["class"] == cls]
        rebuilt = source_clusters(data, ids, {cls: len(slots)}, device)
        anchors = torch.stack([joint_points(data, local_records[i]["ids"]).mean(0) for i in slots])
        centers = torch.stack([joint_points(data, r["ids"]).mean(0) for r in rebuilt])
        rows, cols = linear_sum_assignment(torch.cdist(anchors, centers).square().numpy())
        for row, col in zip(rows, cols):
            records[slots[int(row)]] = rebuilt[int(col)]
    return records


def geometry_error(data, records):
    error, count = 0., 0
    for record in records:
        points = joint_points(data, record["ids"])
        error += float((points - points.mean(0)).square().sum())
        count += len(points)
    return error / count


def candidate_block(data, base, base_records, records, stage):
    changed = [i for i, (a, b) in enumerate(zip(base_records, records))
               if a["class"] != b["class"] or not torch.equal(a["ids"], b["ids"])]
    block = {k: deepcopy(base[k]) for k in FIELDS}
    if changed:
        generated = synthesize_records(data, [records[i] for i in changed], stage)
        for k in FIELDS:
            for j, i in enumerate(changed):
                block[k][i] = generated[k][j]
    block["edge_index"] = SCA.memory_edges(block)
    return block, changed


def make_candidates(data, ids, base, stage, device):
    classes = data["stage_classes"][stage]
    allocation = U.allocate_rows_by_class(data, ids, classes, len(base["x"]), "dispersion")
    records = source_clusters(data, ids, allocation, device)
    validate_base(data, ids, records, base, stage, device)
    output = {"c0": {"block": base, "records": records, "family": "c0",
                     "geometry_error": geometry_error(data, records), "changed_slots": [],
                     "allocation": allocation}}
    skipped = []
    for donor in sorted(classes):
        for recipient in sorted(classes):
            if donor == recipient:
                continue
            result = local_edit(data, records, donor, recipient, device)
            if result is None:
                skipped.append({"donor": donor, "recipient": recipient, "reason": "no_valid_split_merge"})
                continue
            local, edit = result
            for family, proposal in (("local", local), ("recluster", aligned_recluster(
                    data, ids, records, local, edit, device))):
                validate_partition(data, ids, proposal, classes)
                block, changed = candidate_block(data, base, records, proposal, stage)
                quotas = {int(c): sum(r["class"] == c for r in proposal) for c in classes}
                expected = dict(allocation)
                expected[donor] -= 1
                expected[recipient] += 1
                if quotas != expected or (family == "local" and len(changed) != 3):
                    raise AssertionError("unexpected marginal edit")
                name = f"{family}_d{donor}_r{recipient}"
                output[name] = {"block": block, "records": proposal, "family": family,
                                "edit": edit, "changed_slots": changed, "allocation": quotas,
                                "geometry_error": geometry_error(data, proposal)}
    return output, skipped


def select_paired_prefix(observations, names, groups=SELECTION_GROUPS):
    """Read selection-only records, never report metrics or full-horizon scores."""
    if "c0" not in names:
        raise ValueError("missing fallback")
    scores, old_gains, horizon_gains = {}, {}, {}
    for name in names:
        gains, old, by_h = [], [], []
        for seed in sum((list(g) for g in groups), []):
            current, base = observations[name][seed], observations["c0"][seed]
            gain = [current[j]["cumulative_f1"] - base[j]["cumulative_f1"] for j in (1, 2)]
            old.append(float(np.mean([current[j]["old_f1"] - base[j]["old_f1"] for j in (1, 2)])))
            gains.append(float(np.mean(gain)))
            by_h.append(gain)
        scores[name], old_gains[name], horizon_gains[name] = gains, old, by_h
    def best(start, stop):
        return max(names, key=lambda n: (float(np.mean(scores[n][start:stop])), n == "c0", n))
    picks, offset = [], 0
    for group in groups:
        picks.append(best(offset, offset + len(group)))
        offset += len(group)
    name = picks[0]
    gains = np.asarray(scores[name])
    lower_proxy = float(gains.mean() - gains.std(ddof=1) / math.sqrt(len(gains)))
    old_by_group, start = [], 0
    for group in groups:
        old_by_group.append(float(np.mean(old_gains[name][start:start + len(group)])))
        start += len(group)
    min_horizon = float(np.asarray(horizon_gains[name]).mean(axis=0).min())
    accepted = (len(set(picks)) == 1 and name != "c0" and lower_proxy > .5
                and min(old_by_group) >= -2. and min_horizon >= 0.)
    return {"selected": name if accepted else "c0", "accepted": bool(accepted),
            "group_picks": picks, "paired_gain_minus_se": lower_proxy,
            "min_old_group_gain": min(old_by_group), "min_horizon_gain": min_horizon,
            "scope": "selector only; quarter/half; 2 disjoint groups; SE is a risk proxy, not CI"}
