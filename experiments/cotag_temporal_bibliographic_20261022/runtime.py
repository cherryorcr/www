"""Reuse the already audited constructors, student and numeric repair."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "cotag_fidelity_temporal_20261021"
sys.path.insert(0, str(PRIOR))
import common as C
from common import R, E, torch
import solver_precision as S
from temporal_protocol import snapshot_ids, cohort_masks, validate_delta

sys.path.insert(0, str(HERE))
OUT = C.ROOT / "results/cotag_temporal_bibliographic_20261022"
ARMS = ("uq_dm_edgefree", "cat_cctp", "gecc_hard")
CONSTRUCTION_SEEDS = (271006, 271108, 271109)


def source_data(dataset):
    return OUT / "data" / dataset


def load_data(dataset, device):
    folder = source_data(dataset)
    source = C.read(folder / "manifest.json")
    encoded = C.read(folder / "encoding.json")
    if encoded["dataset_manifest_sha256"] != C.sha(folder / "manifest.json"):
        raise AssertionError("Encoded features use a different dataset manifest")
    if encoded["encoded_data_sha256"] != C.sha(folder / "encoded.pt"):
        raise AssertionError("Encoded data changed")
    raw = torch.load(folder / "encoded.pt", map_location="cpu", weights_only=False)
    h = raw["original_ids"] % 10
    pool = ~torch.isin(h, torch.tensor([1, 2, 3]))
    text_emb = raw["x"].clone()
    text_emb[~pool] = 0
    data = dict(raw, pool_mask=pool, selector_dev_mask=h == 1, report_dev_mask=h == 3,
                test_mask=h == 2, text_emb=text_emb, dataset=dataset,
                encoder=R.L.U.FrozenMiniLM(C.MINILM, device))
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    arrival = data["arrival"]
    ids = [torch.where(pool & (arrival == s))[0] for s in range(5)]
    groups = [torch.unique(data["y"][idx]).tolist() for idx in ids]
    budgets, floors = R.V.choose_budgets({}, list(map(len, ids)), groups, "fixed_rate", .05)
    if bool(text_emb[~pool].any()):
        raise AssertionError("Held-out text passed to construction cache")
    stats = []
    for stage, idx in enumerate(ids):
        validate_delta(idx, arrival, pool, stage)
        visible = snapshot_ids(arrival, stage, "arrived_snapshot")
        delta_edge = R.L.induced_edges(data["edge_index"], idx, len(arrival))
        visible_edge = R.L.induced_edges(data["edge_index"], visible, len(arrival))
        endpoints = arrival[data["edge_index"]]
        legal_edge = pool[data["edge_index"]].all(0)
        current_edges = legal_edge & (endpoints == stage).all(0)
        history_edges = legal_edge & (endpoints <= stage).all(0) & (endpoints == stage).any(0) & (endpoints < stage).any(0)
        lost, kept = int(history_edges.sum()), int(current_edges.sum())
        stats.append(dict(stage=stage, year_end=source["boundary_years"][stage],
            arrival_nodes=int((arrival == stage).sum()), pool_nodes=len(idx),
            current_class_counts=torch.bincount(data["y"][idx], minlength=len(data["names"])).tolist(),
            arrived_nodes=len(visible), arrived_edges=visible_edge.shape[1], delta_pool_edges=delta_edge.shape[1],
            excluded_current_to_history_pool_edges=lost,
            fraction_incident_pool_edges_excluded=lost/(lost+kept) if lost+kept else 0.,
            report_rows=int((data["report_dev_mask"] & (arrival == stage)).sum()),
            test_rows=int((data["test_mask"] & (arrival == stage)).sum())))
    return data, arrival, ids, budgets, floors, source, encoded, stats
