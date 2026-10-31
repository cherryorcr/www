from __future__ import annotations

import json

import torch

import direct_student_outer as DSO
import run_student_trajectory_response as TSR


def _memory(rows: int, dim: int = 4) -> dict:
    edge = torch.stack((torch.arange(rows - 1), torch.arange(1, rows)))
    edge = torch.cat((edge, edge.flip(0)), dim=1)
    return {
        "x": torch.arange(rows * dim, dtype=torch.float32).reshape(rows, dim),
        "text_emb": torch.flip(
            torch.arange(rows * dim, dtype=torch.float32).reshape(rows, dim),
            dims=[1]),
        "texts": [f"row-{i}" for i in range(rows)],
        "mass": torch.ones(rows),
        "depth": torch.zeros(rows, dtype=torch.long),
        "stage": torch.zeros(rows, dtype=torch.long),
        "edge_index": edge.long(),
    }


def test_trajectory_uses_same_schedule_and_checkpoints():
    control = _memory(8)
    candidate = DSO.clone_memory(control)
    candidate["x"] = candidate["x"].clone()
    candidate["x"][4:] += 0.05
    support = _memory(8)
    plan = DSO.make_support_plan(support, 4, 4, 9320)
    support_adj = DSO.fixed_norm_adj(support["edge_index"], 8, "cpu")
    left = TSR._trajectory(control, support, support_adj, plan, 150,
                           [0, 2], 4, 1e-3, "cpu")
    right = TSR._trajectory(candidate, support, support_adj, plan, 150,
                            [0, 2], 4, 1e-3, "cpu")
    assert left["schedule_sha256"] == right["schedule_sha256"]
    assert [row["step"] for row in left["trajectory"]] == [0, 2]
    for row in left["trajectory"] + right["trajectory"]:
        assert set(("old", "new", "total")).issubset(row)
        assert all(torch.isfinite(torch.tensor(row[key])) for key in
                   ("old", "new", "total"))


def test_cell_f1_pairs_are_paired_by_seed(tmp_path):
    record = {
        "base": {"rows": [
            {"seed": 150, "cumulative_macro_f1": 40.0,
             "cumulative_accuracy": 30.0, "current_f1": 20.0,
             "old_mean_f1": 60.0},
        ]},
        "candidate": {"rows": [
            {"seed": 150, "cumulative_macro_f1": 41.0,
             "cumulative_accuracy": 31.0, "current_f1": 22.0,
             "old_mean_f1": 59.0},
        ]},
    }
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    pairs = TSR._f1_pairs(record, [150])
    assert pairs[150]["final_f1_pp"] == 1.0
    assert pairs[150]["new_f1_pp"] == 2.0
    assert pairs[150]["old_f1_pp"] == -1.0
