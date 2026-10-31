"""v6-A candidate locality, selection isolation and training-prefix tests."""
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

import v6_local_marginal as V
import run_v6_local_marginal as R

torch.set_num_threads(2)


class Encoder:
    def encode(self, texts):
        result = []
        for text in texts:
            gen = torch.Generator().manual_seed(sum(map(ord, text)))
            result.append(F.normalize(torch.rand(384, generator=gen), dim=0))
        return torch.stack(result)


def data():
    gen = torch.Generator().manual_seed(45)
    d = {"x": torch.randn(72, 8, generator=gen),
         "text_emb": F.normalize(torch.randn(72, 384, generator=gen), dim=1),
         "texts": [f"source text {i} content" for i in range(72)],
         "y": torch.arange(3).repeat_interleave(24), "pool_mask": torch.ones(72, dtype=torch.bool),
         "stage_classes": [[0], [1, 2]], "encoder": Encoder()}
    ids = torch.arange(24, 72)
    b = V.U.build_stage_block(d, ids, 8, d["encoder"], 1, "joint_g5_c320", "concat",
                              "cpu", "dispersion", [1, 2])
    b["edge_index"] = V.SCA.memory_edges(b)
    return d, ids, b


def test_locality_same_budget_partition_and_recluster_slots():
    d, ids, base = data()
    candidates, skipped = V.make_candidates(d, ids, base, 1, "cpu")
    assert not skipped and len(candidates) == 5
    for item in candidates.values():
        assert len(item["block"]["x"]) == len(base["x"])
        V.validate_partition(d, ids, item["records"], [1, 2])
        if item["family"] == "local":
            assert len(item["changed_slots"]) == 3
            other = candidates[f"recluster_d{item['edit']['donor']}_r{item['edit']['recipient']}"]
            assert item["allocation"] == other["allocation"]
            assert [r["class"] for r in item["records"]] == [r["class"] for r in other["records"]]
            for row in set(range(len(base["x"]))) - set(item["changed_slots"]):
                for key in V.FIELDS:
                    assert (item["block"][key][row] == base[key][row] if key == "texts"
                            else torch.equal(item["block"][key][row], base[key][row]))


def test_candidates_preserve_all_old_memory():
    d, ids, base = data()
    candidates, _ = V.make_candidates(d, ids, base, 1, "cpu")
    old = V.SCA.append_memory(None, base, 0)
    for item in candidates.values():
        memory = V.SCA.append_memory(old, item["block"], len(old["x"]))
        V.C.assert_frozen(old, memory)


def test_holdout_or_duplicate_source_is_rejected():
    d, ids, base = data()
    candidates, _ = V.make_candidates(d, ids, base, 1, "cpu")
    records = deepcopy(candidates["c0"]["records"])
    records[0]["ids"][0] = 0
    with pytest.raises(AssertionError):
        V.validate_partition(d, ids, records, [1, 2])
    d["pool_mask"][ids[0]] = False
    with pytest.raises(AssertionError):
        V.validate_partition(d, ids, candidates["c0"]["records"], [1, 2])


def observations(gain=2.):
    scores = {}
    for name in ("c0", "edit"):
        scores[name] = {seed: [{"cumulative_f1": 50. + (gain if name == "edit" else 0),
                              "old_f1": 60.} for _ in range(4)]
                        for group in V.SELECTION_GROUPS for seed in group}
    return scores


def test_selection_uses_only_registered_prefix_not_full_endpoint():
    scores = observations()
    assert V.select_paired_prefix(scores, list(scores))["selected"] == "edit"
    for value in scores["edit"].values():
        value[0] = {"unread": True}
        value[3] = {"unread": True}
    assert V.select_paired_prefix(scores, list(scores))["selected"] == "edit"


def test_selection_group_disagreement_falls_back():
    scores = observations()
    for seed in V.SELECTION_GROUPS[1]:
        for value in scores["edit"][seed]:
            value["cumulative_f1"] = 49.
    assert V.select_paired_prefix(scores, list(scores))["selected"] == "c0"


@pytest.mark.parametrize("failure", ["retention", "horizon", "weak"])
def test_selection_guards(failure):
    scores = observations(.4 if failure == "weak" else 2.)
    for value in scores["edit"].values():
        if failure == "retention":
            value[1]["old_f1"] = value[2]["old_f1"] = 57.
        if failure == "horizon":
            value[1]["cumulative_f1"] = 49.
            value[2]["cumulative_f1"] = 60.
    assert V.select_paired_prefix(scores, list(scores))["selected"] == "c0"


def test_prefix_equals_independent_runs_and_legacy_student():
    _, _, memory = data()
    def noisy_observer(model):
        model.eval()
        torch.rand(97)
        return {}
    curve = R.train_curve(memory, 777, 32, "cpu", noisy_observer)
    for snap in curve["snapshots"]:
        fixed = R.train_curve(memory, 777, snap["step"], "cpu", checkpoints=[snap["step"]])
        for key in ("state_sha256", "optimizer_sha256", "generator_sha256", "last_loss"):
            assert snap[key] == fixed["snapshots"][-1][key]
    official = V.G.train_student(V.G.memory_to_trainable(memory), 777, "cpu", 32)
    assert V.C.tensor_hash(official.state_dict()) == curve["end_state_sha256"]


def test_nested_hash_covers_text_and_source_metadata():
    assert R.nested_hash({"texts": ["a"]}) != R.nested_hash({"texts": ["b"]})
    assert R.nested_hash({"stage": 1}) != R.nested_hash({"stage": 2})


def test_independent_seed_groups():
    all_seeds = sum((list(g) for g in V.SELECTION_GROUPS), []) + list(V.REPORT_SEEDS)
    assert len(all_seeds) == len(set(all_seeds))


def test_horizon_validation():
    assert V.horizons(1000) == [125, 250, 500, 1000]
    with pytest.raises(ValueError):
        V.horizons(3)


def test_frozen_resume_restores_registered_bytes_and_never_changes_values(tmp_path):
    from run_v6_frozen_entry import stable_lock
    original = {"code_sha256": {"previous": "fixed",
        "v2_llm_fusion/v6_local_marginal.py": "a",
        "v2_llm_fusion/run_v6_local_marginal.py": "b"}, "budget": 5}
    path = tmp_path / "freeze.json"
    V.C.save_json(path, original)
    expected = V.C.sha(path)
    V.C.save_json(tmp_path / "registration.json", {"freeze_sha256": expected})
    reordered = deepcopy(original)
    reordered["code_sha256"] = {k: original["code_sha256"][k] for k in (
        "previous", "v2_llm_fusion/run_v6_local_marginal.py", "v2_llm_fusion/v6_local_marginal.py")}
    V.C.save_json(path, reordered)
    stable_lock(path, reordered)
    assert V.C.sha(path) == expected
    before = path.stat().st_mtime_ns
    stable_lock(path, reordered)
    assert path.stat().st_mtime_ns == before
    with pytest.raises(ValueError):
        stable_lock(path, original | {"budget": 10})


def test_summary_cannot_select_using_independent_report_results():
    import audit_v6_local_marginal as A
    manifest = {"dataset": "cora", "order": "canonical", "stage": 1,
        "stage_classes": [[0], [1, 2]], "candidate_build_seconds": .1,
        "shared_blocks": [{"build_seconds": .1}] * 3, "candidates": []}
    fits = {}
    for name, family in (("c0", "c0"), ("local_d1_r2", "local"), ("recluster_d1_r2", "recluster")):
        manifest["candidates"].append({"name": name, "family": family, "geometry_error": 1.,
            "changed_slots": [] if name == "c0" else [0, 1, 2], "singleton_new_rows": 0,
            "rows": 20, "payload": {"logical_payload_bytes": 100}})
        for seed in A.SEEDS:
            metric = {"cumulative_f1": 50. + (2. if family != "c0" else 0.), "old_f1": 60., "new_f1": 40.}
            point = {"selector": {"all": deepcopy(metric)}, "report": {"all": deepcopy(metric)},
                "step": 20, "training_seconds": .1,
                "observation_timing": {"forward_seconds": .01, "selector_seconds": .01}}
            fits[name, seed] = {"curve": [deepcopy(point) for _ in range(4)],
                               "training_seconds": 1., "evaluation_seconds": .1, "steps": 160}
    before = A.summarize_cell(manifest, fits)
    for name in ("local_d1_r2", "recluster_d1_r2"):
        for seed in V.REPORT_SEEDS:
            fits[name, seed]["curve"][-1]["report"]["all"]["cumulative_f1"] = 0.
    after = A.summarize_cell(manifest, fits)
    for family in V.FAMILIES:
        assert before["families"][family]["selection"] == after["families"][family]["selection"]
        assert before["families"][family]["gain"]["mean"] == 2.
        assert after["families"][family]["gain"]["mean"] == -50.


def test_constant_candidate_rank_is_not_silently_dropped():
    import audit_v6_local_marginal as A
    assert A.rank([1., 1., 1.], [2., 3., 1.])["spearman"] == 0.
    assert A.rank([1., 1., 1.], [2., 3., 1.])["constant_rank"]


def test_reload_tolerance_is_explicit():
    import audit_v6_local_marginal as A
    assert A.RELOAD_RTOL == 1e-4 and A.RELOAD_ATOL == 1e-5
