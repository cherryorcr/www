import copy
import hashlib
import json
import struct

import pytest

import audit_large_baseline_handoff_20261008 as A


def fixture(tmp_path):
    package, replays = tmp_path / "package", tmp_path / "replays"
    source_paths = []
    for dataset in ("wikics", "sportsfit", "bookhis"):
        protocol = dict(groups=[[0]], budgets=[2])
        source = dict(protocol=copy.deepcopy(protocol), rows=[dict(arm="random_real", stage=0,
            construction_audit=dict(candidate_delta_ids=[3, 7], historical_raw_reads=0,
                                    report_test_labels_used_for_construction=False))])
        p = package / f"server_snapshot_20261008/formal_150155/baselines/{dataset}_results.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(source))
        source_paths.append(p)
        protocol["stage_id_hashes"] = [hashlib.sha256(struct.pack("<2q", 3, 7)).hexdigest()]
        p = replays / f"handoff_{dataset}/results.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(protocol=protocol)))
    return package, replays, source_paths


def test_original_candidate_pools_match_actual_constructor_delta(tmp_path):
    package, replays, _ = fixture(tmp_path)
    assert A.verify_replay_deltas(package, replays)["checked"] == 3


@pytest.mark.parametrize("key,value", [("candidate_delta_ids", [3, 8]), ("historical_raw_reads", 1)])
def test_changed_candidate_pool_or_access_is_rejected(tmp_path, key, value):
    package, replays, paths = fixture(tmp_path)
    source = json.loads(paths[0].read_text())
    source["rows"][0]["construction_audit"][key] = value
    paths[0].write_text(json.dumps(source))
    with pytest.raises(AssertionError):
        A.verify_replay_deltas(package, replays)
