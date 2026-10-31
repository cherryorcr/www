import torch

import tgc_revision_gate as T
from test_tgc_revision_gate import block
from tgc_r2_response import probe_rows, response_curve, select_indices


def test_probe_sampling_has_no_historical_raw_rows():
    probes = probe_rows(7, 25)
    assert all(((x >= 0) & (x < 7)).all() for x in probes["old"])
    assert all(((x >= 7) & (x < 25)).all() for x in probes["new"])


def test_probe_does_not_change_shared_training_prefix():
    torch.set_num_threads(2)
    memory = block(7)
    support = T._concat_support(block(4), block(5, 40, 1))
    with_probe = response_curve(memory, support, 700, "cpu", (2, 4, 8))
    without_probe = response_curve(memory, support, 700, "cpu", (2, 4, 8), False)
    assert with_probe["end_state_sha256"] == without_probe["end_state_sha256"]
    assert [x["state_sha256"] for x in with_probe["trajectory"]] == [x["state_sha256"] for x in without_probe["trajectory"]]
    assert with_probe["report_labels_used"] is False


def test_guard_can_reject_low_loss_with_historical_damage():
    def candidate(old, new):
        return {"responses": [{"trajectory": [{"step": 32, "old_loss": old,
            "new_loss": new, "balanced_loss": (old + new) / 2, "train_loss": new}]}]}
    decision = select_indices([candidate(1, 1), candidate(1.1, .1)], 32)
    assert decision["balanced"] == 1
    assert decision["guarded"] == 0
