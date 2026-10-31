import copy

import pytest
import torch

import run_handoff_student_transfer_20261008 as H


def inputs():
    delta = dict(ids=torch.tensor([10, 20, 30, 40]),
                 x=torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]]),
                 y=torch.tensor([0, 0, 2, 2]), text_emb=torch.eye(4))
    return delta, torch.tensor([9]), dict(x=torch.tensor([[3., 3.], [0., 1.], [-1., 0.]]))


def test_coreset_labels_recover_without_changing_features():
    delta, parent, memory = inputs()
    saved = memory["x"].clone()
    labels, audit = H.recover_labels(memory, dict(source_ids=[20, 40]), delta, parent)
    assert labels.tolist() == [9, 0, 2]
    assert torch.equal(memory["x"], saved)
    assert "source IDs" in audit["rule"]


@pytest.mark.parametrize("source", [[20, 999], [20], [10, 40]])
def test_invalid_coreset_ids_or_order_are_rejected(source):
    delta, parent, memory = inputs()
    with pytest.raises(AssertionError):
        H.recover_labels(memory, dict(source_ids=source), delta, parent)


def test_synthetic_labels_checked_against_saved_text_sources():
    delta, parent, memory = inputs()
    row = dict(row_classes=[0, 2], text_source_ids=[[10, 20], [40]])
    labels, _ = H.recover_labels(memory, row, delta, parent)
    assert labels.tolist() == [9, 0, 2]
    for sources in ([[30], [40]], [[10], [999]], [[10], []]):
        with pytest.raises(AssertionError):
            H.recover_labels(memory, dict(row, text_source_ids=sources), delta, parent)


def test_tagsam_original_class_allocation_recovers_base_and_append():
    delta, _, _ = inputs()
    row = dict(arm="tagsam_selfloop")
    memory = dict(x=torch.zeros(2, 2), allocation={0: 1, 2: 1})
    labels, _ = H.recover_labels(memory, row, delta, torch.empty(0, dtype=torch.long))
    assert labels.tolist() == [0, 2]
    wrong = dict(memory, allocation={0: 2, 2: 0})
    with pytest.raises(AssertionError):
        H.recover_labels(wrong, row, delta, torch.empty(0, dtype=torch.long))
    appended = dict(x=torch.zeros(4, 2))
    labels, _ = H.recover_labels(appended, row, delta, torch.tensor([9, 9]))
    assert labels.tolist() == [9, 9, 0, 2]


def test_no_new_rows_rejected():
    delta, parent, _ = inputs()
    with pytest.raises(AssertionError):
        H.recover_labels(dict(x=torch.zeros(1, 2)), {}, delta, parent)


def test_tensor_hash_is_order_independent_and_detects_payload_change():
    memory = dict(x=torch.ones(2, 2), text_emb=torch.eye(2), edge_index=torch.empty(2, 0, dtype=torch.long))
    assert H.tensor_dict_hash(memory) == H.tensor_dict_hash(dict(reversed(list(memory.items()))))
    changed = copy.deepcopy(memory)
    changed["text_emb"][0, 0] += .01
    assert H.tensor_dict_hash(memory) != H.tensor_dict_hash(changed)


def test_original_seed_pairs_are_not_confused_with_aggregate():
    source = dict(student_seeds=[150, 151, 152], per_student=[
        dict(cumulative_macro_f1=v) for v in [1., 4., 9.]])
    value = dict(per_student=[dict(seed=152, final_f1=10.), dict(seed=150, final_f1=1.)])
    assert H.legacy_difference(value, source) == {"152": 1., "150": 0.}


def test_common_metric_matches_original_on_same_predictions():
    data=dict(x_device=torch.zeros(6, 2),real_adj=torch.eye(6),prompt_emb=torch.zeros(3,2),
        y=torch.tensor([0,0,1,1,2,2]),report_dev_mask=torch.ones(6,dtype=torch.bool),stage_classes=[[0],[1],[2]])
    pred=torch.tensor([0,2,1,0,2,1])

    class Model:
        def logits(self,*args):
            return torch.nn.functional.one_hot(pred,3).float()

    for stage in range(3):
        common=H.AB.score_prediction(pred,data,stage)
        old=H.OLD.eval_stagewise_exact(Model(),data,data["stage_classes"],stage)
        assert common["final_f1"]==pytest.approx(old["cumulative_macro_f1"])
        assert common["accuracy"]==pytest.approx(old["cumulative_accuracy"])
        assert common["old_f1"]==pytest.approx(old["old_mean_f1"]) if stage else common["old_f1"] is None
        assert common["new_f1"]==pytest.approx(old["current_f1"])
