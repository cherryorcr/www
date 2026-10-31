import torch
import pytest

import graphtext_prompt_student as P
import unified_q_multimodal as U
import unified_q_pairing_readout as D


def memory():
    generator = torch.Generator().manual_seed(10)
    return dict(x=torch.randn(12, 4, generator=generator),
        text_emb=torch.randn(12, 8, generator=generator),
        labels=torch.tensor([0] * 4 + [1] * 4 + [0] * 4),
        stage=torch.tensor([0] * 8 + [1] * 4), mass=torch.ones(12), depth=torch.zeros(12),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        texts=[str(i) for i in range(12)], text_source_ids=[[100 + i] for i in range(12)])


def test_payload_permutation_preserves_arrival_class_and_fixed_fields():
    original = memory()
    state = torch.get_rng_state().clone()
    permutation = D.within_stage_class_permutation(original, 271011)
    assert torch.equal(state, torch.get_rng_state())
    variant = D.permute_payload(original, permutation)
    assert not torch.equal(permutation, torch.arange(12))
    for key in ('x', 'labels', 'stage', 'mass', 'depth', 'edge_index'):
        assert torch.equal(variant[key], original[key])
    for i, source in enumerate(permutation):
        assert variant['texts'][i] == original['texts'][source]
        assert variant['text_source_ids'][i] == original['text_source_ids'][source]
    inverse = permutation.argsort()
    recovered = D.permute_payload(variant, inverse)
    assert recovered['texts'] == original['texts']
    torch.testing.assert_close(recovered['text_emb'], original['text_emb'])
    assert D.text_geometry(variant)['unique_texts'] == 12


@pytest.mark.parametrize('permutation', [torch.arange(12).roll(4), torch.zeros(12, dtype=torch.long)])
def test_illegal_or_nonbijective_mapping_is_rejected(permutation):
    with pytest.raises(ValueError):
        D.permute_payload(memory(), permutation)


@pytest.mark.parametrize('weighting', ['node_mean', 'group_class'])
@pytest.mark.parametrize('use_batch', [False, True])
def test_probe_loss_and_gradients_match_existing_implementation(weighting, use_batch):
    torch.manual_seed(8)
    mem = memory()
    prompt = torch.randn(2, 8)
    model = P.GraphTextPromptStudent(4, hidden=8, out=8, text_dim=8)
    adj = P.normalized_adjacency(mem['edge_index'], 12, 'cpu')
    rows = torch.tensor([0, 1, 4, 8]) if use_batch else None
    expected = U._loss(model, mem, adj, prompt, group_ids=mem['stage'], row_ids=rows,
                       loss_weighting=weighting, chunk_size=3)
    actual = D.encoded_loss(model.encode_graph(mem['x'], adj), model.encode_text(mem['text_emb']),
        mem['labels'], model.encode_text(prompt), model.logit_scale.exp().clamp(max=100),
        mem['stage'], weighting, rows=rows, chunk=3)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual, tuple(model.parameters())),
                    torch.autograd.grad(expected, tuple(model.parameters()))):
        torch.testing.assert_close(a, b)
