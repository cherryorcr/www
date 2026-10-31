import torch

import run_tgc_r14_r11_plasticity as M


def test_disagreement_weighted_assignment_preserves_rows_and_increases_alpha():
    p_initial = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    p_joint = torch.tensor([
        [0.5, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.5, 0.5],
    ])
    low, low_alpha, disagreement = M.plastic_assignment(p_initial, p_joint, .25)
    high, high_alpha, _ = M.plastic_assignment(p_initial, p_joint, .50)
    torch.testing.assert_close(low.sum(1), torch.ones(3))
    torch.testing.assert_close(high.sum(1), torch.ones(3))
    assert disagreement[1] == 0
    assert low_alpha[0] > low_alpha[1]
    assert high_alpha[0] > low_alpha[0]
    assert torch.all(high >= 0)
