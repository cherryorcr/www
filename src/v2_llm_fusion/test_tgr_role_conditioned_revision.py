import torch
import torch.nn.functional as F

import teacher_guided_payload_revision as T
import tgr_role_conditioned_revision as R


def memory(n=8, d=5, seed=7):
    gen = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(n, d, generator=gen),
        "text_emb": F.normalize(torch.randn(n, 6, generator=gen), dim=1),
        "texts": [f"t{i}" for i in range(n)],
        "mass": torch.ones(n),
        "depth": torch.zeros(n, dtype=torch.long),
        "stage": torch.zeros(n, dtype=torch.long),
        "edge_index": torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6],
                                     [1, 2, 2, 3, 4, 5, 6, 7]], dtype=torch.long),
    }


def test_role_projection_respects_different_radii():
    base = torch.ones(5, 3)
    relative = torch.ones_like(base)
    roles = {"radius": torch.tensor([0.01, 0.02, 0.03])}
    R._project_role(relative, base, 3, roles, 0.10)
    ratios = relative.norm(dim=1) / base.norm(dim=1)
    assert float(ratios[0]) <= 0.01001
    assert float(ratios[1]) <= 0.02001
    assert float(ratios[2]) <= 0.03001
    assert float(ratios[3]) <= 0.10001


def test_relation_loss_is_differentiable():
    z = F.normalize(torch.randn(6, 4), dim=1).requires_grad_()
    target = F.normalize(torch.randn(6, 4), dim=1)
    classes = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = R._old_relation_loss(z, target, classes, 6)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_role_selection_is_deterministic_and_covers_old_rows():
    base = memory(8)
    support = memory(12, seed=9)
    classes = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3,
                            4, 4, 5, 5])
    cfg = T.Config(teacher_steps=1, inner_steps=1, outer_steps=1,
                   batch=4, top_r=2, teacher_seeds=(11,),
                   validation_teacher_seeds=(12,))
    ref = {"seed": 11, "params_t": T._clone_params(
        T._params(T._student(11, 5, 6, "cpu")), detach=True)}
    roles_a = R.select_roles(base, support, classes, 8, ref, "cpu", cfg,
                             R.Config())
    roles_b = R.select_roles(base, support, classes, 8, ref, "cpu", cfg,
                             R.Config())
    assert torch.equal(roles_a["role"], roles_b["role"])
    assert len(roles_a["role"]) == 8
    assert sum(roles_a["counts"].values()) == 8

