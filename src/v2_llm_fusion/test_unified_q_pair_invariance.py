from probe_uq_pair_invariance_20261009 import probe


def test_full_candidate_proxy_does_not_identify_within_class_pairs():
    result=probe()
    for row in result['records']:
        assert row['within_class_loss_error']<2e-6
        assert row['within_class_gradient_max_error']<2e-6
        assert row['within_class_gradient_relative_error']<1e-5
        assert row['across_class_loss_difference']>1e-3
        assert row['fixed_minibatch_within_class_loss_difference']>1e-3
