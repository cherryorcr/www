import unittest

import torch

from temporal_protocol import arrival_stages, cohort_masks, snapshot_ids, validate_delta


class TemporalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.arrival = arrival_stages([2014, 2015, 2016, 2017, 2018, 2019, 2020])
        self.pool = torch.tensor([True, False, True, True, True, True, False])

    def test_calendar_boundaries_not_class_balancing(self):
        self.assertEqual(self.arrival.tolist(), [0, 0, 1, 2, 3, 4, 4])

    def test_no_future_snapshot(self):
        for stage in range(5):
            idx = snapshot_ids(self.arrival, stage, 'arrived_snapshot')
            self.assertTrue(bool((self.arrival[idx] <= stage).all()))

    def test_current_snapshot_has_no_historical_nodes(self):
        self.assertEqual(snapshot_ids(self.arrival, 4, 'current_snapshot').tolist(), [5, 6])

    def test_exact_constructor_pool(self):
        validate_delta(torch.tensor([2]), self.arrival, self.pool, 1)
        for ids in (torch.tensor([0, 2]), torch.tensor([2, 3]), torch.tensor([1])):
            with self.assertRaises(AssertionError):
                validate_delta(ids, self.arrival, self.pool, 1)

    def test_cohorts_are_disjoint(self):
        masks = cohort_masks(~self.pool, self.arrival, 4)
        self.assertFalse(bool((masks['historical'] & masks['current']).any()))
        self.assertTrue(torch.equal(masks['historical'] | masks['current'], masks['cumulative']))

    def test_invalid_dates_rejected(self):
        for years in ([0, 2015], [2015, 2021], []):
            with self.assertRaises(ValueError):
                arrival_stages(years)


if __name__ == '__main__':
    unittest.main()
