"""Publication-year boundaries, independent of labels and downstream outcomes."""
import torch

BOUNDS = (2015, 2016, 2017, 2018, 2020)


def arrival_stages(years):
    years = torch.as_tensor(years).flatten().long().cpu()
    if len(years) == 0 or bool((years < 1900).any()) or bool((years > BOUNDS[-1]).any()):
        raise ValueError('Missing or unsupported publication year')
    return torch.bucketize(years, torch.tensor(BOUNDS), right=False)


def validate_delta(ids, arrival, pool, stage):
    if not torch.equal(ids, torch.where(pool & arrival.eq(stage))[0]):
        raise AssertionError('Constructor must receive exactly current-stage pool nodes')
    if not len(ids) or bool((arrival[ids] != stage).any()):
        raise AssertionError('Historical or future source passed to constructor')
    return dict(current_stage_only=True, source_rows=len(ids),
                historical_raw_rows=0, future_raw_rows=0, holdout_rows=0)


def snapshot_ids(arrival, stage, mode):
    if mode == 'arrived_snapshot':
        return torch.where(arrival <= stage)[0]
    if mode == 'current_snapshot':
        return torch.where(arrival == stage)[0]
    raise ValueError(mode)


def cohort_masks(base, arrival, stage):
    return dict(cumulative=base & (arrival <= stage),
                current=base & (arrival == stage),
                historical=base & (arrival < stage))
