"""Declared numerical repair, stored separately from the original failures."""
import argparse
from dataclasses import replace
from pathlib import Path

import common as C
from common import R, torch
import run_temporal as T
import solver_precision as S


def main(args):
    C.RESULTS = C.RESULTS / 'precision_repair_v2'
    original_start = C.start_result
    original_config = R.q_config
    original_compact = R.compact_audit

    def config(*values, **kwargs):
        return replace(original_config(*values, **kwargs), sinkhorn_max_steps=16384)

    def compact(audit):
        return dict(original_compact(audit), sinkhorn_arithmetic='float64 verified at solver output',
                    precision_verified_calls=R.UQ._cotag_precision_calls)

    def start(path, protocol):
        protocol = dict(protocol, solver_arithmetic='float64 normalization; float32 returned Q',
            numerical_repair_hash=C.sha(__file__), precision_wrapper_hash=C.sha(S.__file__),
            repair_reason='Verified float64 normalization plus larger solver cap 16384; tolerance, objectives and 30 construction updates unchanged',
            original_failure_root=str(C.RESULTS.parent/'temporal'),
            solver_tolerance_unchanged=True)
        return original_start(path, protocol)

    C.start_result = start
    R.q_config = config
    R.compact_audit = compact
    S.install(R.UQ)
    args.arm = 'uq_dm_edgefree'
    T.run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=271006)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    main(args)
