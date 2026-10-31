"""Declared numerical repair, stored separately from the original failures."""
import argparse
from pathlib import Path

import common as C
from common import R, torch
import run_temporal as T
import solver_precision as S


def main(args):
    C.RESULTS = C.RESULTS / 'precision_repair'
    original_start = C.start_result

    def start(path, protocol):
        protocol = dict(protocol, solver_arithmetic='float64 normalization; float32 returned Q',
            numerical_repair_hash=C.sha(__file__), precision_wrapper_hash=C.sha(S.__file__),
            repair_reason='Original stage 1 row residual 1.064e-4 exceeded 1e-4 after 4096 iterations; tolerance and objective unchanged',
            original_failure_root=str(C.RESULTS.parent/'temporal'),
            solver_tolerance_unchanged=True)
        return original_start(path, protocol)

    C.start_result = start
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
