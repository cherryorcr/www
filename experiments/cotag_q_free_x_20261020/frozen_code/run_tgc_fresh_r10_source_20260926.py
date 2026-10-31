"""Build a fresh sealed R10 source on previously unused class orders.

This wrapper changes only the class-order seeds used by the frozen R10
construction.  It does not alter the shared R5 implementation, teacher
objective, Student, budget, or report protocol.
"""

import run_tgc_r10_joint_confirmation as M


M.ORDERS = ("order19611", "order19612")
M.V.ORDERS = M.ORDERS
M.V.DATASETS = M.DATASETS
M.V.SEEDS = M.SEEDS
M.V.ARMS = M.ARMS


if __name__ == "__main__":
    M.V.main()
