"""R11 anchor preparation/evaluation on reserved category orders."""
from pathlib import Path

import run_tgc_r11_c0_anchor as M


M.SOURCE = Path("/home/pc/research/TAG_condensation/results/tgc_r10_joint_confirmation_independent_20260925")
M.ORDERS = ("order19605", "order19606", "order19608")


if __name__ == "__main__":
    M.E.main(M.protocol, M.prepare, M.validate)
