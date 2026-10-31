"""R10 shared-assignment preparation on reserved category orders."""
import run_tgc_r10_joint_confirmation as M


M.ORDERS = ("order19605", "order19606", "order19608")
M.V.ORDERS = M.ORDERS
M.V.DATASETS = M.DATASETS
M.V.SEEDS = M.SEEDS
M.V.ARMS = M.ARMS


if __name__ == "__main__":
    M.V.main()
