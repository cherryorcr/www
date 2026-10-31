"""Diagnose archived-vs-fresh raw training reproducibility without selection."""
import json
from pathlib import Path

import torch
import run_small_order_cost_confirmation as R


def main():
    C, U = R.C, R.U
    torch.set_num_threads(2)
    root = C.RESULTS / "small_order_cost_2026_09_17" / "cora"
    data = U.load_unified("cora", U.DATA_CFG["cora"]["path"],
        str(Path(U.__file__).parent / "minilm_from_graphclip"), "cuda:0")
    data["text_emb"] = torch.load(C.artifact_dir("cora", "v4") / "text_emb_pool_only.pt",
                                 map_location="cpu", weights_only=False)
    data["prompt_emb"] = data["encoder"].encode(data["prompts"])
    data["real_adj"] = R.G.norm_adj(data["edge_index"], len(data["x"]), "cuda:0")
    data["x_device"] = data["x"].to("cuda:0")
    R.set_stages(data, R.stage_orders("cora")["canonical"])
    memory = torch.load(root / "canonical/raw_equal_s1_memory.pt", map_location="cpu", weights_only=False)
    archive = json.loads((C.RESULTS / "small_comparison_audited_r2_2026_09_16/cora/raw_equal_s1_seed100.json").read_text())
    fresh = json.loads((root / "canonical/raw_equal_s1_seed100.json").read_text())
    rows = []
    for attempt in range(3):
        row = C.fit(memory, 100, 1072, "cuda:0", data, 1)
        rows.append(row)
        print(json.dumps({"attempt": attempt, "hash": row["end_state_sha256"],
                          "f1": row["metrics"]["all"]["cumulative_f1"]}), flush=True)
    C.save_json(root / "raw_regression_diagnostic.json", {
        "archive": archive, "first_fresh": fresh, "repeats": rows,
        "torch": torch.__version__, "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32})


if __name__ == "__main__":
    main()
