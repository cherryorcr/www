"""Select fixed-budget feasible orders using class counts only, never scores."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path("/home/pc/research/TAG_condensation")
CODE = ROOT / "experiments/uq_text_path_confirmation_20261008/code"
sys.path[:0] = [str(CODE), str(ROOT / "code/v2_llm_fusion"), str(ROOT / "code/frozen_v1_2026_09_09")]
import torch
import run_unified_q_matrix_20261008 as R


def main():
    torch.set_num_threads(2)
    result = dict(rule="scan integer order seeds from 9201; keep first two feasible by pool counts only",
        fixed_budgets=True, no_student_scoring=True, datasets={},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for ds in ("wikics", "sportsfit", "bookhis"):
        ref = R.L.read(ROOT / f"results/unified_q_inloop_cat_20261007/{ds}/protocol.json")
        data = R.L.load_large(ds, "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
            ROOT / f"results/public_baselines_text_cache_20261001/{ds}", "cuda:0", ref)
        accepted, checked = [], []
        budgets = ref["budgets"]
        increments = [v - (budgets[i - 1] if i else 0) for i, v in enumerate(budgets)]
        for seed in range(9201, 9301):
            groups = R.V.choose_groups(ref, seed)
            data["stage_classes"] = groups
            ids = R.L.stage_ids(data)
            feasible = all(len(g) <= m <= len(raw) for g, m, raw in zip(groups, increments, ids))
            checked.append(dict(order_seed=seed, groups=groups, raw_rows=list(map(len, ids)),
                new_rows=increments, feasible=feasible,
                failures=[dict(stage=i, raw_rows=len(raw), new_rows=m, classes=len(g))
                          for i, (g,m,raw) in enumerate(zip(groups,increments,ids)) if not len(g) <= m <= len(raw)]))
            if feasible:
                accepted.append(seed)
            if len(accepted) == 2:
                break
        if len(accepted) != 2:
            raise RuntimeError("could not find two feasible orders without changing budget")
        result["datasets"][ds] = dict(selected_orders=accepted, budgets=budgets, checked=checked)
        del data
        torch.cuda.empty_cache()
    R.V.write_json(ROOT / "results/uq_text_path_20261008/order_feasibility.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
