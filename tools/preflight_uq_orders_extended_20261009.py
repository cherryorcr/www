"""Find three feasible fixed-schedule class orders without looking at scores."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path("/home/pc/research/TAG_condensation")
CODE = ROOT / "experiments/uq_text_path_confirmation_20261008/code"
sys.path[:0] = [str(CODE), str(ROOT / "code/v2_llm_fusion"), str(ROOT / "code/frozen_v1_2026_09_09")]
import run_unified_q_matrix_20261008 as R


def main():
    torch.set_num_threads(2)
    output = dict(
        rule="scan order seeds from 9201; retain first three feasible fixed-schedule orders by pool counts only",
        fixed_budgets=True, no_student_scoring=True, datasets={},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for dataset in ("wikics", "sportsfit", "bookhis"):
        reference = R.L.read(ROOT / f"results/unified_q_inloop_cat_20261007/{dataset}/protocol.json")
        data = R.L.load_large(dataset,
            "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
            ROOT / f"results/public_baselines_text_cache_20261001/{dataset}",
            "cuda:0", reference)
        budgets = reference["budgets"]
        increments = [int(v) - (int(budgets[i - 1]) if i else 0)
                      for i, v in enumerate(budgets)]
        accepted, checked = [], []
        for seed in range(9201, 10001):
            groups = R.V.choose_groups(reference, seed)
            data["stage_classes"] = groups
            ids = R.L.stage_ids(data)
            failures = [dict(stage=i, raw_rows=len(raw), new_rows=m,
                             classes=len(group))
                        for i, (group, m, raw) in enumerate(
                            zip(groups, increments, ids))
                        if not len(group) <= m <= len(raw)]
            feasible = not failures
            checked.append(dict(order_seed=seed, groups=groups,
                                raw_rows=list(map(len, ids)),
                                new_rows=increments, feasible=feasible,
                                failures=failures))
            if feasible:
                accepted.append(seed)
            if len(accepted) == 3:
                break
        if len(accepted) != 3:
            raise RuntimeError(f"{dataset}: only {len(accepted)} feasible orders")
        output["datasets"][dataset] = dict(selected_orders=accepted,
            budgets=budgets, checked=checked)
        del data
        torch.cuda.empty_cache()
    print(json.dumps(output, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
