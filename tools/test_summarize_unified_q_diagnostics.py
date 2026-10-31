import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("summary_uq", Path(__file__).with_name("summarize_unified_q_diagnostics_20261007.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def test_paired_uses_order_units_not_student_repetitions():
    rows = []
    for order, delta in [(9101, 1), (9102, 2), (9103, 3)]:
        for seed in (1, 2, 3):
            for arm in ("cat_cctp", "uq_v3"):
                rows.append({"dataset": "wikics", "order_seed": order,
                    "construction_seed": seed, "arm": arm,
                    "final_f1": 50 + (delta if arm == "uq_v3" else 0)})
    result = S.paired(rows)[0]
    assert result["paired_cells"] == 9
    assert result["orders"] == 3
    assert result["positive_cells"] == 9
    assert result["mean_delta_f1"] == 2
    assert result["descriptive_order_t_ci_low"] < 0


def test_no_confirmation_ci_with_one_order():
    result = S.paired([
        {"dataset": "bookhis", "order_seed": 9101, "construction_seed": 1,
         "arm": a, "final_f1": f} for a, f in (("cat_cctp", 20), ("uq_v3", 30))])[0]
    assert result["paired_cells"] == 1
    assert result["descriptive_order_t_ci_low"] is None
