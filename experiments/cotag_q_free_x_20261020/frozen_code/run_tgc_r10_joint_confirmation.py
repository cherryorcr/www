"""Independent confirmation of the content/topology joint construction.

This runner deliberately keeps the R5 implementation unchanged.  It only
changes the confirmation orders, construction seeds, and report seeds, then
evaluates the four matched factorial arms: C0, content-only, topology-only,
and joint.  The extra R5 branches are still materialized by the shared
runner for compatibility, but they are not part of this confirmation table.
"""

from dataclasses import replace
import json

import run_tgc_r5_joint as V


DATASETS = ("cora", "citeseer")
ORDERS = ("order19601", "order19602", "order19603")
SEEDS = tuple(range(150, 156))
CORE_ARMS = ("c0", "content", "topology", "joint")
# The shared R5 constructor materializes these branches internally and uses
# their stage-0 files while preparing later stages.  They are compatibility
# arms here, not part of the primary joint-confirmation claim.
ARMS = CORE_ARMS + ("recursive", "maintenance", "capacity")

# New construction seeds avoid treating the previous R5/R6 realization as
# the only possible endpoint of the teacher optimization.
V.TEACHER = replace(
    V.TEACHER,
    teacher_seeds=(9000, 9001),
    train_seeds=(9100, 9101),
    validation_seeds=(9110, 9111),
    probe_seed=9090,
)
V.DATASETS = DATASETS
V.ORDERS = ORDERS
V.SEEDS = SEEDS
V.ARMS = ARMS

_base_protocol = V.protocol


def protocol():
    spec = _base_protocol()
    spec.update({
        "name": "tgc-r10-joint-confirmation-v1",
        "datasets": list(DATASETS),
        "orders": list(ORDERS),
        "report_seeds": list(SEEDS),
        "arms": list(ARMS),
        "primary_arms": list(CORE_ARMS),
        "development_only": False,
        "independent_confirmation": True,
        "construction_seed_change": "R5 shared assignment with teacher seeds 9000/9001",
        "purpose": (
            "test whether positive content-topology interaction replicates on reserved "
            "orders under matched C0/content/topology/joint evaluation"
        ),
        "promotion_rule": {
            "mean_final_min_pp": 0.5,
            "stage_mean_min_pp": 0.0,
            "worst_final_min_pp": -2.0,
            "worst_old_min_pp": -2.0,
            "min_cells_ge_minus1": 4,
            "mean_new_min_pp": 0.0,
        },
        "selection": "no arm selected from report F1; core four-arm factorial is primary",
    })
    return json.loads(json.dumps(spec))


V.protocol = protocol


if __name__ == "__main__":
    V.main()
