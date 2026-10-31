"""Deploy/run the conserved-payload role-permutation diagnostic."""
from pathlib import Path
import os

import phase12_remote as remote


ROOT = Path(__file__).resolve().parents[1]
remote.SNAPSHOT = remote.REMOTE + "/work_runs/tgc_two_prototype_smoke_20260924"
remote.FILES = ["code/v2_llm_fusion/" + name for name in (
    "run_tgc_two_prototype_role_permutation.py",
    "run_tgc_two_prototype_role_diagnostic.py",
    "run_tgc_two_prototype_smoke.py",
    "run_tgc_two_prototype_textcap_confirmation.py",
    "run_tgc_joint_student_response.py",
    "run_tgc_r11_c0_anchor.py",
    "tgc_shared_assignment.py",
    "run_student_compatible_append.py",
    "protocol_v3.py",
)]


if __name__ == "__main__":
    os.environ.setdefault("TAG_REMOTE_PASSWORD", "")
    remote.main()
