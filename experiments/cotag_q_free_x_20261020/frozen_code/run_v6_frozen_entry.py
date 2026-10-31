"""Resume frozen v6-A without rewriting semantically identical configuration."""
import hashlib
import json
from pathlib import Path

import run_v6_local_marginal as R


def stable_lock(path, value):
    if not path.exists():
        R.C.save_json(path, value)
        return
    saved = json.loads(path.read_text())
    if saved != json.loads(json.dumps(value)):
        raise ValueError(f"frozen configuration mismatch: {path}")
    registration = path.parent / "registration.json"
    if path.name != "freeze.json" or not registration.exists():
        return
    expected = json.loads(registration.read_text())["freeze_sha256"]
    if R.C.sha(path) == expected:
        return
    # The original writer reordered only the two entries from an unordered set.
    codes = saved["code_sha256"]
    last = list(codes)[-2:]
    if set(last) != {"v2_llm_fusion/run_v6_local_marginal.py", "v2_llm_fusion/v6_local_marginal.py"}:
        raise AssertionError("unrecognized frozen serialization change")
    restored = dict(saved)
    restored["code_sha256"] = {k: codes[k] for k in list(codes)[:-2] + last[::-1]}
    newline = "\r\n" if b"\r\n" in path.read_bytes() else "\n"
    encoded = json.dumps(restored, indent=2, allow_nan=False).replace("\n", newline).encode()
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise AssertionError("cannot restore registered byte representation")
    before = R.C.sha(path)
    path.write_bytes(encoded)
    if R.C.sha(path) != expected:
        raise AssertionError("registered hash restoration failed")
    R.C.save_json(path.parent / "serialization_repair.json", {
        "before_sha256": before, "restored_sha256": expected,
        "configuration_values_changed": False, "training_code_changed": False,
        "reason": "restore exact registered JSON field order; stop rewriting unchanged configuration",
        "entry_sha256": R.C.sha(__file__)})


if __name__ == "__main__":
    R.O.lock_json = stable_lock
    R.main()
