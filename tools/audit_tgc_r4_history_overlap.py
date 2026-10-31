"""Check actual class-group overlap; a new order name is not a new partition."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    files = subprocess.check_output(["rg", "--files", str(args.results), "-g", "manifest.json"], text=True).splitlines()
    matches, malformed = [], []
    for name in files:
        path = Path(name)
        if "tgc_r4_relational" in str(path):
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            malformed.append(name)
            continue
        dataset = item.get("dataset")
        if dataset not in protocol["datasets"]:
            continue
        group = item.get("stage_classes")
        if group is None:
            continue
        for order, expected in protocol["class_groups"][dataset].items():
            if group == expected:
                matches.append({"dataset": dataset, "new_order": order,
                                "prior_order": item.get("order"), "prior_manifest": name})
    result = {"scope": "available local manifest.json files, not all historical configs",
              "files_scanned": len(files), "malformed": malformed, "exact_matches": matches}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
