"""Choose unseen class partitions before any R6 report evaluation."""
import argparse
import json
from pathlib import Path
import random


def groups(dataset, seed):
    n = 7 if dataset == "cora" else 6
    values = list(range(n))
    random.Random(seed).shuffle(values)
    return [sorted(values[:2]), sorted(values[2:4]), sorted(values[4:])]


def signature(value):
    return tuple(tuple(sorted(map(int, stage))) for stage in value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seen = {d: set() for d in ("cora", "citeseer")}
    files, errors = [], []
    for path in sorted(args.results.rglob("*.json")):
        if path.name not in ("protocol.json", "manifest.json", "config.json"):
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8-sig"))
            dataset = item.get("dataset")
            if dataset not in seen:
                dataset = next((d for d in seen if d in path.parts or any(p.startswith(d + "_") for p in path.parts)), None)
            value = item.get("stage_classes")
            if dataset in seen and isinstance(value, list) and len(value) == 3:
                seen[dataset].add(signature(value))
            for dataset, orders in item.get("class_groups", {}).items():
                if dataset in seen and isinstance(orders, dict):
                    for value in orders.values():
                        if isinstance(value, list) and len(value) == 3:
                            seen[dataset].add(signature(value))
            files.append(str(path))
        except (ValueError, TypeError, OSError) as error:
            errors.append({"file": str(path), "error": str(error)})
    prior_count = {d: len(v) for d, v in seen.items()}
    selected = []
    for seed in range(19601, 25000):
        if all(signature(groups(d, seed)) not in seen[d] for d in seen):
            selected.append(seed)
            for dataset in seen:
                seen[dataset].add(signature(groups(dataset, seed)))
        if len(selected) == 3:
            break
    if len(selected) != 3 or errors:
        raise RuntimeError({"selection": selected, "errors": errors})
    result = {"orders": [f"order{s}" for s in selected], "class_groups": {
        d: {f"order{s}": groups(d, s) for s in selected} for d in seen},
        "files_scanned": len(files), "unique_prior_three_stage_partitions": prior_count,
        "errors": errors, "scope": "available local protocol/manifest/config JSON, not missing historical files",
        "report_scores_read": False, "files": files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
