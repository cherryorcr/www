"""Extract a current edge-free reference manifest from an existing protocol JSON.

The source may use either the large-run ``groups`` name or the small-graph
``stage_classes`` name.  No metrics are copied; only the predeclared class
stream and cumulative memory budgets are materialized.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    protocol = payload.get("protocol", payload)
    if not isinstance(protocol, dict):
        protocol = payload
    # Small-graph result files keep the stream beside the protocol metadata;
    # large-graph manifests usually keep it inside the protocol object.
    stream = protocol if (protocol.get("groups") or protocol.get("stage_classes")) else payload
    groups = stream.get("groups") or stream.get("stage_classes")
    budgets = stream.get("budgets")
    if not groups or not budgets or len(groups) != len(budgets):
        raise ValueError("source must contain same-length groups/stage_classes and budgets")

    result = {
        "name": "current-edgefree-reference-20261011",
        "dataset": args.dataset,
        "groups": [[int(v) for v in group] for group in groups],
        "budgets": [int(v) for v in budgets],
        "report_dev_mod": int(stream.get("report_dev_mod", protocol.get("report_dev_mod", 3))),
        "source_protocol": str(args.source),
        "source_name": protocol.get("name"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
