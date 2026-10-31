"""Extract the frozen ArXiv base+4 stream into the current-run manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.source.read_text(encoding="utf-8"))
    protocol = payload.get("protocol", payload)
    result = {
        "name": "arxiv-current-edgefree-20261011",
        "dataset": "arxiv",
        "groups": protocol["groups"],
        "budgets": protocol["budgets"],
        "memory_rate": protocol.get("memory_rate", 0.05),
        "report_dev_mod": protocol.get("report_dev_mod", 3),
        "source_protocol": str(args.source),
        "source_construction_seed": protocol.get("construction_seed"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
