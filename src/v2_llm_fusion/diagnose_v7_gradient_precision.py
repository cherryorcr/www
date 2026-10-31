"""Measure audit arithmetic differences without changing frozen v7 scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def diagnose(root):
    cells = []
    for path in sorted(Path(root).glob("*/*/stage*/selection.json")):
        selected = json.loads(path.read_text())
        manifest = json.loads((path.parent / "candidates.json").read_text())
        entries = {i["name"]: i for i in manifest["candidates"]}
        groups = ((7300, 7301), (7400, 7401))
        scores64 = {n: {} for n in entries}
        discrepancies = []
        for seed in (7300, 7301, 7400, 7401):
            base = torch.load(path.parent / f"c0_gradient_seed{seed}.pt", map_location="cpu", weights_only=False)
            for name in entries:
                curve = base if name == "c0" else torch.load(
                    path.parent / f"{name}_gradient_seed{seed}.pt", map_location="cpu", weights_only=False)
                distances32, distances64 = [], []
                for a, b in zip(curve["observations"], base["observations"]):
                    x, y = a["gradient"], b["gradient"]
                    distances32.append(float(np.clip(1. - F.cosine_similarity(x[None], y[None]).item(), 0., 2.)))
                    xd, yd = x.double(), y.double()
                    cosine = float(xd.dot(yd) / (xd.norm().clamp_min(1e-8) * yd.norm().clamp_min(1e-8)))
                    distances64.append(float(np.clip(1. - cosine, 0., 2.)))
                coverage = ((entries["c0"]["geometry_error"] - entries[name]["geometry_error"])
                            / max(abs(entries["c0"]["geometry_error"]), 1e-8))
                saved = selected["scores"][name][str(seed)]["score"]
                q32 = coverage - .5 * float(np.mean(distances32))
                q64 = coverage - .5 * float(np.mean(distances64))
                scores64[name][seed] = 0. if name == "c0" else q64
                discrepancies.append({"candidate": name, "seed": seed, "saved": saved,
                                      "q32": q32, "q64": q64, "gap32": abs(saved - q32),
                                      "gap64": abs(saved - q64)})
        picks = [max(entries, key=lambda n: (np.mean([scores64[n][s] for s in group]), n == "c0", n))
                 for group in groups]
        selected64 = picks[0] if len(set(picks)) == 1 else "c0"
        if np.mean(list(scores64[selected64].values())) <= 0:
            selected64 = "c0"
        cells.append({"cell": str(path.parent.relative_to(root)),
                      "selected32": selected["selection"]["selected"], "selected64": selected64,
                      "group_picks64": picks, "max_gap32": max(r["gap32"] for r in discrepancies),
                      "max_gap64": max(r["gap64"] for r in discrepancies),
                      "worst64": max(discrepancies, key=lambda r: r["gap64"]), "observations": discrepancies})
    return {"cells": cells, "max_gap32": max(c["max_gap32"] for c in cells),
            "max_gap64": max(c["max_gap64"] for c in cells),
            "selection_changes": sum(c["selected32"] != c["selected64"] for c in cells),
            "scope": "post-run numerical diagnosis; frozen selections unchanged; no report metrics read"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = diagnose(Path(args.root))
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}))
    for cell in result["cells"]:
        print(json.dumps({k: v for k, v in cell.items() if k != "observations"}))
