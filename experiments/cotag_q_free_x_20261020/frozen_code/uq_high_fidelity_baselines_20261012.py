"""Pinned native baseline cores under an explicitly adapted continual interface."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from sklearn.metrics import pairwise_distances
import torch

import run_unified_q_matrix_20261008 as R

GECC_COMMIT = "2a120ed49e501a39a7c94bdabfc4c33c8c032235"
GECC_SHA256 = "fcf042cddfeb795e578d46e995602f476814a81cb9c3a84d5edb3153bac22508"
GECC_FILE = Path(__file__).with_name("graphslim_gecc_pinned.py")


def pinned_functions(path, names, namespace, expected_hash=None):
    if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"source hash differs: {path.name}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"missing native definitions: {names}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def verified_repo(repo, commit, files):
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError(f"source pin differs: {repo}")
    hashes = {}
    for name in files:
        path = repo / name
        canonical = subprocess.check_output(["git", "-C", str(repo), "show", f"HEAD:{name}"])
        if canonical != path.read_bytes():
            raise ValueError(f"modified official source: {name}")
        hashes[name] = hashlib.sha256(canonical).hexdigest()
    return dict(commit=commit, files=hashes, official_reproduction=False)


def native_fuzzy_centers(values, rows, seed, source=GECC_FILE, restarts=20, max_iters=300):
    """Execute upstream FCM/restart selection, observing rather than replacing it."""
    import skfuzzy

    records, best = [], {}

    def observed_cmeans(*args, **kwargs):
        result = skfuzzy.cluster.cmeans(*args, **kwargs)
        center, membership, _, _, objective, iterations, fpc = result
        sizes = membership.sum(1)
        balance = float(np.square(sizes - len(values) / rows).sum())
        score = float(objective[-1]) + balance
        records.append(dict(restart=len(records), objective=score, fuzzy_objective=float(objective[-1]),
                            balance_penalty=balance, iterations=int(iterations), fpc=float(fpc)))
        if not best or score < best["objective"]:
            safe = np.maximum(membership, np.finfo(np.float64).tiny)
            best.clear()
            best.update(objective=score, centers=center.copy(), restart=len(records) - 1,
                        entropy=float(-(membership * np.log(safe)).sum(0).mean()),
                        min_slot_mass=float(sizes.min()), max_slot_mass=float(sizes.max()),
                        row_sum_error=float(np.max(np.abs(membership.sum(0) - 1))))
        return result

    namespace = dict(torch=torch, np=np, random=np.random, pairwise_distances=pairwise_distances,
                     fuzz=SimpleNamespace(cluster=SimpleNamespace(cmeans=observed_cmeans)))
    pinned_functions(source, {"perform_balance_evolve_clusteringjk"}, namespace, GECC_SHA256)
    arguments = SimpleNamespace(device="cpu", seed=seed, fuzziness=1.3,
                                rep_fuzz=restarts, balance_alpha=1.)
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        centers = namespace["perform_balance_evolve_clusteringjk"](
            np.asarray(values), None, 0, None, arguments, rows, epsilon=1e-8,
            max_iters=max_iters, save_p=False)
    finally:
        np.random.set_state(state)
    discrepancy = float(np.max(np.abs(centers.numpy() - best.pop("centers"))))
    if discrepancy > 1e-5 or best["row_sum_error"] > 1e-8:
        raise AssertionError("native fuzzy centers or memberships differ")
    return centers, dict(**best, native_center_error=discrepancy, restarts=records)


def gecc_soft(delta, rows, encoder, seed, device):
    cfg = R.M.Config(seed=seed, text_sources=5, text_mode="nearest")
    allocation = R.M.ratio_allocation(delta["y"], rows)
    propagated = R.M.propagated_features(delta, cfg.gecc_weights, device).cpu()
    centers, records = [], []
    for label, count in allocation.items():
        values = propagated[delta["y"] == label].numpy()
        x, record = native_fuzzy_centers(values, count, seed + 1009 * delta["stage"])
        centers.append(x)
        records.append(dict(label=int(label), raw_rows=len(values), slots=count, **record))
    labels = R.M.row_labels(allocation)
    block, attachment = R.M.attach_text(delta, propagated, torch.cat(centers), labels, encoder, cfg)
    block["labels"] = labels
    return block, dict(implementation="native GraphSlim GECC FCM + balanced restart selection",
        repository="https://github.com/Emory-Melody/GraphSlim", commit=GECC_COMMIT,
        source_sha256=GECC_SHA256, fuzziness=1.3, restarts=20, max_iters=300, tolerance=1e-8,
        native_center_rule="sum(u**m * x) / sum(u**m); not linear Q pooling",
        initialization="native random centers and membership; unusual axis preserved",
        allocation=allocation, clustering=records, attachment=attachment,
        adaptations=["current delta only; frozen prefix", "common MiniLM/CCTP nearest-5 text",
                     "common prompt Student", "fuzziness 1.3 supplemental soft branch, not tuned native ArXiv config"],
        official_reproduction=False)


def build(arm, delta, parent, labels, encoder, prompt, seed, device):
    if arm == "gecc_soft":
        return gecc_soft(delta, len(labels), encoder, seed, device)
    if arm == "tagsam_selector_response":
        from uq_tagsam_native_selector_20261012 import build_tagsam
        return build_tagsam(delta, parent, len(labels), encoder, seed, device)
    raise ValueError("STAD must use its topology-preserving runner")
