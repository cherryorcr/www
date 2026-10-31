"""Freeze transitive adaptation provenance before a common-Student baseline batch."""
import argparse
import json
from pathlib import Path
import sys

import torch

import run_uq_final_confirmation_20261012 as E
import uq_high_fidelity_baselines_20261012 as H
import uq_stad_native_core_20261012 as S


def run(args):
    modules = [E, H, S, E.R, E.R.M, E.R.P, E.R.L, E.R.L.U, E.R.L.U.protocol_v3,
               E.R.V, E.R.I, E.R.I.D, E.R.UQ]
    official = None
    if args.arm == "tagsam_selector_response":
        import uq_tagsam_native_selector_20261012 as T
        import run_unified_q_tagsam_20261008 as A
        modules.extend((T, A, A.T, A.T.C, A.T.G, A.T.SCA))
        for name in ("run_multiscale_asm", "run_student_compatible_append", "run_gate1_continual"):
            if name in sys.modules:
                modules.append(sys.modules[name])
        official = H.verified_repo(T.REPO, T.COMMIT,
            ("model.py", "selection.py", "distill.py", "dataset.py", "epoch.py"))
    elif args.arm == "stad_native_local":
        import run_uq_stad_continual_20261012 as C
        modules.append(C)
        official = S.official_components()[-1]
    elif args.arm == "gecc_soft":
        official = dict(commit=H.GECC_COMMIT, source_sha256=E.file_hash(H.GECC_FILE))
        if official["source_sha256"] != H.GECC_SHA256:
            raise ValueError("GECC source hash differs")
    else:
        raise ValueError(args.arm)
    files = sorted(set(Path(module.__file__).resolve() for module in modules))
    manifest = dict(arm=args.arm, dataset=args.dataset, order_seed=args.order_seed,
        construction_seed=args.construction_seed, source_files={str(path): E.file_hash(path) for path in files},
        wrapper_sha256=E.file_hash(__file__), native_source=official,
        official_reproduction=False, method_selection_from_test=False,
        common_student_steps=2048, common_student_seeds=[150, 151, 152], common_student_batch=24,
        memory_rate=.05, access="current legal delta + own frozen synthetic prefix",
        synthetic_topology="STAD native support retained; GECC/TAGSAM edge-free")
    target = args.output / "runtime_manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("baseline source/protocol differs; choose a fresh output")
    S.save(target, manifest)
    if args.arm == "stad_native_local":
        C.run(args)
    else:
        args.arms, args.save_final = [args.arm], True
        E.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("gecc_soft", "tagsam_selector_response", "stad_native_local"), required=True)
    parser.add_argument("--dataset", choices=("arxiv", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, default=9101)
    parser.add_argument("--construction-seed", type=int, default=271006)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    run(args)
