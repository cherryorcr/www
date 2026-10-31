"""Full-budget baseline interface checks on a small graph, never method selection."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

import run_uq_final_confirmation_20261012 as E
import uq_high_fidelity_baselines_20261012 as H
import uq_stad_native_core_20261012 as S

R = E.R


def fixture(encoder):
    rng = torch.Generator().manual_seed(481)
    y = torch.arange(3).repeat_interleave(20)
    centers = torch.randn((3, 8), generator=rng) * 2
    x = centers[y] + .05 * torch.randn((60, 8), generator=rng)
    documents = (
        "This book discusses ancient history, archaeological evidence and historical events.",
        "This book describes military history, battles, armies and historical strategy.",
        "This book examines modern history, political institutions and social change.",
    )
    texts = [documents[int(label)] + f" Volume {i}." for i, label in enumerate(y)]
    edges = []
    for node in range(60):
        target = (node // 20) * 20 + (node + 1) % 20
        edges.extend(((node, target), (target, node)))
    return dict(ids=torch.arange(60), stage=0, x=x, y=y, texts=texts,
                text_emb=encoder.encode(texts).cpu(), edge_index=torch.tensor(edges).T,
                boundary_edge_index=torch.empty((2, 0), dtype=torch.long))


def run(args):
    path = args.output / "audit.json"
    if path.exists():
        raise FileExistsError("preserve smoke audit; choose a fresh output")
    encoder = R.L.U.FrozenMiniLM(args.minilm, args.device)
    delta = fixture(encoder)
    prompt = encoder.encode(["a book about ancient history", "a book about military history",
                             "a book about modern history"]).to(args.device)
    labels = torch.arange(3)
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    if args.arm == "stad_native_local":
        official = S.official_components()
        block, audit = S.construct(delta, 3, encoder, 271006, args.device, args.output,
                                  "bookhis", ["Ancient History", "Military History", "Modern History"], official)
    else:
        block, audit = H.build(args.arm, delta, None, labels, encoder, prompt, 271006, args.device)
    checks = R.V.validate_memory(block, None, 3)
    if any(checks.get(key) is False for key in ("budget_ok", "prefix_ok", "finite_ok")):
        raise AssertionError(checks)
    reference = encoder.encode(block["texts"]).cpu()
    error = float((block["text_emb"] - reference).abs().max())
    if error > 1e-6 or any(len(text) > 320 for text in block["texts"]):
        raise AssertionError("stored text differs from common encoder/cap")
    edges = block["edge_index"]
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= 3):
        raise AssertionError("synthetic edge index outside memory")
    # Exercise the common readout without reducing the baseline constructor budget.
    student = R.P.train_prompt_student(block, prompt, 150, args.device, 8, lr=.001, batch=3)
    with torch.no_grad():
        values = student.logits(delta["x"].to(args.device),
            R.P.normalized_adjacency(delta["edge_index"], 60, args.device), prompt)
    if values.shape != (60, 3) or not bool(values.isfinite().all()):
        raise AssertionError("common Student interface failed")
    result = dict(passed=True, arm=args.arm, fixture_rows=60, memory_rows=3,
        memory_rate=.05, construction_budget="native registered baseline budget, no smoke reductions",
        student_interface_steps=8, not_an_accuracy_experiment=True,
        stored_text_max_error=error, synthetic_edges=edges.shape[1], checks=checks,
        seconds=time.perf_counter() - started,
        peak_mb=torch.cuda.max_memory_allocated(args.device) / 2**20,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), audit=audit)
    S.save(path, result)
    print(json.dumps({key: value for key, value in result.items() if key != "audit"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("gecc_soft", "tagsam_selector_response", "stad_native_local"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    run(args)
