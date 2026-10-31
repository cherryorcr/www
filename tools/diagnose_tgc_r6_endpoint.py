"""Compare freshly reconstructed endpoints without changing frozen artifacts."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/v2_llm_fusion"))
import torch
import run_tgc_r6_mechanisms as V

E, K, D, T = V.E, V.K, V.D, V.T


def differences(a, b):
    result = {"same_memory_hash": T.memory_hash(a) == T.memory_hash(b)}
    for key in set(a) | set(b):
        if key not in a or key not in b:
            result[key] = "missing"
        elif isinstance(a[key], torch.Tensor):
            same = torch.equal(a[key], b[key])
            result[key] = {"equal": same, "max_abs": float((a[key].double()-b[key].double()).abs().max())
                           if a[key].shape == b[key].shape and a[key].numel() else None}
        elif isinstance(a[key], (str, int, bool, float, list)):
            result[key] = a[key] == b[key]
    return result


def main(args):
    torch.set_num_threads(2)
    E.configure(V.protocol())
    cache = Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919") / f"{args.dataset}_r005/text_emb_pool_only.pt"
    data = V.R.R.load_data(args.dataset, args.minilm, args.device, cache)
    ids, _ = D.set_stream(data, args.order)
    folder = V.SOURCE / args.dataset / args.order
    c0 = E.load(folder / "c0_s2_memory.pt")
    block = E.load(folder / "c0_s2_block.pt")
    delta = T.raw_delta_block(data, ids[2], 2)
    result = {}
    for arm, endpoint, prefix in (("c0", "joint", "shared"), ("recursive", "recursive", "recursive")):
        old = E.load(folder / f"{arm}_s1_memory.pt")
        base = V.V.append_parent(old, block, args.device)
        q = D.current_assignment(data, ids[2], c0, len(old["x"]), 2, args.device)
        ctx = K.context(old, delta, base, q, V.V.TEACHER, args.device)
        p, coarse, info = K.learn(q, data["y"][ids[2]], ctx, len(old["x"]), V.CFG)
        memory, _ = K.materialize(base, delta, p, coarse, data["encoder"], V.CFG)
        original, jp, ja = V.J.construct(old, delta, base, q, data["y"][ids[2]], data["encoder"], V.V.TEACHER, args.device, V.CFG)
        target = E.load(folder / f"{endpoint}_s2_memory.pt")
        oldp = E.load(folder / f"{prefix}_s2_assignment.pt")
        result[arm] = {"base_vs_c0": differences(base, c0), "replay_vs_frozen": differences(memory, target),
                       "K_vs_J": differences(memory, original["joint"]),
                       "p_error_vs_frozen": float((p-oldp).abs().max()),
                       "p_error_K_vs_J": float((p-jp).abs().max()),
                       "teacher_now": ctx["audit"]["teachers"],
                       "teacher_frozen": E.read(folder / f"{prefix}_s2_construction.json")["teachers"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    E.save(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--order", default="order19341")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    main(parser.parse_args())
