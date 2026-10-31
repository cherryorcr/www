"""ONE-TIME test read on the four never-touched TAG benchmarks.

Protocol (PROTOCOL v2 sec.6): the test partition (node_id % 10 == 2) is read
EXACTLY ONCE, after the method is frozen, by student seeds that did not
participate in any selection. The frozen method is: deterministic constructor
(maxmin+Lloyd on [L2norm(X)||text_emb], mean X, top-5 fused text re-encoded,
text-kNN degree=1.0) at rate=5%. NO X optimisation, NO perturbation (all such
modules were falsified; see GATE_THIRD_FALSIFICATION).

The script is idempotent-guarded: it refuses to run if its output dir already
contains results.json, and it writes confirmation_accessed: true with the access
reason, so the one-time read is auditable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

import protocol_v3

sys.path.insert(0, "/home/pc/research/TAG_condensation/code/v2_codex")
from run_multiscale_asm import norm_adj, train_student, build_initial, load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(protocol_v3.STUDENT_SEEDS))
    ap.add_argument("--student-batch", type=int, default=protocol_v3.STUDENT.batch)
    ap.add_argument("--student-steps", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("test already read for this output dir (one-time rule)")
    device = args.device

    D = load_dataset(args.dataset, "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", device)
    x, texts, y, n = D["x"], D["texts"], D["y"], D["n"]
    pool = torch.where(D["pool_mask"])[0]
    k = math.floor(args.rate * len(pool))
    encoder = D["encoder"]
    prompt_emb = encoder.encode(D["prompts"])
    real_adj = norm_adj(D["edge_index"], n, device)
    xg = x.float().to(device)
    nmask = D["dev_mask"]
    tmask = ~nmask & ~D["pool_mask"]
    print(f"[{args.dataset}] n={n} pool={len(pool)} k={k} "
          f"dev={int(nmask.sum())} test={int(tmask.sum())}", flush=True)

    text_emb_pool = encoder.encode([texts[int(i)] for i in pool])
    art = build_initial(x[pool], text_emb_pool, [texts[int(i)] for i in pool],
                        k, encoder, device)

    def score(mask, tag):
        f1s, acs = [], []
        for seed in args.seeds:
            steps = protocol_v3.student_steps(k, override=args.student_steps)
            model = train_student(art, seed, device, steps, batch=args.student_batch)
            pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
            yy, pp = y[mask].numpy(), pred[mask].numpy()
            f1s.append(100 * f1_score(yy, pp, average="macro", zero_division=0))
            acs.append(100 * accuracy_score(yy, pp))
            del model
        print(f"  {tag}: MacroF1={np.mean(f1s):.2f}±{np.std(f1s):.2f} "
              f"Acc={np.mean(acs):.2f}±{np.std(acs):.2f} (n={int(mask.sum())})", flush=True)
        return {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                "accuracy_mean": float(np.mean(acs)), "accuracy_std": float(np.std(acs)),
                "n": int(mask.sum()), "per_student": [{"macro_f1": a, "accuracy": b}
                                                      for a, b in zip(f1s, acs)]}

    res = {"dataset": args.dataset, "n": int(n), "n_pool": int(len(pool)), "k": int(k),
           "rate": args.rate, "student_seeds": args.seeds,
           "confirmation_accessed": True,
           "access_reason": "one-time test read after method freeze (no X optimisation)",
           "test_rule": "node_id % 10 == 2",
           "method": "frozen_v1 constructive (no ASM, no jitter)",
           "dev": score(nmask, "dev"),
           "test": score(tmask, "test")}
    res.update(protocol_v3.config_dict(dataset=args.dataset, rate=args.rate,
                                       student_batch=args.student_batch,
                                       student_steps=(args.student_steps if
                                                      args.student_steps is not None else
                                                      protocol_v3.student_steps(k)),
                                       student_seeds=args.seeds))
    json.dump(res, open(out / "results.json", "w"), indent=1)
    print(f"  -> {out/'results.json'}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
