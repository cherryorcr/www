"""Verify the FUSE_CHARS root-cause fix: does length-adaptive text fusion
recover the loss on long-text datasets?

Diagnosis: FUSE_CHARS=320 split over g=5 texts gives 64 chars/text, which
truncates long texts (Cora abstracts ~989 chars) mid-sentence. Short-text
datasets (sportsfit titles ~122 chars) are unaffected.

Arms (all same constructor; only the TEXT FUSION changes):
  orig_320_64     current frozen rule: join of 5 x 64 chars
  adaptive_300    per-text budget = max(64, min(300, remaining)) -> long text kept whole-ish
  adaptive_600    per-text budget 600
  sent_cut        sentence-boundary truncation at ~300 chars per text
  textfull        use the FIRST selected text in full (g=1, no truncation)
  ours_reencode_ref  = orig, for reference

Metric: dev macro-F1 / accuracy, paired over student seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

import protocol_v3

sys.path.insert(0, "/home/pc/research/TAG_condensation/code/v2_codex")
from run_multiscale_asm import (norm_adj, train_student, load_dataset, maxmin_lloyd_gpu,
                                knn_topology_gpu)


def fuse(joint, members, texts, mode, g=5, base_chars=320):
    cen = joint[members].mean(0, keepdim=True)
    d = torch.cdist(joint[members], cen).flatten()
    order = members[torch.argsort(d)][:g]

    def cut(s, n):
        if len(s) <= n:
            return s
        if mode == "sent_cut":
            # cut at the last sentence boundary within n chars, else at a space
            head = s[:n]
            m = list(re.finditer(r"[.!?]\s", head))
            if m:
                return head[:m[-1].end()].strip()
            sp = head.rfind(" ")
            return head[:sp].strip() if sp > n // 2 else head
        sp = s[:n].rfind(" ")
        return s[:sp].strip() if sp > n // 2 else s[:n]

    if mode == "orig_320_64":
        per = max(48, base_chars // max(1, len(order)))
        return " ".join(texts[int(i)][:per] for i in order)
    if mode in ("adaptive_300", "adaptive_600"):
        per = int(mode.split("_")[1])
        return " ".join(cut(texts[int(i)], per) for i in order)
    if mode == "sent_cut":
        return " ".join(cut(texts[int(i)], 300) for i in order)
    if mode == "textfull":
        return texts[int(order[0])]
    raise ValueError(mode)


def build(x, te, texts, k, encoder, device, mode):
    joint = torch.cat((F.normalize(x.float(), dim=1), te), dim=1)
    assign = maxmin_lloyd_gpu(joint, k, device)
    jg = joint.to(device)
    syn_x, fused, mass = [], [], []
    for c in assign.unique():
        m = torch.where(assign == c)[0]
        if len(m) == 0:
            continue
        syn_x.append(x[m].float().mean(0))
        fused.append(fuse(jg, m.to(device), texts, mode))
        mass.append(float(len(m)))
    X = torch.stack(syn_x)
    T = encoder.encode(fused)
    edges = knn_topology_gpu(T, 1.0, device)
    return {"x": X, "text_emb": T, "texts": fused, "edge_index": edges,
            "weights": torch.tensor(mass)}


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
    ap.add_argument("--modes", nargs="+",
                    default=["orig_320_64", "adaptive_300", "adaptive_600", "sent_cut", "textfull"])
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("immutable")
    device = args.device

    D = load_dataset(args.dataset, "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", device)
    x, texts, y, n = D["x"], D["texts"], D["y"], D["n"]
    pool = torch.where(D["pool_mask"])[0]
    k = max(2, math.floor(args.rate * len(pool)))
    encoder = D["encoder"]
    prompt_emb = encoder.encode(D["prompts"])
    real_adj = norm_adj(D["edge_index"], n, device)
    xg = x.float().to(device)
    te = encoder.encode([texts[int(i)] for i in pool])
    Xp = x[pool]
    raw_len = float(np.mean([len(texts[int(i)]) for i in pool]))
    print(f"[{args.dataset}] pool={len(pool)} k={k} raw_text_len={raw_len:.0f}", flush=True)

    res = {}
    for mode in args.modes:
        art = build(Xp, te, [texts[int(i)] for i in pool], k, encoder, device, mode)
        fl = float(np.mean([len(t) for t in art["texts"]]))
        f1s, acs = [], []
        for s in args.seeds:
            steps = protocol_v3.student_steps(k, override=args.student_steps)
            model = train_student(art, s, device, steps, batch=args.student_batch)
            pred = model.logits(xg, real_adj, prompt_emb.to(device)).argmax(1).cpu()
            yy, pp = y[D["dev_mask"]].numpy(), pred[D["dev_mask"]].numpy()
            f1s.append(100 * f1_score(yy, pp, average="macro", zero_division=0))
            acs.append(100 * accuracy_score(yy, pp))
            del model
        res[mode] = {"macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                     "accuracy": float(np.mean(acs)), "fused_len": fl,
                     "example": art["texts"][0][:160], "scores": f1s}
        print(f"  {mode:14s} fused_len={fl:6.0f} F1={np.mean(f1s):6.2f}±{np.std(f1s):.2f} "
              f"Acc={np.mean(acs):6.2f}", flush=True)
        del art

    json.dump({**protocol_v3.config_dict(dataset=args.dataset, rate=args.rate,
               student_batch=args.student_batch,
               student_steps=(args.student_steps if args.student_steps is not None else
                              protocol_v3.student_steps(k)),
               student_seeds=args.seeds),
               "dataset": args.dataset, "k": k, "raw_text_len": raw_len,
               "results": res, "test_accessed": False},
              open(out / "results.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
