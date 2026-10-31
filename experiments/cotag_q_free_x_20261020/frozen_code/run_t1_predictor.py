"""T1: training-free retrainability predictor validation.

The law says retrainability = distributional isomorphism x coverage. v3 showed
pairwise-cos (self-discriminability) does NOT predict F1. Test the law's
prediction: distance from synthetic text embeddings to the REAL text manifold
(and coverage of the real manifold by synthetic roots) should predict F1.

Uses 13 saved artifacts with known dev F1 (v2_full: 4, v3_disc: 5,
headroom: 4). Computes candidate predictors per artifact:
  shift_nn      mean nearest-real-neighbor cosine distance of syn text embs
  shift_mmd     RBF-MMD^2 between syn text embs and real text embs
  cover_mean    mean over real nodes of distance to nearest syn text emb
  cover_max     max (covering radius)
  pair_cos      mean pairwise cos among syn embs (known-bad control)
Then Spearman correlation of each predictor with dev F1.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from transformers import AutoModel, AutoTokenizer

DEV_RULE = 1
BASE = Path("/home/pc/research/problem_validation_results")
RUNS = {
    "v2_full_2026_09_09": {
        "v1_concat_320": "syn_v1_concat_320.pt", "v1_concat_700": "syn_v1_concat_700.pt",
        "v2_llm_50w": "syn_v2_llm_50w.pt", "v2_llm_100w": "syn_v2_llm_100w.pt"},
    "v3_disc_2026_09_09": {
        "disc_prompt": "syn_disc_prompt.pt", "keyword_bag": "syn_keyword_bag.pt",
        "concat_kw": "syn_concat_kw.pt", "best_of_n": "syn_best_of_n.pt"},
    "headroom_2026_09_09": {
        "kmeans_multi": "syn_kmeans_multi.pt", "text_only_cluster": "syn_text_only_cluster.pt",
        "oracle_label": "syn_oracle_label.pt"},
}
# note: baseline appears in all three runs with identical artifact; keep one copy (v1_concat_320)


class FrozenMiniLM:
    def __init__(self, path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts, batch_size=64, max_length=256):
        parts = []
        for s in range(0, len(texts), batch_size):
            b = self.tokenizer(texts[s:s + batch_size], padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(self.device)
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            parts.append(F.normalize(((h * m).sum(1) / m.sum(1).clamp_min(1e-9)).float(), dim=1).cpu())
        return torch.cat(parts)


def mmd_rbf(a, b, n_sub=500, seed=0):
    g = torch.Generator().manual_seed(seed)
    if len(a) > n_sub:
        a = a[torch.randperm(len(a), generator=g)[:n_sub]]
    if len(b) > n_sub:
        b = b[torch.randperm(len(b), generator=g)[:n_sub]]
    z = torch.cat([a, b])
    d2 = torch.cdist(z, z).square()
    sigma2 = d2[d2 > 0].median()
    k = torch.exp(-d2 / (2 * sigma2))
    na, nb = len(a), len(b)
    kaa = (k[:na, :na].sum() - na) / (na * (na - 1))
    kbb = (k[na:, na:].sum() - nb) / (nb * (nb - 1))
    kab = k[:na, na:].mean()
    return float(kaa + kbb - 2 * kab)


def main():
    device = "cuda:0"
    data = torch.load("/home/pc/research/GraphCLIP/processed_data/cora.pt",
                      map_location="cpu", weights_only=False)
    encoder = FrozenMiniLM("/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", device)
    real_emb = encoder.encode(list(data.raw_texts))

    # gather F1s from results.json files
    f1 = {}
    for run in RUNS:
        res = json.load(open(BASE / run / "results.json"))
        for r in res["results"]:
            f1[r["arm"]] = r["dev_macro_f1_mean"]
    # headroom baseline_maxmin == v1_concat/v1_concat_320 (identical artifact)

    rows = []
    for run, arms in RUNS.items():
        for arm, fname in arms.items():
            p = BASE / run / fname
            if not p.exists():
                print(f"missing: {p}")
                continue
            syn = torch.load(p, map_location="cpu", weights_only=False)
            e = syn["text_emb"]
            e = F.normalize(e.float(), dim=1)
            # shift: syn -> real nearest neighbor cosine distance
            sims = e @ real_emb.T
            shift_nn = float((1 - sims.max(dim=1).values).mean())
            # coverage: real -> nearest syn
            cover = 1 - sims.max(dim=0).values
            cover_mean = float(cover.mean())
            cover_max = float(cover.max())
            # mmd
            shift_mmd = mmd_rbf(e, real_emb)
            # pairwise cos control
            k = len(e)
            pairs = torch.triu_indices(k, k, offset=1)
            pair_cos = float((e[pairs[0]] * e[pairs[1]]).sum(1).mean())
            arm_f1 = f1.get(arm)
            if arm_f1 is None:
                continue
            rows.append(dict(arm=arm, f1=arm_f1, shift_nn=shift_nn, shift_mmd=shift_mmd,
                             cover_mean=cover_mean, cover_max=cover_max, pair_cos=pair_cos))
            print(f"{arm:20s} f1={arm_f1:6.2f} shift_nn={shift_nn:.4f} mmd={shift_mmd:.4f} "
                  f"cover_mean={cover_mean:.4f} pair_cos={pair_cos:.4f}", flush=True)

    print(f"\n{len(rows)} artifacts")
    f1s = [r["f1"] for r in rows]
    print("\npredictor        spearman   pearson")
    for key in ("shift_nn", "shift_mmd", "cover_mean", "cover_max", "pair_cos"):
        vals = [r[key] for r in rows]
        sp = spearmanr(vals, f1s)
        pe = pearsonr(vals, f1s)
        print(f"{key:15s} {sp.statistic:+.3f} (p={sp.pvalue:.3f})   {pe.statistic:+.3f} (p={pe.pvalue:.3f})")
    # combined: shift + coverage (both should be low). simple sum of ranks
    sn = np.argsort(np.argsort([r["shift_nn"] for r in rows]))
    cm = np.argsort(np.argsort([r["cover_mean"] for r in rows]))
    combo = sn + cm
    sp = spearmanr(combo, f1s)
    print(f"{'rank(shift)+rank(cover)':15s} {sp.statistic:+.3f} (p={sp.pvalue:.3f})")

    json.dump({"rows": rows}, open(BASE / "t1_predictor_validation.json", "w"), indent=1)
    print("\nsaved t1_predictor_validation.json")


if __name__ == "__main__":
    main()
