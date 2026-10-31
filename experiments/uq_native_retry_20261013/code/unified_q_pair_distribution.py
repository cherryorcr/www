"""Class-conditional product-kernel MMD without raw pairwise matrices.

The fixed nonlinear feature maps define k((x,t),(x',t')) =
<phi_x(x),phi_x(x')> * <phi_t(t),phi_t(t')>. Its empirical mean
embedding is phi_x.T @ phi_t / n; no n-by-n kernel is materialized.
"""
import math

import torch
import torch.nn.functional as F


def features(x, weight):
    return F.normalize(F.relu(x @ weight), dim=1)


@torch.no_grad()
def target(message, text, labels, hidden=64, banks=2, chunk=2048, seed=0):
    if min(hidden, banks, chunk) < 1:
        raise ValueError("positive joint kernel dimensions required")
    generator = torch.Generator(device=message.device).manual_seed(int(seed))
    wx = torch.randn((banks, message.shape[1], hidden), generator=generator,
                     device=message.device) / math.sqrt(message.shape[1])
    wt = torch.randn((banks, text.shape[1], hidden), generator=generator,
                     device=message.device) / math.sqrt(text.shape[1])
    classes = torch.unique(labels, sorted=True)
    pairs, shuffled, marg_x, marg_t = [], [], [], []
    # One permutation per class is shared by the random-feature banks. It
    # changes only the target pairing, never the actual payload candidate pool.
    permutations = [torch.randperm(int((labels == c).sum()), generator=generator,
                                   device=message.device) for c in classes]
    for bx, bt in zip(wx, wt):
        bp, bs, mx, mt = [], [], [], []
        for cls, permutation in zip(classes, permutations):
            ids = torch.where(labels == cls)[0]
            fg, ft = [], []
            for part in ids.split(chunk):
                fg.append(features(message[part], bx))
                ft.append(features(text[part], bt))
            g, t = torch.cat(fg), torch.cat(ft)
            bp.append(g.T @ t / len(ids))
            bs.append(g.T @ t[permutation] / len(ids))
            mx.append(g.mean(0))
            mt.append(t.mean(0))
        pairs.append(torch.stack(bp))
        shuffled.append(torch.stack(bs))
        marg_x.append(torch.stack(mx))
        marg_t.append(torch.stack(mt))
    result = dict(wx=wx, wt=wt, classes=classes, paired=torch.stack(pairs),
                  shuffled=torch.stack(shuffled), marginal_x=torch.stack(marg_x),
                  marginal_t=torch.stack(marg_t), seed=int(seed))
    difference=(result["paired"]-result["shuffled"]).square().sum((-1,-2))
    result["pair_shuffle_mmd"] = float(difference.sum(1).mean())
    return result


def loss(message, text, labels, reference, mode="paired"):
    if mode not in ("paired", "marginal", "shuffled"):
        raise ValueError("unknown joint distribution control")
    classes = torch.unique(labels, sorted=True)
    if not torch.equal(classes, reference["classes"].to(labels.device)):
        raise ValueError("joint distribution classes differ")
    losses = []
    for bank, (wx, wt) in enumerate(zip(reference["wx"], reference["wt"])):
        g, t = features(message, wx), features(text, wt)
        class_losses = []
        for index, cls in enumerate(classes):
            select = labels == cls
            gc, tc = g[select], t[select]
            if mode == "marginal":
                value = ((gc.mean(0) - reference["marginal_x"][bank,index]).square().sum()
                        +(tc.mean(0) - reference["marginal_t"][bank,index]).square().sum()) / 2
            else:
                joint = gc.T @ tc / len(gc)
                value = (joint - reference[mode][bank,index]).square().sum()
            class_losses.append(value)
        losses.append(torch.stack(class_losses).sum())
    return torch.stack(losses).mean()
