"""Trace construction without changing the historical R5 implementation."""
from contextlib import contextmanager
import time
from unittest.mock import patch

import torch
import tgc_r6_mechanisms as K

J, R, T, M = K.J, K.R, K.T, K.M


def fingerprint(value):
    if isinstance(value, torch.Tensor):
        value = {"value": value}
    return R.C.tensor_hash({k: v.detach().cpu() for k, v in value.items()})


def sparse_fingerprint(a):
    a = a.coalesce()
    return fingerprint({"indices": a.indices(), "values": a.values()})


@contextmanager
def deterministic(enabled):
    before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    if enabled:
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(before[0])
        torch.backends.cuda.matmul.allow_tf32 = before[1]
        torch.backends.cudnn.allow_tf32 = before[2]
        torch.backends.cudnn.benchmark = before[3]


def traced_teacher(support, base, seed, mode, cfg, device):
    R.sync(device)
    started = time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = R.G.CLIPStudent(support["x"].shape[1], text_dim=support["text_emb"].shape[1]).to(device)
    x, text, adj = support["x"].to(device), support["text_emb"].to(device), M.adjacency(support, device)
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    gen = torch.Generator().manual_seed(seed * 7 + 1)
    sequence, losses, trace = [], [], [{"step": 0, "state": fingerprint(model.state_dict())}]
    old_n, sampled_old, total = support["old_rows"], 0, 0
    for step in range(cfg.teacher_steps):
        ids = M.sample_rows(len(x), old_n, cfg.teacher_batch, gen, mode, old_n / len(base["x"]))
        sequence.append(ids)
        sampled_old += int((ids < old_n).sum())
        total += len(ids)
        loss = T._info_nce(model, x, text, adj, ids.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gradient = fingerprint({name: value.grad for name, value in model.named_parameters()})
        opt.step()
        losses.append(float(loss.detach()))
        trace.append({"step": step + 1, "state": fingerprint(model.state_dict()),
                      "gradient": gradient, "loss": losses[-1]})
    model.eval()
    R.sync(device)
    return model, {"seed": seed, "sampler": mode, "steps": cfg.teacher_steps,
                   "sampled_old_fraction": sampled_old / total,
                   "batch_sequence_hash": fingerprint({"ids": torch.stack(sequence)}),
                   "adjacency_hash": sparse_fingerprint(adj), "trace": trace,
                   "loss_first": losses[0], "loss_last": losses[-1],
                   "seconds": time.perf_counter() - started, "state_sha256": fingerprint(model.state_dict())}


def construct(old, delta, base, q, labels, encoder, teacher_cfg, device, cfg, stable=False):
    trace = []
    original = J.objective

    def objective(p, *args, **kwargs):
        result = original(p, *args, **kwargs)
        trace.append({"step": len(trace), "p_hash": fingerprint(p), "loss": float(result[0].detach())})
        return result

    with deterministic(stable), patch.object(M, "fit_teacher", traced_teacher):
        ctx = K.context(old, delta, base, q, teacher_cfg, device)
        with patch.object(J, "objective", objective):
            p, coarse, audit = J.learn(q, labels, ctx["h"], ctx["adjacency"], len(old["x"]), cfg)
        memory, material = K.materialize(base, delta, p, coarse, encoder, cfg)
    return memory, p, {"context": ctx["audit"], "assignment": audit, "materialization": material,
                       "h_hash": fingerprint(ctx["h"]), "raw_adjacency_hash": sparse_fingerprint(ctx["adjacency"]),
                       "q_hash": fingerprint(q), "assignment_trace": trace,
                       "deterministic_construction": stable, "report_labels_used": False,
                       "historical_raw_reads": 0}


def compare(a, b, pa, pb, aa, ab):
    teachers = []
    for ta, tb in zip(aa["context"]["teachers"], ab["context"]["teachers"]):
        first = next((x["step"] for x, y in zip(ta["trace"], tb["trace"]) if x != y), None)
        teachers.append({"first_different_step": first,
                         "same_batches": ta["batch_sequence_hash"] == tb["batch_sequence_hash"],
                         "same_adjacency": ta["adjacency_hash"] == tb["adjacency_hash"]})
    return {"teachers": teachers, "same_h": aa["h_hash"] == ab["h_hash"],
            "first_assignment_difference": next((x["step"] for x, y in zip(aa["assignment_trace"], ab["assignment_trace"])
                                                  if x != y), None),
            "same_p": torch.equal(pa, pb), "p_max_abs": float((pa - pb).abs().max()),
            "same_memory": T.memory_hash(a) == T.memory_hash(b),
            "x_max_abs": float((a["x"] - b["x"]).abs().max()),
            "same_text": a["texts"] == b["texts"], "same_edges": torch.equal(a["edge_index"], b["edge_index"])}
