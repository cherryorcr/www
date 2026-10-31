"""Evaluate a completed native-core memory with a fixed native student protocol.

This is a separate diagnostic. It does not alter the common-student results or
select an epoch from a test mask.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import common as C
from common import R, torch
import native_tagsam as N
import tagsam_response as T
sys.path.append(str(C.ROOT/'tools'))
import run_uq_baseline_expansion_20261013 as B


def load_memory(path: Path):
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def train_native(native, memory, text, seed, device, graph_lr, text_lr, epochs):
    torch.manual_seed(seed)
    model = native.model(native.args).to(device)
    optimizer = torch.optim.SGD([
        {"params": model.graph_encoder.parameters(), "lr": graph_lr,
         "momentum": 0.9, "weight_decay": 5e-4},
        {"params": model.text_encoder.parameters(), "lr": text_lr,
         "momentum": 0.9, "weight_decay": 5e-4},
    ])
    x = memory["x"].float().to(device)
    text = text.float().to(device)
    losses = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x), device=device)
        epoch_losses = []
        for start in range(0, len(x), 32):
            idx = order[start:start+32]
            # A self-loop memory has no neighbors outside the minibatch.
            edge = torch.arange(len(idx), device=device).repeat(2, 1)
            syn_adj = N.full_adjs(edge, len(idx), device)
            loss, _ = model(x[idx], syn_adj, text[idx])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite native student loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(dict(epoch=epoch+1, batch_losses=epoch_losses))
    return model.eval(), losses


def score(native, model, data, prompt_text, device):
    edge = data["edge_index"].long()
    adj = N.full_adjs(edge, len(data["x"]), device)
    with torch.no_grad():
        logits = model(data["x"].float().to(device), adj, prompt_text.to(device),
                       is_eval=True).cpu().argmax(1)
    output = {}
    for split, mask in (("report_dev", data["report_dev_mask"]),
                        ("sealed_test", data["test_mask"])):
        allowed = mask.cpu()
        y = data["y"][allowed].numpy()
        pred = logits[allowed].numpy()
        output[split] = dict(nodes=int(allowed.sum()),
            macro_f1=float(100*f1_score(y, pred, labels=list(range(len(data["prompts"]))),
                                        average="macro", zero_division=0)),
            accuracy=float(100*accuracy_score(y, pred)))
    return output


def run(args):
    root = C.RESULTS / "native_epochs" / f"{args.dataset}_outer{args.outer}"
    result_path = root / "results.json"
    result = C.read(result_path)
    if result["status"] != "complete":
        raise RuntimeError("Native construction chain is not complete")
    stage = len(result["protocol"]["budgets"]) - 1
    row = next(row for row in result["rows"] if row["stage"] == stage)
    audit = row["construction"]
    memory = load_memory(root / f"s{stage}" / "memory.pt.gz")
    if R.V.memory_hash(memory) != audit["memory_hash"]:
        raise AssertionError("Final native memory hash mismatch")
    if memory["edge_index"].numel():
        raise AssertionError("Native diagnostic expects edge-free memory")

    loader = SimpleNamespace(dataset=args.dataset, minilm=C.MINILM, device=args.device,
        order_seed=9101, text_cache=C.ROOT/f'results/uq_baseline_expansion_20261013/text_cache/{args.dataset}')
    data = B.load_data(loader, B.read_reference(None, args.dataset), R)
    for field, mask in (('pool_mask_hash', 'pool_mask'), ('report_mask_hash', 'report_dev_mask'),
                        ('test_mask_hash', 'test_mask')):
        assert R.V.hash_tensor(data[mask]) == result['protocol'][field]
    assert [R.V.hash_tensor(v) for v in R.L.stage_ids(data)] == result['protocol']['stage_id_hashes']
    native = N.components(args.device, data["x"].shape[1])
    bert = N.BertFeatures(args.device)
    memory_text = bert.encode(memory["texts"])
    prompt_text = bert.encode(data["prompts"])
    response = audit["response"]
    graph_lr = float(response["final_graph_lr"])
    text_lr = float(response["final_text_lr"])

    protocol = dict(name="native-tagsam-student-diagnostic-v1",
        source=str(result_path), source_memory_hash=audit["memory_hash"],
        dataset=args.dataset, stage=stage, construction_protocol=result["protocol"],
        native_student="upstream CLIP graph encoder and residual text projection",
        student_seeds=C.STUDENT_SEEDS, epochs=args.epochs, batch=32,
        batch_mode="shuffle synthetic rows once per epoch, retain final incomplete batch",
        graph_lr=graph_lr, text_lr=text_lr, optimizer="SGD, momentum=.9, weight_decay=5e-4",
        prompt_rule="common class prompts encoded with frozen BERT CLS",
        report_masks="same common report-dev and sealed-test masks; no epoch selection",
        evaluation_split="diagnostic only; separate from common MiniLM student",
        source_result_hash=C.sha(result_path), runner_hash=C.sha(__file__),
        changes_from_official_eval=['fixed last epoch rather than best test epoch',
            'common all-seen-category prompts rather than independent five-way tasks',
            'batch-local self-loop equivalent instead of NeighborSampler'],
        teacher_check='Reconstruct the final-stage fixed 3-epoch teacher with the original seed and legal inputs; evaluate once, no tuning')
    out = root / "native_student_diagnostic"
    path = out / "results.json"
    if path.exists():
        old = C.read(path)
        if old["protocol"] != protocol:
            raise ValueError("Existing native diagnostic protocol differs")
        print(json.dumps(old, indent=2))
        return
    rows = []
    for seed in C.STUDENT_SEEDS:
        R.V.sync(args.device)
        if str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        model, losses = train_native(native, memory, memory_text, seed, args.device,
                                      graph_lr, text_lr, args.epochs)
        scores = score(native, model, data, prompt_text, args.device)
        R.V.sync(args.device)
        rows.append(dict(seed=seed, losses=losses, scores=scores,
                         seconds=time.perf_counter()-started,
                         peak_mb=torch.cuda.max_memory_allocated(args.device)/2**20
                         if str(args.device).startswith("cuda") else None))
        del model
        torch.cuda.empty_cache()
    summary = {}
    for split in ("report_dev", "sealed_test"):
        summary[split] = {
            "macro_f1": float(np.mean([row["scores"][split]["macro_f1"] for row in rows])),
            "macro_f1_sd": float(np.std([row["scores"][split]["macro_f1"] for row in rows], ddof=1)),
            "accuracy": float(np.mean([row["scores"][split]["accuracy"] for row in rows])),
            "accuracy_sd": float(np.std([row["scores"][split]["accuracy"] for row in rows], ddof=1)),
        }
    parent = None if stage == 0 else load_memory(root/f's{stage-1}'/'memory.pt.gz')
    delta = R.L.delta_view(data, R.L.stage_ids(data)[stage], stage)
    raw = T.A.reference_memory(delta, parent)
    raw_text = bert.encode(raw['texts'])
    teacher, teacher_audit = N.train_teacher(native, raw, raw_text, stage, args.device)
    teacher_scores = score(native, teacher, data, prompt_text, args.device)
    old_losses, new_losses = audit['teacher']['losses'], teacher_audit['losses']
    assert len(old_losses) == len(new_losses)
    teacher_check = dict(scores=teacher_scores, training=teacher_audit,
        original_loss_max_abs_difference=max(abs(a-b) for a,b in zip(old_losses, new_losses)),
        legal_reference_rows=len(raw['x']), reconstruction_only=True,
        note='Default teacher budget transferred to the small continual delta; not an optimized native teacher upper bound')
    output = dict(protocol=protocol, status="complete", rows=rows, summary=summary,
                  teacher_check=teacher_check)
    C.save(path, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), default="cora")
    parser.add_argument("--outer", type=int, default=5000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    run(args)
