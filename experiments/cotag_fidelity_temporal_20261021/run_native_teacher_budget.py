"""Predeclared teacher-budget diagnostic; retain the native condensation core."""
from __future__ import annotations

import argparse
from pathlib import Path

import common as C
from common import torch
import native_tagsam as N
import run_native_small as RUN


def train_teacher(native, raw, text, stage, device):
    torch.manual_seed(9000+stage)
    model = native.model(native.args).to(device)
    optimizer = torch.optim.Adam([
        {'params': model.graph_encoder.parameters(), 'lr': 2e-5},
        {'params': model.text_encoder.parameters(), 'lr': 2e-5}])
    x, text = raw['x'].to(device), text.to(device)
    adjs = N.full_adjs(raw['edge_index'], len(x), device)
    generator = torch.Generator().manual_seed(9000+stage)
    losses, epochs = [], 0
    while len(losses) < 2048:
        epochs += 1
        model.train()
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), 1024):
            idx = order[start:start+1024].to(device)
            graph, txt = model.encode_graph(x, adjs)[idx], model.encode_text(text[idx])
            logits = (1/.07)*graph@txt.T
            targets = torch.arange(len(idx), device=device)
            loss = (N.F.cross_entropy(logits, targets)+N.F.cross_entropy(logits.T, targets))/2
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite native teacher loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if len(losses) == 2048:
                break
    return model.eval(), dict(epochs_started=epochs, minibatch=1024, lr=2e-5,
        updates=len(losses), losses=losses,
        selection='fixed last update 2048; no validation/test selection',
        message_passing='full legal source graph, not native NeighborSampler batches')


def main(args):
    original_root = C.RESULTS
    C.RESULTS = original_root/'teacher_budget2048'
    original_start, original_save = C.start_result, C.save
    original_initial = RUN.T.initial_block

    def initial(data, delta, rows, seed, device, cache):
        return original_initial(data, delta, rows, seed, device,
                                original_root/'native_epochs'/cache.name)

    def start(path, protocol):
        protocol = dict(protocol,
            name='native-tagsam-full-epoch-teacher2048-continual-v1',
            teacher_updates=2048, teacher_budget_wrapper_hash=C.sha(__file__))
        protocol['adaptations'] = [
            'teacher fixed final 2048-update native model, not official 3-epoch default or test-based selection'
            if item.startswith('teacher fixed final') else item for item in protocol['adaptations']]
        return original_start(path, protocol)

    def save(path, value):
        if 'native_student_diagnostic' in Path(path).parts and 'protocol' in value:
            value['protocol']['teacher_check'] = 'Reconstruct the original fixed 2048-update native teacher with the same legal data and seed'
            value['protocol']['teacher_budget_wrapper_hash'] = C.sha(__file__)
            if 'teacher_check' in value:
                value['teacher_check']['note'] = 'Fixed 2048-update teacher budget diagnostic; no outcome-based stopping'
        return original_save(path, value)

    RUN.T.initial_block = initial
    N.train_teacher = train_teacher
    C.start_result, C.save = start, save
    if not args.evaluate_only:
        RUN.run(args)
    import native_student_diagnostic as D
    args.epochs = 15
    D.run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('cora',), default='cora')
    parser.add_argument('--outer', type=int, default=5000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    main(args)
