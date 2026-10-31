"""Loss-level probe, not a new trained Student or a downstream benchmark."""
import argparse
import json
from pathlib import Path

import torch

import unified_q_multimodal as U
import graphtext_prompt_student as P


def probe():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(271009)
        labels=torch.arange(3).repeat_interleave(4)
        memory=dict(x=torch.randn(12,4),text_emb=torch.randn(12,8),labels=labels)
        prompt=torch.randn(3,8)
        model=P.GraphTextPromptStudent(4,hidden=8,out=8,text_dim=8)
        adj=P.normalized_adjacency(torch.empty(2,0,dtype=torch.long),12,'cpu')
        within=torch.tensor([3,0,1,2,7,4,5,6,11,8,9,10])
        across=torch.arange(12).roll(4)
        groups=(labels>0).long()
        output=[]
        for weighting in ('node_mean','group_class'):
            for chunk in (0,3):
                def measure(permutation, rows=None):
                    current=dict(memory,text_emb=memory['text_emb'][permutation])
                    loss=U._loss(model,current,adj,prompt,row_ids=rows,group_ids=groups,
                                 loss_weighting=weighting,chunk_size=chunk)
                    gradient=torch.cat([g.flatten() for g in torch.autograd.grad(loss,tuple(model.parameters()))])
                    return float(loss.detach()),gradient
                original, grad=measure(torch.arange(12))
                permuted, pg=measure(within)
                changed, cg=measure(across)
                rows=torch.tensor([0,1,4,8,9])
                batch,_=measure(torch.arange(12),rows)
                batch_permuted,_=measure(within,rows)
                output.append(dict(weighting=weighting,chunk_size=chunk,
                    full_loss=original,within_class_loss_error=abs(original-permuted),
                    within_class_gradient_max_error=float((grad-pg).abs().max()),
                    within_class_gradient_relative_error=float((grad-pg).norm()/grad.norm()),
                    across_class_loss_difference=abs(original-changed),
                    across_class_gradient_max_difference=float((grad-cg).abs().max()),
                    fixed_minibatch_within_class_loss_difference=abs(batch-batch_permuted)))
    return dict(scope='12-row deterministic CPU loss/gradient probe, no dataset labels or report/test evaluation',
        property='Full-candidate multimodal loss and theta gradients are invariant to within-class text permutations when weights are constant within each permuted group.',
        caveat='Actual Student samples paired row minibatches. Global within-class text permutation can change those minibatch text marginals, so this does NOT prove identical training trajectories or classification scores.',
        records=output)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    result=probe()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
