"""Controlled construction replay and callback side-effect audit; no F1 tuning."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import unified_q_multimodal as U
import external_continual_mechanisms as M
import run_unified_q_large_smoke_20261007 as L
import run_unified_q_revision_20261007 as V
import run_unified_q_inloop_cat_20261007 as I
import run_unified_q_fixed_content_20261007 as D


def numpy_equal(a,b):
    return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("/home/pc/research/TAG_condensation"))
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--deterministic",action="store_true")
    parser.add_argument("--arm",choices=("uq_dm_edgefree","uq_dm_edgefree_no_gm"),default="uq_dm_edgefree")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.root
    torch.set_num_threads(2)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    ref=L.read(root/"results/unified_q_inloop_cat_20261007/wikics/protocol.json")
    data=L.load_large("wikics","/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
        root/"results/public_baselines_text_cache_20261001/wikics",args.device,ref)
    delta=L.delta_view(data,L.stage_ids(data)[0],0)
    labels=M.row_labels(M.ratio_allocation(delta["y"],ref["budgets"][0]))
    cfg=I.config(args.arm,271006)
    memories={}
    callback_report={}
    records=[]
    for name in ("plain_a","plain_b","with_callback"):
        random.seed(271006)
        np.random.seed(271006)
        torch.manual_seed(271006)
        def callback(state):
            before_hash=V.memory_hash(U.append_memory(None,state["block"]))
            before_q=state["q"].clone()
            py,np_state,cpu=random.getstate(),np.random.get_state(),torch.get_rng_state()
            cuda=torch.cuda.get_rng_state_all()
            D.topology_snapshot(state)
            callback_report.update(
                memory_unchanged=before_hash==V.memory_hash(U.append_memory(None,state["block"])),
                q_unchanged=torch.equal(before_q,state["q"]),
                python_rng_unchanged=py==random.getstate(),
                numpy_rng_unchanged=numpy_equal(np_state,np.random.get_state()),
                torch_cpu_rng_unchanged=torch.equal(cpu,torch.get_rng_state()),
                cuda_rng_unchanged=all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())))
        block,audit=U.optimize_stage(delta,None,labels,data["encoder"],data["prompt_emb"],cfg,args.device,
            use_boundary=False,use_topology=False,
            diagnostic_callback=callback if name=="with_callback" else None)
        memories[name]=U.append_memory(None,block)
        records.append(dict(name=name,memory_hash=V.memory_hash(memories[name]),
            final_distribution=audit["distribution_matching"]["final_train"]))
    contrasts=[]
    for left,right in (("plain_b","plain_a"),("with_callback","plain_a")):
        a,b=memories[left],memories[right]
        contrasts.append(dict(left=left,right=right,exact=V.memory_hash(a)==V.memory_hash(b),
            x_relative_error=float((a["x"]-b["x"]).norm()/b["x"].norm()),
            text_source_changes=sum(x!=y for x,y in zip(a["text_source_ids"],b["text_source_ids"]))))
    result=dict(scope="WikiCS first-stage construction only; no metric/model selection",arm=args.arm,records=records,
        callback_side_effects=callback_report,contrasts=contrasts,
        implementation=str(U.__file__),cuda_deterministic=torch.are_deterministic_algorithms_enabled())
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not all(callback_report.values()):
        raise SystemExit(1)


if __name__=="__main__":
    main()
