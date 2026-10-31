"""Exploratory legacy readout of already sealed shared/independent Q memories."""
import argparse
import json
from pathlib import Path
import time

import torch

import run_handoff_student_transfer_20261008 as H


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--minilm",required=True)
    p.add_argument("--device",default="cuda:1")
    args=p.parse_args()
    torch.set_num_threads(2)
    path=args.root/"shared_legacy_transfer.json"
    protocol=dict(name="exploratory-legacy-readout-of-sealed-shared-Q",datasets=["wikics","sportsfit","bookhis"],
        orders=[9101,9102,9103],arms=["uq_dm_edgefree","split_q"],seeds=[150,151,152],steps=2048,batch=16,
        final_stage_only=True,construction_repeated=False,preregistered_before_prompt_results=False,
        runner_sha256=H.sha(__file__),legacy_trainer_sha256=H.sha(H.L.U.G.__file__),
        purpose="test Student-protocol sensitivity; all nine cells, no favorable-result selection")
    out=json.loads(path.read_text()) if path.exists() else dict(protocol=protocol,rows=[],status="running")
    if out["protocol"]!=protocol:
        raise ValueError("changed readout protocol")
    for dataset in protocol["datasets"]:
        data=None
        for order in protocol["orders"]:
            source=args.root/f"shared_{dataset}_o{order}"
            result=json.loads((source/"results.json").read_text())
            sp=result["protocol"]
            if result["status"]!="complete":
                raise AssertionError("source chain incomplete")
            if data is None:
                data=H.L.load_large(dataset,args.minilm,args.cache_root/dataset,args.device,sp)
            data["stage_classes"]=sp["groups"]
            if [H.V.hash_tensor(ids) for ids in H.L.stage_ids(data)]!=sp["stage_id_hashes"]:
                raise AssertionError("source IDs differ")
            if H.V.hash_tensor(data["report_dev_mask"])!=sp["report_mask_hash"]:
                raise AssertionError("report mask differs")
            for arm in protocol["arms"]:
                memory=None
                for stage in range(5):
                    block,_=H.A.load_block(source,arm,stage)
                    memory=H.U.append_memory(memory,block)
                row=next(r for r in result["rows"] if (r["arm"],r["stage"],r["readout"])==(arm,4,"native"))
                digest=H.V.memory_hash(memory)
                if digest!=row["memory_hash"]:
                    raise AssertionError("sealed source memory changed")
                prior=[r for r in out["rows"] if (r["dataset"],r["order"],r["arm"])==(dataset,order,arm)]
                if prior:
                    if len(prior)!=1 or prior[0]["memory_hash"]!=digest:
                        raise AssertionError("resume source differs")
                    continue
                H.V.sync(args.device)
                start=time.perf_counter()
                with H.A.preserve_rng():
                    score=H.score(memory,data,4,"legacy",args.device,protocol["seeds"],protocol["steps"])
                H.V.sync(args.device)
                record=dict(dataset=dataset,order=order,arm=arm,memory_hash=digest,score=score,
                    prompt_score=row["score"],source_result_sha256=H.sha(source/"results.json"),
                    seconds=time.perf_counter()-start)
                out["rows"].append(record)
                H.V.write_json(path,out)
                print(json.dumps(dict(dataset=dataset,order=order,arm=arm,legacy_f1=score["final_f1"],
                    accuracy=score["accuracy"])),flush=True)
        del data
        torch.cuda.empty_cache()
    if len(out["rows"])!=18:
        raise AssertionError("incomplete legacy readout")
    out["status"]="complete"
    H.V.write_json(path,out)


if __name__=="__main__":
    main()
