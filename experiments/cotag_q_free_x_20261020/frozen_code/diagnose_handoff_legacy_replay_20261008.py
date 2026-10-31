"""Diagnose archived score drift without selecting/tuning memory or Student."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

import run_handoff_student_transfer_20261008 as H


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=("wikics","bookhis"),required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--text-cache",type=Path,required=True)
    p.add_argument("--minilm",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:1")
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostic is immutable")
    torch.set_num_threads(2)
    manifest=json.loads(args.manifest.read_text())
    paired=json.loads((args.run/"results.json").read_text())
    data=H.L.load_large(args.dataset,args.minilm,args.text_cache,args.device,manifest)
    H.OLD.configure_eval(data,manifest["groups"],args.device)
    if H.L.U.G is not H.OLD.U.G:
        raise AssertionError("legacy trainers are not the same imported module")
    arms=("k_center","gecc_hard_tag") if args.dataset=="wikics" else ("random_real","gecc_hard_tag")
    results=[]
    for arm in arms:
        source=next(r for r in manifest["rows"] if r["arm"]==arm and r["stage"]==0)
        memory=torch.load(source["memory_path"],map_location="cpu",weights_only=False)
        if H.original_hash(memory,source["hash_kind"])!=source["memory_hash"]:
            raise AssertionError("source memory changed")
        prior=next(r for r in paired["rows"] if (r["arm"],r["stage"],r["student"])==(arm,0,"legacy"))
        start=time.perf_counter()
        with H.A.preserve_rng():
            original=H.OLD.student_score(memory,0,data,[150,151,152],args.device,16,2048)
        with H.A.preserve_rng():
            wrapper=H.score(memory,data,0,"legacy",args.device,[150,151,152],2048)
        original_seeds={seed:row for seed,row in zip(original["student_seeds"],original["per_student"])}
        prior_seeds={row["seed"]:row for row in prior["score"]["per_student"]}
        comparison=[]
        for row in wrapper["per_student"]:
            seed=row["seed"]
            comparison.append(dict(seed=seed,original_entry_f1=original_seeds[seed]["cumulative_macro_f1"],
                wrapper_f1=row["final_f1"],paired_run_f1=prior_seeds[seed]["final_f1"],
                original_entry_accuracy=original_seeds[seed]["cumulative_accuracy"],wrapper_accuracy=row["accuracy"]))
        results.append(dict(arm=arm,stage=0,memory_hash=source["memory_hash"],
            nonself_edges=int((memory["edge_index"][0]!=memory["edge_index"][1]).sum()),
            per_seed=comparison,seconds=time.perf_counter()-start))
        print(json.dumps(results[-1]),flush=True)
    out=dict(dataset=args.dataset,status="complete",purpose="score-drift diagnosis only; no favorable rerun replaces the main paired table",
        torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(args.device),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        legacy_trainer_sha256=H.sha(H.L.U.G.__file__),wrapper_sha256=H.sha(H.__file__),rows=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    H.V.write_json(args.output,out)


if __name__=="__main__":
    main()
