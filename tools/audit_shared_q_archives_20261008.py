"""Reconstruct shared-Q readouts and distinguish replay drift from new seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/v2_llm_fusion"))
import run_unified_q_constructor_attribution_20261008 as B
from summarize_shared_q_baseline_20261008 import load, MAIN, handoff_path


def block_difference(block, reference):
    if len(block["x"]) != len(reference["x"]) or not torch.equal(block["labels"], reference["labels"]):
        raise AssertionError("replay/control slot allocation differs")
    return dict(relative_x_difference=float((block["x"]-reference["x"]).norm()/reference["x"].norm().clamp_min(1e-12)),
        payload_source_rows_changed=sum(a!=b for a,b in zip(block["text_source_ids"],reference["text_source_ids"])),
        rows=len(block["x"]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT/"results/shared_q_baseline_20261008")
    p.add_argument("--source",type=Path,default=ROOT/"results/unified_q_attribution_20261008")
    args=p.parse_args()
    torch.set_num_threads(2)
    shared,handoff,incomplete=load(args.root)
    compared,replay=[],[]
    native_count,common_count,built_handoff_count=0,0,0
    for c in shared:
        proto=c["protocol"]
        name=f"{proto['dataset']}_o{proto['order_seed']}"
        path,source=args.root/f"shared_{name}",args.source/name
        if hashlib.sha256((source/"results.json").read_bytes()).hexdigest()!=proto["source_cell_sha256"]:
            raise AssertionError("source experiment changed")
        old_result=json.loads((source/"results.json").read_text())
        common=None
        parents={MAIN:None,"split_q":None}
        for stage in range(5):
            common_block,_=B.A.load_block(source,"cat_cctp",stage)
            common=B.A.append_block(common,common_block)
            full,_=B.A.load_block(path,MAIN,stage)
            old_full,old_audit=B.A.load_block(source,MAIN,stage)
            full_score=next(r["score"] for r in c["rows"] if (r["arm"],r["stage"],r["readout"])==(MAIN,stage,"native"))
            old_score=next(r["score"] for r in old_result["rows"] if (r["arm"],r["stage"],r["student_family"])==(MAIN,stage,"prompt"))
            for arm in (MAIN,"split_q"):
                block,audit=B.A.load_block(path,arm,stage)
                if json.loads(json.dumps(audit["config"]))!=proto["q_config"]:
                    raise AssertionError("archive configuration differs")
                if audit["historical_raw_reads"] or audit["old_x_relative_change_max"]:
                    raise AssertionError("access or old memory changed")
                memory=B.U.append_memory(parents[arm],block)
                B.V.validate_memory(memory,parents[arm],proto["budgets"][stage])
                if memory["edge_index"].numel():
                    raise AssertionError("unexpected synthetic edges")
                for row in c["rows"]:
                    if row["arm"]!=arm or row["stage"]!=stage:
                        continue
                    view=memory
                    if row["readout"]=="common_cat_payload":
                        view=B.fixed_payload(memory,common)
                        B.A.assert_fixed_content(memory,view)
                        common_count+=1
                    else:
                        native_count+=1
                    if B.V.memory_hash(view)!=row["memory_hash"]:
                        raise AssertionError("evaluated hash differs from reconstructed archive")
                parents[arm]=memory
                if arm==MAIN:
                    old_prototypes=old_audit["q_cost"]["prototype_source_ids"]
                    new_prototypes=audit["q_cost"]["prototype_source_ids"]
                    replay.append(dict(cell=name,stage=stage,**block_difference(block,old_full),
                        prototype_rows_changed=sum(a!=b for a,b in zip(old_prototypes,new_prototypes)),
                        f1_delta=full_score["final_f1"]-old_score["final_f1"],
                        accuracy_delta=full_score["accuracy"]-old_score["accuracy"]))
                else:
                    compared.append(dict(cell=name,stage=stage,**block_difference(block,full)))
    for c in handoff:
        proto=c["protocol"]
        path=handoff_path(args.root, proto['dataset'])
        for arm in (MAIN,"puma_full"):
            parent=None
            for stage in range(5):
                block,_=B.A.load_block(path,arm,stage)
                memory=B.A.append_block(parent,block)
                B.V.validate_memory(memory,parent,proto["budgets"][stage])
                rows=[r for r in c["rows"] if r["arm"]==arm and r["stage"]==stage]
                if any(B.V.memory_hash(memory)!=r["memory_hash"] for r in rows):
                    raise AssertionError("handoff reconstructed memory differs from Student input")
                built_handoff_count+=len(rows)
                parent=memory
    repaired_count=0
    repair_path=args.root/"handoff_cctp_repaired_sportsfit"
    if (repair_path/"results.json").exists():
        repaired=json.loads((repair_path/"results.json").read_text())
        if repaired["status"]=="complete":
            parent=None
            for stage in range(5):
                block,audit=B.A.load_block(repair_path,"cat_cctp_repaired",stage)
                if not audit["fixed_content"]["fixed_content_exact"] or audit["historical_raw_reads"]:
                    raise AssertionError("invalid repaired block")
                memory=B.A.append_block(parent,block)
                B.V.validate_memory(memory,parent,repaired["protocol"]["budgets"][stage])
                rows=[r for r in repaired["rows"] if r["stage"]==stage]
                if any(B.V.memory_hash(memory)!=r["memory_hash"] for r in rows):
                    raise AssertionError("repaired archive differs from evaluated memory")
                repaired_count+=len(rows)
                parent=memory
    result=dict(shared_cells=len(shared),handoff_datasets=len(handoff),
        native_readouts_reconstructed=native_count,common_readouts_reconstructed=common_count,
        newly_built_handoff_readouts_reconstructed=built_handoff_count,
        repaired_payload_readouts_reconstructed=repaired_count,
        same_seed_replay=replay,split_vs_shared=compared,incomplete=incomplete,
        note="Archive audit reconstructs our saved blocks and exact common payloads. External memory hashes were verified on server before label recovery. Re-encoding/current-delta checks here are recorded construction checks, not a second encoder run or OS access trace.")
    (args.root/"archive_audit.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,(list,dict))}))


if __name__=="__main__":
    main()
