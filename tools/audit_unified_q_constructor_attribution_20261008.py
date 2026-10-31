"""Reconstruct every evaluated memory, including the shared-payload readout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code/v2_llm_fusion"))
import run_unified_q_constructor_attribution_20261008 as R
from summarize_unified_q_constructor_attribution_20261008 import load


def diagnostic_record(block, audit, full):
    trace = audit["trace"]
    dm = audit["distribution_matching"]
    n = len(block["x"])
    changed = sum(a != b for a, b in zip(block["text_source_ids"], full["text_source_ids"]))
    return dict(rows=n, unique_text_ratio=len(set(block["texts"]))/n,
        relative_x_difference_from_full=float((block["x"]-full["x"]).norm()/full["x"].norm().clamp_min(1e-12)),
        changed_payload_rows_from_full=changed,
        x_residual_ratio_max=audit["x_residual_ratio_max"],
        gradient_match_first=trace[0]["gradient_match"], gradient_match_last=trace[-1]["gradient_match"],
        q_gradient_norm_first=trace[0]["q_gradient_norm"], q_gradient_norm_last=trace[-1]["q_gradient_norm"],
        dm_train_initial=dm["initial_train"], dm_train_final=dm["final_train"],
        dm_holdout_initial=dm["initial_holdout_bank"], dm_holdout_final=dm["final_holdout_bank"])


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT/"results/unified_q_constructor_attribution_20261008_v2")
    p.add_argument("--source",type=Path,default=ROOT/"results/unified_q_attribution_20261008")
    args=p.parse_args()
    cells,incomplete=load(args.root)
    evaluated,fixed_pairs=0,0
    records, diagnostics, replay=[], [], []
    for c in cells:
        proto=c["protocol"]
        name=f"{proto['dataset']}_o{proto['order_seed']}"
        output,source=args.root/name,args.source/name
        if hashlib.sha256((source/"results.json").read_bytes()).hexdigest() != proto["source_cell_sha256"]:
            raise AssertionError("source cell content changed")
        source_result=json.loads((source/"results.json").read_text())
        common=None
        commons={}
        full_blocks={}
        for stage in range(5):
            block,_=R.A.load_block(source,"cat_cctp",stage)
            common=R.A.append_block(common,block)
            commons[stage]=common
            full_blocks[stage],full_audit=R.A.load_block(output,R.C.MAIN,stage)
            old_full,old_audit=R.A.load_block(source,R.C.MAIN,stage)
            old_score=next(r["score"] for r in source_result["rows"] if
                r["arm"]==R.C.MAIN and r["stage"]==stage and r["student_family"]=="prompt")
            new_score=next(r["score"] for r in c["rows"] if
                r["arm"]==R.C.MAIN and r["stage"]==stage and r["readout"]=="native")
            prototypes=full_audit["q_cost"]["prototype_source_ids"]
            old_prototypes=old_audit["q_cost"]["prototype_source_ids"]
            replay.append(dict(cell=name,stage=stage,rows=len(prototypes),
                prototype_rows_changed=sum(a!=b for a,b in zip(prototypes,old_prototypes)),
                **full_audit["unchanged_main_replay"],
                final_f1_delta=new_score["final_f1"]-old_score["final_f1"],
                accuracy_delta=new_score["accuracy"]-old_score["accuracy"]))
        for arm in proto["arms"]:
            parent=None
            for stage in range(5):
                block,audit=R.A.load_block(output,arm,stage)
                if json.loads(json.dumps(audit["config"])) != proto["control_configs"][arm]:
                    raise AssertionError("archived intervention config differs from protocol")
                if audit["historical_raw_reads"] or audit["old_x_relative_change_max"]:
                    raise AssertionError("access/prefix audit changed")
                payload=audit["payload_validation"]
                if (not payload["current_delta_text_sources"] or not payload["same_class_text_sources"]
                        or payload["text_reencode_max_error"] > 1e-4):
                    raise AssertionError("recorded text provenance/re-encoding check failed")
                memory=R.U.append_memory(parent,block)
                checks=R.V.validate_memory(memory,parent,proto["budgets"][stage])
                if not all(checks.values()) or memory["edge_index"].numel():
                    raise AssertionError("reconstructed memory invalid")
                if not audit["assignment_checks"]["sinkhorn_feasible"]:
                    raise AssertionError("assignment infeasible")
                if audit["x_residual_ratio_max"] > proto["q_config"]["trust_radius"] + 1e-5:
                    raise AssertionError("feature trust region violated")
                if arm=="no_residual" and audit["x_residual_norm"] != 0:
                    raise AssertionError("residual not zero")
                if arm=="no_distribution" and audit["distribution_matching"]["enabled"]:
                    raise AssertionError("DM not disabled")
                if arm=="no_node_text" and (audit["config"]["text_weight"] or audit["config"]["pair_weight"]):
                    raise AssertionError("node-text intervention not applied")
                if not torch.equal(block["labels"],full_blocks[stage]["labels"]):
                    raise AssertionError("slot labels differ across interventions")
                diagnostics.append(dict(cell=name,arm=arm,stage=stage,
                    **diagnostic_record(block,audit,full_blocks[stage])))
                rows=[r for r in c["rows"] if r["arm"]==arm and r["stage"]==stage]
                for row in rows:
                    view=memory
                    if row["readout"]=="common_cat_payload":
                        view=R.fixed_payload(memory,commons[stage])
                        R.A.assert_fixed_content(memory,view)
                        fixed_pairs+=1
                    if R.V.memory_hash(view) != row["memory_hash"]:
                        raise AssertionError("evaluated memory hash mismatch")
                    if R.R.digest(view["x"]) != row["x_hash"] or R.R.digest(view["text_emb"]) != row["text_hash"]:
                        raise AssertionError("feature/payload hash mismatch")
                    evaluated+=1
                parent=memory
        records.append(dict(cell=name,arms=len(proto["arms"]),records=len(c["rows"]),passed=True))
    result=dict(complete_cells=len(cells),formal_matrix_complete=len(cells)==9 and not incomplete,
        evaluated_memories=evaluated,fixed_payload_pairs=fixed_pairs,incomplete=incomplete,cells=records)
    (args.root/"archive_audit.json").write_text(json.dumps(result,indent=2))
    (args.root/"construction_diagnostics.json").write_text(json.dumps(dict(
        note="Posthoc observed traces, not independent causal or extra-seed evidence. Archive audit checks recorded provenance; it does not re-encode texts or trace OS file access.",
        rows=diagnostics,same_seed_replay=replay),indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("cells","incomplete")}))


if __name__ == "__main__":
    main()
