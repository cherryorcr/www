"""Check final stored memories, operators and protocol artifacts for this batch."""
import argparse
import json
from pathlib import Path
import torch
import unified_q_multimodal as U
import run_unified_q_revision_20261007 as V


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--require-complete",action="store_true")
    args=p.parse_args()
    records,pending,warnings=[],[],[]
    for path in sorted(args.root.glob("*/results.json")):
        result=json.loads(path.read_text())
        protocol=json.loads(path.with_name("protocol.json").read_text())
        if result["protocol"] != protocol:
            raise AssertionError(f"protocol artifact mismatch {path}")
        if result["status"] != "complete":
            pending.append(str(path)); continue
        diagnostic=path.with_name("diagnostic_memory.pt")
        bundle=torch.load(diagnostic,map_location="cpu",weights_only=False) if diagnostic.exists() else None
        for arm in protocol["arms"]:
            rows=sorted([r for r in result["rows"] if r["arm"]==arm],key=lambda r:r["stage"])
            if [r["stage"] for r in rows] != list(range(len(protocol["groups"]))):
                raise AssertionError("stage coverage")
            checkpoint=diagnostic if bundle else path.with_name(f"latest_{arm}.pt")
            saved=None if bundle else torch.load(checkpoint,map_location="cpu",weights_only=False)
            memory=bundle["memories"][arm] if bundle else saved["memory"]
            if not bundle and saved["stage"]!=rows[-1]["stage"]:
                raise AssertionError("checkpoint stage")
            if V.memory_hash(memory)!=rows[-1]["memory_hash"]:
                raise AssertionError(f"memory hash {checkpoint} {arm}")
            if bundle:
                reference=bundle["memories"]["unchanged"]
                if any(not torch.equal(memory[key],reference[key]) for key in ("x","labels","mass")):
                    raise AssertionError("fixed-content diagnostic changed X, labels, or mass")
                if arm.startswith("topology") and (
                    not torch.equal(memory["text_emb"],reference["text_emb"])
                    or memory["texts"]!=reference["texts"]):
                    raise AssertionError("fixed-content topology changed text")
                if V.hash_tensor(memory["x"])!=rows[-1]["content_hash"] or V.hash_tensor(memory["text_emb"])!=rows[-1]["text_hash"]:
                    raise AssertionError("fixed-content artifact hashes differ")
            V.validate_memory(memory,None,protocol["budgets"][-1])
            if any(len(t)>320 for t in memory["texts"]):
                raise AssertionError("text budget")
            operator_ok=True
            operator_symmetry_max_abs=None
            if bundle and arm=="topology_weighted":
                adj=U._normalize_dense(torch.block_diag(*bundle["weighted_blocks"]))
                operator_symmetry_max_abs=float((adj-adj.T).abs().max())
                operator_ok=(V.hash_tensor(adj)==rows[-1]["operator_hash"] and
                             bool(torch.isfinite(adj).all()) and bool((adj>=0).all()) and
                             torch.allclose(adj,adj.T,atol=1e-5,rtol=1e-6))
                if operator_symmetry_max_abs>1e-6:
                    warnings.append(dict(source=str(path),arm=arm,
                        warning="float32 Q-transpose A Q is not exactly symmetric; measured operator retained, not altered",
                        symmetry_max_abs=operator_symmetry_max_abs,tolerance=1e-5))
            if not operator_ok:
                raise AssertionError("weighted operator artifact mismatch")
            records.append(dict(dataset=protocol["dataset"],arm=arm,source=str(path),
                memory_rows=len(memory["x"]),checkpoint_bytes=checkpoint.stat().st_size,
                shared_container=bool(bundle),hash_ok=True,operator_ok=True,
                operator_symmetry_max_abs=operator_symmetry_max_abs))
    if args.require_complete:
        for phase in ("confirmation","baselines","diagnostics"):
            status=args.root/f"{phase}_status.json"
            if not status.exists() or json.loads(status.read_text())["status"]!="complete":
                pending.append(str(status))
        expected = 4 + (3 * 3 * 3 - 3) * 3 + 3 * 4 + 3 * 7
        if len(records)!=expected:
            pending.append(f"expected {expected} new method cells, got {len(records)}")
    audit=dict(status=("partial" if pending else "passed_with_numerical_warning" if warnings else "passed"),
               records=records,pending=pending,warnings=warnings,
               access_audit="constructor input/interface audit; no OS-level read tracing",
               disk_bytes=sum(p.stat().st_size for p in args.root.rglob("*.pt")))
    (args.root/"artifact_audit.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(dict(status=audit["status"],cells=len(records),pending=pending,warnings=warnings,disk_bytes=audit["disk_bytes"])))
    if args.require_complete and pending:
        raise SystemExit(1)


if __name__=="__main__":
    main()
