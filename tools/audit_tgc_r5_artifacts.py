"""Independent source/materialization audit without changing frozen runners."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.snapshot / "code/v2_llm_fusion"))
    import torch
    import run_tgc_r5_old_confirmation as O
    R, T, M = O.R, O.T, O.M
    torch.set_num_threads(2)
    protocol = read(args.root / "protocol.json")
    assert R.source_hashes() == protocol["source_hashes"]
    seal = read(args.root / "selection_seal.json")
    assert seal["protocol_sha256"] == digest(args.root / "protocol.json")
    for filename, expected in seal["decisions"].items():
        assert digest(args.root / filename) == expected
    joint = "joint" in protocol["arms"]
    records = []
    for dataset in protocol["datasets"]:
        data = None
        if joint:
            data = R.R.load_data(dataset, "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", "cuda:0",
                                Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919") /
                                f"{dataset}_r005" / "text_emb_pool_only.pt")
        for order in protocol["orders"]:
            folder = args.root / dataset / order
            manifest = read(folder / "manifest.json")
            all_ids = [set(v) for v in manifest["stage_source_ids"]]
            assert all(not all_ids[i] & all_ids[j] for i in range(3) for j in range(i))
            for stage in (1, 2):
                ids = torch.tensor(manifest["stage_source_ids"][stage])
                old_n = manifest["budgets"][stage-1]
                if not joint:
                    ctx = read(folder / f"s{stage}_support_diagnostics.json")
                    assert ctx["current_delta_ids"] == ids.tolist()
                    base = O.load(folder / f"c0_s{stage}_memory.pt")
                    rel = O.load(folder / f"relation_old_s{stage}_memory.pt")
                    rand = O.load(folder / f"random_old_s{stage}_memory.pt")
                    opt = read(folder / f"relation_old_s{stage}_optimization.json")
                    assert set(opt["rows"]) <= set(range(old_n))
                    assert set(opt["rows"]).isdisjoint(ctx["protected"])
                    assert len(opt["rows"]) <= max(1, int(.1 * old_n))
                    assert opt["input_memory_sha256"] == T.memory_hash(base)
                    norms = (rel["x"] - base["x"]).norm(dim=1)
                    random_norms = (rand["x"] - base["x"]).norm(dim=1)
                    torch.testing.assert_close(norms, random_norms, atol=2e-6, rtol=2e-5)
                    records.append({"dataset": dataset, "order": order, "stage": stage,
                                    "same_rows_random_passed": True,
                                    "max_random_norm_error": float((norms - random_norms).abs().max())})
                    continue
                assert data["pool_mask"][ids].all()
                for prefix in ("shared", "recursive", "maintenance", "capacity"):
                    audit = read(folder / f"{prefix}_s{stage}_construction.json")
                    assert audit["current_delta_ids"] == ids.tolist()
                    p = O.load(folder / f"{prefix}_s{stage}_assignment.pt")
                    assert R.C.tensor_hash({"p": p}) == audit["assignment"]["assignment_sha256"]
                    classes = torch.tensor(audit["assignment"]["row_classes"])
                    assert p.shape == (len(ids), manifest["budgets"][stage] - old_n)
                    assert torch.isfinite(p).all() and (p >= 0).all()
                    assert torch.count_nonzero(p[data["y"][ids, None] != classes[None]]) == 0
                    torch.testing.assert_close(p.sum(1), torch.ones(len(p)), atol=2e-6, rtol=1e-5)
                    arm = "joint" if prefix == "shared" else prefix
                    memory = O.load(folder / f"{arm}_s{stage}_memory.pt")
                    coefficients = p.double() / p.double().sum(0)
                    expected_x = coefficients.T @ data["x"][ids].double()
                    error = float((expected_x - memory["x"][old_n:].double()).abs().max())
                    torch.testing.assert_close(expected_x, memory["x"][old_n:].double(), atol=2e-5, rtol=2e-5)
                    torch.testing.assert_close(memory["mass"][old_n:], p.sum(0), atol=1e-5, rtol=1e-5)
                    source_groups = audit["materialization"]["selected_delta_local_text_ids"]
                    for j, local_ids in enumerate(source_groups):
                        assert len(local_ids) <= protocol["assignment"]["text_sources"]
                        assert len(local_ids) == len(set(local_ids))
                        assert all(0 <= i < len(ids) and p[i, j] > 0 for i in local_ids)
                        chars = protocol["assignment"]["text_chars"]
                        per = max(1, chars // len(local_ids))
                        expected = " ".join(str(data["texts"][int(ids[i])])[:per] for i in local_ids)[:chars]
                        assert expected == memory["texts"][old_n + j]
                    embedded = data["encoder"].encode(memory["texts"][old_n:]).cpu()
                    torch.testing.assert_close(embedded, memory["text_emb"][old_n:], rtol=2e-5, atol=2e-6)
                    if prefix == "maintenance":
                        before = O.load(folder / f"maintenance_pre_revision_s{stage}_memory.pt")
                        opt = read(folder / f"maintenance_s{stage}_optimization.json")
                        assert opt["input_memory_sha256"] == T.memory_hash(before)
                        M.validate_memory(before, memory, opt["rows"], protocol["teacher_and_maintenance"]["radius"])
                    records.append({"dataset": dataset, "order": order, "stage": stage, "branch": prefix,
                                    "source_weighted_x_max_error": error,
                                    "max_single_source_weight": float(coefficients.max()),
                                    "min_effective_sources": float((1 / coefficients.square().sum(0)).min()),
                                    "text_reencoded_and_verified": True})
    output = {"status": "passed", "protocol_sha256": digest(args.root / "protocol.json"),
              "audit_source_sha256": digest(Path(__file__)), "python": sys.version,
              "torch": torch.__version__, "cuda": torch.version.cuda, "records": records,
              "scope": "snapshot integrity, current-source IDs, weighted synthetic X, text provenance/re-encoding, matched displacement; score reload in cell audit.json"}
    (args.root / "artifact_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "records": len(records), "root": str(args.root)}))


if __name__ == "__main__":
    main()
