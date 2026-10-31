"""Read-only handoff audit; export small, traceable replay inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/pc/research/TAG_condensation/results"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def verify_checksums(root):
    checked, failures = 0, []
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines():
        expected, relative = line.split(None, 1)
        path = (root / relative.lstrip(" *").replace("\\", "/")).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("checksum path escapes the supplied package")
        actual = sha(path) if path.exists() else None
        checked += 1
        if actual != expected:
            failures.append(dict(path=str(path), expected=expected, actual=actual))
    return dict(checked=checked, failures=failures)


def make_manifest(root, dataset):
    formal = root / "server_snapshot_20261008/formal_150155"
    path = formal / f"baselines/{dataset}_results.json"
    core = read(path)
    core_hash = sha(path)
    protocol = core["protocol"]
    rows = []
    for r in core["rows"]:
        rows.append(dict(arm=r["arm"], stage=r["stage"], memory_rows=r["memory_rows"],
            memory_path=r["source_memory_file"], memory_hash=r["memory_sha256"],
            hash_kind="large_baseline_memory", source_json=str(path), source_sha256=core_hash,
            legacy_score=r["score"], construction_seconds=r.get("construction_seconds"),
            construction_seed=protocol.get("construction_seed"),
            source_ids=r.get("construction_audit", {}).get("selected_source_ids"),
            row_classes=r.get("construction_audit", {}).get("row_classes"),
            text_source_ids=r.get("construction_audit", {}).get("text_source_ids"),
            delta_source_ids=r.get("construction_audit", {}).get("source_ids"),
            text_mode=r.get("construction_audit", {}).get("text_mode"),
            text_adapter=r.get("construction_audit", {}).get("text_adapter"),
            selection_config={key: r.get("construction_audit", {}).get(key)
                for key in ("method", "selection_seed", "approximate", "candidate_cap",
                            "selection_batch", "projection_dim")},
            audit=r.get("construction_audit", {})))
    puma_path = formal / f"puma/{dataset}_results.json"
    puma = read(puma_path)
    puma_hash = sha(puma_path)
    for r in puma["records"]:
        rows.append(dict(arm="puma_full", stage=r["stage"], memory_rows=r["memory_rows"],
            memory_path=f"{REMOTE}/puma_formal_20261008_{dataset}/{dataset}_base4/puma_full_s{r['stage']}_memory.pt",
            memory_hash=r["memory_sha256"], hash_kind="large_baseline_memory",
            source_json=str(puma_path), source_sha256=puma_hash, legacy_score=r["score"],
            replay_status="construction hash only; original runner did not save memory; requires rebuild",
            construction_seconds=r.get("construction_seconds"), audit=r.get("construction_audit", {})))
    date = "20261007" if dataset in ("wikics", "bookhis") else "20261008"
    tagsam_path = root / f"server_snapshot_{date}/tagsam_formal/{dataset}/results.json"
    tag = read(tagsam_path)
    tag_matches = (tag["manifest"]["stage_classes"] == protocol["groups"]
                   and tag["manifest"]["budgets"] == protocol["budgets"])
    tag_hash = sha(tagsam_path)
    for r in tag["results"]:
        if r["arm"] != "tagsam_selfloop" or not tag_matches:
            continue
        audit = next(v for v in tag["audits"] if v["stage"] == r["stage"] and v["arm"] == r["arm"])
        rows.append(dict(arm=r["arm"], stage=r["stage"], memory_rows=r["rows"],
            memory_path=f"{REMOTE}/tagsam_large_formal_{date}_{dataset}/{r['arm']}_s{r['stage']}_memory.pt",
            memory_hash=audit["tensor_sha256"], hash_kind="small_comparison_tensor",
            source_json=str(tagsam_path), source_sha256=tag_hash, legacy_score=r,
            construction_seconds=audit.get("construction_seconds"), audit=audit))
    return dict(dataset=dataset, groups=protocol["groups"], budgets=protocol["budgets"],
        report_dev_mod=3, construction_seed=271006, source_protocol=protocol,
        expected_stage_ids=tag["manifest"]["stage_source_ids_sha256"] if tag_matches else None,
        expected_report_mask=tag["manifest"]["report_dev_mask_sha256"],
        expected_report_mask_kind="small_comparison_tensor_mask",
        text_cache_sha256=tag["manifest"]["text_cache_sha256"],
        student_seeds=[150,151,152], student_steps=2048,
        students=dict(legacy=dict(batch=16, loss="row-wise bidirectional InfoNCE"),
                      prompt=dict(batch=24, loss="multi-positive + prompt CE + margin")),
        payload_reencoding=False, tagsam_same_protocol=tag_matches,
        tagsam_protocol=dict(groups=tag["manifest"]["stage_classes"], budgets=tag["manifest"]["budgets"]), rows=rows)


def verify_replay_deltas(root, replays):
    records = []
    for dataset in ("wikics", "sportsfit", "bookhis"):
        source = root / f"server_snapshot_20261008/formal_150155/baselines/{dataset}_results.json"
        core = read(source)
        path = replays / f"handoff_{dataset}_verified/results.json"
        if not path.exists():
            path = replays / f"handoff_{dataset}/results.json"
        protocol = read(path)["protocol"]
        for key in ("groups", "budgets"):
            if core["protocol"][key] != protocol[key]:
                raise AssertionError(f"{dataset}: source {key} differ")
        for row in core["rows"]:
            audit = row["construction_audit"]
            ids = audit.get("source_ids", audit.get("candidate_delta_ids"))
            if not ids:
                raise AssertionError("source audit lacks the constructor candidate delta")
            digest = hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest()
            if digest != protocol["stage_id_hashes"][row["stage"]]:
                raise AssertionError(f"{dataset}/{row['arm']}/s{row['stage']}: candidate delta differs")
            if audit.get("historical_raw_reads") != 0 or audit.get("report_test_labels_used_for_construction") is not False:
                raise AssertionError("source audit does not declare legal construction access")
            records.append(dict(dataset=dataset, arm=row["arm"], stage=row["stage"],
                candidate_rows=len(ids), candidate_ids_sha256=digest, source_json_sha256=sha(source),
                expected_same_as_main=True, declaration_historical_raw_reads=0))
    return dict(status="complete", checked=len(records), rows=records,
        scope="Full candidate ID arrays from the original core-baseline construction audits equal the main runner's actual current-delta hashes. Access declarations and interface checks are not OS traces.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--handoff", type=Path, default=Path("D:/Aprojects/LARGE_GRAPH_BASELINE_HANDOFF_20261008/large_graph_baseline_20261008"))
    p.add_argument("--output", type=Path, default=ROOT / "results/shared_q_baseline_20261008/handoff_audit")
    p.add_argument("--verify-replays", type=Path)
    args = p.parse_args()
    if args.verify_replays is not None:
        audit = verify_replay_deltas(args.handoff, args.verify_replays)
        write(args.verify_replays / "source_delta_audit.json", audit)
        print(json.dumps(dict(status=audit["status"], checked=audit["checked"])))
        return
    checksums = verify_checksums(args.handoff)
    if checksums["failures"]:
        write(args.output / "checksum_failures.json", checksums)
        raise ValueError("handoff checksum mismatch")
    comparisons = []
    for dataset in ("wikics", "sportsfit", "bookhis", "arxiv"):
        manifest = make_manifest(args.handoff, dataset)
        # Large construction audits stay in the source package, not duplicated
        # for every replay. Labels are reconstructed and checked on the server.
        for row in manifest["rows"]:
            row["audit_fields"] = sorted(row.pop("audit").keys())
        write(args.output / f"{dataset}_manifest.json", manifest)
        current = ROOT / f"results/unified_q_constructor_attribution_20261008_v2/{dataset}_o9101/results.json"
        current_protocol = read(current)["protocol"] if current.exists() else None
        comparisons.append(dict(dataset=dataset, handoff_groups=manifest["groups"], handoff_budgets=manifest["budgets"],
            current_groups=current_protocol["groups"] if current_protocol else None,
            current_budgets=current_protocol["budgets"] if current_protocol else None,
            same_groups=current_protocol is not None and current_protocol["groups"] == manifest["groups"],
            same_budgets=current_protocol is not None and current_protocol["budgets"] == manifest["budgets"],
            direct_score_reuse=False, required="paired handoff-order construction + per-Student replay",
            available_memory_records=len(manifest["rows"])))
    summary = dict(checksums=checksums, comparisons=comparisons,
        puma_student_label="misleading: calls legacy L.student_score, not current train_prompt_student",
        baseline_fidelity="CaT/GECC/PUMA/TAGSAM are declared adaptations, not official full pipelines",
        raw_reuse="read archived memories only after content-hash and split checks",
        stats="150-152 subset paired with new runs; 153-155 retained in original package, not silently pooled")
    write(args.output / "audit.json", summary)
    print(json.dumps(dict(checked=checksums["checked"], datasets=comparisons), indent=2))


if __name__ == "__main__":
    main()
