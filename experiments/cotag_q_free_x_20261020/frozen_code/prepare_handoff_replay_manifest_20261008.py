"""Recover missing real-coreset provenance by exact current-delta contents.

Write a new manifest; never alter the supplied package or a completed protocol.
Ambiguous labels are rejected instead of inferred from a downstream model.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

import run_handoff_student_transfer_20261008 as H


class AmbiguousSourceLabelsError(AssertionError):
    pass


def content_keys(x, text_emb, texts):
    x = x.detach().cpu().contiguous().numpy()
    text_emb = text_emb.detach().cpu().contiguous().numpy()
    if len(x) != len(text_emb) or len(x) != len(texts):
        raise AssertionError("multimodal content length mismatch")
    for feature, embedding, text in zip(x, text_emb, texts):
        h = hashlib.sha256()
        h.update(feature.tobytes())
        h.update(embedding.tobytes())
        h.update(text.encode("utf-8"))
        yield h.hexdigest()


def exact_sources(memory, delta, count):
    if count <= 0 or count > len(memory["x"]) or count > len(delta["x"]):
        raise AssertionError("invalid coreset block size")
    index = {}
    for pos, key in enumerate(content_keys(delta["x"], delta["text_emb"], delta["texts"])):
        index.setdefault(key, []).append(pos)
    selected, used, duplicate_contents = [], set(), 0
    keys = content_keys(memory["x"][-count:], memory["text_emb"][-count:], memory["texts"][-count:])
    for offset, key in enumerate(keys):
        candidates = index.get(key, [])
        if not candidates:
            raise AssertionError(f"coreset row {offset} has no exact current-delta multimodal source")
        if len({int(delta["y"][i]) for i in candidates}) != 1:
            raise AmbiguousSourceLabelsError("identical contents have ambiguous source labels")
        unused = [i for i in candidates if i not in used]
        if not unused:
            raise AssertionError("coreset has more copies than its current-delta sources")
        pos = unused[0]
        used.add(pos)
        selected.append(int(delta["ids"][pos]))
        duplicate_contents += int(len(candidates) > 1)
        if (not torch.equal(memory["x"][-count + offset], delta["x"][pos])
                or not torch.equal(memory["text_emb"][-count + offset], delta["text_emb"][pos])
                or memory["texts"][-count + offset] != delta["texts"][pos]):
            raise AssertionError("content hash collision or dtype mismatch")
    return selected, dict(rule="exact X, text embedding and string in current delta; no outcome labels",
        resolved_rows=count, duplicate_contents_same_label=duplicate_contents,
        duplicate_policy="distinct equivalent raw IDs; all matching IDs must share the same class",
        historical_raw_reads=0, report_test_labels_used=False)


def replay_sources(memory, delta, count, config, device):
    # Identical content can have different raw labels. Replay the declared
    # label-free selector, not a label choice among otherwise identical rows.
    if not config or config.get("approximate"):
        raise AssertionError("unsupported or missing selector provenance")
    with H.A.preserve_rng():
        block, audit = H.L.U.build_real_selection_block(delta, torch.arange(len(delta["x"])),
            count, config["method"], delta["stage"], device)
    if audit["selection_seed"] != config["selection_seed"]:
        raise AssertionError("selector replay seed differs")
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if not torch.equal(block[key], memory[key][-count:]):
            raise AssertionError(f"selector replay differs from archived {key}")
    if block["texts"] != memory["texts"][-count:]:
        raise AssertionError("selector replay differs from archived text strings")
    local = torch.tensor(audit["selected_source_ids"], dtype=torch.long)
    if len(local) != count or len(torch.unique(local)) != count:
        raise AssertionError("selector replay did not retain distinct source IDs")
    return delta["ids"][local].tolist(), dict(
        rule="label-free original selector replay; exact X/text/mass/depth/stage agreement",
        selection_config=config, selector_module_sha256=H.sha(H.L.U.__file__),
        identity_scope="reconstructed provenance, not originally saved source IDs",
        resolved_rows=count, historical_raw_reads=0, report_test_labels_used=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    original = json.loads(args.manifest.read_text())
    manifest = copy.deepcopy(original)
    data = H.L.load_large(manifest["dataset"], args.minilm, args.text_cache, args.device, manifest)
    if H.tensor_dict_hash({"mask": data["report_dev_mask"]}) != manifest["expected_report_mask"]:
        raise AssertionError("report mask differs")
    if H.sha(args.text_cache / "text_emb_pool_only.pt") != manifest["text_cache_sha256"]:
        raise AssertionError("text cache differs")
    ids = H.L.stage_ids(data)
    audits = []
    for stage, raw_ids in enumerate(ids):
        delta = H.L.delta_view(data, raw_ids, stage)
        count = manifest["budgets"][stage] - (manifest["budgets"][stage-1] if stage else 0)
        for row in manifest["rows"]:
            if row["stage"] != stage or row["arm"] not in ("random_real", "k_center", "medoid"):
                continue
            memory = torch.load(row["memory_path"], map_location="cpu", weights_only=False)
            if H.original_hash(memory, row["hash_kind"]) != row["memory_hash"]:
                raise AssertionError("source memory changed")
            if len(memory["x"]) != manifest["budgets"][stage]:
                raise AssertionError("source budget differs")
            if row.get("source_ids") is None:
                try:
                    row["source_ids"], audit = exact_sources(memory, delta, count)
                except AmbiguousSourceLabelsError:
                    row["source_ids"], audit = replay_sources(memory, delta, count,
                        row["selection_config"], args.device)
                row["source_id_recovery"] = audit
                audits.append(dict(arm=row["arm"], stage=stage, **audit))
    manifest["manifest_revision"] = dict(name="exact-content-source-recovery-v1",
        source_manifest_sha256=H.sha(args.manifest), resolver_sha256=H.sha(__file__),
        changed_memory=False, source_id_recovery=audits)
    if args.output.exists() and json.loads(args.output.read_text()) != manifest:
        raise ValueError("refuse to replace a different resolved manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    H.V.write_json(args.output, manifest)
    print(json.dumps(dict(status="complete", output=str(args.output), resolved_blocks=len(audits))), flush=True)


if __name__ == "__main__":
    main()
