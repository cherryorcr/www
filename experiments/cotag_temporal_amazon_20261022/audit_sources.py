"""Audit chronology provenance before allowing a temporal benchmark run."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil


SNAPSHOT = "c9fc08d286d11f164ea8cb61ded3179d23037f05"
SOURCE_ROOT = Path("/home/pc/.cache/huggingface/hub/datasets--Graph-COM--Text-Attributed-Graphs/snapshots") / SNAPSHOT
CONVERTED_ROOT = Path("/home/pc/research/TAGSAM/data/tagbench")
DATASETS = ("sportsfit", "bookhis")
CSV_IDS = {
    "sportsfit": "1XyI6TbW9sXU0rIDcQLi0UJTrlKWHuabI",
    "bookhis": "1gpBLHC6dbcpy9Ug_cvaEEzEegnRJ9dsQ",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def schema(obj):
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a mapping, received {type(obj).__name__}")
    return {
        str(key): {"type": type(value).__name__,
                   "shape": list(value.shape) if hasattr(value, "shape") else None}
        for key, value in obj.items()
    }


def candidate_fields(keys):
    keys = list(keys)
    return {
        "time_fields": [key for key in keys if any(
            token in key.lower() for token in ("date", "year", "timestamp", "time", "arrival"))],
        "external_id_fields": [key for key in keys if key.lower() in {
            "asin", "parent_asin", "isbn", "product_id", "item_id", "external_id"}],
    }


def inspect_local():
    import torch

    results = {}
    for name in DATASETS:
        processed_path = SOURCE_ROOT / name / "processed_data.pt"
        text_path = SOURCE_ROOT / name / "raw_texts.pt"
        converted_path = CONVERTED_ROOT / f"{name}.pt"
        processed = torch.load(processed_path, map_location="cpu", weights_only=False)
        texts = torch.load(text_path, map_location="cpu", weights_only=False)
        converted = torch.load(converted_path, map_location="cpu", weights_only=False)
        source_schema, converted_schema = schema(processed), schema(converted)
        ids = converted["original_node_ids"].long().flatten()
        row = {
            "source_schema": source_schema,
            "converted_schema": converted_schema,
            "source_candidate_fields": candidate_fields(source_schema),
            "converted_candidate_fields": candidate_fields(converted_schema),
            "nodes": len(texts),
            "raw_text_container": type(texts).__name__,
            "all_raw_text_entries_are_strings": all(isinstance(x, str) for x in texts),
            "converted_original_ids_are_sequential": torch.equal(ids, torch.arange(len(texts))),
            "text_lengths_equal": len(texts) == len(converted["text_list"]),
            "exact_text_matches_by_row": sum(a == b for a, b in zip(texts, converted["text_list"])),
            "files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}
                      for p in (processed_path, text_path, converted_path)],
            "verified_node_timestamp_pairs": 0,
            "temporal_run_status": "not_launched_no_verified_timestamp_mapping",
        }
        results[name] = row
        print(json.dumps({"dataset": name, "nodes": row["nodes"],
                          "candidate_fields": row["source_candidate_fields"]}), flush=True)
    free = shutil.disk_usage(CONVERTED_ROOT).free
    return {"snapshot": SNAPSHOT, "free_disk_bytes": free,
            "free_disk_floor_bytes": 8 * 2**30, "datasets": results}


def inspect_public():
    import requests
    from bs4 import BeautifulSoup

    session = requests.Session()
    results = {}
    commit_response = session.get("https://api.github.com/repos/CurryTang/TSGFM/commits/master", timeout=30)
    commit_response.raise_for_status()
    commit = commit_response.json()["sha"]
    for name in DATASETS:
        generator_url = f"https://raw.githubusercontent.com/CurryTang/TSGFM/{commit}/data/single_graph/{name}/gen_data.py"
        generator = session.get(generator_url, timeout=30)
        generator.raise_for_status()
        if CSV_IDS[name] not in generator.text:
            raise AssertionError("Expected CSV link is absent from the pinned public generator")
        with session.get("https://drive.google.com/uc", params={"export": "download", "id": CSV_IDS[name]},
                         stream=True, timeout=(15, 30)) as response:
            response.raise_for_status()
            prefix = next(response.iter_content(8192))
            fields = next(csv.reader(io.StringIO(prefix.decode("utf-8-sig"))))
            if "text" not in fields or "node_id" not in fields:
                raise AssertionError("Public endpoint did not return the expected CSV header")
            results[name] = {
                "generator_url": generator_url,
                "generator_sha256": hashlib.sha256(generator.content).hexdigest(),
                "csv_url": response.url,
                "csv_fields": fields,
                "csv_candidate_fields": candidate_fields(fields),
                "csv_reported_total_bytes": response.headers.get("content-length"),
                "inspected_prefix_bytes": len(prefix),
                "prefix_sha256": hashlib.sha256(prefix).hexdigest(),
                "scope": "CSV header and bounded prefix only; no claim to a full CSV join audit",
            }
    official_url = "https://nijianmo.github.io/amazon/index.html"
    page = session.get(official_url, timeout=30)
    page.raise_for_status()
    soup = BeautifulSoup(page.text, "html.parser")
    links = []
    for row in soup.find_all("tr"):
        label = row.get_text(" ", strip=True)
        if label.startswith(("Books ", "Sports and Outdoors ")):
            links.append({"label": label, "links": [
                {"label": a.get_text(" ", strip=True), "url": a["href"]}
                for a in row.find_all("a", href=True)]})
    return {
        "datasets": results, "upstream_generator_commit": commit,
        "official_review_source": {"url": official_url,
            "page_sha256": hashlib.sha256(page.content).hexdigest(),
            "documents_review_timestamp": "unixReviewTime" in page.text,
            "download_links": links},
        "mapping_status": "No verified ASIN-to-node and timestamp join obtained",
        "exclusions": ["sequential node IDs are not dates", "book-description years are not arrival times",
                       "first observed review time, if recovered, is not necessarily product launch time"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("local", "public"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = inspect_local() if args.mode == "local" else inspect_public()
    result.update(audited_at_utc=datetime.now(timezone.utc).isoformat(),
                  script_sha256=sha(__file__), mode=args.mode,
                  purpose="Chronology feasibility only; no downstream outcomes or paper edits")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(str(args.output))


if __name__ == "__main__":
    main()
