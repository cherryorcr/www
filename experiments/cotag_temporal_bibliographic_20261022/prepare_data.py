"""Prepare chronological citation subsets without using downstream outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
import shutil
import unicodedata
import zipfile

SOURCES = {
    "acm": ("https://lfs.aminer.cn/lab-datasets/citation/acm.v9.zip", 686542584),
    "dblp": ("https://lfs.aminer.cn/lab-datasets/citation/dblp.v10.zip", 1848832450),
}
RULES = {
    "acm": {
        "databases": ["sigmod", "vldb", "very large data bases", "transactions on database systems"],
        "data mining": ["sigkdd", "knowledge discovery and data mining", "transactions on knowledge discovery from data"],
        "computer networks": ["sigcomm", "mobicom", "mobile computing and networking", "computer communication review"],
    },
    "dblp": {
        "artificial intelligence": ["aaai", "ijcai", "national conference on artificial intelligence", "international joint conference on artificial intelligence", "journal of artificial intelligence research"],
        "computer vision": ["cvpr", "iccv", "eccv", "computer vision and pattern recognition", "international conference on computer vision", "european conference on computer vision", "international journal of computer vision"],
        "natural language processing": ["acl", "emnlp", "naacl", "association for computational linguistics", "empirical methods in natural language processing"],
        "machine learning": ["icml", "nips", "neural information processing systems", "international conference on machine learning", "journal of machine learning research"],
    },
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def canonical(value):
    value = unicodedata.normalize("NFKC", html.unescape(str(value))).casefold()
    return " ".join(re.findall(r"\w+", value))


def stable_id(dataset, source_id):
    return int(hashlib.sha256(f"{dataset}:{source_id}".encode()).hexdigest()[:15], 16)


def labels_for(dataset, venue):
    text = " " + canonical(venue) + " "
    return [name for name, aliases in RULES[dataset].items()
            if any(" " + canonical(alias) + " " in text for alias in aliases)]


def acm_records(stream):
    row = None
    for line in stream:
        line = line.rstrip("\r\n")
        if line.startswith("#*"):
            if row is not None:
                yield row
            row = {"title": line[2:], "references": []}
        elif row is not None:
            if line.startswith("#index"):
                row["id"] = line[6:]
            elif line.startswith("#t"):
                row["year"] = line[2:]
            elif line.startswith("#c"):
                row["venue"] = line[2:]
            elif line.startswith("#!"):
                row["abstract"] = line[2:]
            elif line.startswith("#%"):
                row["references"].append(line[2:])
    if row is not None:
        yield row


def iter_records(path, dataset):
    with zipfile.ZipFile(path) as archive:
        members = [v for v in archive.infolist() if not v.is_dir()]
        for member in members:
            if not member.filename.endswith(".txt" if dataset == "acm" else ".json"):
                raise ValueError(f"Unexpected archive member: {member.filename}")
            with archive.open(member) as raw:
                stream = io.TextIOWrapper(raw, encoding="utf-8", errors="strict")
                if dataset == "acm":
                    yield from acm_records(stream)
                else:
                    for line in stream:
                        if line.strip():
                            yield json.loads(line)


def download(dataset, folder):
    import requests

    url, size = SOURCES[dataset]
    path = folder / Path(url).name
    folder.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != size:
            raise ValueError("Existing archive size mismatch; preserve it for inspection")
        return path
    partial = path.with_suffix(path.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if shutil.disk_usage(folder).free < size - offset + 8 * 2**30:
        raise RuntimeError("Download would cross the 8 GiB disk floor")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
            raise ValueError("Server did not honor the exact resume offset")
        count = offset
        checkpoint = count // (128 * 2**20)
        with partial.open("ab" if offset else "xb") as f:
            for block in response.iter_content(2**20):
                count += len(block)
                if count > size:
                    raise ValueError("Download exceeds registered archive length")
                f.write(block)
                if count // (128 * 2**20) > checkpoint:
                    checkpoint = count // (128 * 2**20)
                    print(json.dumps({"event": "download", "dataset": dataset, "bytes": count, "total": size}), flush=True)
    if count != size:
        raise ValueError("Incomplete archive; preserve .part for resume")
    partial.replace(path)
    return path


def select_records(path, dataset):
    counts, venues, years, grouped = Counter(), Counter(), Counter(), {}
    generic = ("proceedings of", "preface", "author index", "table of contents", "front matter", "back matter", "editorial")
    for row in iter_records(path, dataset):
        counts["source_records"] += 1
        venue = str(row.get("venue", ""))
        venues[venue] += 1
        try:
            year = int(row.get("year", ""))
        except (TypeError, ValueError):
            counts["invalid_year"] += 1
            continue
        years[year] += 1
        if not 1990 <= year <= 2015:
            counts["outside_year_range"] += 1
            continue
        matches = labels_for(dataset, venue)
        if len(matches) != 1:
            counts["no_venue_match" if not matches else "ambiguous_venue"] += 1
            continue
        title = html.unescape(str(row.get("title", ""))).strip()
        key = canonical(title)
        if len(key) < 12 or key.startswith(generic) or not row.get("id"):
            counts["invalid_or_generic_title_or_id"] += 1
            continue
        abstract = html.unescape(str(row.get("abstract", ""))).strip()
        rec = dict(id=str(row["id"]), title=title, abstract=abstract, year=year,
                   label=matches[0], venue=venue, references=list(map(str, row.get("references", []))))
        grouped.setdefault(key, []).append(rec)
    chosen, aliases = [], {}
    for key, rows in grouped.items():
        if len({(r["year"], r["label"]) for r in rows}) != 1:
            counts["ambiguous_title_records_removed"] += len(rows)
            continue
        rows.sort(key=lambda r: r["id"])
        first = dict(rows[0])
        first["references"] = sorted({ref for r in rows for ref in r["references"]})
        chosen.append(first)
        for row in rows:
            aliases[row["id"]] = first["id"]
        counts["equivalent_duplicate_records_merged"] += len(rows) - 1
    if len({r["id"] for r in chosen}) != len(chosen):
        raise AssertionError("Same source ID identifies multiple different titles")
    profile = {"filter_counts": dict(counts), "source_year_counts": dict(sorted(years.items())),
               "top_source_venues": venues.most_common(100), "eligible_unique_titles": len(chosen)}
    return chosen, aliases, profile


def temporal_bounds(rows):
    years = sorted(r["year"] for r in rows if stable_id(r["dataset"], r["id"]) % 10 not in (1, 2, 3))
    if not years:
        raise ValueError("Empty training pool")
    bounds = [years[math.ceil(len(years) * p) - 1] for p in (.4, .55, .7, .85, 1.)]
    if len(set(bounds)) != 5:
        raise ValueError(f"Five distinct training-year boundaries unavailable: {bounds}")
    return bounds


def finalize(rows, aliases, dataset, excluded_titles=()):
    excluded_titles = set(excluded_titles)
    overlap = sum(canonical(r["title"]) in excluded_titles for r in rows)
    rows = [dict(r, dataset=dataset) for r in rows if canonical(r["title"]) not in excluded_titles]
    before_cap = len(rows)
    rows.sort(key=lambda r: hashlib.sha256(f"{dataset}:{r['id']}".encode()).hexdigest())
    rows = rows[:60000]
    rows.sort(key=lambda r: r["id"])
    bounds = temporal_bounds(rows)
    lookup = {r["id"]: i for i, r in enumerate(rows)}
    edges, counts = set(), Counter()
    for i, row in enumerate(rows):
        for ref in row.pop("references"):
            target = lookup.get(aliases.get(ref, ref))
            if target is None:
                counts["references_outside_retained_nodes"] += 1
            elif target == i:
                counts["self_references_removed"] += 1
            elif rows[target]["year"] > row["year"]:
                counts["forward_year_references_removed"] += 1
            else:
                edges.add((i, target))
    for row in rows:
        row["split_hash_id"] = stable_id(dataset, row["id"])
        row["stage"] = next(i for i, b in enumerate(bounds) if row["year"] <= b)
        row["text"] = "Title: " + row["title"] + (". Abstract: " + row["abstract"] if row["abstract"] else "")
    if len({r["split_hash_id"] for r in rows}) != len(rows):
        raise AssertionError("Hash collision in row split identifiers")
    stages = []
    for stage in range(5):
        cur = [r for r in rows if r["stage"] == stage]
        train = [r for r in cur if r["split_hash_id"] % 10 not in (1, 2, 3)]
        if not train:
            raise ValueError("Empty training time bin")
        stages.append(dict(stage=stage, year_end=bounds[stage], nodes=len(cur), train_nodes=len(train),
                           train_class_counts=dict(Counter(r["label"] for r in train)),
                           split_counts=dict(Counter(str(r["split_hash_id"] % 10) for r in cur))))
    undirected = sorted(edges | {(b, a) for a, b in edges})
    report = dict(dataset=dataset, nodes=len(rows), classes=list(RULES[dataset]),
                  class_counts=dict(Counter(r["label"] for r in rows)), boundary_years=bounds,
                  deduplicated_across_datasets=overlap, nodes_before_cap=before_cap,
                  abstract_present=sum(bool(r["abstract"]) for r in rows),
                  directed_citations=len(edges), symmetric_edges=len(undirected), edge_filter_counts=dict(counts),
                  stages=stages, venue_rules=RULES[dataset])
    return rows, undirected, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    acm_titles = set()
    for dataset in SOURCES:
        output = args.root / dataset
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["preparation_code_sha256"] != sha(__file__):
                raise ValueError("Preparation changed; use a new output root")
            if dataset == "acm":
                with (output / "records.jsonl").open(encoding="utf-8") as f:
                    acm_titles = {canonical(json.loads(line)["title"]) for line in f}
            print(json.dumps({"event": "skip_prepared", "dataset": dataset}), flush=True)
            continue
        archive = download(dataset, args.root / "raw")
        chosen, aliases, profile = select_records(archive, dataset)
        records, edges, report = finalize(chosen, aliases, dataset, acm_titles)
        if dataset == "acm":
            acm_titles = {canonical(r["title"]) for r in records}
        output.mkdir(parents=True, exist_ok=True)
        with (output / "records.jsonl").open("x", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        write_new(output / "edges.json", edges)
        write_new(output / "profile.json", profile)
        report.update(source_url=SOURCES[dataset][0], archive_bytes=archive.stat().st_size,
                      archive_sha256=sha(archive), preparation_code_sha256=sha(__file__),
                      created_at_utc=datetime.now(timezone.utc).isoformat(),
                      records_sha256=sha(output / "records.jsonl"), edges_sha256=sha(output / "edges.json"))
        write_new(manifest_path, report)
        print(json.dumps({"event": "prepared", **report}), flush=True)


if __name__ == "__main__":
    main()
