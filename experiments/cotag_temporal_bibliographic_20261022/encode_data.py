"""Frozen pointwise MiniLM features, with no graph-wide propagation or training."""
import argparse
import json
from pathlib import Path

from runtime import C, R, torch, source_data


def main(args):
    folder = source_data(args.dataset)
    manifest = C.read(folder / "manifest.json")
    if (folder / "encoding.json").exists():
        audit = C.read(folder / "encoding.json")
        if audit["dataset_manifest_sha256"] != C.sha(folder / "manifest.json") or audit["encoded_data_sha256"] != C.sha(folder / "encoded.pt"):
            raise AssertionError("Existing feature provenance differs")
        C.emit(event="skip_encoded", dataset=args.dataset)
        return
    if (folder / "encoded.pt").exists():
        raise FileExistsError("Unverified encoded data already exists; do not overwrite")
    for name in ("records", "edges"):
        filename = "records.jsonl" if name == "records" else "edges.json"
        if C.sha(folder / filename) != manifest[f"{name}_sha256"]:
            raise AssertionError("Prepared source data changed")
    with (folder / "records.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    names = manifest["classes"]
    texts = [r["text"] for r in rows]
    encoder = R.L.U.FrozenMiniLM(C.MINILM, args.device)
    x = encoder.encode(texts)
    y = torch.tensor([names.index(r["label"]) for r in rows], dtype=torch.long)
    edges = torch.tensor(C.read(folder / "edges.json"), dtype=torch.long).reshape(-1, 2).T.contiguous()
    payload = dict(x=x.cpu(), y=y, texts=texts, edge_index=edges, names=names,
        original_ids=torch.tensor([r["split_hash_id"] for r in rows], dtype=torch.long),
        source_ids=[r["id"] for r in rows], node_year=torch.tensor([r["year"] for r in rows]),
        arrival=torch.tensor([r["stage"] for r in rows]), prompts=[f"a research paper about {c}" for c in names])
    if tuple(x.shape) != (len(rows), 384) or not bool(torch.isfinite(x).all()):
        raise AssertionError("Invalid MiniLM features")
    C.disk_check()
    torch.save(payload, folder / "encoded.pt")
    weights = {str(p.relative_to(C.MINILM)): C.sha(p) for p in Path(C.MINILM).glob("*")
               if p.is_file() and p.suffix in {".bin", ".safetensors", ".json", ".txt"}}
    audit = dict(dataset=args.dataset, dataset_manifest_sha256=C.sha(folder / "manifest.json"),
        encoded_data_sha256=C.sha(folder / "encoded.pt"), nodes=len(rows), dimension=384,
        encoder_path=C.MINILM, encoder_hashes=weights, encoder_code_sha256=C.sha(R.L.U.FrozenMiniLM.encode.__code__.co_filename),
        encoding_code_sha256=C.sha(__file__), pointwise_frozen=True, graph_propagation=False,
        max_tokens=256, training_on_future_or_heldout_nodes=False,
        heldout_features="pointwise CPU features for inference only; masked out of construction text cache")
    C.save(folder / "encoding.json", audit)
    C.emit(event="encoded", dataset=args.dataset, nodes=len(rows), classes=names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("acm", "dblp"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    main(args)
