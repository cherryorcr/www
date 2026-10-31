"""Prepare native Cora class-stream artifacts, with labels isolated from training.

Each zero-based stage file contains only its arriving class block. The runner
forms cumulative references by disjoint union. No cross-stage edges survive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import torch
from protocol_data import GRAPH_KEYS, subset_graph

STAGE_CLASSES = ((0, 1), (2, 3), (4, 5, 6))
SPLIT_NAMES = ("train", "checkpoint", "test")
ENCODING_VERSION = "native-tag-bert-cls-eval-512-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _partition(original_id: int, seed: int = 0) -> str:
    """ID-only SHA256 buckets; no labels and no class-specific rebalancing."""
    number = int.from_bytes(hashlib.sha256(f"cora:native-tag-v1:{seed}:{original_id}".encode()).digest(), "big")
    return "train" if number * 100 < 70 * 2**256 else ("checkpoint" if number * 100 < 85 * 2**256 else "test")


def build_stage_partitions(source: dict[str, Any], text_emb: torch.Tensor, split_seed: int = 0):
    """Pure preparation boundary, testable without BERT or downloads."""
    required = ("node_f", "edge_index", "text_list", "label_list", "all_labels", "labels_desc")
    missing = [key for key in required if key not in source]
    if missing:
        raise KeyError(f"source Cora asset missing fields: {missing}")
    n = len(source["text_list"])
    if source["node_f"].shape[0] != n or len(source["label_list"]) != n or text_emb.shape[0] != n:
        raise ValueError("source node/text/label/embedding lengths do not match")
    labels = [str(label) for label in source["all_labels"]]
    if len(labels) != 7 or len(set(labels)) != 7 or len(source["labels_desc"]) != 7:
        raise ValueError("Cora stream requires seven unique classes and descriptions")
    label_to_index = {label: i for i, label in enumerate(labels)}
    y = torch.tensor([label_to_index[str(value)] for value in source["label_list"]], dtype=torch.long)
    original_ids = torch.as_tensor(source.get("original_node_ids", torch.arange(n)), dtype=torch.long).cpu()
    if original_ids.shape != (n,) or original_ids.unique().numel() != n:
        raise ValueError("original_node_ids must identify each source row uniquely")
    split = [_partition(int(i), split_seed) for i in original_ids]
    stage = torch.tensor([0, 0, 1, 1, 2, 2, 2])[y]
    graph = {
        "x": torch.as_tensor(source["node_f"]).detach().cpu().float(),
        "edge_index": torch.as_tensor(source["edge_index"]).detach().cpu().long(),
        "text_emb": text_emb.detach().cpu().float(),
        "texts": [str(value) for value in source["text_list"]],
        "weights": torch.ones(n, dtype=torch.float32),
        "origin_ids": original_ids, "stage": stage,
    }
    stages, audits = [], []
    edges = graph["edge_index"]
    cross = stage[edges[0]] != stage[edges[1]]
    for t, classes in enumerate(STAGE_CLASSES):
        rows = torch.where(stage == t)[0]
        masks = {name: torch.tensor([split[i] == name for i in rows.tolist()], dtype=torch.bool) for name in SPLIT_NAMES}
        train_rows = rows[masks["train"]]
        train_graph, eval_graph = subset_graph(graph, train_rows), subset_graph(graph, rows)
        evaluation = {
            "graph": eval_graph, "y": y[rows], "masks": masks,
            "indices": {name: torch.where(mask)[0] for name, mask in masks.items()},
            "origin_ids": original_ids[rows].clone(),
        }
        stages.append({"train": train_graph, "eval": evaluation})
        audits.append({
            "stage": t, "class_indices": list(classes), "num_nodes": len(rows),
            "num_edges_directed": eval_graph["edge_index"].shape[1],
            "train_nodes": len(train_rows), "train_edges_directed": train_graph["edge_index"].shape[1],
            "split_counts": {name: int(mask.sum()) for name, mask in masks.items()},
            "split_ids_sha256": {name: _tensor_sha(original_ids[rows[mask]]) for name, mask in masks.items()},
            "all_origin_ids_sha256": _tensor_sha(original_ids[rows]),
            "edge_index_sha256": _tensor_sha(eval_graph["edge_index"]),
        })
    return stages, {
        "stages": audits, "cross_stage_edges_dropped": int(cross.sum()),
        "cross_stage_edge_fraction": float(cross.float().mean()) if edges.shape[1] else 0.0,
        "original_id_source": "original_node_ids" if "original_node_ids" in source else "source artifact row index; scoped to source SHA256",
        "all_origin_ids_sha256": _tensor_sha(original_ids),
    }


def _encoder_signature(bert_path: str):
    path = Path(bert_path).expanduser()
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(bert_path, allow_patterns=["*.json", "vocab.txt", "*.safetensors", "pytorch_model*.bin"]))
    path = path.resolve()
    selected = [file for file in path.iterdir() if file.is_file() and (file.suffix in {".json", ".safetensors"} or file.name == "vocab.txt" or file.name.startswith("pytorch_model") and file.suffix == ".bin")]
    if not any(file.name == "config.json" for file in selected) or not any(file.suffix in {".bin", ".safetensors"} for file in selected):
        raise ValueError(f"BERT directory must contain config.json and model weights: {path}")
    print(f"Hashing BERT configuration/tokenizer/weights in {path}", flush=True)
    signature = {
        "version": ENCODING_VERSION,
        "files": [{"name": file.name, "sha256": _sha256(file), "bytes": file.stat().st_size} for file in sorted(selected)],
        "model_class": "BertModel", "tokenizer_class": "BertTokenizer",
        "pooling": "last_hidden_state[:,0,:]", "frozen": True, "eval_mode": True,
        "max_length": 512, "padding": "longest_in_batch", "truncation": True,
        "prompt_template": "{label} {description}", "feature_normalization": "none",
    }
    return path, signature


def _encode_texts(texts, labels, descriptions, args, source_sha):
    from transformers import BertModel, BertTokenizer
    model_path, signature = _encoder_signature(str(args.bert_path))
    prompts = [f"{label} {desc}" for label, desc in zip(labels, descriptions)]
    cache_key = _json_sha({"source_sha256": source_sha, "encoder": signature, "text_sha256": _json_sha(texts), "prompts_sha256": _json_sha(prompts)})
    cache_path = Path(args.cache_path) if args.cache_path else None
    if cache_path and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("cache_key") != cache_key:
            raise ValueError(f"Cache differs or lacks provenance; choose a new --cache-path: {cache_path}")
        for name, count in (("text_emb", len(texts)), ("class_text_emb", 7)):
            value = cached[name]
            if value.ndim != 2 or value.shape[0] != count or not torch.isfinite(value).all() or _tensor_sha(value) != cached["tensor_sha256"][name]:
                raise ValueError(f"Invalid cache tensor: {name}")
        print(f"Using verified frozen-eval cache {cache_path}", flush=True)
        # A slice may own storage for every node plus class prompt.  Clone on
        # both cache reads and writes so a class-only file cannot serialize it.
        return (cached["text_emb"].detach().clone().contiguous(),
                cached["class_text_emb"].detach().clone().contiguous(), cached["encoder"], prompts)
    tokenizer = BertTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = BertModel.from_pretrained(str(model_path), local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)
    all_texts, outputs = list(texts) + prompts, []
    with torch.inference_mode():
        for start in range(0, len(all_texts), int(args.batch_size)):
            tokenized = tokenizer(all_texts[start:start + int(args.batch_size)], return_tensors="pt", padding=True, truncation=True, max_length=512).to(args.device)
            outputs.append(model(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"]).last_hidden_state[:, 0, :].cpu().float())
            if start == 0 or (start // int(args.batch_size)) % 25 == 0 or start + int(args.batch_size) >= len(all_texts):
                print(f"BERT encoded {min(start + int(args.batch_size), len(all_texts))}/{len(all_texts)} texts", flush=True)
    embedding = torch.cat(outputs)
    text_emb = embedding[:len(texts)].detach().clone().contiguous()
    class_text_emb = embedding[len(texts):].detach().clone().contiguous()
    encoder = {**signature, "model_path": str(model_path), "cache_key": cache_key, "source_sha256": source_sha}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cache_key": cache_key, "text_emb": text_emb, "class_text_emb": class_text_emb, "encoder": encoder,
                    "tensor_sha256": {"text_emb": _tensor_sha(text_emb), "class_text_emb": _tensor_sha(class_text_emb)}}, cache_path)
    return text_emb, class_text_emb, encoder, prompts


def prepare(args):
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    source_path, output_dir = Path(args.source).resolve(), Path(args.output).resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_sha = _sha256(source_path)
    text_emb, class_text_emb, encoder, prompts = _encode_texts(
        [str(value) for value in source["text_list"]], [str(value) for value in source["all_labels"]],
        [str(value) for value in source["labels_desc"]], args, source_sha)
    stages, audit = build_stage_partitions(source, text_emb, args.split_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for t, item in enumerate(stages):
        for name in ("train", "eval"):
            path = output_dir / f"stage_{t}_{name}.pt"
            torch.save(item[name], path)
            files[path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
    class_path = output_dir / "class_text_emb.pt"
    torch.save(class_text_emb.detach().clone().contiguous(), class_path)
    files[class_path.name] = {"sha256": _sha256(class_path), "bytes": class_path.stat().st_size}
    manifest = {
        "protocol": "native_tag_v1", "dataset": "cora", "source": str(source_path), "source_sha256": source_sha,
        "source_audit_metadata": source.get("audit_metadata"),
        "num_nodes": len(source["text_list"]), "num_edges_directed": int(source["edge_index"].shape[1]),
        "feature_dim": int(source["node_f"].shape[1]), "stage_classes": [list(c) for c in STAGE_CLASSES],
        "stage_numbering": "zero-based; each train/eval file is a delta stage, never cumulative",
        "split_rule": "SHA256(cora:native-tag-v1:{seed}:{original_id}); 70/15/15 integer thresholds, no label stratification",
        "split_seed": args.split_seed, "split_names": list(SPLIT_NAMES),
        "topology": "induced within each stage; all cross-stage edges removed; cumulative graphs are disjoint unions",
        "train_graph_excludes_checkpoint_and_test": True,
        "eval_graph": "fixed whole stage graph including holdout inputs; isolated from parameter/condensation training",
        "bert": encoder, "text_embedding_dim": int(text_emb.shape[1]), "class_prompts": prompts,
        "graph_schema": list(GRAPH_KEYS), "files": files, **audit,
        "label_boundary": "only preparation/evaluation read labels; train files contain neither y nor class text embeddings",
        "encoding_boundary": "frozen pretrained BERT independently encodes all strings; no fitting or adaptation on heldout texts",
        "text_selection_note": "joint_medoid uses joint normalized k-means, weighted X and a real representative text/embedding; no GPT2 CMI. Random keeps real X/text with nearest-representative mass.",
        "legacy_cache": "not reused: old TAGSAM cache lacks source/encoder provenance and did not explicitly force BERT eval mode",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "cora.pt")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "cora_class_stream")
    parser.add_argument("--bert-path", default="bert-base-uncased")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=0)
    print(json.dumps(prepare(parser.parse_args()), ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
