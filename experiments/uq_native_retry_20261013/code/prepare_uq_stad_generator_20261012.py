"""Prepare the pinned local STAD generator, including a verified transfer path."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
WEIGHT_SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
WEIGHT_BYTES = 3087467144
TARGET = "/home/pc/research/TAG_condensation/experiments/uq_final_evidence_20261012/models/qwen2.5-1.5b"
REQUIRED = ("config.json", "generation_config.json", "tokenizer_config.json",
            "tokenizer.json", "vocab.json", "merges.txt", "model.safetensors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--target", type=Path, default=Path(TARGET))
    parser.add_argument("--verify-local", action="store_true")
    parser.add_argument("--transfer", choices=("ssh", "tailnet", "modelscope"), default="ssh")
    args = parser.parse_args()
    if not args.verify_local:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=REVISION, local_dir=str(args.target),
            endpoint=args.endpoint, allow_patterns=list(REQUIRED), max_workers=3)
    for name in REQUIRED:
        if not (args.target / name).is_file():
            raise FileNotFoundError(f"missing pinned generator file: {name}")
    weight = args.target / "model.safetensors"
    if weight.stat().st_size != WEIGHT_BYTES:
        raise ValueError("generator weights have an unexpected byte count")
    hashes = {}
    for name in REQUIRED:
        digest = hashlib.sha256()
        with (args.target / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        hashes[name] = digest.hexdigest()
    if hashes[weight.name] != WEIGHT_SHA256:
        raise ValueError("generator weights differ from the pinned official LFS digest")
    transfer = "snapshot_download"
    if args.verify_local:
        if args.transfer == "modelscope":
            transfer = ("trusted ModelScope HTTPS mirror; no TLS bypass; official Hugging Face "
                        "pinned LFS SHA-256 and byte count verified at destination")
        else:
            transfer = ("canonical Hugging Face HTTPS via Windows curl; SHA-256 checked locally; "
                        + ("encrypted Tailnet HTTP transfer" if args.transfer == "tailnet" else "SSH transfer")
                        + "; SHA-256 checked at destination")
    manifest = dict(repository=REPO, revision=REVISION, path=str(args.target),
        download_endpoint=args.endpoint, tls_verification=True,
        weight_expected_sha256=WEIGHT_SHA256, weight_expected_bytes=WEIGHT_BYTES,
        file_sha256=hashes,
        transfer=transfer,
        native_generator="qwen3-max API", adapted_generator="local transformers Qwen2.5-1.5B-Instruct",
        rationale="explicit lower-capacity adaptation; no external API credentials",
        generation=dict(temperature=1.2, top_p=.85, max_new_tokens=256, candidates_per_cluster=3))
    path = args.target / "generator_provenance.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("existing generator provenance differs")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
