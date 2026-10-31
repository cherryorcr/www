"""Run the predeclared experiment, audit and summary as a fail-fast pipeline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("arxiv", "bookchild"), required=True)
    p.add_argument("--device", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    code = root / "code/v2_llm_fusion"
    result = Path("/home/pc/research/TAG_condensation/results/degree_confirmation_20260920") / args.dataset
    previous = Path("/home/pc/research/TAG_condensation/results/p5_large_base4_20260919") / (args.dataset + "_base4")
    data = ("/home/pc/research/TAGSAM/data/ogbn_arxiv_full_minilm_semantic.pt" if args.dataset == "arxiv"
            else "/home/pc/research/TAGSAM/data/tagbench/bookchild.pt")
    result.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, "-u", str(code / "run_degree_cross_task_confirmation.py"), "--dataset", args.dataset,
         "--data", data, "--text", str(previous / "text_emb_pool_only.pt"), "--c0-root", str(previous),
         "--output", str(result), "--device", args.device],
        [sys.executable, "-u", str(code / "audit_degree_confirmation.py"), "--root", str(result), "--device", args.device],
        [sys.executable, "-u", str(code / "summarize_degree_confirmation.py"), "--root", str(result)],
    ]
    for command in commands:
        print(json.dumps({"command": command}), flush=True)
        subprocess.run(command, check=True, cwd=root)
    (result / "pipeline_complete.json").write_text(json.dumps({"completed": True}), encoding="utf-8")


if __name__ == "__main__":
    main()
