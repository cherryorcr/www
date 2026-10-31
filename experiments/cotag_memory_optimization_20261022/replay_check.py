"""Post-hoc repeatability check, excluded from construction-level statistics.

The SportsFit 271108 pair was selected because its first comparison dropped by
about one F1 point. Re-evaluate the EXACT saved memories with the SAME students;
do not replace their original scores or count these as independent constructions.
"""
import argparse
import json
from pathlib import Path

import run as A


def main(args):
    A.torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    root = args.input
    ref = A.R.L.read(A.ROOT / "experiments/uq_final_evidence_20261012/sportsfit_reference.json")
    groups = A.R.V.choose_groups(ref, 9101)
    data = A.R.L.load_large("sportsfit",
        "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip",
        A.ROOT / "results/public_baselines_text_cache_20261001/sportsfit", "cuda:0",
        dict(groups=groups, report_dev_mod=ref.get("report_dev_mod", 3)))
    records = []
    for variant in ("original", "combined"):
        folder = root / f"sportsfit_{variant}_c271108"
        before = A.read(folder / "results.json")
        assert before["status"] == "complete"
        row = before["rows"][-1]
        file = folder / row["memory_path"]
        assert A.digest(file) == row["memory_file_sha256"]
        memory = A.load_memory(file)
        assert A.R.V.memory_hash(memory) == row["memory_hash"]
        out = args.output / variant
        out.mkdir(exist_ok=True)
        score = A.evaluate(memory, data, len(groups)-1, "cuda:0", out)
        predictions = []
        for old, new in zip(row["scores"]["per_student"], score["per_student"]):
            assert old["seed"] == new["seed"]
            with A.np.load(folder / old["prediction_path"]) as a, A.np.load(out / new["prediction_path"]) as b:
                assert A.np.array_equal(a["ids"], b["ids"])
                assert A.np.array_equal(a["truth"], b["truth"])
                predictions.append(dict(seed=old["seed"], old_f1=old["f1"], new_f1=new["f1"],
                    prediction_agreement=float(A.np.mean(a["pred"] == b["pred"]))))
        records.append(dict(variant=variant, memory_hash=row["memory_hash"],
            original_scores=row["scores"], repeated_scores=score, comparisons=predictions))
        A.emit(event="repeat_complete", variant=variant, original_f1=row["scores"]["f1"], repeated_f1=score["f1"])
    A.save(result_path, dict(status="complete", purpose=__doc__, source_hash=A.digest(Path(__file__)), rows=records))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
