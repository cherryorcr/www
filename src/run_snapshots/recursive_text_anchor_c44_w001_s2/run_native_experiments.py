"""Audited development runner for native static/continual TAG condensation.

Only this evaluator/orchestrator knows holdout labels. Model/condensation
functions receive label-free graph dictionaries; raw-history oracle inputs
are separated explicitly from deployable memory-plus-delta inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from native_core import clean_data, condense_asm, evaluate_logits, train_model
from protocol_data import concat_graphs, select_initial, subset_graph
from topology import apply_link_predictor, fit_link_predictor


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(y, pred):
    if len(y) == 0:
        return {"macro_f1": None, "accuracy": None, "n": 0}
    labels = np.unique(y)
    return {"macro_f1": 100 * float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "accuracy": 100 * float(accuracy_score(y, pred)), "n": len(y),
            "support": {str(int(c)): int((y == c).sum()) for c in labels}}


class Experiment:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "run_config.json").exists():
            raise FileExistsError("choose a new output directory; existing runs are immutable")
        code_snapshot = self.output / "code"
        code_snapshot.mkdir(exist_ok=True)
        for source_file in Path(__file__).parent.glob("*.py"):
            shutil.copy2(source_file, code_snapshot / source_file.name)
        torch.set_num_threads(args.threads)
        self.prepared = Path(args.prepared)
        self.manifest = json.loads((self.prepared / "manifest.json").read_text(encoding="utf-8"))
        self.verify_input("class_text_emb.pt")
        self.classes = torch.load(self.prepared / "class_text_emb.pt", map_location="cpu", weights_only=True)
        self.evals = {}
        self.rows = []
        self.counts = []
        self.started = time.time()
        configuration = vars(args).copy()
        configuration.update({"development_only": True,
                              "source_manifest_sha256": file_sha(self.prepared / "manifest.json"),
                              "source_sha256": {p.name: file_sha(p) for p in code_snapshot.glob("*.py")},
                              "torch_version": str(torch.__version__),
                              "gpu": torch.cuda.get_device_name(torch.device(self.device)) if "cuda" in self.device else "cpu"})
        dump(self.output / "run_config.json", configuration)

    def verify_input(self, filename):
        expected = self.manifest["files"][filename]["sha256"]
        if file_sha(self.prepared / filename) != expected:
            raise ValueError(f"input hash mismatch: {filename}")

    def load_stage(self, stage):
        self.verify_input(f"stage_{stage}_train.pt")
        self.verify_input(f"stage_{stage}_eval.pt")
        value = torch.load(self.prepared / f"stage_{stage}_train.pt", map_location="cpu", weights_only=False)
        if "y" in value or "labels" in value:
            raise ValueError("training input contains labels")
        if not bool((value["stage"] == stage).all()):
            raise ValueError("stage train must contain only this delta")
        self.evals[stage] = torch.load(self.prepared / f"stage_{stage}_eval.pt", map_location="cpu", weights_only=False)
        holdout = self.evals[stage]
        forbidden = torch.cat([holdout["graph"]["origin_ids"][holdout["masks"][name]] for name in ("checkpoint", "test")])
        if len(set(value["origin_ids"].tolist()) & set(forbidden.tolist())):
            raise ValueError("holdout roots entered training data")
        self.counts.append(len(value["x"]))
        return value

    def metric(self, model, stage, partition):
        all_y, all_p, per_stage = [], [], []
        for j in range(stage + 1):
            evaluation = self.evals[j]
            pred = evaluate_logits(model, evaluation["graph"], self.classes).argmax(1).cpu()
            mask = evaluation["masks"][partition]
            y = evaluation["y"][mask].cpu().numpy()
            p = pred[mask].numpy()
            all_y.append(y)
            all_p.append(p)
            per_stage.append(score(y, p))
        result = score(np.concatenate(all_y), np.concatenate(all_p))
        result["per_stage"] = per_stage
        result["old_mean_f1"] = float(np.mean([r["macro_f1"] for r in per_stage[:-1]])) if stage else None
        result["current_f1"] = per_stage[-1]["macro_f1"]
        return result

    def fit(self, data, stage, seed, steps=None, label="student", weighted=False, save=False):
        steps = self.args.student_steps if steps is None else steps
        k = max(2, math.floor(self.args.rate * sum(self.counts[:stage + 1])))
        batch = min(self.args.batch_size, k, len(data["x"]))
        if batch < 2:
            raise ValueError("InfoNCE needs at least two roots")
        selected, best = None, -math.inf
        step0_checkpoint = None
        curve = []

        def checkpoint(model, row):
            nonlocal selected, best, step0_checkpoint
            if row["step"] % self.args.check_every == 0 or row["step"] == steps:
                metric = self.metric(model, stage, "checkpoint")
                if row["step"] == 0:
                    step0_checkpoint = metric
                curve.append({"step": row["step"], "f1": metric["macro_f1"], "accuracy": metric["accuracy"],
                              "per_stage": metric["per_stage"], "old_mean_f1": metric["old_mean_f1"],
                              "current_f1": metric["current_f1"],
                              "unique_rows_seen": row["unique_rows_seen"], "examples_seen": row["examples_seen"]})
                if metric["macro_f1"] > best:
                    best = metric["macro_f1"]
                    selected = (row["step"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        start = time.time()
        model, logs = train_model(data, seed, steps, self.args.lr, self.args.lr, batch, self.device,
                                  hidden_dim=self.args.hidden, out_dim=self.args.hidden,
                                  callback=checkpoint, weighted=weighted)
        endpoint = self.metric(model, stage, "checkpoint")
        model.load_state_dict(selected[1])
        info = {"seed": seed, "steps": steps, "selected_step": selected[0], "best_checkpoint_f1": best,
                "step0_checkpoint": step0_checkpoint, "best_minus_step0": best - step0_checkpoint["macro_f1"],
                "endpoint_checkpoint": endpoint, "curve": curve, "train_seconds": time.time() - start,
                "unique_rows_seen": logs[-1]["unique_rows_seen"], "examples_seen": logs[-1]["examples_seen"],
                "input_roots": len(data["x"]), "contrastive_batch": batch, "weighted": weighted}
        if save:
            dump(self.output / f"{label}_s{stage}_m{seed}_train.json", info)
            torch.save({"state_dict": selected[1], "model_config": model.model_config,
                        "train_origin_ids": data["origin_ids"], "stage": stage,
                        "training_labels_used": False}, self.output / f"{label}_s{stage}_m{seed}_model.pt")
        return model, info

    def record(self, data, stage, arm, access, weighted=False):
        key = f"{arm}_s{stage}"
        payload = clean_data(data, "cpu")
        payload["metadata"] = {**data.get("metadata", {}), "arm": arm, "stage": stage,
                               "access": access, "training_labels_used": False,
                               "source_manifest_sha256": file_sha(self.prepared / "manifest.json")}
        artifact_path = self.output / f"{key}_artifact.pt"
        torch.save(payload, artifact_path)
        # Reload the exported object before independent training.
        payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
        row = {"arm": arm, "stage": stage, "access": access, "roots": len(payload["x"]),
               "root_rate": len(payload["x"]) / sum(self.counts[:stage + 1]),
               "train_denominator": sum(self.counts[:stage + 1]),
               "artifact_bytes": artifact_path.stat().st_size, "artifact_sha256": file_sha(artifact_path),
               "text_utf8_bytes": sum(len(t.encode()) for t in payload["texts"]),
               "tensor_bytes": sum(v.numel() * v.element_size() for v in payload.values() if torch.is_tensor(v)),
               "mass": float(payload["weights"].sum()), "model_runs": []}
        for seed in self.args.model_seeds:
            model, info = self.fit(payload, stage, seed, label=arm, weighted=weighted, save=True)
            info["checkpoint"] = self.metric(model, stage, "checkpoint")
            if self.args.confirm:
                info["confirmation"] = self.metric(model, stage, "test")
            row["model_runs"].append(info)
            del model
        row["mean_checkpoint_f1"] = float(np.mean([v["checkpoint"]["macro_f1"] for v in row["model_runs"]]))
        if self.args.confirm:
            row["mean_confirmation_f1"] = float(np.mean([v["confirmation"]["macro_f1"] for v in row["model_runs"]]))
        self.rows.append(row)
        dump(self.output / "results.json", {"results": self.rows, "elapsed_seconds": time.time() - self.started,
                                            "development_only": True, "confirmation_accessed": self.args.confirm})
        print(json.dumps({k: row[k] for k in ("stage", "arm", "roots", "mean_checkpoint_f1")}), flush=True)

    def teacher(self, source, stage, suffix, weighted=False):
        teacher, info = self.fit(source, stage, self.args.seed + 700, self.args.teacher_steps,
                                 label=f"teacher_{suffix}", weighted=weighted, save=True)
        print(json.dumps({"teacher": suffix, "stage": stage, "checkpoint_f1": info["best_checkpoint_f1"]}), flush=True)
        return teacher

    def initializer(self, source, k, stage, teacher, mode="joint_medoid", real_x=False):
        initial = select_initial(source, k, self.args.seed + 31 * stage, mode)
        if real_x:
            lookup = {int(v): i for i, v in enumerate(source["origin_ids"])}
            initial["x"] = torch.stack([source["x"][lookup[int(v)]] for v in initial["origin_ids"]])
        decoder, info = fit_link_predictor(source, teacher, self.args.seed + stage, self.device,
                                            steps=self.args.link_steps)
        initial = apply_link_predictor(initial, teacher, decoder, degree=self.args.degree)
        initial["metadata"] = {"initializer": mode, "real_x": real_x, "topology": info,
                               "text_selection": "real representative; not GPT2-CMI"}
        return initial

    def optimize(self, source, initial, teacher, stage, name, weighted=False, extra_config=None, append_old=None,
                 select_validation=True, retain_outer_steps=False):
        config = {"outer_steps": self.args.outer_steps, "outer_lr": self.args.outer_lr,
                  "inner_steps": self.args.inner_steps, "inner_lr_graph": self.args.inner_lr,
                  "inner_lr_text": self.args.inner_lr, "inner_lr_mode": self.args.inner_lr_mode,
                  "batch_size": min(self.args.batch_size, len(initial["x"])),
                  "reference_size": self.args.reference_size, "eval_every": self.args.outer_eval_every,
                  "weighted": weighted, "feature_norm_cap": self.args.norm_cap}
        config.update(extra_config or {})
        validations = []

        def validator(candidate, step):
            validation_data = concat_graphs((append_old, candidate)) if append_old is not None else candidate
            model, info = self.fit(validation_data, stage, 9000 + self.args.seed,
                                   self.args.validation_steps, weighted=weighted)
            validations.append({"outer_step": step, **info})
            print(json.dumps({"condense": name, "stage": stage, "outer_step": step,
                              "validation_f1": info["best_checkpoint_f1"]}), flush=True)
            return info["best_checkpoint_f1"]

        def snapshot(candidate, step, kind):
            torch.save(candidate, self.output / f"{name}_s{stage}_{kind}.pt")
            if retain_outer_steps and kind in ("initial", "last"):
                torch.save(candidate, self.output / f"{name}_s{stage}_outer_{int(step)}.pt")

        config["artifact_callback"] = snapshot
        start = time.time()
        artifact, logs = condense_asm(source, initial, teacher, config, self.args.seed + stage,
                                     self.device, validator=validator if select_validation else None)
        dump(self.output / f"{name}_s{stage}_condense.json", {"logs": logs, "validations": validations,
                                                             "seconds": time.time() - start})
        artifact["metadata"]["initializer"] = initial.get("metadata", {})
        return artifact

    def run(self):
        if {"old_only", "union_no_recondense"}.intersection(self.args.arms) and "union" not in self.args.arms:
            raise ValueError("old_only and union_no_recondense require the union arm to supply its previous memory")
        history, previous = [], {}
        for stage in range(self.args.stages):
            delta = self.load_stage(stage)
            history.append(delta)
            full = concat_graphs(history)
            k = math.floor(self.args.rate * len(full["x"]))
            if k < 2:
                raise ValueError(f"root budget {k} cannot support contrastive condensation")
            if "full" in self.args.arms:
                self.record(full, stage, "full", "full_history_reference")
            shared = self.teacher(full, stage, "offline")
            for mode, real, arm in [("random", True, "random"), ("joint_medoid", True, "coreset"),
                                    ("joint_medoid", False, "frozen")]:
                if arm in self.args.arms:
                    init = self.initializer(full, k, stage, shared, mode, real)
                    self.record(init, stage, arm, "offline_selection_reference" if stage else "current_delta")
            if self.args.phase == "headroom":
                del shared
                continue
            if "offline_asm" in self.args.arms:
                init = self.initializer(full, k, stage, shared)
                compact = self.optimize(full, init, shared, stage, "offline_asm")
                self.record(compact, stage, "offline_asm", "full_history_reference")
            if self.args.phase == "static":
                del shared
                continue
            if "delta_only" in self.args.arms:
                self.record(delta, stage, "delta_only", "current_delta_over_budget_diagnostic")
            for arm in ("append", "union", "mass_union"):
                if arm not in self.args.arms:
                    continue
                old = previous.get(arm)
                if old is not None and arm == "union":
                    if "old_only" in self.args.arms:
                        self.record(old, stage, "old_only", "previous_union_memory_diagnostic")
                    if "union_no_recondense" in self.args.arms:
                        self.record(concat_graphs((old, delta)), stage, "union_no_recondense", "memory_plus_delta_over_budget")
                if arm == "append":
                    new_k = k - (len(old["x"]) if old is not None else 0)
                    if new_k < 2:
                        raise ValueError("Append needs at least two new roots for contrastive condensation")
                    source = delta
                    teacher = self.teacher(source, stage, arm)
                    init = self.initializer(source, new_k, stage, teacher)
                    new = self.optimize(source, init, teacher, stage, arm, append_old=old)
                    compact = concat_graphs((old, new)) if old is not None else new
                else:
                    source = concat_graphs((old, delta)) if old is not None else delta
                    teacher = self.teacher(source, stage, arm, weighted=arm == "mass_union")
                    init = self.initializer(source, k, stage, teacher)
                    compact = self.optimize(source, init, teacher, stage, arm, weighted=arm == "mass_union")
                previous[arm] = compact
                self.record(compact, stage, arm, "memory_plus_current_delta", weighted=arm == "mass_union")
                del teacher
            del shared
        dump(self.output / "completed.json", {"status": "complete", "elapsed_seconds": time.time() - self.started})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--phase", choices=["headroom", "static", "continual"], default="headroom")
    p.add_argument("--stages", type=int, default=1)
    p.add_argument("--rate", type=float, default=.01)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--model-seeds", type=int, nargs="+", default=[100, 101])
    p.add_argument("--arms", nargs="+", default=["full", "random", "coreset", "frozen", "offline_asm"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--student-steps", type=int, default=512)
    p.add_argument("--teacher-steps", type=int, default=800)
    p.add_argument("--validation-steps", type=int, default=512)
    p.add_argument("--check-every", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--link-steps", type=int, default=160)
    p.add_argument("--degree", type=float, default=1.5)
    p.add_argument("--outer-steps", type=int, default=100)
    p.add_argument("--outer-lr", type=float, default=.01)
    p.add_argument("--inner-steps", type=int, default=10)
    p.add_argument("--inner-lr", type=float, default=.005)
    p.add_argument("--inner-lr-mode", choices=["fixed", "bounded", "unbounded"], default="fixed")
    p.add_argument("--reference-size", type=int, default=128)
    p.add_argument("--outer-eval-every", type=int, default=50)
    p.add_argument("--norm-cap", type=float, default=None)
    p.add_argument("--confirm", action="store_true")
    args = p.parse_args()
    Experiment(args).run()


if __name__ == "__main__":
    main()
