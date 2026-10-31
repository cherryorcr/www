"""Fill small-graph baseline gaps without changing historical experiments."""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


ARMS = ("tagsam_native_bounded", "stad_native_local", "dance_adapter",
        "sr_gm_adapter", "stad_adapter")
INSPIRED = {"dance_adapter", "sr_gm_adapter", "stad_adapter"}
STUDENT_SEEDS = [150, 151, 152]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def sr_gm_latent_refine(block, delta, device, seed, legacy, steps=3):
    """Inspired control: conflict projection uses the shared 128-D latent space."""
    torch.manual_seed(seed)
    out = {key: value.clone() if torch.is_tensor(value) else value
           for key, value in block.items()}
    count = min(512, len(out["x"]), len(delta["x"]))
    if count < 2:
        raise ValueError("SR-GM contrastive refinement requires at least two rows")
    generator = torch.Generator().manual_seed(seed + 77)
    raw_idx = torch.randperm(len(delta["x"]), generator=generator)[:count]
    syn_idx = torch.randperm(len(out["x"]), generator=generator)[:count]
    x0 = out["x"][syn_idx].float().to(device)
    text = out["text_emb"][syn_idx].float().to(device)
    xr = delta["x"][raw_idx].float().to(device)
    tr = delta["text_emb"][raw_idx].float().to(device)
    model = legacy.TinyMM(xr.shape[1]).to(device)
    params = tuple(model.parameters())
    x_var = torch.nn.Parameter(x0.clone())
    optimizer = torch.optim.Adam([x_var], lr=.03)
    trajectory = []
    for update in range(steps):
        gr, trh = model(xr, tr)
        gs, tsh = model(x_var, text)
        real_logits, syn_logits = gr @ trh.T / .07, gs @ tsh.T / .07
        target = torch.arange(count, device=device)
        real_loss = (F.cross_entropy(real_logits, target) +
                     F.cross_entropy(real_logits.T, target)) / 2
        syn_loss = (F.cross_entropy(syn_logits, target) +
                    F.cross_entropy(syn_logits.T, target)) / 2
        real_grad = torch.autograd.grad(real_loss, params, retain_graph=True)
        syn_grad = torch.autograd.grad(syn_loss, params, create_graph=True,
                                      retain_graph=True)
        vr = torch.cat([g.detach().reshape(-1) for g in real_grad])
        vs = torch.cat([g.reshape(-1) for g in syn_grad])
        gm = 1 - F.cosine_similarity(vs, vr, dim=0)
        gx, gt = torch.autograd.grad(syn_loss, (gs, tsh), create_graph=True)
        dot = (gx * gt).sum(1, keepdim=True)
        projection = gx - dot / (gt.square().sum(1, keepdim=True) + 1e-8) * gt
        decoupled = torch.where(dot < 0, projection, gx)
        with torch.no_grad():
            z = F.normalize(x_var.detach(), dim=1)
            similarity = z @ z.T
            similarity.fill_diagonal_(-2)
            neighbors = similarity.topk(min(4, count - 1), dim=1).indices
        smooth = (decoupled - decoupled[neighbors].mean(1)).square().mean()
        objective = gm + .05 * smooth + .01 * (x_var - x0).square().mean()
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError("SR-GM-inspired objective is nonfinite")
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        objective.backward()
        if x_var.grad is None or not bool(torch.isfinite(x_var.grad).all()):
            raise FloatingPointError("SR-GM-inspired gradient is nonfinite")
        trajectory.append(dict(update=update + 1, objective=float(objective.detach()),
            gradient_matching=float(gm.detach()), damping=float(smooth.detach()),
            feature_gradient_norm=float(x_var.grad.norm())))
        optimizer.step()
    out["x"][syn_idx] = x_var.detach().cpu()
    return out, dict(steps=steps, active_rows=count, projection_space="shared 128-D latent outputs",
        reason="source X and MiniLM dimensions differ; coordinatewise raw-gradient projection is undefined",
        optimized="X only; fixed real-source text payload", trajectory=trajectory,
        official_reproduction=False)


def load_modules(root):
    frozen = root / "experiments/uq_final_evidence_20261012/code"
    sys.path.insert(0, str(frozen))
    sys.path.insert(1, str(root / "code/v2_llm_fusion"))
    import run_unified_q_matrix_20261008 as R
    import run_uq_final_confirmation_20261012 as E
    import run_uq_baseline_expansion_20261013 as B
    source = root / "tools/uq_stad_native_core_20261012.py"
    spec = importlib.util.spec_from_file_location("missing_small_stad_native_core", source)
    S = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = S
    spec.loader.exec_module(S)
    if Path(R.__file__).resolve().parent != frozen.resolve():
        raise AssertionError("immutable construction snapshot was not imported")
    return R, E, B, S


def install_tagsam_prefix_compatibility():
    import run_unified_q_tagsam_20261008 as A
    target = A.T.SCA.G
    if target is not A.T.G:
        raise AssertionError("TAGSAM response and append modules use different prefix helpers")
    if hasattr(target, "shrink_by_mass"):
        raise AssertionError("expected audited frozen helper without shrink_by_mass")

    def unchanged_prefix(memory, keep):
        if int(keep) != len(memory["x"]):
            raise ValueError("compatibility helper only permits the exact frozen prefix")
        return memory

    target.shrink_by_mass = unchanged_prefix
    return dict(target=str(Path(target.__file__).resolve()),
        rule="missing shrink_by_mass is identity only when keep equals the full old prefix",
        shrinking_allowed=False, frozen_source_modified=False)


def migrate_stage_zero(source, destination, protocol, R):
    """Reuse a completed stage only across the two explicitly audited repairs."""
    if protocol["arm"] != "tagsam_native_bounded":
        raise ValueError("stage-zero migration is registered for TAGSAM only")
    old_path = source / "results.json"
    old = json.loads(old_path.read_text())
    before = dict(old["protocol"])
    after = dict(protocol)
    if old.get("status") == "complete" or [r["stage"] for r in old["rows"]] != [0]:
        raise ValueError("migration requires exactly one completed stage of an unfinished run")
    expected_runner = "2ebc9bb4c16205c4032d7268f7c347d676a461de3f3b6491328ba26d98b62bfb"
    expected_stad = "75af643e7a856077a9d56cb910db3bf5846e90ad4b533ac35ef0bd5931f89fc5"
    if before["runner_hash"] != expected_runner:
        raise ValueError("unreviewed source runner cannot be migrated")
    if before["source_hashes"]["uq_stad_native_core_20261012.py"] != expected_stad:
        raise ValueError("unreviewed source append helper cannot be migrated")
    if before["name"] != "cotag-q-missing-small-baselines-20261019-v1":
        raise ValueError("unexpected source protocol version")
    for value in (before, after):
        for name in ("name", "runner_hash", "compatibility_repairs"):
            value.pop(name, None)
        value["source_hashes"] = dict(value["source_hashes"])
        value["source_hashes"].pop("uq_stad_native_core_20261012.py")
    if before != after:
        raise ValueError("invariant experiment protocol changed; stage migration refused")
    block_path = source / "s0/block.pt.gz"
    audit_path = source / "s0/construction.json"
    with gzip.open(block_path, "rb") as stream:
        block = torch.load(stream, map_location="cpu", weights_only=False)
    audit = json.loads(audit_path.read_text())
    row = old["rows"][0]
    R.V.validate_memory(block, None, protocol["budgets"][0])
    if not (R.V.memory_hash(block) == audit["memory_hash"] == row["memory_hash"]):
        raise ValueError("source artifact and completed metrics do not agree")
    if row["memory_rows"] != protocol["budgets"][0] or row["scores"] is None:
        raise ValueError("source stage is not fully evaluated")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("migration destination must be empty")
    (destination / "s0").mkdir()
    for path in (block_path, audit_path):
        shutil.copy2(path, destination / "s0" / path.name)
    migration = dict(source=str(source.resolve()), source_results_sha256=sha(old_path),
        source_protocol=old["protocol"], stage=0, memory_hash=row["memory_hash"],
        artifact_sha256=sha(block_path), source_audit_sha256=sha(audit_path),
        reason="stage zero has no old prefix; updated STAD append returns the same block",
        reused_student_metrics=True)
    save(destination / "migration.json", migration)
    return dict(protocol=protocol, rows=[row], status="running", migration=migration)


def inspired_construct(delta, rows, arm, encoder, device, R, legacy):
    seed = 7000 + int(delta["stage"])
    local = dict(x=delta["x"], y=delta["y"], texts=delta["texts"],
                 edge_index=delta["edge_index"])
    quotas = R.M.ratio_allocation(delta["y"], rows)
    block = legacy.build_clusters(local, delta["text_emb"], torch.arange(len(delta["x"])),
        quotas, arm, int(delta["stage"]), seed, encoder, device)
    refinement = None
    if arm == "sr_gm_adapter":
        block, refinement = sr_gm_latent_refine(block, delta, device, seed, legacy)
    block["mass"] = block["weights"].clone()
    block["depth"] = torch.zeros(len(block["x"]), dtype=torch.long)
    if len(block["x"]) != rows:
        raise AssertionError("inspired constructor underfilled its registered budget")
    return block, dict(method=arm, seed=seed, allocation=quotas,
        quota_rule="shared exact ratio_allocation; repairs legacy saturated-class underfill",
        source_ids_hash=R.V.hash_tensor(delta["ids"]), historical_raw_reads=0,
        external_labels_used_for_selection=False, official_reproduction=False,
        identity="inspired reference; legacy Student; excluded from common-Student ranking",
        topology="new-block text kNN degree 4; frozen block-diagonal prefix",
        refinement=refinement)


def run(args):
    R, E, B, S = load_modules(args.root)
    reference = B.read_reference(None, args.dataset)
    data = B.load_data(args, reference, R)
    ids = R.L.stage_ids(data)
    sizes = list(map(len, ids))
    budgets, floors = R.V.choose_budgets(reference, sizes, data["stage_classes"], "fixed_rate", .05)
    modules = [R, E, B, S, R.P, R.M, R.L, R.V]
    native = None
    legacy = None
    repairs = dict(stad_module="explicit tools source; avoids frozen module name shadowing")
    if args.arm == "tagsam_native_bounded":
        import uq_tagsam_native_selector_20261012 as native
        modules.append(native)
        repairs["tagsam_prefix"] = install_tagsam_prefix_compatibility()
    elif args.arm == "stad_native_local":
        native = S.official_components()
    else:
        import run_external_large_adapters_20261011 as legacy
        modules.append(legacy)
    inspired = args.arm in INSPIRED
    protocol = dict(name="cotag-q-missing-small-baselines-20261019-repair-v2",
        dataset=args.dataset, arm=args.arm, groups=data["stage_classes"],
        budgets=budgets, stage_raw_rows=sizes, class_floor_adjustments=floors,
        memory_rate=.05, actual_final_rate=budgets[-1] / sum(sizes),
        order_seed=args.order_seed, construction_seed=(7000 if inspired else args.construction_seed),
        construction_seed_rule=("7000 + stage" if inspired else "271006, native stage offsets"),
        student_seeds=STUDENT_SEEDS, student_steps=(1024 if inspired else 2048), student_batch=24,
        student=("run_multiscale_asm single-positive legacy" if inspired else R.P.model_config()),
        encoder=args.minilm, pool_mask_hash=R.V.hash_tensor(data["pool_mask"]),
        report_mask_hash=R.V.hash_tensor(data["report_dev_mask"]),
        test_mask_hash=R.V.hash_tensor(data["test_mask"]),
        stage_id_hashes=[R.V.hash_tensor(v) for v in ids],
        source_hashes={Path(m.__file__).name: sha(m.__file__) for m in modules},
        runner_hash=sha(__file__), official_reproduction=False,
        access="current delta only; frozen synthetic prefix; no historical raw reads",
        evaluation="report-dev; additionally historical test mask for common-Student arms",
        test_status="previously inspected dataset masks; not a newly blind sealed confirmation",
        method_selection_from_test=False, rank_with_common_student=not inspired,
        dispersion="sample SD across three Students; one order/construction cell",
        stage_evaluation=("final only" if inspired else "every stage"),
        compatibility_repairs=repairs)
    if args.arm == "stad_native_local":
        protocol.update(official_source=native[-1],
            generator=json.loads((S.MODEL / "generator_provenance.json").read_text()),
            sparse_audit=S.audit_sparse(), internal_split_policy=S.INTERNAL_SPLIT_POLICY,
            adaptations="native core and selection; local Qwen2.5-1.5B; shared MiniLM/Student; binary stored support")
    elif args.arm == "tagsam_native_bounded":
        protocol["adaptations"] = "native GPT2-large PMI selector; bounded 100x10 response; shared MiniLM/Student"
    else:
        protocol["adaptations"] = "legacy inspired constructor and Student; exact shared row quotas"
        if args.arm == "sr_gm_adapter":
            protocol["gradient_projection"] = "shared 128-D latent outputs; explicit dimension adaptation"
    path = args.output / "results.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["protocol"] != protocol:
            raise ValueError("protocol mismatch: preserve this output and use a new directory")
    elif args.resume_from is not None:
        result = migrate_stage_zero(args.resume_from, args.output, protocol, R)
        print(json.dumps(dict(event="migrated_stage_zero", source=str(args.resume_from),
                              memory_hash=result["rows"][0]["memory_hash"])), flush=True)
    else:
        result = dict(protocol=protocol, rows=[], status="running")
    save(args.output / "protocol.json", protocol)
    save(path, result)
    parent = None
    class_names = list(R.L.U.G.CLASS_NAMES)
    if len(class_names) != len(data["prompts"]):
        raise AssertionError("class names and common prompts differ")
    for stage, raw_ids in enumerate(ids):
        stage_dir = args.output / f"s{stage}"
        block_path = stage_dir / "block.pt.gz"
        audit_path = stage_dir / "construction.json"
        previous = next((row for row in result["rows"] if row["stage"] == stage), None)
        if block_path.exists() and audit_path.exists():
            with gzip.open(block_path, "rb") as stream:
                block = torch.load(stream, map_location="cpu", weights_only=False)
            construction = json.loads(audit_path.read_text())
            memory = S.append(parent, block)
            checks = R.V.validate_memory(memory, parent, budgets[stage])
            if R.V.memory_hash(memory) != construction["memory_hash"]:
                raise AssertionError("resume artifact hash differs from construction record")
            if previous:
                if previous["memory_hash"] != construction["memory_hash"]:
                    raise AssertionError("resume metrics and artifact disagree")
                parent = memory
                continue
            print(json.dumps(dict(event="resume_constructed", stage=stage)), flush=True)
        else:
            if previous or block_path.exists() or audit_path.exists():
                raise ValueError("incomplete artifact pair; preserve files for inspection")
            delta = R.L.delta_view(data, raw_ids, stage)
            new_rows = budgets[stage] - (0 if parent is None else len(parent["x"]))
            R.V.sync(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            started = time.perf_counter()
            if args.arm == "tagsam_native_bounded":
                block, audit = native.build_tagsam(delta, parent, new_rows, data["encoder"],
                    args.construction_seed, args.device)
            elif args.arm == "stad_native_local":
                block, audit = S.construct(delta, new_rows, data["encoder"], args.construction_seed,
                    args.device, stage_dir, args.dataset, class_names, native)
            else:
                block, audit = inspired_construct(delta, new_rows, args.arm, data["encoder"], args.device, R, legacy)
            memory = S.append(parent, block)
            R.V.sync(args.device)
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(args.device) / 2**20
            checks = R.V.validate_memory(memory, parent, budgets[stage])
            construction = dict(stage=stage, construction_seconds=seconds, construction_peak_mb=peak,
                memory_hash=R.V.memory_hash(memory), audit=audit)
            stage_dir.mkdir(parents=True, exist_ok=True)
            temporary = block_path.with_suffix(".gz.tmp")
            with gzip.open(temporary, "wb", compresslevel=3) as stream:
                torch.save(block, stream)
            temporary.replace(block_path)
            save(audit_path, construction)
            print(json.dumps(dict(event="constructed", dataset=args.dataset, arm=args.arm, stage=stage,
                rows=len(memory["x"]), seconds=seconds, peak_mb=peak)), flush=True)
        started = time.perf_counter()
        if inspired:
            scores = None
            if stage == len(ids) - 1:
                legacy_memory = dict(memory, weights=memory["mass"])
                score = legacy.evaluate(legacy_memory, data, args.device, STUDENT_SEEDS, stage, 1024)
                for metric in ("macro_f1", "accuracy"):
                    score[metric + "_std"] = float(np.std([v[metric] for v in score["per_seed"]], ddof=1))
                scores = {"report_dev": score}
        else:
            scores = E.evaluate(memory, data, stage, STUDENT_SEEDS, args.device, steps=2048, batch=24)
        R.V.sync(args.device)
        row = dict(arm=args.arm, stage=stage, raw_rows=len(raw_ids),
            new_rows=budgets[stage] - (0 if parent is None else len(parent["x"])),
            memory_rows=len(memory["x"]), scores=scores,
            score=scores["report_dev"] if scores else None,
            construction_seconds=construction["construction_seconds"],
            construction_peak_mb=construction["construction_peak_mb"],
            student_seconds=time.perf_counter()-started, checks=checks,
            memory_hash=construction["memory_hash"], audit=construction["audit"])
        result["rows"].append(row)
        save(path, result)
        print(json.dumps(dict(event="evaluated" if scores else "stored", dataset=args.dataset,
            arm=args.arm, stage=stage, score=row["score"])), flush=True)
        parent = memory
        gc.collect()
        torch.cuda.empty_cache()
    result["status"] = "complete"
    save(path, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/pc/research/TAG_condensation"))
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--order-seed", type=int, default=9101)
    parser.add_argument("--construction-seed", type=int, default=271006)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    run(args)


if __name__ == "__main__":
    main()
