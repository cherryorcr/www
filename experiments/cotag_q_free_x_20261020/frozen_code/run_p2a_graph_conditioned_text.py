"""P2-A text-only mechanism experiment; frozen v4 and P3-B remain unchanged."""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import p2a_graph_text as T
import p2_text_operators as BT
import run_small_comparison_audited as C
import run_small_order_cost_confirmation as O
import run_p3b_topology_diagnostic as B

U, G, SCA, P = C.U, C.G, C.SCA, C.P
FIELDS = ("x", "text_emb", "texts", "mass", "depth", "stage")


class MeteredEncoder:
    def __init__(self, encoder, device):
        self.encoder, self.device = encoder, device
        self.tokenizer = encoder.tokenizer
        self.seconds, self.sequences = 0., 0

    def encode(self, texts):
        B.sync(self.device)
        started = time.perf_counter()
        result = self.encoder.encode(texts)
        B.sync(self.device)
        self.seconds += time.perf_counter() - started
        self.sequences += len(texts)
        return result


def stage_orders(dataset, names):
    orders = B.stage_orders(dataset)
    sizes = [2, 2, 3] if dataset == "cora" else [2, 2, 2]
    for seed in (1703, 1704):
        classes = list(range(sum(sizes)))
        random.Random(seed).shuffle(classes)
        groups, offset = [], 0
        for size in sizes:
            groups.append(sorted(classes[offset:offset + size]))
            offset += size
        orders[f"order{seed}"] = groups
    return {name: P.canonicalize_class_groups(orders[name]) for name in names}


def legal_delta(data, ids):
    if len(ids) == 0 or not data["pool_mask"][ids].all():
        raise ValueError("nonconstruction source requested")
    if len(ids.unique()) != len(ids):
        raise ValueError("duplicate source ID")
    return dict(x=data["x"][ids].cpu(), text_emb=data["text_emb"][ids].cpu(),
                y=data["y"][ids].cpu(), texts=[data["texts"][int(i)] for i in ids],
                edge_index=C.induced_edges(data["edge_index"], ids, len(data["x"])))


def shared_clusters(delta, rows, stage, encoder, device):
    """Reproduce legacy per-class Mean-X/concat before varying text targets."""
    ids = torch.arange(len(delta["x"]))
    classes = sorted(delta["y"].unique().tolist())
    allocation = U.allocate_rows_by_class(delta, ids, classes, rows, "dispersion")
    clusters, xs, legacy_texts, legacy_embs = [], [], [], []
    joint = torch.cat((F.normalize(delta["x"].float(), dim=1), delta["text_emb"]), dim=1)
    for cls in classes:
        local = ids[delta["y"] == cls]
        assign = SCA.cluster_assign(joint[local], torch.ones(len(local)), allocation[cls], device)
        class_texts = []
        for cluster in range(int(assign.max()) + 1):
            members = local[assign == cluster]
            clusters.append(members)
            xs.append(delta["x"][members].float().mean(0))
            class_texts.append(G.fuse_new_only(joint, members, delta["texts"], budget_chars=320, g=5))
        legacy_texts.extend(class_texts)
        legacy_embs.append(encoder.encode(class_texts))
    block = dict(x=torch.stack(xs), text_emb=torch.cat(legacy_embs), texts=legacy_texts,
                 mass=torch.tensor([float(len(c)) for c in clusters]),
                 depth=torch.zeros(len(clusters), dtype=torch.long),
                 stage=torch.full((len(clusters),), stage, dtype=torch.long))
    if len(clusters) != rows:
        raise AssertionError("cluster count differs from row budget")
    block["edge_index"] = SCA.memory_edges(block, device)
    return block, joint, clusters, allocation


def append_fixed_edges(old, block, reference):
    memory = {key: (block[key] if old is None else
                   old[key] + block[key] if key == "texts" else
                   torch.cat((old[key], block[key]))) for key in FIELDS}
    memory["edge_index"] = reference["edge_index"].clone()
    C.assert_frozen(old, memory)
    for key in ("x", "mass", "depth", "stage", "edge_index"):
        if not torch.equal(memory[key], reference[key]):
            raise AssertionError(f"text-only control failed: {key}")
    if old is not None and not torch.equal(old["depth"], memory["depth"][:len(old["x"])]):
        raise AssertionError("historical depth changed")
    return memory


def fit_or_load(memory, data, stage, seed, arm, out, device, steps_override):
    path = out / f"{arm}_s{stage}_seed{seed}.json"
    checkpoint = path.with_suffix(".pt")
    steps = P.student_steps(len(memory["x"]), override=steps_override)
    if path.exists() and checkpoint.exists():
        row = json.loads(path.read_text())
        if (row["support_sha256"] != C.tensor_hash(memory) or row["steps"] != steps
                or row["seed"] != seed):
            raise ValueError("cached student fit mismatch")
    else:
        started = time.perf_counter()
        row = C.fit(memory, seed, steps, device, data, stage, checkpoint=checkpoint)
        B.sync(device)
        row["wall_seconds"] = time.perf_counter() - started
        C.save_json(path, row)
    O.verify_fit(memory, row, checkpoint, data, stage, device)
    return row


def run_order(data, name, groups, args, plan_hash):
    out = args.output / name
    out.mkdir(parents=True, exist_ok=True)
    stages, budgets = O.set_stages(data, groups)
    O.lock_json(out / "manifest.json", dict(plan_sha256=plan_hash, stage_classes=groups,
        budgets=budgets, counts=list(map(len, stages)),
        source_ids_sha256=C.tensor_hash({str(s): ids for s, ids in enumerate(stages)})))
    memories, reference_old, results, costs = {}, None, [], []
    for stage, ids in enumerate(stages):
        source = legal_delta(data, ids)
        encoder = MeteredEncoder(data["encoder"], args.device)
        started = time.perf_counter()
        keep = budgets[stage - 1] if stage else 0
        block, joint, clusters, allocation = shared_clusters(
            source, budgets[stage] - keep, stage, encoder, args.device)
        reference = SCA.append_memory(reference_old, block, keep, device=args.device)
        cluster_seconds = time.perf_counter() - started
        archive = args.reference_root / data["dataset"] / name / f"mean_x_s{stage}_memory.pt"
        if archive.exists():
            previous = torch.load(archive, map_location="cpu", weights_only=False)
            if C.tensor_hash(previous) != C.tensor_hash(reference) or previous["texts"] != reference["texts"]:
                raise AssertionError("legacy shared X/text/topology regression vs P1-X")
        elif name in {"canonical", "order1701", "order1702"}:
            raise FileNotFoundError(archive)
        shared_start, encode_start = time.perf_counter(), encoder.seconds
        bank = T.snippet_bank(source["texts"], encoder)
        query_start = time.perf_counter()
        queries = T.make_queries(source["x"], source["text_emb"], clusters,
                                 source["edge_index"], 9001 + 1009 * stage)
        shared = dict(stage=stage, cluster_legacy_topology_seconds=cluster_seconds,
                      bank_query_seconds=time.perf_counter() - shared_start,
                      query_seconds=time.perf_counter() - query_start,
                      bank_encoding_seconds=encoder.seconds - encode_start,
                      source_snippets=len(bank["texts"]), source_content_tokens=bank["content_tokens"],
                      source_tokens_with_special=bank["tokens_with_special"],
                      source_ids=ids.tolist(), clusters_global_ids=[ids[c].tolist() for c in clusters],
                      allocation=allocation, graph_query_stats=queries["stats"],
                      shuffled_permutation=queries["permutation"].tolist(),
                      local_edge_sha256=T.tensor_sha(source["edge_index"]),
                      reference_sha256=C.tensor_hash(reference),
                      historical_raw_reads=0, holdout_source_reads=0,
                      query_sha256={k: T.tensor_sha(queries[k]) for k in
                                    ("attribute", "text", "graph", "shuffled", "neighbors")})
        shared_path = out / f"shared_s{stage}.json"
        if shared_path.exists():
            archived = json.loads(shared_path.read_text())
            for key in ("reference_sha256", "query_sha256", "clusters_global_ids", "local_edge_sha256"):
                if archived[key] != shared[key]:
                    raise AssertionError(f"resumed construction mismatch: {key}")
            shared = archived
        else:
            C.save_json(shared_path, shared)
        stage_results = []
        for arm in args.arms:
            memory_path = out / f"{arm}_s{stage}_memory.pt"
            audit_path = out / f"{arm}_s{stage}_text.json"
            if memory_path.exists() and audit_path.exists():
                memory = torch.load(memory_path, map_location="cpu", weights_only=False)
                audit = json.loads(audit_path.read_text())
                if C.sha(memory_path) != audit["file_sha256"]:
                    raise AssertionError("resumed memory file changed")
                checked_block = {k: memory[k][keep:] for k in FIELDS}
                checked = append_fixed_edges(memories.get(arm), checked_block, reference)
                if C.tensor_hash(memory) != C.tensor_hash(checked):
                    raise AssertionError("cached memory control mismatch")
            else:
                texts, targets, row_audits = [], [], []
                started, encode_start = time.perf_counter(), encoder.seconds
                for row, members in enumerate(clusters):
                    text, target, row_audit = T.construct(
                        arm, joint, source["text_emb"], members, source["texts"], encoder,
                        bank, queries, row)
                    texts.append(text)
                    targets.append(target)
                    row_audits.append(row_audit)
                varied = dict(block, texts=texts, text_emb=torch.stack(targets))
                memory = append_fixed_edges(memories.get(arm), varied, reference)
                B.sync(args.device)
                seconds = time.perf_counter() - started
                torch.save(memory, memory_path)
                audit = dict(stage=stage, arm=arm, rows=row_audits,
                             aggregate=BT.aggregate_audit(row_audits),
                             arm_text_seconds=seconds, encoding_seconds=encoder.seconds - encode_start,
                             non_encoding_seconds=max(0., seconds - (encoder.seconds - encode_start)),
                             file_sha256=C.sha(memory_path), support_sha256=C.tensor_hash(memory),
                             control_sha256=C.tensor_hash({k: memory[k] for k in
                                 ("x", "mass", "depth", "stage", "edge_index")}),
                             payload=O.payload_stats(memory))
                if audit["aggregate"]["budget_violations"]:
                    raise AssertionError("budget audit failed")
                C.save_json(audit_path, audit)
            seed_rows = [fit_or_load(memory, data, stage, seed, arm, out, args.device, args.steps)
                         for seed in args.seeds]
            result = dict(arm=arm, stage=stage, rows=len(memory["x"]), per_seed=seed_rows,
                          metrics=C.aggregate(seed_rows))
            results.append(result)
            costs.append(dict(arm=arm, stage=stage, shared=shared,
                              text_seconds=audit["arm_text_seconds"], encoding_seconds=audit["encoding_seconds"],
                              student_seconds=sum(r["seconds"] for r in seed_rows),
                              student_steps=sum(r["steps"] for r in seed_rows), payload=audit["payload"]))
            memories[arm] = memory
            stage_results.append(result)
            C.save_json(out / "progress.json", dict(results=results, costs=costs))
            print(json.dumps(dict(event="stage_done", dataset=data["dataset"], order=name,
                                  stage=stage, arm=arm,
                                  f1=result["metrics"]["all"]["cumulative_f1"]["mean"])), flush=True)
        # Paired initializations are an invariant, not inferred from seed labels.
        for i in range(len(args.seeds)):
            if len({r["per_seed"][i]["start_state_sha256"] for r in stage_results}) != 1:
                raise AssertionError("between-arm student initialization mismatch")
        reference_old = reference
        del source, bank, queries
    C.save_json(out / "results.json", dict(results=results, costs=costs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default=str(Path(U.__file__).parent / "minilm_from_graphclip"))
    ap.add_argument("--reference-root", type=Path, default=C.RESULTS / "p1x_x_operator_20260918")
    ap.add_argument("--arms", nargs="+", choices=T.ARMS, default=list(T.ARMS))
    ap.add_argument("--orders", nargs="+", default=["canonical", "order1701", "order1702"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(100, 106)))
    ap.add_argument("--steps", type=int, default=None, help="smoke override; never canonical evidence")
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    orders = stage_orders(args.dataset, args.orders)
    plan = dict(version="p2a-graph-text-v1", dataset=args.dataset, orders=orders,
                arms=args.arms, seeds=args.seeds, steps_override=args.steps,
                token_cap=T.BUDGET, source_token_cap=T.SOURCE_CAP,
                snippet_tokens=T.SNIPPET_TOKENS, g=T.MAX_PARTS,
                temperature=T.TEMPERATURE, relevance=T.RELEVANCE,
                rate=.05, allocation="dispersion", x="mean", response=False,
                topology="shared legacy joint_g5_c320 frozen-old cross-degree=1",
                text_encoder="frozen MiniLM; no trained query projection",
                attribute_query="softmax(cos(X_i,meanX)/0.2) weighted source text embeddings",
                graph_query="normalize(attribute_query + unique legal external-neighbor text mean)",
                shuffle="deterministic derangement of neighbor summaries; own attribute query unchanged",
                access="current delta induced graph only; no historical raw or holdout text for construction",
                evaluation="report-dev, all prompts primary; seen prompts secondary; full transductive graph",
                test_labels_evaluated=False, report_used_for_training=False,
                construction_labels="class partition and dispersion allocation only",
                cost_scope="shared cached source text encoding excluded; bank/query costs charged separately",
                trainable_text_parameters=0, classification="smoke" if args.steps else "known-order diagnostic",
                promotion="requires independent orders1703/1704 and student seeds110..115; no dev retuning",
                screening_gate=dict(final_gain_each_dataset=.5, stage_mean_gain_each_dataset=0.,
                    min_order_final_gain=-2., min_old_new_gain=-2.,
                    controls=["center_concat", "text_selection", "attribute_selection", "shuffled_graph"]),
                student=dict(batch=P.STUDENT.batch, lr=P.STUDENT.lr, steps="max(50,min(2048,16*rows))"),
                source_sha256=C.sha(U.DATA_CFG[args.dataset]["path"]),
                text_cache_sha256=C.sha(C.artifact_dir(args.dataset, "v4") / "text_emb_pool_only.pt"),
                minilm_sha256={p.name: C.sha(p) for p in Path(args.minilm).glob("*") if p.is_file()},
                python=platform.python_version(), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(args.device),
                code_sha256={Path(p).name: C.sha(p) for p in
                    (__file__, T.__file__, BT.__file__, C.__file__, O.__file__, B.__file__,
                     U.__file__, G.__file__, SCA.__file__, P.__file__,
                     Path(C.__file__).with_name("small_comparison_metrics.py"))})
    O.lock_json(args.output / "plan.json", plan)
    data = B.prepare(args.dataset, args.minilm, args.device, args.output)
    for name, groups in orders.items():
        run_order(data, name, groups, args, C.sha(args.output / "plan.json"))
    C.save_json(args.output / "acceptance.json", dict(complete=True,
        scope=plan["classification"], fits=len(orders)*len(args.arms)*3*len(args.seeds),
        checkpoint_reload_verified=True, paired_initialization_verified=True,
        fixed_x_rows_edges=True, old_text_frozen=True, test_labels_evaluated=False))
    print("P2A_DONE", flush=True)


if __name__ == "__main__":
    main()
