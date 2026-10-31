# External Baseline Fidelity Audit

本表核对的是方法身份与协议可比性，不把机制适配结果写成官方复现。
严格主表的默认要求：current-delta-only、不可读取历史 raw、相同 memory budget、相同 Prompt Student/文本编码器、同一 Student seed 集、同一 report/test 划分。

本轮没有重新从 GitHub 在线拉取 commit；固定 commit、论文链接和审计结论来自已交付的 baseline handoff package。由于当前网络代理无法连接 GitHub，本表不新增“在线复核通过”的声明。

## Classification

| Method | Category | Strict status | Original mechanism | Current adaptation | Data access | Memory/topology | Student | Seeds/orders | Caveat |
|---|---|---|---|---|---|---|---|---|---|
| CaT-MMD-TAG | mechanism-adapted | eligible | class-balanced MMD condensation of synthetic features against current raw/teacher statistics | upstream CaT construction and MMD objective adapted to fixed 5% TAG stream | current delta only; no cumulative raw in constructor | edge-free synthetic rows with TAG text | canonical scratch GCN/MiniLM/bidirectional InfoNCE | seeds=150, 151, 152, 153, 154, 155; orders=order19611, order19612, order19613, order19614, order19615 | none recorded |
| external_continual_adapters_card.json | unclassified | conditional | not recorded | TAG text attachment, 5% cumulative stream, shared scratch InfoNCE replaces upstream supervised/distilled student, GECC hard variant and current-delta restriction | not recorded | CaT/GECC edge-free synthetic rows; k-center/medoid real-node coreset; semantic-recursive sealed artifact | two-layer GCN, frozen MiniLM/text projection, bidirectional InfoNCE, scratch per stage | seeds=150, 151, 152, 153, 154, 155; orders=cora, citeseer, order19611, order19612, order19613, order19614, order19615 (CaT/k-center) | current-delta locality not explicit |
| GCond | reference-only | exclude | gradient matching over synthetic features and parametrized adjacency | not recorded | repository readable; no test labels used in this verification | not recorded | not recorded | seeds=not recorded; orders=not recorded | current-delta locality not explicit; reference-only classification |
| GCondenser | reference-only | exclude | benchmark toolkit wrapping multiple graph condensation methods with Hydra and DGL/PyG backends | not recorded | repository readable; no test labels used in this verification | not recorded | not recorded | seeds=not recorded; orders=not recorded | current-delta locality not explicit; reference-only classification |
| GECC-hard-TAG | mechanism-adapted | eligible | hard hop-weight propagation followed by Lloyd-style structure-free clustering | hard hop-weight branch, sklearn Lloyd, fixed TAG text attachment | current delta only; cumulative-raw and fuzzy variants excluded from strict table | edge-free synthetic rows with TAG text | canonical scratch GCN/MiniLM/bidirectional InfoNCE | seeds=150, 151, 152, 153, 154, 155; orders=order19611, order19612, order19613 partial, order19614 cora partial | none recorded |
| Official GraphCLIP frozen checkpoint | reference-only | exclude | official pretrained graph-text model inference, not a condensation constructor | local deterministic dev split and stage slices, official checkpoint evaluated as an upper/reference model, not continual training and not a current-delta adapter | complete original TAG used for transductive ego-subgraph features | none; no synthetic memory constructed | none; training_steps=0 and model state unchanged | seeds=; orders=not recorded | current-delta locality not explicit; reference-only classification |
| k-center/FPS | mechanism-adapted | eligible | farthest-point/k-center coreset selection in the feature/text space | TAG text and fixed cumulative 5% stream | current legal pool for farthest-point selection; no historical raw constructor access | real-node coreset selected in joint feature/text space | canonical scratch GCN/MiniLM/bidirectional InfoNCE | seeds=150, 151, 152, 153, 154, 155; orders=order19611, order19612, order19613, order19614, order19615 | none recorded |
| medoid | mechanism-adapted | eligible | medoid coreset selection in the feature/text space | TAG text and fixed cumulative 5% stream | current legal pool for medoid selection; no historical raw constructor access | real-node medoid coreset selected from joint feature/text space | canonical scratch GCN/MiniLM/bidirectional InfoNCE | seeds=150, 151, 152, 153, 154, 155; orders=order19611, order19612, order19613 partial, order19614 cora partial | none recorded |
| PUMA-inspired TAG adaptation | mechanism-adapted | eligible | pseudo-labeling, confidence filtering, one-time normalized propagation, edge-free memory, scratch retraining per stage | GraphCLIP Cora/Citeseer instead of official PUMA data, frozen MiniLM plus TAG feature fusion, InfoNCE GCN Student instead of PUMA classifier, current-delta teacher/candidate split, cumulative 5% memory budget | current delta plus previous synthetic parent only; no historical raw graph/text/edge access | edge-free synthetic rows with TAG text; self-loops only at Student evaluation | canonical two-layer GCN, frozen MiniLM, bidirectional InfoNCE, scratch per stage | seeds=150, 151, 152, 153, 154, 155; orders=datasets=cora, citeseer; orders=order19611, order19612; stages=3 | none recorded |
| semantic-recursive | mechanism-adapted | conditional | recursive semantic append of synthetic memory with class-aware text/topology payloads | shared TAG protocol and student replace upstream evaluation | sealed stage artifact and current protocol stream; no test labels | sealed recursive synthetic artifact with TAG text | canonical scratch GCN/MiniLM/bidirectional InfoNCE | seeds=150, 151, 152, 153, 154, 155; orders=order19611, order19612 | current-delta locality not explicit |
| SFGC | reference-only | exclude | structure-free graph condensation by trajectory-buffer imitation and feature distillation | not recorded | repository readable; no test labels used in this verification | not recorded | not recorded | seeds=not recorded; orders=not recorded | current-delta locality not explicit; reference-only classification |
| STAD | mechanism-adapted | supplementary | not recorded | official STAD components loaded from pinned checkout, test scoring disabled by AST extraction, canonical TAG student and protocol | not recorded | STAD condensed features, adjacency and generated text | canonical GraphCLIP transfer student; local Qwen text generation | seeds=not recorded; orders=cora, citeseer, 3 stages, student seeds 100-105 | current-delta locality not explicit; card does not declare strict eligibility |
| TAGSAM | mechanism-adapted | supplementary | text-attributed graph condensation with graph/text-aware synthetic memory | canonical encoders, dispersion/concat initialization, fixed-rate continual blocks, 100 outer steps and 10 inner updates | not recorded | synthetic node features with self-loop or semantic topology | canonical two-layer GCN, frozen MiniLM, bidirectional InfoNCE, scratch per stage | seeds=100, 101, 102, 103, 104, 105; orders=cora, citeseer, order19611, order19612 | current-delta locality not explicit; card does not declare strict eligibility |

## Interpretation

- `eligible` 只表示 card 中记录的适配版本满足进入严格比较的元数据条件，不表示它是官方 native reproduction。
- `conditional` 表示方法机制可比较，但种子、顺序覆盖或访问描述仍不完整；应进入补充表或补跑后再进主表。
- `reference-only` 不参与严格排序，包括静态/离线方法、官方 checkpoint 或 Student/loss 不同的方法。
- CaT、GECC、PUMA、k-center 和 medoid 的当前结果属于 mechanism-adapted；TAGSAM/STAD 有额外的 seed/order 警告；GCond/SFGC/GCondenser/GraphCLIP 等 reference-only 不应和 Unified-Q 直接横排。

## Per-card Detail

### CaT-MMD-TAG

- Card: `cat_mmd_tag_card.json`
- Paper/repository: CaT: Balanced Continual Graph Learning with Graph Condensation; upstream `superallen13/CaT-CGL@43d6fcfd7b1754fe53b9f911492d7cb0c113a2eb`
- Mechanism: class-balanced MMD condensation of synthetic features against current raw/teacher statistics
- Adaptation: upstream CaT construction and MMD objective adapted to fixed 5% TAG stream
- Access/memory: current delta only; no cumulative raw in constructor; edge-free synthetic rows with TAG text
- Student: canonical scratch GCN/MiniLM/bidirectional InfoNCE
- Audits: source_locality=passed; current-delta protocol; budget=passed; prefix=passed; checkpoint_reload=passed
- Strict status: **eligible**

### external_continual_adapters_card.json

- Card: `external_continual_adapters_card.json`
- Paper/repository: External continual adapter bundle; upstream `CaT=superallen13/CaT-CGL@43d6fcfd7b1754fe53b9f911492d7cb0c113a2eb; GECC=rockcor/GECC@d731be00cc3beb7585e8fa7b60ca21a6c3cb4e3f`
- Mechanism: not recorded
- Adaptation: TAG text attachment, 5% cumulative stream, shared scratch InfoNCE replaces upstream supervised/distilled student, GECC hard variant and current-delta restriction
- Access/memory: not recorded; CaT/GECC edge-free synthetic rows; k-center/medoid real-node coreset; semantic-recursive sealed artifact
- Student: two-layer GCN, frozen MiniLM/text projection, bidirectional InfoNCE, scratch per stage
- Audits: source_locality=protocol records physical fixture limitation; budget=results include memory_rows; prefix=existing per-stage artifacts; checkpoint_reload=independent audit passed for all formal continual roots; order19615 maximum metric error 0.0
- Strict status: **conditional**

### GCond

- Card: `gcond_reference_only_card.json`
- Paper/repository: Graph Condensation for Graph Neural Networks; upstream `https://github.com/ChandlerBang/GCond`
- Mechanism: gradient matching over synthetic features and parametrized adjacency
- Adaptation: not recorded
- Access/memory: repository readable; no test labels used in this verification; not recorded
- Student: not recorded
- Audits: source_locality=repository-only; budget=not-applicable; prefix=not-applicable; checkpoint_reload=not-applicable
- Strict status: **exclude**

### GCondenser

- Card: `gcondenser_reference_only_card.json`
- Paper/repository: GCondenser: Benchmarking Graph Condensation; upstream `https://github.com/superallen13/GCondenser`
- Mechanism: benchmark toolkit wrapping multiple graph condensation methods with Hydra and DGL/PyG backends
- Adaptation: not recorded
- Access/memory: repository readable; no test labels used in this verification; not recorded
- Student: not recorded
- Audits: source_locality=repository-only; budget=not-applicable; prefix=not-applicable; checkpoint_reload=not-applicable
- Strict status: **exclude**

### GECC-hard-TAG

- Card: `gecc_hard_tag_card.json`
- Paper/repository: GECC: Scalable and Structure-free Graph Condensation with Evolving Capabilities; upstream `rockcor/GECC@d731be00cc3beb7585e8fa7b60ca21a6c3cb4e3f`
- Mechanism: hard hop-weight propagation followed by Lloyd-style structure-free clustering
- Adaptation: hard hop-weight branch, sklearn Lloyd, fixed TAG text attachment
- Access/memory: current delta only; cumulative-raw and fuzzy variants excluded from strict table; edge-free synthetic rows with TAG text
- Student: canonical scratch GCN/MiniLM/bidirectional InfoNCE
- Audits: source_locality=passed; current delta only; budget=passed; prefix=passed; checkpoint_reload=passed
- Strict status: **eligible**

### Official GraphCLIP frozen checkpoint

- Card: `graphclip_reference_only_card.json`
- Paper/repository: GraphCLIP: Enhancing Transferability in Graph Foundation Models for Text-Attributed Graphs; upstream `local official source tree: /home/pc/research/GraphCLIP`
- Mechanism: official pretrained graph-text model inference, not a condensation constructor
- Adaptation: local deterministic dev split and stage slices, official checkpoint evaluated as an upper/reference model, not continual training and not a current-delta adapter
- Access/memory: complete original TAG used for transductive ego-subgraph features; none; no synthetic memory constructed
- Student: none; training_steps=0 and model state unchanged
- Audits: source_locality=reference-only; complete original graph features; budget=not-applicable; prefix=not-applicable; checkpoint_reload=strict graph/text load and unchanged state recorded
- Strict status: **exclude**

### k-center/FPS

- Card: `k_center_native_card.json`
- Paper/repository: k-center/Farthest Point Sampling coreset reference; upstream `not recorded`
- Mechanism: farthest-point/k-center coreset selection in the feature/text space
- Adaptation: TAG text and fixed cumulative 5% stream
- Access/memory: current legal pool for farthest-point selection; no historical raw constructor access; real-node coreset selected in joint feature/text space
- Student: canonical scratch GCN/MiniLM/bidirectional InfoNCE
- Audits: source_locality=passed; current legal pool only; budget=passed; prefix=passed; checkpoint_reload=passed
- Strict status: **eligible**

### medoid

- Card: `medoid_native_card.json`
- Paper/repository: Medoid coreset reference; upstream `not recorded`
- Mechanism: medoid coreset selection in the feature/text space
- Adaptation: TAG text and fixed cumulative 5% stream
- Access/memory: current legal pool for medoid selection; no historical raw constructor access; real-node medoid coreset selected from joint feature/text space
- Student: canonical scratch GCN/MiniLM/bidirectional InfoNCE
- Audits: source_locality=passed; current legal pool only; budget=passed; prefix=passed; checkpoint_reload=passed
- Strict status: **eligible**

### PUMA-inspired TAG adaptation

- Card: `puma_adapter_card.json`
- Paper/repository: PUMA: Efficient Continual Graph Learning with Graph Condensation; upstream `superallen13/PUMA@9e4e87f53db0da6aacad4a0a9d08f1c2970413d0`
- Mechanism: pseudo-labeling, confidence filtering, one-time normalized propagation, edge-free memory, scratch retraining per stage
- Adaptation: GraphCLIP Cora/Citeseer instead of official PUMA data, frozen MiniLM plus TAG feature fusion, InfoNCE GCN Student instead of PUMA classifier, current-delta teacher/candidate split, cumulative 5% memory budget
- Access/memory: current delta plus previous synthetic parent only; no historical raw graph/text/edge access; edge-free synthetic rows with TAG text; self-loops only at Student evaluation
- Student: canonical two-layer GCN, frozen MiniLM, bidirectional InfoNCE, scratch per stage
- Audits: source_locality=passed per stage manifest; no historical raw graph/text/edge access; budget=passed; cumulative five-percent rows; prefix=passed by stage manifest; checkpoint_reload=passed in formal runner audits
- Strict status: **eligible**

### semantic-recursive

- Card: `semantic_recursive_card.json`
- Paper/repository: Semantic-Recursive Graph Condensation; upstream `not recorded`
- Mechanism: recursive semantic append of synthetic memory with class-aware text/topology payloads
- Adaptation: shared TAG protocol and student replace upstream evaluation
- Access/memory: sealed stage artifact and current protocol stream; no test labels; sealed recursive synthetic artifact with TAG text
- Student: canonical scratch GCN/MiniLM/bidirectional InfoNCE
- Audits: source_locality=passed; sealed artifact and current stream; budget=passed; prefix=passed; checkpoint_reload=passed
- Strict status: **conditional**

### SFGC

- Card: `sfgc_reference_only_card.json`
- Paper/repository: Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data; upstream `https://github.com/Amanda-Zheng/SFGC`
- Mechanism: structure-free graph condensation by trajectory-buffer imitation and feature distillation
- Adaptation: not recorded
- Access/memory: repository readable; no test labels used in this verification; not recorded
- Student: not recorded
- Audits: source_locality=repository-only; budget=not-applicable; prefix=not-applicable; checkpoint_reload=not-applicable
- Strict status: **exclude**

### STAD

- Card: `stad_baseline_card.json`
- Paper/repository: Semi-Supervised Text-Attributed Graph Distillation; upstream `https://github.com/laiyurui/STAD-Semi-Supervised-Text-Attributed-Graph-Distillation`
- Mechanism: not recorded
- Adaptation: official STAD components loaded from pinned checkout, test scoring disabled by AST extraction, canonical TAG student and protocol
- Access/memory: not recorded; STAD condensed features, adjacency and generated text
- Student: canonical GraphCLIP transfer student; local Qwen text generation
- Audits: source_locality=protocol/acceptance recorded; budget=acceptance rows checked; prefix=acceptance recorded; checkpoint_reload=independent acceptance passed
- Strict status: **supplementary**

### TAGSAM

- Card: `tagsam_baseline_card.json`
- Paper/repository: TAGSAM: Text-Attributed Graph Condensation; upstream `https://github.com/SundayVHan/TAGSAM`
- Mechanism: text-attributed graph condensation with graph/text-aware synthetic memory
- Adaptation: canonical encoders, dispersion/concat initialization, fixed-rate continual blocks, 100 outer steps and 10 inner updates
- Access/memory: not recorded; synthetic node features with self-loop or semantic topology
- Student: canonical two-layer GCN, frozen MiniLM, bidirectional InfoNCE, scratch per stage
- Audits: source_locality=recorded in manifest; budget=recorded in manifest; prefix=recorded in construction artifacts; checkpoint_reload=independent reload passed for 72 fits per dataset; maximum metric error 0.0
- Strict status: **supplementary**
