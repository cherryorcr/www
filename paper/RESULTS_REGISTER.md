# Current Experiment Results Register

Updated: 2026-10-15

The paper uses ArXiv, SportsFit, and BookHis as the three main large-graph datasets. Cora, Citeseer, and WikiCS provide supplementary scale/domain checks. Results are separated into matched mechanism adaptations, qualified references, historical comparisons, and internal controls.

The paper-facing name is **CoTAG-Q: Learning Graph--Text Support for History-Constrained Continual TAG Condensation**. This renames the edge-free Unified-Q-EF implementation without changing its algorithm. CoTAG-Q-T denotes the separate topology extension. Artifact paths, source identifiers, hashes, and explicitly historical implementation names remain unchanged.

## 1. Primary Exact-5% Nine-Cell Fixed-Mask Evaluation

Source: `../results/uq_final_evidence_report_20261013_132047/summary.md` (snapshot SHA-256 recorded there). On each dataset/method, three class orders x three construction seeds give nine matched cells; three Student seeds are averaged inside each cell. The common Prompt Student has 2048 steps and MiniLM text embeddings. Access is current-delta-only with a frozen edge-free synthetic prefix. Paired intervals use nine cell differences. The external rows are current-protocol mechanism adaptations, not official end-to-end reproductions. The evaluation masks have been inspected repeatedly: this is post-hoc fixed-mask evidence, not a fresh blind/sealed test. The intervals describe sampled order/seed variation conditional on these masks and do not correct for adaptive method selection.

The current legal-pool 5% targets round to 5,927, 6,057, and 1,454 final rows on ArXiv, SportsFit, and BookHis, respectively. The paper's three-dataset paired figure at `figures/confirmation_paired.pdf` is a source-backed redraw from `pairs.csv` in the same immutable snapshot; only presentation labels were updated to CoTAG-Q and fixed-mask wording. It replaces the ArXiv-only figure in the main narrative; the older image remains in the repository for provenance.

| Dataset | Method | Final F1 | Cell SD | Acc. | Old F1 | New F1 | Stage F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ArXiv | CoTAG-Q | 52.91 | 0.73 | 71.15 | 52.25 | 54.27 | 62.23 |
| ArXiv | GECC-hard-TAG | 52.68 | 0.45 | 70.94 | 51.45 | 55.24 | 62.34 |
| ArXiv | CaT-CCTP | 34.98 | 1.79 | 60.41 | 33.77 | 37.50 | 47.36 |
| SportsFit | CoTAG-Q | 80.08 | 0.60 | 85.52 | 80.00 | 80.51 | 82.85 |
| SportsFit | GECC-hard-TAG | 80.57 | 1.16 | 89.01 | 80.57 | 80.58 | 83.59 |
| SportsFit | CaT-CCTP | 74.50 | 2.14 | 85.56 | 74.31 | 75.52 | 79.50 |
| BookHis | CoTAG-Q | 48.81 | 1.57 | 81.05 | 48.39 | 53.45 | 51.82 |
| BookHis | GECC-hard-TAG | 45.34 | 0.74 | 81.26 | 45.48 | 43.86 | 50.43 |
| BookHis | CaT-CCTP | 29.16 | 1.05 | 79.23 | 29.16 | 29.09 | 34.87 |

| Paired F1 difference (CoTAG-Q minus) | Mean pp | 95% CI | Positive | Worst cell pp |
|---|---:|---|---:|---:|
| ArXiv / GECC-hard | +0.22 | [-0.34, 0.79] | 4/9 | -0.47 |
| SportsFit / GECC-hard | -0.49 | [-1.50, 0.52] | 2/9 | -2.14 |
| BookHis / GECC-hard | +3.47 | [2.20, 4.74] | 9/9 | +0.09 |
| ArXiv / CaT-CCTP | +17.93 | [16.81, 19.05] | 9/9 | +15.92 |
| SportsFit / CaT-CCTP | +5.58 | [3.77, 7.39] | 9/9 | +2.22 |
| BookHis / CaT-CCTP | +19.66 | [18.38, 20.93] | 9/9 | +17.63 |

BookHis is the only dataset with a positive nine-cell interval against GECC-hard. SportsFit Accuracy favors GECC-hard by 3.49 points. Construction time and peak CUDA allocated GiB for CoTAG-Q / GECC-hard / CaT-CCTP are ArXiv 243.3 / 189.5 / 76.1 seconds and 17.33 / 0.93 / 0.94 GiB; SportsFit 170.3 / 735.1 / 174.6 seconds and 24.08 / 0.98 / 1.05 GiB; BookHis 68.9 / 52.0 / 28.6 seconds and 2.91 / 0.49 / 0.46 GiB. Times exclude loading and common text-cache preparation and were measured with concurrent jobs; they are not isolated throughput benchmarks.

### Separate fidelity references and rejected replacement

GECC-soft with native FCM completed one BookHis cell; the user stopped its ArXiv and SportsFit runs. Those two cells are intentionally absent and must not be described as queued or filled with hard-branch values. Bounded TAGSAM and local-Qwen STAD completed all three large graphs but have reduced unrolling or altered supervision/generator, respectively. They remain qualified references, not official end-to-end reproductions. Their current fixed-mask results appear in Section 2; earlier development values are not substituted into that table.

The single joint graph-text DM replacement is **not** the main method. One matched cell per dataset gives report-development F1 deltas against CoTAG-Q of +0.45 (ArXiv), +0.15 (SportsFit), and -2.65 (BookHis) points. Post-hoc fixed-mask deltas are -0.21, -0.84, and -1.15 points, respectively. The source is `../results/uq_joint_replacement_20261013/summary.md`; these readings are not an independent confirmation of a new candidate.

## 2. Current Five-Dataset Baseline Expansion

Source: `../results/uq_baseline_expansion_20261013/report_expansion_utc_20261014_021125_595059/summary_zh.md`; receipt: `../results/uq_baseline_expansion_20261013/latest_report_receipt.json`. The ten primary method entries completed 50/50 dataset combinations. Including qualified references, 57/59 registered combinations completed; the remaining two GECC-soft runs were stopped. Immutable report queue markers reflect collection time, not live authorization to resume them.

This panel uses one order (9101), one construction seed (271006), and three Student seeds (150/151/152). All primary entries use the common current-delta-only, frozen-prefix, cumulative legal-pool 5% schedule, MiniLM encoder, and Prompt GCN budget (2048 steps, batch 24). Construction budgets differ and are documented in the source. Values below are Student means, not means over construction seeds or orders. The one-cell panel must remain separate from Section 1's nine-cell evidence.

| Method | ArXiv F1/Acc. | SportsFit F1/Acc. | BookHis F1/Acc. | Cora F1/Acc. | Citeseer F1/Acc. |
|---|---:|---:|---:|---:|---:|
| CoTAG-Q | 53.20 / 71.05 | 80.35 / 86.32 | 49.24 / 82.01 | 87.24 / 87.82 | 70.14 / 74.40 |
| CaT-CCTP | 36.35 / 61.59 | 77.50 / 86.58 | 31.11 / 79.70 | 85.94 / 87.45 | 69.74 / 75.34 |
| GECC-hard | 52.98 / 71.15 | 78.85 / 88.71 | 45.71 / 81.03 | 85.37 / 86.59 | 67.15 / 73.25 |
| Random | 47.92 / 69.52 | 76.32 / 86.21 | 35.67 / 80.00 | 77.03 / 78.47 | 63.35 / 70.74 |
| K-center | 28.26 / 56.52 | 58.38 / 76.72 | 23.16 / 76.03 | 68.56 / 69.50 | 47.67 / 50.99 |
| Cluster-medoid | 46.87 / 69.06 | 77.33 / 87.33 | 35.92 / 80.81 | 81.74 / 84.26 | 66.37 / 72.52 |
| Herding | 49.20 / 69.80 | 74.07 / 85.27 | 36.86 / 80.69 | 80.71 / 82.29 | 67.78 / 74.09 |
| PUMA-inspired | 46.95 / 69.88 | 81.00 / 89.07 | 35.98 / 80.91 | 87.60 / 89.05 | 70.42 / 75.97 |
| GCond-x core | 49.09 / 67.80 | 67.75 / 64.44 | 30.03 / 75.54 | 83.17 / 84.13 | 67.09 / 74.40 |
| SFGC bounded core | 22.40 / 43.31 | 38.50 / 63.38 | 14.52 / 29.78 | 72.64 / 73.19 | 55.91 / 64.58 |

The panel contains nine comparison entries, not nine official graph-condensation reproductions: Random, K-center, Cluster-medoid, and Herding are selection controls; PUMA-inspired is an inspired reference; GCond-x and SFGC retain official cores within a changed continual wrapper. PUMA-inspired has higher F1 on SportsFit, Cora, and Citeseer in this panel. These results do not show universal dominance of CoTAG-Q.

| Qualified reference | ArXiv F1/Acc. | SportsFit F1/Acc. | BookHis F1/Acc. |
|---|---:|---:|---:|
| TAGSAM bounded core | 39.23 / 53.22 | 71.33 / 83.01 | 11.36 / 59.51 |
| STAD local-Qwen | 50.35 / 65.98 | 72.63 / 77.85 | 37.43 / 77.87 |
| GECC-soft native FCM | stopped | stopped | 40.54 / 74.26 |

TAGSAM reduces the official selector/response construction budget from 5000 x 15 to 100 x 10. STAD changes supervision, the LLM, and synthetic topology. SFGC has a bounded teacher/outer budget. The final BookHis GCond-x/SFGC stage has one proxy class and zero discriminative feature gradients. These limitations qualify low scores; the results do not establish superiority over complete native algorithms. Cora/Citeseer use the project's canonical split, not the Planetoid 20-label-per-class protocol.

## 3. Archived Nominal-Rate Development Table

The following older table uses the Prompt Student and a nominal 5% schedule but is superseded for primary claims. ArXiv has only one order and an actual 5.71% current-pool ratio. These rows must not be mixed with Section 1 or treated as an independent confirmation.

| Dataset | Method | Final Macro-F1 | Accuracy | Old F1 | New F1 | Orders |
|---|---|---:|---:|---:|---:|---:|
| ArXiv | Unified-Q edge-free | 52.25 | 71.59 | 51.04 | 54.77 | 1 |
| ArXiv | GECC-hard TAG adapter | 51.09 | 70.27 | 49.93 | 53.50 | 1 |
| ArXiv | CaT-CCTP adapter | 36.74 | 63.62 | 35.36 | 39.59 | 1 |
| SportsFit | Unified-Q edge-free | 78.60 | 83.93 | 78.58 | 78.74 | 3 |
| SportsFit | GECC-hard TAG adapter | 77.32 | 85.52 | 77.61 | 75.70 | 3 |
| SportsFit | CaT-CCTP adapter | 73.75 | 81.94 | 73.54 | 74.95 | 3 |
| BookHis | Unified-Q edge-free | 41.97 | 80.34 | 41.39 | 48.38 | 3 |
| BookHis | GECC-hard TAG adapter | 41.23 | 79.97 | 40.81 | 45.83 | 3 |
| BookHis | CaT-CCTP adapter | 24.83 | 75.45 | 24.47 | 28.75 | 3 |

## 4. Expanded Three-Dataset Coverage (Archived)

This panel contains more than ten named comparison entries per main dataset. Its first three rows refer to the archived development table above, not the current fixed-mask main table. The remaining rows are mechanism adaptations, coreset controls, historical controls, or reference-only comparisons and must retain their tier labels in the paper. Several rows also use an older Student and cannot be merged with the newer common-Student panel in Section 2.

| Method | ArXiv F1/Acc. | SportsFit F1/Acc. | BookHis F1/Acc. | Status |
|---|---:|---:|---:|---|
| Unified-Q edge-free | 52.25 / 71.59 | 78.60 / 83.93 | 41.97 / 80.34 | archived development |
| GECC-hard TAG adapter | 51.09 / 70.27 | 77.32 / 85.52 | 41.23 / 79.97 | archived development mechanism adapter |
| CaT-CCTP adapter | 36.74 / 63.62 | 73.75 / 81.94 | 24.83 / 75.45 | archived development mechanism adapter |
| PUMA-inspired TAG | 18.86 / 27.39 | 29.20 / 29.88 | 17.85 / 32.51 | mechanism adaptation / coverage |
| TAGSAM mechanism | 20.60 / 32.56 | 33.89 / 38.96 | 19.57 / 42.70 | mechanism adaptation / coverage |
| Random real coreset | 17.81 / 24.66 | 27.44 / 30.55 | 18.68 / 33.59 | legal coreset coverage |
| K-center/FPS | 19.43 / 30.69 | 29.42 / 32.01 | 20.04 / 38.01 | legal coreset coverage |
| Medoid coreset | 18.06 / 24.71 | 27.00 / 27.78 | 17.01 / 30.21 | legal coreset coverage |
| Adaptive coherent | 25.59 / 50.70 | 47.47 / 68.14 | 17.67 / 60.18 | internal historical candidate |
| Static-TAG full-history | 18.85 / 27.53 | 27.32 / 35.15 | 18.39 / 31.25 | offline reference; not continual |
| Full-history edge-free raw | 47.72 / 71.57 | 78.26 / 88.96 | 44.78 / 83.09 | retention reference; not a constructor |

The expanded panel therefore covers eleven named rows including Unified-Q, or ten baseline/reference entries excluding it. It should not be described as eleven official native reproductions.

## 5. Current Matched Full History and Archival Retention

### Current legal-pool 5% matched reference

Source: `../results/cotag_q_four_gap_20261015_v1/report_final_20261015T190000/`, particularly `fullhistory_retention.csv` and `fullhistory_student_pairs.csv`. This panel supersedes the older retention run for the current abstract and main text. It uses order 9101, the same legal pool and masks, 320-character preprocessing, MiniLM, prompts, original-graph inference, 2048 Student updates and batch size 24. Condensed scores average three construction seeds and three Student seeds; each full-history reference uses only the three Student seeds, without artificial construction replication.

| Dataset | Reference | Full F1 | Condensed F1 | F1 retention | Full Acc. | Condensed Acc. | Acc. retention |
|---|---|---:|---:|---:|---:|---:|---:|
| ArXiv | EF | 50.93 | 52.96 | 103.99% | 71.11 | 70.64 | 99.34% |
| ArXiv | Induced topology | 53.47 | 52.96 | 99.04% | 72.37 | 70.64 | 97.60% |
| SportsFit | EF | 79.98 | 80.02 | 100.05% | 88.29 | 85.38 | 96.71% |
| SportsFit | Induced topology | 81.46 | 80.02 | 98.23% | 89.33 | 85.38 | 95.58% |
| BookHis | EF | 40.16 | 48.53 | 120.84% | 81.48 | 81.38 | 99.88% |
| BookHis | Induced topology | 46.96 | 48.53 | 103.33% | 82.09 | 81.38 | 99.14% |

Ratios above divide unrounded mean scores; they are not means of Student-wise ratios. The latter are exported separately and reported in Appendix B. Final retained/legal rows are 5927/118539, 6057/121137 and 1454/29086. Topology is the legal training-pool induced graph, not a fully supervised entire-graph optimum. Loss-row sampling covers 33.91%, 33.30% and 81.76% of the pools; all pool nodes still participate in graph forward and neighbors may receive indirect gradients. Shared finite training steps do not establish full-history convergence or equal per-node exposure. These masks have been inspected previously; this is not fresh blind confirmation.

### Archived nominal-rate development reference

This is a separate matched report-development retention run and not the same statistical run as the current fixed-mask main table. The reference retains cumulative raw node features/text with edge-free (self-loop) training; it is not a full-topology oracle. ArXiv's condensed memory occupies 5.71% of the current legal pool; the SportsFit and BookHis method values also differ from the main table. Each retention ratio uses the numerator and denominator from this same run, not a newer primary-table numerator.

| Dataset | Full F1 | Condensed F1 | F1 retention | Full Acc. | Condensed Acc. | Acc. retention |
|---|---:|---:|---:|---:|---:|---:|
| ArXiv | 47.72 | 52.25 | 109.50% | 71.57 | 71.59 | 100.03% |
| SportsFit | 78.26 | 79.44 | 101.50% | 88.96 | 85.72 | 96.36% |
| BookHis | 44.78 | 44.98 | 100.45% | 83.09 | 82.55 | 99.35% |

These values support a retention claim, not a claim that condensation universally improves the raw graph. Ratios above 100% can arise from finite-Student optimization, sampling, and removal of redundant or noisy raw rows.

## 6. Published-Method Identity and Reproduction Status

An official/native reproduction means that the published code and loss are run with the original input assumptions, teacher/Student, data access, and evaluation protocol. A mechanism adaptation preserves the published mechanism but changes enough of the interface to satisfy current-delta-only access, frozen prefix, a cumulative 5% memory ratio, TAG text storage, and the common Prompt Student. These are not interchangeable claims.

| Original method | Current paper status | F1/Accuracy under the current three-dataset protocol |
|---|---|---|
| CaT | mechanism-adapted TAG continual baseline | CaT-CCTP in primary nine-cell and five-dataset tables |
| GECC | hard-clustering mechanism adaptation; separate native FCM-core adaptation | GECC-hard in primary tables; GECC-soft BookHis one-cell reference only; other two stopped |
| PUMA | inspired pseudo-teacher reference, not official PUMA core | common-Student five-dataset panel and separate archived coverage |
| TAGSAM | official selector/response core with reduced construction budget and continual wrapper | qualified common-Student reference on all three large graphs |
| GCond/GCond-x | official SGC/gradient-distance core, identity adjacency, changed continual proxy | GCond-x core in five-dataset panel; not native end-to-end reproduction |
| SFGC | official trajectory-matching core with bounded expert/outer budget | bounded core in five-dataset panel; not native end-to-end reproduction |
| GCondenser | official benchmark toolkit checked; no legal continual TAG constructor | not reported under current protocol |
| STAD | native-core/local-Qwen reference with changed supervision, LLM, topology, and continual access | qualified common-Student reference on all three large graphs; legacy inspired rows kept separate |
| DANCE / SR-GM | simplified inspired controls using a legacy Student | archived references only; not current common-Student ranking |
| GraphCLIP | official frozen checkpoint reference; no condensation or Student training | no condensed-memory F1/Accuracy |

Published-code cores within a changed wrapper are adaptations, even when the source commit is verified. GCondenser is a toolkit and GraphCLIP is a pretraining/checkpoint reference, not additional continual TAG constructors. Missing native end-to-end scores must not be estimated from adapted results. Native evaluations would answer a different question because access boundaries, objectives, models, or memory definitions differ.

## 7. Exploratory EM-GECC-X Branch

The bounded EM-GECC-X pilot is retained as a negative exploratory result and is not part of the main ranking. It used GECC-style propagated-feature initialization, a class-masked latent responsibility matrix, direct updates of new synthetic features, frozen historical prefixes, the cumulative 5% schedule, and the common Prompt Student. The report-development scope was one order and one construction seed on Cora/Citeseer, with three Student seeds.

| Dataset | Method | Macro-F1 | Accuracy | Status |
|---|---|---:|---:|---|
| Cora | GECC-hard + nearest-5 TAG | 82.11 | 85.36 | matched comparator |
| Cora | EM-GECC-X | 70.14 | 74.29 | failed continuation gate |
| Cora | EM-no-GM | 79.65 | 82.41 | component control |
| Citeseer | GECC-hard + nearest-5 TAG | 65.81 | 69.80 | matched comparator |
| Citeseer | EM-GECC-X | 56.31 | 64.26 | failed continuation gate |
| Citeseer | EM-no-GM | 67.89 | 72.00 | component control |

The complete branch was 12.17 F1 points below the stronger matched comparator on average and therefore did not pass the precommitted continuation gate. The pilot does not establish that EM is generally ineffective; this implementation was not promoted into the CoTAG-Q result. Full diagnostics and hashes are recorded in `../docs/current/EM_GECC_X_PILOT_20261013_ZH.md` and `../results/em_gecc_x_pilot_20261013/`.

## 8. Cross-Student Transfer

### Current exact-rate ArXiv memories

Source: `../results/uq_cross_arch_20261015_arxiv_v2/aggregate/summary.md`. The audit passes all 54 expected method/architecture/cell records, with identical memory/archive hashes and evaluation masks across Students. The constructor is not rerun. Each row averages nine order/construction cells after averaging Student seeds 150/151/152 within each cell; both Students use 2048 steps and batch 24. The tested GraphSAGE uses two PyG mean SAGEConv layers with explicit self-loop neighbors and the root transform.

| Student | Method | Final F1 +/- cell SD | Accuracy +/- cell SD |
|---|---|---:|---:|
| Prompt GCN | CoTAG-Q | 52.91 +/- 0.73 | 71.15 +/- 0.59 |
| Prompt GCN | GECC-hard-TAG | 52.68 +/- 0.45 | 70.94 +/- 0.49 |
| Prompt GCN | CaT-CCTP | 34.98 +/- 1.79 | 60.41 +/- 2.64 |
| Prompt GraphSAGE | CoTAG-Q | 53.93 +/- 0.60 | 71.27 +/- 0.35 |
| Prompt GraphSAGE | GECC-hard-TAG | 52.60 +/- 0.93 | 70.62 +/- 0.46 |
| Prompt GraphSAGE | CaT-CCTP | 50.24 +/- 0.53 | 70.78 +/- 0.65 |

On GraphSAGE, CoTAG-Q minus GECC-hard-TAG is +1.34 F1 points, exploratory crossed-factor bootstrap 95% CI [0.74, 1.91], positive 9/9, worst +0.37. Its difference against CaT-CCTP is +3.69 [2.80, 4.63], positive 9/9. CoTAG-Q's same-memory GraphSAGE-minus-GCN difference is +1.02 [0.27, 1.60]. These exploratory crossed intervals are distinct from the primary nine-cell paired t intervals.

### Archived SportsFit and BookHis transfer

The rows below reuse historical memory snapshots and remain separate from the current exact-rate ArXiv transfer and primary ranking.

| Dataset | Method | Prompt GraphSAGE Final F1 | Accuracy | Cells |
|---|---|---:|---:|---:|
| BookHis | CoTAG-Q | 36.83 +/- 4.22 | 71.23 +/- 6.48 | 9 |
| BookHis | GECC-hard TAG | 35.50 +/- 4.32 | 69.45 +/- 8.08 | 9 |
| BookHis | CaT-CCTP | 29.24 +/- 4.05 | 63.26 +/- 14.92 | 9 |
| SportsFit | CoTAG-Q | 75.27 +/- 2.06 | 80.00 +/- 1.79 | 9 |
| SportsFit | GECC-hard TAG | 74.91 +/- 1.93 | 81.86 +/- 3.44 | 9 |
| SportsFit | CaT-CCTP | 77.39 +/- 2.97 | 84.83 +/- 2.67 | 9 |

The GraphSAGE table is a transfer check, not a second construction protocol. The same frozen memories are evaluated without retuning the constructor.

## 9. Supplementary and Boundary Results

WikiCS, Cora, and Citeseer provide supplementary scale/domain checks. WikiCS has richer text and a smaller graph than the main ArXiv/product datasets. Its placement is based on that scope, not on method ranking. The edge-free CoTAG-Q main method does not make a link-prediction claim; older topology-enabled Cora/Citeseer audits are reported separately under CoTAG-Q-T.

## 10. Current Component Evidence and Manuscript Organization

### Original-constructor loss and support controls

Source: `../results/uq_component_ablations_20261014/report_utc_20261013_201811_267213/`. Use `summary.csv` and `pairs.csv` with `family=original_loss`, `split=sealed_test`; the artifact's split name is not a fresh blind-test claim. Each arm is complete for three construction seeds on order 9101, with three Student seeds averaged per construction cell. Fixed-Q changes only assignment trainability, retaining the initialization, residual, objectives, and budget.

| Arm | ArXiv F1 +/- cell SD | SportsFit F1 +/- cell SD | BookHis F1 +/- cell SD |
|---|---:|---:|---:|
| Full CoTAG-Q | 52.96 +/- 0.21 | 80.02 +/- 0.30 | 48.53 +/- 0.85 |
| Fixed-Q, trainable residual | 40.20 +/- 0.20 | 58.35 +/- 0.48 | 39.62 +/- 0.17 |
| No DM | 51.05 +/- 0.64 | 65.28 +/- 0.82 | 47.23 +/- 0.97 |
| No GM | 53.20 +/- 0.19 | 80.47 +/- 0.56 | 48.47 +/- 1.11 |

| Full minus control, F1 points [paired 95% CI] | ArXiv | SportsFit | BookHis |
|---|---:|---:|---:|
| Fixed-Q | +12.76 [11.78, 13.73] | +21.67 [20.62, 22.71] | +8.91 [7.19, 10.64] |
| No DM | +1.91 [0.82, 3.00] | +14.74 [13.26, 16.21] | +1.30 [0.57, 2.04] |
| No GM | -0.24 [-1.17, 0.69] | -0.45 [-2.21, 1.31] | +0.06 [-1.38, 1.50] |

All nine dataset/construction cells favor learned over fixed Q and DM over no-DM. GM has no consistent positive F1 contribution here and remains in the frozen primary implementation. `tools/make_core_ablation_figure.py` creates `figures/core_ablation_effects.pdf`, its PNG preview, and a JSON containing the CSV hash, exact filter, and plotted records.

### Construction-time text and controlled assignment implementation

Source: `../results/cotag_q_four_gap_20261015_v1/report_final_20261015T190000/`. The completed 2x2 factorial adds separate no-text-initialization and no-pair-feedback arms to the earlier joint-removal control. All four arms use order 9101, three construction seeds and three Student seeds per construction. No-init changes only `text_weight` from 0.35 to zero in prototype selection and initial logits. No-pair changes only `pair_weight` from 0.5 to zero in real and synthetic proxy losses; prompt supervision remains.

| Arm | ArXiv F1 +/- cell SD | SportsFit F1 +/- cell SD | BookHis F1 +/- cell SD |
|---|---:|---:|---:|
| Full | 52.96 +/- 0.21 | 80.02 +/- 0.30 | 48.53 +/- 0.85 |
| No text initialization | 53.52 +/- 0.79 | 77.50 +/- 0.83 | 46.79 +/- 1.21 |
| No pair feedback | 53.13 +/- 0.28 | 80.30 +/- 0.98 | 48.14 +/- 2.04 |
| Both off | 53.32 +/- 0.41 | 77.64 +/- 0.69 | 47.87 +/- 0.63 |

| Full minus control, F1 points [paired 95% CI] | ArXiv | SportsFit | BookHis |
|---|---:|---:|---:|
| No text initialization | -0.56 [-2.05, 0.93] | +2.51 [0.48, 4.55] | +1.74 [-3.32, 6.80] |
| No pair feedback | -0.16 [-1.36, 1.03] | -0.28 [-3.47, 2.91] | +0.39 [-2.68, 3.46] |
| Both off | -0.36 [-0.89, 0.18] | +2.38 [0.51, 4.25] | +0.66 [-2.37, 3.68] |

SportsFit's full-minus-no-init Accuracy difference is +6.87 [5.42, 8.32]. Its two positive F1 contrasts each favor full in all three construction seeds. The evidence locates the clearest node-text benefit in initialization; pair-feedback intervals all cross zero. Factorial interactions (full minus no-init minus no-pair plus both-off) are -0.37 [-0.72, -0.01], -0.15 [-4.76, 4.46] and +1.48 [-0.48, 3.44]. These nominal, unadjusted three-cell intervals do not establish a general positive interaction. All contrasts are conditional on the frozen constructor and previously inspected masks.

The component snapshot also contains a separate matched shared/independent implementation. Its absolute scores are not pooled with the original constructor. Shared-minus-independent F1 differences are +0.04 [-0.43, 0.50], -0.37 [-1.21, 0.46], and -1.92 [-5.02, 1.17]. Shared/independent peak GiB are 12.08/15.99, 24.72/32.60, and 2.77/3.26; times are 241.3/292.5, 177.9/226.7, and 66.9/81.6 seconds. These costs support a resource benefit within the controlled implementation, not universal efficiency or accuracy superiority.

New-block diagnostics use `diagnostics.csv`, filtered to `arm=control_shared` and averaged uniformly across stage/seed records (9 ArXiv blocks; 15 each on SportsFit/BookHis). Hard coverage is 1.0000/0.9998/1.0000; normalized row entropy 0.1200/0.2385/0.0827; unique text ratio 1.0000/0.9995/0.9990; first-up-to-512-slot pairwise text cosine 0.0947/0.3988/0.1194; soft/hard cosine gap 0.3853/0.2037/0.4693. Column-support entropy fields are missing, not zero. `coverage.csv` counts experiment completion, not slot coverage.

### Placement and interpretation

| Manuscript location | Evidence and purpose |
|---|---|
| Section 6, primary comparison | Current nine-cell exact-rate table; dataset-specific paired differences, Old/New and stage means |
| Section 6, broader baseline panel | Five-dataset common-Student comparison; controls, adaptations, and qualified references identified |
| Section 6, architecture transfer | Current ArXiv same-memory GCN/GraphSAGE transfer |
| Section 6, compression and cost | Rate sensitivity, current matched full-history references, and primary construction costs |
| Section 7, core mechanism | Complete fixed-Q/no-DM/no-GM controls and four-arm construction-time text factorial |
| Section 7, diagnostics | Support coverage, row entropy, text uniqueness, pairwise cosine, and soft/hard gap |
| Appendix B | Shared/independent, residual, initialization, storage and graph controls; full text-factorial contrasts; Student-paired retention; TAGSAM last-stage budget; identity and dense-Q costs; historical and exploratory results |

Storage-only class-shared and random-five results come from `../results/uq_priority_controls_20261014/summary.md`. They keep constructed features fixed and cannot replace the construction-time text control. SportsFit/BookHis replays are paired with their same-run primary, not the archived primary scores. Soft-storage outcomes are mixed, with no three-seed F1 interval excluding zero. Negative and uncertain contrasts are retained in Appendix B; none is used to silently simplify or replace the frozen primary method.

### TAGSAM budget and identity-cost diagnostics

The same immutable four-gap batch includes a last-stage TAGSAM 100-versus-500 outer-update comparison, with ten inner steps and a common prefix, native PMI initialization and teacher. F1 changes 42.22 to 38.68 on ArXiv, 69.10 to 61.13 on SportsFit and 11.03 to 10.48 on BookHis. The reconstructed 100-update feature errors relative to the archive are 0.03497/0.02004/0.01910 in relative norm; labels/text agree. Only same-run contrasts are valid. This is not all-stage or native 5000-by-15 reproduction and does not repair baseline-fidelity limitations.

Stage-zero dense/sparse identity costs are 82.02/82.23, 123.08/123.15 and 20.00/19.65 seconds, with peak allocated GiB 11.56/11.50, 23.44/23.33 and 1.17/1.17. Each arm was measured once, dense before sparse. Operator forward, first- and second-derivative errors are zero in the checks, but optimization trajectories are not identical: feature maximum errors are 0.000941/0.002671/0.000336, and ArXiv/SportsFit hard text choices differ. Dense-Q logits/gradients/two Adam states alone account for 2726/4446/102 MiB at stage zero. Replacing the identity saves little relative to dense assignments; these are diagnostics, not substituted primary memories or speed claims.

### Independent verification and figure provenance

`../tools/verify_cotag_four_gap_independent_20261015.py` independently recomputes exports with stdlib statistics from saved raw Student records, without importing aggregation helpers. The passing audit at `../results/cotag_q_four_gap_20261015_v1/verification_20261015/independent_audit_v1.json` verifies all 48 source JSON hashes, the aggregation snapshot hash, and 8061 fields across 741 CSV rows. Protocol/intervention, construction-cell SD/paired t intervals, full-history ratios, cost accounting and figure-source hashes pass. Three saved ArXiv numerical-warning records are retained; this is not a claim of warning-free numerical execution or an audit of unobserved training behavior.

- Data JSON SHA-256: `e281341687735590d9a6b35750fc210b2226f36d1efb807c2213ed1b9bf6cba8`.
- Verifier SHA-256: `8518d114d463c98f5d3bd9cf7dd1894aa510ad9b70a62f8a94032ba8d4950a95`.
- Aggregation snapshot SHA-256: `c17ee6079c02d222a3702da5ad66669feecb1facf27c6aedc54bbea16f7edf8b`.
- Current aggregation/renderer SHA-256: `1c7de4e251c3ef472c607fc04e821444a3710ae751942a28681c76bbca5d37de`; later rendering/provenance changes explain the difference from the snapshotted aggregation script.

Figures `four_gap_text_factorial`, `four_gap_fullhistory_matched` and `four_gap_identity_cost` are exact copies of the immutable batch's PDF/PNG/source-JSON artifacts. Copies are checked by SHA-256. Primary nine-cell metrics and sealed source files remain unchanged.

## 11. Paper Guardrails

- The main claim is history-constrained continual TAG condensation with shared graph-text support and differentiable text feedback. The primary nine-cell gain over GECC-hard is BookHis-specific.
- CaT/GECC are mechanism adaptations; GCond-x/SFGC are adapted official cores; TAGSAM/STAD are qualified references; PUMA-inspired is an inspired reference. None is a native end-to-end reproduction under the present contract.
- Shared-Q is not claimed to be universally better than split-Q.
- Full-history rows are retention/reference comparisons and may access cumulative raw data.
- Prompt Student is a prompt-based graph-text readout; it is not strict label-free unseen-class zero-shot learning.
- Sharing reduces assignment logits from two matrices to one; it does not guarantee lower total construction memory or greater accuracy. Controlled results are reported separately from the final-version primary table.
- Evaluation intervals are conditional on repeatedly inspected masks. Historical filenames containing `sealed` are artifact identities, not evidence of a fresh blind test.
