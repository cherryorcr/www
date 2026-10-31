# External baseline status (2026-10-02)

This file records the externally sourced baselines already present on the
server. It is an inventory and comparability guide, not a claim that every
method is an official reproduction.

## Formal or protocol-complete results

| Method | Server result root | Coverage | Classification | Main caveat |
|---|---|---|---|---|
| PUMA-inspired TAG adaptation | `/home/pc/research/TAG_condensation/results/puma_adapter_formal_20261002` | Cora/Citeseer x `order19611`/`order19612`; 3 arms; Student seeds 150-155 | mechanism-adapted, strict continual access | not native PUMA; official commit and differences are recorded in `baseline_cards_20261002/puma_adapter_card.json` |
| TAGSAM GraphCLIP adapter | `/home/pc/research/TAG_condensation/results/tagsam_graphclip_comparison_2026_09_16` | Cora/Citeseer; 4 arms; seeds 100-105 | supplementary adapted/reference result | official TAGSAM loss/components are pinned, but encoder, initialization, budget, optimization and Student seeds differ from the strict 150-155 table |
| STAD local-canonical adapter | `/home/pc/research/TAG_condensation/results/external_tag_stad_2026_09_17` | Cora/Citeseer; 3 stages; seeds 100-105 | supplementary adapted/reference result | uses local Qwen text generation and canonical GraphCLIP student; independent acceptance exists, but Student seeds differ from strict table |
| semantic-recursive, CaT-MMD, GECC-hard, k-center, medoid | `/home/pc/research/TAG_condensation/results/external_continual_adapters_20260927_v2` | Cora/Citeseer x `order19611`/`order19612`; 5 arms; Student seeds 150-155 | source-checked mechanism adaptations | upstream methods are adapted to TAG text, the legal 5% stream and scratch InfoNCE Student |
| CaT-MMD-TAG, k-center/FPS | `/home/pc/research/TAG_condensation/results/external_continual_order19613_20261001_*`, `external_continual_order19614_*`, `external_continual_order19615_20261002` | `order19613/19614/19615`; Cora/Citeseer coverage; Student seeds 150-155 | mechanism-adapted new-order extension | `order19615` construction/evaluation/audit/checkpoint reload passed for both datasets |
| GCond, SFGC, GCondenser | `/home/pc/research/TAG_condensation/external_repos/{GCond,SFGC,GCondenser}` | official commit pinned; no shared-protocol result | reference-only | compile smoke passed; static/legacy pipeline cannot be claimed as continual TAG reproduction |

The external-adapter protocol explicitly says “source-checked mechanism
adaptations; not official end-to-end reproduction”. Its source pins include
CaT `superallen13/CaT-CGL@43d6fcfd7b1754fe53b9f911492d7cb0c113a2eb` and GECC
`rockcor/GECC@d731be00cc3beb7585e8fa7b60ca21a6c3cb4e3f`, and it records the
local source hashes and no test-label construction access.

## Reference-only or feasibility results

| Method | Server result root | Classification | Why excluded from continual main table |
|---|---|---|---|
| Official GraphCLIP frozen checkpoint | `/home/pc/research/TAG_condensation/results/official_graphclip_report_2026_09_17` | official checkpoint reference | complete original TAG features, no synthetic memory, no Student training, and no continual append |

The GraphCLIP reference is recorded in `baseline_cards_20261002/graphclip_reference_only_card.json`; its checkpoint, graph-state and text-state hashes are pinned.

`/home/pc/research/TAG_condensation/results/cat_multimodal_minimal_full_20261002_cora`
and the corresponding Citeseer directory contain a feasibility study with
`cat_x`, `cat_t_only`, `cat_xt`, and `random_text`. Their protocol is explicitly
“feasibility only”; only `order19611` and `order19612` result files are present,
so these should not be placed in the formal external-baseline table.

The older static TAGSAM runner is also reference-only when it rebuilds from the
full legal seen pool. It is not a strict continual comparison because it does
not reuse frozen parent memory.

## Handoff notes

1. Use the protocol-complete roots above for the external-baseline table,
   with each method's classification and adaptation notice shown beside its
   metrics.
2. Keep feasibility/static outputs in a separate appendix or diagnostic table.
3. If a native official reproduction is required, it is still missing for all
   methods except the pinned source-mechanism checks; native reproduction would
   require the original training/evaluation pipeline and its compatible data,
   not just the shared TAG student.
4. Do not merge these results with one-stage static `run_external_baselines.py`
   numbers without stating the protocol mismatch.
