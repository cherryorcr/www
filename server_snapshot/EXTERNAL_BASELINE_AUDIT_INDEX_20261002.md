# External Baseline Audit Index

| Scope | Source locality | Budget | Prefix preservation | Checkpoint reload | Evidence |
|---|---|---|---|---|---|
| CaT/GECC/k-center/medoid, `order19611/19612` | passed; current-delta protocol recorded | passed | passed | passed | `external_continual_adapters_20260927_v2/*/audit.json` |
| CaT/GECC/k-center/medoid, `order19613` | passed; current-delta protocol recorded | passed | passed | passed | `external_continual_order19613_20261001_*/**/audit.json` |
| CaT/k-center, `order19614` | passed; current-delta protocol recorded | passed | passed | passed | `external_continual_order19614_*/**/audit.json` |
| CaT/k-center, `order19615` | passed; current-delta protocol recorded | passed | passed | passed | `external_continual_order19615_20261002/{cora,citeseer}/order19615/audit.json` |
| PUMA-inspired | passed; sealed source IDs and no test construction labels | passed | passed by stage manifests | passed in formal runner audits | `puma_adapter_formal_20261002/*/results.json` |
| TAGSAM | recorded in manifest; adapted current-delta input | passed | recorded in construction artifacts | passed; 72 fits per dataset, maximum metric error 0.0 | `tagsam_graphclip_comparison_2026_09_16/{cora,citeseer}/audit.json` |
| STAD | acceptance records source access and test exclusion | passed | acceptance passed | independent acceptance passed | `external_tag_stad_2026_09_17/{cora,citeseer}/acceptance.json` |
| GCond/SFGC/GCondenser | repository readable; no TAG construction run | not applicable | not applicable | not applicable | `baseline_cards_20261002/*reference_only_card.json` |
| Official GraphCLIP frozen checkpoint | reference-only; complete original TAG features | not applicable | not applicable | strict checkpoint load recorded | `official_graphclip_report_2026_09_17/{cora,citeseer}.json` and `audit/*` |

The physical server loader still has the full offline fixture available for slicing/evaluation. The formal continual cards therefore state the logical access restriction and the fixture limitation explicitly; they do not claim OS-level isolation.

## Delivery normalization

- Twelve cards are present under `baseline_cards_20261002/`, including the standalone PUMA card.
- `EXTERNAL_BASELINE_CARD_AUDIT_20261002.md` reports no missing required card fields.
- `EXTERNAL_BASELINE_COMPARISON_TABLE_20261002.md` contains 42 strict final-stage rows from 180 normalized result rows.
- Four TAGSAM/STAD final-stage rows are retained in `EXTERNAL_BASELINE_ADAPTED_REFERENCE_TABLE_20261003.md` because their Student seeds are 100-105 rather than the frozen 150-155.
- Official GraphCLIP is recorded as a section 3.C reference-only card and is excluded from the 42-row continual comparison table.
- TAGSAM and STAD source manifests omit an order field; their rows are labeled `order-unspecified`.
- Nested manifests omit some construction/text/edge cost fields; the comparison table leaves those cells blank and the summary JSON records the warning.
