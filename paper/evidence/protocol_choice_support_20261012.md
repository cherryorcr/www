# Evidence for Protocol Choices

This audit supports the setting rationale in Sections 2 and 5 and Appendix A.
It does not change the constructor, Student, splits, or sealed ranking.

## Decisions and Scope

- Use a cumulative 5% node-count ratio, with the recorded class-coverage floor.
  Memory grows with cumulative construction-pool size; it is not a fixed row
  count or a byte-level 5% compression claim.
- Treat historical-raw non-access as the task boundary and frozen synthetic
  storage as an additional append-only design choice. Do not imply that
  continual learning in general requires freezing synthetic memory.
- Keep the common Prompt Student and fixed training budget for constructor
  comparisons. Attribute the large Legacy-to-Prompt score changes to the
  Student configuration, not to Unified-Q alone.
- Use the name supervised prompt-based cosine readout. GraphCLIP-style
  target-without-training zero-shot transfer has not been established here.

## Primary Literature

| Source | Verified support | Scope limitation |
| --- | --- | --- |
| [CaT](https://arxiv.org/html/2309.09455) | Class-incremental transductive evaluation; incoming-graph condensation, accumulating synthetic memory, Training in Memory; ratio sweep at 0.5%, 1%, 5%, 10% | Default ratio is 1%, not 5%; our cumulative budget and class allocation are adaptations |
| [GECC](https://arxiv.org/html/2502.17614) | Evolving condensation and a GCondX-Grow control that freezes inherited synthetic nodes and appends new nodes | Native GECC is not claimed to freeze representatives; our frozen-prefix GECC is an adaptation |
| [SFGC](https://arxiv.org/html/2306.02664) | Graph-aware construction can produce synthetic attributes without explicit synthetic edges | This does not establish universal structural-task utility for an edge-free memory |
| [GraphCLIP](https://arxiv.org/html/2410.10329), Section 3.3.1 | Class text prompts and cosine readout; zero-shot means no additional target-dataset training | Our target-supervised Student is not an official GraphCLIP reproduction or equivalent zero-shot protocol |

## Experimental Records

Paths below are relative to the project root.

### Historical Ratio Pilot

- Aggregate: `results/p5_rate_curve_small_20260919/summary/rate_final_summary.csv`
- Protocol: `docs/current/P5_SMALL_EXTENSION_COMPLETION_ZH_2026-09-19.md`
- Early C0 constructor, stored topology, Legacy Student, three stages,
  Student seeds 100--105, report-development data.

| Dataset | 1% Final F1 | 5% Final F1 | 10% Final F1 |
| --- | ---: | ---: | ---: |
| Cora | 42.00 | 49.85 | 45.38 |
| Citeseer | 52.20 | 59.17 | 55.82 |

Student steps are not held fixed: Cora 3456/17856/29376 and Citeseer
4416/22560/31968. First-stage effective batches at 1% are 5 and 7 rather
than 16. This is pilot motivation for an operating point, not a final-version
budget-only ablation or proof that 5% is optimal.

### Historical Old-Feature Updates

- Aggregate: `results/unified_q_diagnostics_20261007/method_cells.csv`
- Protocol: `docs/current/UNIFIED_Q_DIAGNOSTICS_RESULTS_ZH_2026-10-07.md`, Sections 3 and 6
- Reference: `uq_v3`, not `uq_30` or `uq_init`; topology-enabled early version.
- Order 9101, construction seed 271006, Student seeds 150/151/152.
- Bounded old-X update uses a 1% per-stage radius; unbounded removes the hard
  radius but retains a soft anchor. Old text and labels are unchanged.

| Dataset | Frozen F1 / Acc | Bounded F1 / Acc | Unbounded F1 / Acc |
| --- | ---: | ---: | ---: |
| WikiCS | 74.76 / 77.41 | 73.94 / 76.78 | 71.22 / 73.33 |
| BookHis | 42.69 / 78.36 | 42.13 / 78.40 | 38.96 / 79.86 |

Macro-F1 favors the frozen reference, but BookHis Accuracy does not. One
early-version order/seed does not prove that all memory refresh is ineffective.

### Matched Student Factorial

- Aggregate: `results/student_method_factorial_20261010/final_matched_operator/method_means.csv`
- Report: `results/student_method_factorial_20261010/final_matched_operator/summary.md`
- Same stored memories and full-raw inference operator; 2048 steps, batch 24,
  orders 9101/9102/9103, Student seeds 150/151/152; report-development data.
- Student change is a bundle: row-wise InfoNCE becomes multi-positive alignment
  plus prompt CE/margin. Individual loss necessity is not isolated.
- Final F1 Legacy -> Prompt: Unified-Q BookHis 16.96 -> 41.97, SportsFit
  26.43 -> 78.60, WikiCS 51.18 -> 79.86. GECC and CaT also improve.
- Implementation: `code/v2_llm_fusion/graphtext_prompt_student.py`,
  `multi_positive_pair_loss` and `train_prompt_student`.

### Current Supporting Evaluations

- Full-history retention: `paper_www/sections/06_results.tex`, `tab:retention`.
  Matched finite-training reference; separate from the sealed main ranking.
  F1 retention ArXiv/SportsFit/BookHis: 109.50%/101.50%/100.45%;
  Accuracy retention: 100.03%/96.36%/99.35%.
  Above-100% values are not an oracle or a universal condensation guarantee.
- Cross-architecture transfer: `results/unified_q_overnight_20261012/summary.md`
  and `paper_www/sections/06_results.tex`, `tab:sage`. Same stored memory,
  nine construction cells per dataset and three Student seeds per cell;
  no constructor retuning. Rankings remain dataset-dependent.

## Remaining Evidence Gaps

1. The matched final Unified-Q-EF rate sensitivity panel is complete for
   ArXiv and BookHis at 1%/5%/10%, and for SportsFit at 1%/5%. SportsFit 10%
   reproducibly exceeds the 48-GB GPU during the dense assignment backward
   pass; the failure is retained as a resource boundary rather than replaced
   by a retuned configuration. See
   `results/uq_rate_sensitivity_20261012/rate_sensitivity_summary.md`.
2. Matched final Unified-Q-EF frozen-versus-refresh comparison across orders
   and construction seeds. Historical pilots cannot replace it.
3. An independent unseen-class/cross-dataset transfer experiment if the paper
   is to claim GraphCLIP-style zero-shot learning.
