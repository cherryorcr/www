# CoTAG-Q: WWW Manuscript

The English manuscript uses the ACM `acmart` class with the `sigconf`
two-column format. The source entry point is `main.tex`.

Current title: **CoTAG-Q: Learning Graph--Text Support for
History-Constrained Continual TAG Condensation**.

`CoTAG-Q` is the paper-facing name for the edge-free method previously
called `Unified-Q-EF`. Historical artifact paths, hashes, and implementation
identifiers retain their original names. `CoTAG-Q-T` denotes the separate
topology extension. Renaming does not change the algorithm or results.

## Files

- `main.tex`: metadata, abstract, section order, and bibliography.
- `sections/01_introduction.tex`: motivation and contributions.
- `sections/02_problem_protocol.tex`: legal access, fixed-rate budget, and readout.
- `sections/03_related_baselines.tex`: related work and adaptation identities.
- `sections/04_method.tex`: assignment, text feedback, losses, and complexity.
- `sections/05_experimental_setup.tex`: datasets, Student, and evaluation design.
- `sections/06_results.tex`: matched results, broader controls, retention, and cost.
- `sections/07_ablation_discussion.tex`: component interventions and diagnostics.
- `sections/08_limitations.tex`: baseline fidelity, evaluation scope, and scaling.
- `sections/09_conclusion.tex`: supported conclusions.
- `sections/10_protocol_choice_support.tex`: historical protocol pilots.
- `sections/11_extended_evidence.tex`: full component controls and archived evaluations.
- `RESULTS_REGISTER.md`: evidence sources and protocol boundaries.
- `figures/`: result figures referenced by the manuscript.

## Evidence

The primary table uses the immutable `uq_final_evidence_report_20261013_132047`
snapshot: three orders by three construction seeds, each averaging three
Student seeds, at exact cumulative 5% of the legal construction pool.
Because the masks have been inspected repeatedly, this is a post-hoc
fixed-mask evaluation rather than a new blind test.

The expanded five-dataset common-Student panel is recorded in
`uq_baseline_expansion_20261013/report_expansion_utc_20261014_021125_595059`.
It uses one order and one construction seed. Selection controls, inspired
references, bounded official-core adaptations, and legacy-Student results
are identified separately; they are not interchangeable official reproductions.

Shared/independent assignment controls use their own matched implementation.
The core fixed-Q/no-DM/no-GM study is complete for three construction seeds
on one order. The current ArXiv GraphSAGE transfer reuses all nine primary
memories per method; earlier SportsFit/BookHis transfer uses historical
snapshots. These scopes remain separate in the manuscript.
Current full-history retention uses the same legal pool, class order,
text preprocessing, Student seeds and 2048-update budget as the 5% memory,
with both edge-free and induced-training-topology references. It is a
one-order, finite-budget, post-hoc comparison. Older 5.71%-ArXiv retention
results remain archival and are not used in the current abstract.
GECC-soft ArXiv/SportsFit retries were stopped at the user's request.

The immutable `cotag_q_four_gap_20261015_v1/report_final_20261015T190000`
batch adds the four-arm text factorial, matched full-history references,
a TAGSAM last-stage response-budget check, and dense/sparse identity costs.
Its independent verifier checks 48 raw JSON hashes and recomputes 8061
exported fields. The result register records scope and figure provenance.

`tools/make_core_ablation_figure.py` redraws the core paired effects from
the immutable component `pairs.csv` and writes a figure-source hash record.
No experiment metric is rewritten by the plotting script.

## Compilation

Keep the existing `main.tex` editor open and use its built-in compiler for
diagnostics. The project requires the local `sections/`, `figures/`, and
`acmart` class. Overleaf can compile the full project with pdfLaTeX.
Edits do not automatically commit or push the repository.
