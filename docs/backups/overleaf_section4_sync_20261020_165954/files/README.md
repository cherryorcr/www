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
- `sections/06_results.tex`: RQ1, replicated comparisons and method/dataset breadth.
- `sections/07_ablation_discussion.tex`: RQ2--RQ4, mechanisms, transfer, retention, and cost.
- `sections/08_limitations.tex`: baseline fidelity, evaluation scope, and scaling.
- `sections/09_conclusion.tex`: supported conclusions.
- `sections/10_protocol_choice_support.tex`: historical protocol pilots.
- `sections/11_extended_evidence.tex`: full component controls and archived evaluations.
- `RESULTS_REGISTER.md`: evidence sources and protocol boundaries.
- `figures/`: conceptual diagrams, result figures, and archived visualizations.

## Narrative and Figures

The 2026-10-17 revision follows the exposition of GraphCLIP (WWW 2025,
arXiv:2410.10329v4): state the challenges, locate the gaps in existing
interfaces, explain how the design responds, and evaluate those responses.
Its layout remains the anonymous ACM/WWW 2027 submission format. The
reference's algorithm, benchmark setting, and claims are not transferred to
this paper.

The experiments form a single section with four research questions:
comparative utility, mechanisms, transfer/full-history utility, and resources.
Replicated primary results and the single-construction breadth panel remain
separate. Supplementary controls and detailed qualifications are in the appendix.

- Figure 1, `challenge_overview`: the historical-access and continuous/discrete-payload challenges, grounded in CaT/PUMA and TAGSAM/STAD
  and TAGSAM/STAD, followed by the proposed continual feature/text construction.
  The named methods are shown without an invented text-adaptation stage.
- Figure 2, `method_overview`: explicit input objects, class-balanced support,
  two payload branches, optimizer feedback, and frozen-prefix append operation.
- Figure 3, `primary_confirmation`: source-backed paired effects on the primary
  nine-cell mask. The additional held-out-mask check remains in the appendix;
  `split_confirmation` is retained as an audit figure showing both scopes.
- Figure 4, `construction_loss_traces`: archived no-GM update trajectories in
  the appendix, not a convergence proof for the primary GM constructor.

The first three figures can be regenerated from the project root with Python,
NumPy, and Matplotlib:

```powershell
python paper_www/tools/make_challenge_overview.py
python paper_www/tools/make_method_overview.py
python paper_www/tools/make_split_confirmation.py
```

`tools/diagram_primitives.py` supplies shared drawing elements. The paired
figure also writes `figures/split_confirmation.source.json` with source hashes,
filters, and the plotted values. Older result figures are retained for
provenance and are not all active in the current manuscript.

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

Compile the complete project with the installed MiKTeX:

```powershell
./tools/compile.ps1
```

The script normally writes `build/main.pdf`. The current revision was verified
with MiKTeX/pdfLaTeX (three passes) at
`build/graphclip_revision/main.pdf`, because the ordinary PDF was open and
locked by another application. Both builds use this same `main.tex` source.
The verified PDF has 11 pages; the conclusion ends on page 7.

Keep the existing `main.tex` editor open. Its isolated preview compiler
currently cannot resolve the external `sections/` files; this does not affect
the successful complete-project MiKTeX build. The project requires `sections/`,
`figures/`, and the `acmart` class. Overleaf can compile the full project with
pdfLaTeX. Edits do not automatically commit or push the repository.
