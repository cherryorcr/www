# CoTAG-Q: Continual Text-Attributed Graph Condensation

This repository contains the public code, experiment protocols, compact
evidence, and paper sources for the CoTAG-Q study submitted to The Web
Conference. It combines the local paper workspace with a verified source
snapshot from the remote experiment server.

## Repository Layout

- `src/`: research and evaluation source code used by the selected protocols.
- `experiments/`: registered CoTAG experiment queues, protocol files, and
  summarizers. Queue registrations record intended runs and do not start new
  experiments by themselves.
- `results/`: compact JSON, CSV, Markdown, and text evidence. Raw run trees
  and tensors remain on the experiment machines.
- `paper/`: LaTeX source, figures, tables, bibliography, and the Chinese
  translation of the paper.
- `docs/`: experiment notes, protocol decisions, and evidence interpretation.
- `environment/`: tested environment and dependency information.
- `server_snapshot/`: read-only provenance material copied from the remote
  server on 2026-10-31.
- `provenance/`: release scope and source-hash records.

## Reproduction Starting Points

The paper source is `paper/main.tex`. The principal registered comparisons
are described by:

```text
experiments/cotag_baseline_repetitions_20261024/queue.py
experiments/cotag_arxiv_five_stage_20261024/queue.py
experiments/cotag_arxiv_base4_20261026/queue.py
```

The queues reference the Linux/CUDA paths used on the experiment server.
Reproducing GPU numbers therefore requires the datasets, text encoder cache,
and recorded environment described in `environment/ENVIRONMENT.md`. Those
inputs are not redistributed in this public repository.

To compile the manuscript locally, enter `paper/` and run the normal
`acmart` pdfLaTeX workflow. The anonymous data and code link is listed in the
abstract footnote.

## Public Release Boundary

The repository keeps source code, protocol registrations, compact summaries,
checksums, and paper assets. It does not include raw graph datasets, model
weights, transformer caches, synthetic tensor dumps, virtual environments,
GPU logs, or temporary build products. These files are large, regenerable, or
contain local-machine state. The exact exclusions and source snapshot hashes
are recorded in `provenance/RELEASE_MANIFEST.md`.

No credentials or access tokens are required by the source tree. Do not add
private data, checkpoints, or secrets to future commits.
