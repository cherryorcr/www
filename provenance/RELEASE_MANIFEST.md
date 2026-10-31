# Public Release Manifest

Release date: 2026-10-31

## Inputs Combined

1. Local workspace: `D:/Aprojects/continue_tag_condensation`.
2. Remote experiment workspace: `/home/pc/research/TAG_condensation`, reached
   through SSH alias `pc-rack-server`.
3. GitHub target: `https://github.com/cherryorcr/www`, branch `main`.

The local paper source is authoritative for `paper/`. The remote source
snapshot is authoritative for server-side experiment files when both trees
contain the same path. The queue files checked during preparation had these
matching SHA-256 digests:

```text
experiments/cotag_baseline_repetitions_20261024/queue.py
49e5f2b5f158820a2786fb6b0312c5a33300cc3d88d9fbfa36ac509c77df707f
experiments/cotag_arxiv_five_stage_20261024/queue.py
e9ff6db7f24704e6ae463d191d7edc4325a1129979763d602103afd33ac3f11f
experiments/cotag_arxiv_base4_20261026/queue.py
b50ab0911558527e953a45eb600047410763422c930222e8c903b6e95676f48a
```

The compressed remote source snapshot used during staging had SHA-256:

```text
d4635ab29724e7e94c192106ff403a2921c65ab021baf9215c6450c9d975fe20
```

## Included

- Source files from the selected research and evaluation modules.
- Registered CoTAG protocols and their compact audit inputs.
- Current paper source, figures, tables, bibliography, and translation.
- Environment and dependency notes.
- Compact JSON, CSV, Markdown, and text evidence selected from the current
  CoTAG experiment directories.
- Read-only server documentation and baseline audit cards.

## Excluded

- Raw graph and text datasets.
- Transformer and language-model weights.
- Synthetic tensor and checkpoint files.
- Virtual environments and package caches.
- Full run directories, GPU logs, temporary files, and generated PDFs.
- Legacy API deployment helpers containing credential-shaped configuration.
- Local credentials, tokens, and editor state.

These exclusions are deliberate. They keep the repository suitable for public
review while preserving the information needed to inspect the method,
protocol, calculations, and manuscript claims. The experiment notes identify
the external data and environment assumptions for a full rerun.
