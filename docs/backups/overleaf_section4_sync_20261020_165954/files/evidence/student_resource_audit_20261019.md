# Isolated Student Resource Audit

Date: 2026-10-19

Source directories on the remote server:

- `/home/pc/research/TAG_condensation/results/cotag_training_resources_20261019/arxiv/results.json`
- `/home/pc/research/TAG_condensation/results/cotag_training_resources_20261019/sportsfit/results.json`
- `/home/pc/research/TAG_condensation/results/cotag_training_resources_20261019/bookhis/results.json`

The audit uses the same Prompt Student, frozen MiniLM embeddings, 2,048
updates, batch size 24, and Student seeds 150/151/152. It measures training
only: construction, evaluation, the text encoder, and evaluation caches are
excluded. Memory is peak `torch.cuda.max_memory_allocated`; time is
synchronized wall time. The full-raw reference uses the complete final legal
pool and its induced graph. A full-raw edge-free control is also present in
the source JSON.

| Dataset | 5% memory peak / time | Full raw graph peak / time | Peak reduction | Time reduction |
|---|---:|---:|---:|---:|
| ArXiv | 100.7 MiB / 7.75 s | 1614.6 MiB / 63.61 s | 93.8% | 87.8% |
| SportsFit | 102.2 MiB / 7.73 s | 1656.3 MiB / 66.64 s | 93.8% | 88.4% |
| BookHis | 43.1 MiB / 7.42 s | 411.2 MiB / 15.11 s | 89.5% | 50.9% |

The full-raw edge-free controls are 1592.1/72.29 s, 1628.1/72.19 s, and
406.4/14.45 s for ArXiv, SportsFit, and BookHis, respectively. Their peak
memory is close to the full raw graph, so the Student-training reduction is
primarily caused by the smaller stored memory and propagation support rather
than by removing edges alone.

These measurements support a downstream-training resource claim only. They
do not establish end-to-end construction efficiency. Current construction
peaks are dominated by dense support/Sinkhorn states, retained proxy
intermediates, and an evaluation loader that can keep full-graph caches on
the GPU. The logical current-delta-only protocol and physical GPU residency
must therefore be reported separately.
