# What the new comparisons do and do not restore

Upstream TAGSAM pin: `d812f06fd16db3cd4d497804a8e504d2c738faa8`.
The runtime checks the contents against the Git commit before executing
extracted definitions. Local copies of upstream files are audit references,
not replacements for the pinned runtime source.

| Component | Historical bounded adapter / new budget sweep | Native-epoch Cora adaptation |
|---|---|---|
| Text selector | GPT-2-large word PMI and neighborhood redundancy | Same selector |
| Selected text budget | Common 320-character cap | Common 320-character cap |
| Construction text encoder | MiniLM | Native frozen BERT CLS |
| Internal graph encoder | Common CLIP proxy | Upstream GCN and CLIP definitions |
| Internal text projection | Linear | Upstream residual projection, dropout and layer norm |
| Temperature | Learned proxy scale | Native fixed 1/0.07 |
| Outer objective | Upstream weighted BCE | Upstream weighted BCE |
| Inner training | 10 or 15 sampled minibatch updates | Literal upstream function; 15 full epochs, batch 20, dropped incomplete tails |
| Response samples | One fixed subset, at most 256 | Resampled each outer iteration, at most 2000 |
| Outer optimizer | SGD with learned inner rates, no scheduler | Native SGD and StepLR(200, 0.95) |
| Teacher | Common proxy, 2048 updates | Native model; buffer defaults: 3 epochs, batch 1024, Adam 2e-5 |
| Teacher selection | Fixed training budget | Final teacher; external-accuracy selection removed |
| Real graph processing | Full legal source graph | Full legal source graph instead of NeighborSampler |
| Continual access | Current delta and own frozen synthetic prefix | Same |
| Final evaluation | Common MiniLM prompt student, 2048 updates | Same; reconstruct final MiniLM embeddings from saved texts |

The 5000x15 minibatch experiment is **not** the official 5000x15-epoch
budget. At a memory size K, native inner updates per outer iteration are
`15 * floor(K / 20)`, not 15. Its scheduler is another material difference:
running a fixed large learning rate longer is not equivalent to native
optimization. Both differences must accompany the large-graph budget table.

The Cora native-core run resolves several of these specific concerns, but
does not establish native large-graph performance. Nor does a common final
student test the original method's distilled-learning-rate evaluation. A
native downstream result, if later added, must be a separate protocol.

The completed default Cora run also has a separate native-student diagnostic:
15 epochs, batch 32, native model and distilled learning rates, common prompts,
all-seen-category evaluation, and fixed last epoch. This is not the upstream
five-way, best-test-epoch evaluator. Reconstructed final-stage teacher loss
matched the saved trace exactly; the default 3 epochs meant only 3 updates
for 745 legal reference rows. A separately registered complete Cora chain
therefore uses 2048 native teacher updates, retaining all native condensation
updates. Both budgets must be shown. This follow-up changes the native teacher
default and must not be labeled an unchanged official run. The upstream README
uses default `buffer.py` and `distill.py` commands, without an additional
dataset-specific teacher schedule in that file.

## STAD

The existing STAD adaptation already retains dual self-training, native
coarsening (with an independently checked sparse equivalent), keyword and
text candidate construction, and internal Wasserstein/validation selection.
The reviewed adapter remains different in its local Qwen2.5-1.5B generator,
internal 10%/10%/80% split, current-delta access, binary stored synthetic
topology and common student. Repeating these runs does not repair those
differences. This batch does not silently relabel STAD as an official run or
claim its large-graph fidelity gap is closed.

## Fair Use

Do not weaken the text adapter or select a low-performing baseline run.
Present complete predeclared budget chains, failures, and uncertainty at
the correct unit. Distinguish student-seed dispersion for one memory from
construction-seed dispersion. Negative results establish a limitation of
the tested transfer setting, not the upper performance limit of the source
method in its native task.
