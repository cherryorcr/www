# Matched longer-budget development check

Registered on 2026-10-20 after the first paired construction seed, before
running any longer-budget outcomes. The primary 30-update experiment is
unchanged. This diagnostic is analyzed separately and does not select a
paper method or stopping point using test scores.

Trigger: independent-X loss is still decreasing at 30 updates in every
ArXiv and BookHis pilot stage. Changes between the last two adjacent
10-update means range from 4.5% to 54.6%, above the existing 1% practical
plateau diagnostic. A 30-update deficit alone cannot establish that the
parameterization generalizes better than optimized free features.

Run both CoTAG-Q and independent X for 100 construction updates, one
construction seed (271006), order 9101, on ArXiv, SportsFit and BookHis.
Keep initialization, learning rates, objectives, text encoder, memory rate,
access constraints and Student budgets unchanged. Each arm builds all its
own stage prefixes at 100 updates. There is no continuation of a memory
constructed with another budget. Evaluate the final memory with GCN and
the same frozen GraphSAGE variant, using Student seeds 150, 151, 152.

The initial attempt overlapped the primary GPU queue and was stopped before
any completed stage. Its immutable result file still says `running`, but
the process was verified absent; reports classify it as interrupted, not
as an active or completed experiment.

The original solver tolerance is `1e-4`. After the v2 longer-budget attempt
failed at a row residual of `0.0001029372`, the tolerance was amended to
`2e-4` for both arms, before obtaining any completed 100-update outcome.
This is a post-failure numerical amendment, not an initially registered
choice. The objective is unchanged, but numerical assignments and optimization
paths may change. The diagnostic is therefore not a pure step-only contrast
against the primary experiment. The v3 attempt exposed a checker still using
the old hardcoded tolerance; v4 checks the actual recorded tolerance. Failed
and interrupted attempts remain intact and are listed in the final report.

Record all per-update objective components and final-memory audits. Check
the final three consecutive 10-update windows. Two adjacent changes below
1% are a practical plateau diagnosis, not a mathematical convergence proof.
Inspect DM, GM and propagation separately. Do not automatically extend to
200, retune either arm, or combine this single-construction check with the
three-construction primary mean/SD. If neither arm stabilizes, state the
remaining optimization limitation rather than selecting by test accuracy.

Save to results/cotag_q_free_x_budget100_20261020. Preserve failed attempts.
No paper modifications or Overleaf push are part of this experiment.

On 2026-10-21 the v4 SportsFit full arm failed with CUDA OOM during backward
in its first stage. It is not retried. `run_budget_arm.py` finishes the
independent-X arm separately under the same 100-update, 2e-4 tolerance
configuration. It constructs its own prefixes from scratch; the v4 failure
remains an unavailable full-arm outcome, not a zero score. The new runner
does not retune a loss, learning rate, Student, or feature constraint.

The separate SportsFit independent-X attempt also failed with CUDA OOM.
Neither failed arm is retried. ArXiv v4 finished stages 0 and 1, but SSH
disconnected and a progress-print call raised BrokenPipeError. No final or
intermediate memory archive had been written. Its completed trajectories
are retained. To obtain a final-memory outcome, ArXiv is rerun in v5 with
the same numerical configuration. This infrastructure recovery necessarily
repeats those two uncheckpointed stages; it is not another independent seed.
BookHis had not started and is launched separately in v5. The runner now
saves stage memories and its output goes to a remote log, so a lost SSH
stream cannot invalidate construction. These changes do not alter either
constructor. There is no additional SportsFit run or tolerance adjustment.
