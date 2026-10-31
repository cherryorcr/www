# 大图五层基线实验状态（2026-09-26）

## 当前状态

本轮目标是在统一 `base+4`、固定 5% memory、同一 Student 协议下完成五层比较：

1. C0 / semantic-recursive continual append；
2. random、k-center、real-node FPS coreset；
3. full-history synthetic re-condensation；
4. complete raw graph with equal steps；
5. complete raw graph with 8192 steps。

目前本地已完成并通过协议审计的是前 3 层及其构造/Student 变体；第 4、5 层的远程 raw 参考尚未同步到本地，不能视为完成。

## 已完成结果

本地汇总文件：

- `results/tgc_large_baseline_comparison_20260926.json`
- `results/tgc_large_baseline_comparison_20260926.md`
- `results/tgc_large_baseline_protocol_audit_20260926.json`

三个大图 `arXiv`、`BookChild`、`SportsFit` 均已完成五阶段的 C0、semantic-recursive、random real-node、k-center、real-node FPS、full-history synthetic 和 GPU full-history 结果。协议审计为 `all_pass=true`；其中 full-history 两个臂明确标记为 offline reference，因为它们可回读历史 raw 并重写历史 memory。

semantic-recursive 相对 paired C0 的增量阶段结果：

| 数据集 | cumulative 平均增益 | new-class 平均增益 | 正阶段 | 最差阶段 | 最终 C0 / semantic |
|---|---:|---:|---:|---:|---:|
| arXiv | +0.637 pp | +2.971 pp | 3/4 | -0.421 pp | 18.817 / 19.227 |
| BookChild | +0.203 pp | +2.241 pp | 3/4 | -1.552 pp | 12.607 / 13.313 |
| SportsFit | +1.735 pp | +0.080 pp | 3/4 | -2.381 pp | 25.013 / 28.076 |
| 合计 | **+0.858 pp** | **+1.764 pp** | **9/12** | **-2.381 pp** | — |

该结果支持 semantic-recursive 在大图上的方向性机制信号，但不能支持其已经稳定优于 C0、coreset 或所有离线参考。

## 尚未完成的 raw 参考

远程服务器上的任务为：

```text
run_tgc_large_raw_reference_20260926.py --dataset arxiv
```

其后台链会在 `arXiv` 完成后依次运行 `BookChild` 和 `SportsFit`。目标输出目录为：

```text
/home/pc/research/TAG_condensation/results/tgc_large_raw_reference_20260926/
```

每个数据集应最终出现：

```text
<dataset>_base4/results.json
```

并包含 `raw_equal` 与 `raw_long` 两个 arm，各 5 个 stage。截止本状态记录，远程节点 `100.126.175.112` 的 Tailscale 节点仍可见，但 SSH、TSMP 和新连接均超时；本地没有同步到任何 raw `results.json`。因此：

- 没有重新启动 raw 任务；
- 没有删除或覆盖远程输出；
- 没有把 raw 结果填入五层主表；
- 没有据此宣称完整 raw 参考已经完成。

## 恢复后的收尾顺序

服务器恢复后，先检查远程进程和结果完整性，再同步整个目录：

```powershell
ssh pc@100.126.175.112 "ps -p 3540989 -o pid=,etime=,stat=,cmd=; find /home/pc/research/TAG_condensation/results/tgc_large_raw_reference_20260926 -name results.json -printf '%P %s\\n' | sort"

scp -r pc@100.126.175.112:/home/pc/research/TAG_condensation/results/tgc_large_raw_reference_20260926/* `
  results/tgc_large_raw_reference_20260926/
```

同步后必须确认三个文件各含 10 行结果、两个 arm 和五个 stage，然后运行：

```powershell
python code/v2_llm_fusion/summarize_tgc_large_baseline_comparison_20260926.py `
  --c0-root results/p5_large_base4_20260919 `
  --semantic-root results/tgc_semantic_recursive_large_20260926 `
  --coreset-root results/tgc_large_coreset_approx_bfps_20260926_remote `
  --full-history-root results/tgc_large_full_history_20260926_remote `
  --full-history-gpu-root results/tgc_large_full_history_gpu_20260926 `
  --raw-root results/tgc_large_raw_reference_20260926 `
  --output results/tgc_large_baseline_comparison_20260926

python code/v2_llm_fusion/audit_tgc_large_baseline_protocol_20260926.py `
  --c0-root results/p5_large_base4_20260919 `
  --semantic-root results/tgc_semantic_recursive_large_20260926 `
  --coreset-root results/tgc_large_coreset_approx_bfps_20260926_remote `
  --full-history-root results/tgc_large_full_history_20260926_remote `
  --full-history-gpu-root results/tgc_large_full_history_gpu_20260926 `
  --raw-root results/tgc_large_raw_reference_20260926 `
  --output results/tgc_large_baseline_protocol_audit_20260926

python code/v2_llm_fusion/write_tgc_large_five_level_report_20260926.py `
  --summary-json results/tgc_large_baseline_comparison_20260926.json `
  --audit-json results/tgc_large_baseline_protocol_audit_20260926.json `
  --output docs/current/TGC_LARGE_FIVE_LEVEL_BASELINE_COMPARISON_ZH_2026-09-26.md
```

只有在 raw 两个 arm 通过行数、stage、steps 和历史访问标记审计后，才更新 `docs/INDEX.md` 与论文草稿；论文中要把合法 continual 方法、coreset 参考、full-history synthetic 离线参考、complete raw equal-step 和 complete raw 8192-step 明确分栏。

## 解释边界

`raw_equal` 使用与 synthetic Student 相同的 2048 optimizer steps；`raw_long` 使用 8192 steps。两者都读取累计历史 raw 并允许重写历史支持，因此是离线参考，不是合法 continual constructor。`raw_long` 也不是理论上界或充分收敛证明。

在 raw 结果归档前，当前可引用的结论仅是：semantic-recursive 在三大图上的配对平均增益为 +0.858 pp，new-class 平均增益为 +1.764 pp，9/12 个增量 cell 为正，但存在数据集和阶段依赖。
