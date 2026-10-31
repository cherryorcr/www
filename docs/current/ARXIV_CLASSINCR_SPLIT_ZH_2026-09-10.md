# arxiv 类别增量切分：已生成并冻结（协议 ④ 执行记录）

日期：2026-09-10
状态：**已冻结**（预注册方案见 `ARXIV_CLASSINCR_SPLIT_DESIGN_ZH_2026-09-10.md`；本文件是执行记录与验证）。
数据：`/home/pc/research/TAGSAM/data/ogbn_arxiv_full_classincr3_*`（本地镜像 manifest：`docs/current/arxiv_classincr_2026_09_10/`）
生成代码：`code/v2_llm_fusion/make_arxiv_classincr.py`
协议版本：`arxiv-classincr3-v1`；源数据 SHA256 前缀 `6ad58b190cf08ce2a7c39bb7e5d0796c`

## 1. 切分结果

| 阶段 | 类别数 | pool 节点 | 占池比例 |
|---|---:|---:|---:|
| stage 1 | 13 | 45,199 | 33.4% |
| stage 2 | 14 | 45,122 | 33.3% |
| stage 3 | 13 | 45,152 | 33.3% |
| 合计 | 40 | 135,473 | 100% |

**类别组成**

- **stage 1（13 类）**：cs.cv, cs.ds, cs.dc, cs.si, cs.ir, cs.ne, cs.cy, cs.hc, cs.sd, cs.sc, cs.na, cs.ms, cs.os
- **stage 2（14 类）**：cs.lg, cs.ai, cs.ni, cs.lo, cs.gt, cs.se, cs.db, cs.dl, cs.fl, cs.dm, cs.mm, cs.gr, cs.ar, cs.gl
- **stage 3（13 类）**：cs.it, cs.cl, cs.cr, cs.ro, cs.sy, cs.cc, cs.cg, cs.pl, cs.et, cs.ma, cs.ce, cs.pf, cs.oh

**评测分区**（与 Cora/Citeseer 完全一致）：

```
dev  = node_id % 10 == 1    16,935 节点
test = node_id % 10 == 2    16,935 节点   test_accessed: False（尚未读取）
pool = 其余                 135,473 节点
```

## 2. 生成的文件

| 文件 | 内容 |
|---|---|
| `_delta_t{1,2,3}.pt` | 各阶段类别的 pool 节点（45,199 / 45,122 / 45,152） |
| `_cumulative_t{1,2,3}.pt` | 累积 pool（45,199 / 90,321 / 135,473） |
| `_real_t{1,2,3}.pt` | 含 dev/test 的完整诱导子图（56,592 / 56,295 / 56,456） |
| `_manifest.json` | 切分规则、类别映射、SHA256、评测分区 |

命名与既有 `temporal3` 文件一致，现有 runner 只需替换路径前缀。

## 3. 验证记录

- ✅ 三阶段 pool 规模几乎完全相等（45,199/45,122/45,152），最大偏差 <0.2%；
- ✅ 类别数 13/14/13；
- ✅ dev 与 test 各 16,935；test 未读取（`test_accessed: False`）；
- ✅ 类别名与索引的对应正确（由生成器直接读 `all_labels`，非硬编码）；
- ⚠️ **长尾未消除**：`cs.gl` 仅 24 个 pool 节点、dev 上仅 **1** 个；`cs.os` 102、`cs.oh` 311、`cs.pf` 326、`cs.ms` 326、`cs.ar` 325、`cs.gr` 404。

## 4. 必须处理的长尾问题（沿用设计文档 §2 方案 B1）

**不做类别合并**（合并会破坏与官方 GraphCLIP 及同领域工作的可比性）。改为**双指标报告**：

1. 全部 40 类的 Macro-F1（主指标，如实报告）；
2. **去掉 pool<500 的 7 个类**（cs.gl 24、cs.os 102、cs.oh 311、cs.ar 325、cs.pf 326、cs.ms 326、cs.gr 404）后的 Macro-F1（辅助指标）。

理由：`cs.gl` 在 dev 上只有 1 个节点，其 F1 只能是 0 或 100，进入宏平均等同于掷硬币——这不是方法差异，是评估噪声。**必须在结果 JSON 中逐类记录节点数与 F1**，以便审阅核查。

## 5. 两个必须记下的过程教训

**教训 1：不要把索引—名字的对应关系"假设"出来。**

首次设计时我把 `all_labels` 的顺序假设为按池规模降序（沿用早前 dump 的行序），据此生成了错误的名字表（把 cs.cv(21844) 标成了 cs.na(445) 的位次）。**实际 `all_labels` 的顺序与池规模无关。**

修正：生成器**从数据集直接读取 `all_labels` 并建立 name→index 映射**，而不用硬编码名字表；生成后逐类交叉核对（名字↔计数）才发现并排除。

**教训：涉及索引—名字映射的数据集处理，必须从数据源构造映射并逐项验证。** 此类错误会静默产生看似合理的输出（本切分的规模是均衡的，只有名字标错），极难从统计量上发现。

**教训 2：快照的 schema 必须与既有运行器的**语义**一致，而不只是字段名一致。**

第一版生成器把 `arrival_stage` 填成了**全 0**（每个快照内所有节点同值）。字段名与 temporal3 相同，但语义错了：`eval_stagewise` 用 `arrival_stage` 把评测节点**按来源阶段分组**（`mask = dev_mask & (stage_arr == s)`），因此需要每个节点记录"它的类别是第几阶段引入的"。

结果是持续部分全部返回 `NaN`（`per_stage` 的 mask 为空 → `Mean of empty slice`），而静态部分正常——**如果不检查输出只看是否崩溃，很容易漏掉**。

修正后的正确语义：

| 快照 | arrival_stage 取值 |
|---|---|
| `delta_t1` | {0} |
| `delta_t2` | {1} |
| `cumulative_t2` | {0, 1} ← 关键：累积图含多个阶段，必须区分 |
| `real_t2` | {1}（该阶段自身的类） |

**教训：复用既有 runner 时，必须逐字段确认语义（尤其是任何被用作分组键的字段），并用一个小样例验证输出不是 NaN。**

## 6. 下一步

1. 改造 runner 以读取 classincr 切分（替换 temporal 路径前缀 + 适配 `node_idx` 字段）；
2. 重跑静态三阶段主对照：method（append+c15）/ recluster / full_history / raw；
3. 重跑持续场景，验证预期（**类别增量下 full_history 应明显变差，与 Cora/Citeseer 同向**，方法与基线的差距应拉大）；
4. 补 v1.2 扰动模块在持续场景下的判定（此前只在静态测过）；
5. 全部结果写入新文档；给旧 temporal 文档加"边界实验"标记。
