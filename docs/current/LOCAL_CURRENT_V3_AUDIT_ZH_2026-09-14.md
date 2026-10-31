# 当前 TAG 持续凝聚主线本地审计

日期：2026-09-14  
目的：在远程重跑前，确认本地项目当前真正推进的版本，避免把 D2/A2 或早期 native TAG 原型混入主线。

## 1. 版本边界

当前主线是 **GraphCLIP continual v3 candidate**：

- 核心通用构造与诊断代码：`code/v2_llm_fusion/`；
- Cora/Citeseer 类别增量 runner：`code/frozen_v1_2026_09_09/`；
- 共享协议：`code/v2_llm_fusion/protocol_v3.py`；
- arxiv continual 与诊断脚本使用同一构造器族，但属于规模/流式扩展实验。

以下内容不属于当前主方法：

- D2/A2：历史的递归 moment、coverage、GraphSAGE 专家和固定融合路线；
- `code/native_tag_v1/`：早期 TAGSAM 风格诊断后端；
- `code/graphclip_native/`：较早的 GraphCLIP Gate 快照；
- `run_v2_comparison.py`、`run_v2_full_pipeline.py`、`run_v3_discriminative.py` 等：早期 LLM/判别性文本融合实验。

## 2. 当前方法的实际流程

1. 从 GraphCLIP 数据读取节点特征 `X`、原始文本、真实边和类别提示。
2. 用冻结 MiniLM 编码节点文本和类别 prompt，得到 384 维向量。
3. 构造联合空间：`[L2Norm(X) || text_emb]`，图特征与文本各占 384 维。
4. 在联合空间中进行确定性 max-min 播种和 Lloyd 聚类。
5. 每个簇生成一行 synthetic TAG：
   - `X` 为簇内原始特征均值；
   - 选取最靠近簇心的最多 5 条文本；
   - 按约 320 字符预算截断拼接；
   - 对拼接后的完整字符串重新用 MiniLM 编码；
   - 保存 `mass`/`weights`、`depth` 和来源 `stage`。
6. synthetic 拓扑由融合文本 embedding 的全局 top-E 相似边生成，`DEGREE=1.0`。
7. student 是两层 GCN 加文本投影头，用双向 InfoNCE 在 synthetic TAG 上从头训练。
8. 评测在真实图上进行全图 transductive message passing，再与类别 prompt 比较完成 zero-shot 节点分类。

这里的“无标签凝聚”只表示凝聚和 InfoNCE 不直接读取节点标签；阶段划分和 dev 评估仍然使用标签，评估图也不是严格 inductive。

## 3. 当前持续协议

Cora：`[0,1] -> [2,3] -> [4,5,6]`。  
Citeseer：`[0,1] -> [2,3] -> [4,5]`。

预算为累计训练节点数的 5%。当前 append 规则为：

- 旧 memory 默认保持原行内容不变；
- 当前 delta 单独凝聚成新块；
- 新类至少保留 `15 × 新类数` 行；
- 若预算不足，按 `mass` 从小到大删除旧行；
- 合并后重建全体 synthetic 文本相似边。

三类持续对照：

- `append_fixed`：合规主方法，历史原图不可重读；
- `recluster`：把旧 synthetic 行和当前 raw delta 混合重聚类，允许旧新行合并；
- `full_history`：重新读取全部历史 raw 数据，是违规诊断参照，不是可部署方法。

## 4. v3 统一协议

当前共享协议文件定义：

| 项目 | 当前值 |
|---|---:|
| student seeds | `100..105` |
| teacher seed | `9000` |
| reference seed | `777` |
| method seed | `9001` |
| student batch | `16` |
| student lr | `1e-3` |
| student steps | `max(50, min(2048, 16 × roots))` |
| teacher steps | `4096` |
| raw reference steps | `8192` |
| ASM outer steps | `100` |
| ASM inner steps | `10` |
| ASM batch | `16` |
| ASM outer lr | `1e-2` |
| text fusion | `g=5`, `320` chars |
| synthetic degree | `1.0` |

协议名为 `tagsam-unified-v3-2026-09-13`。本次本地检查已把 append 配额函数、Cora/Citeseer runner 和 arxiv split 入口接到该协议上。

## 5. 已有科研证据的正确读法

当前证据支持：

- 图文对应确实影响下游；融合文本必须重新编码，而不是简单平均多个 embedding；
- append 在类别增量下有明确的历史保持解释，full-history 的表现受混合池类别主导影响；
- ASM、response-retention、几何去偏置、LLM/判别性文本融合和若干自适应判据没有形成稳定的跨数据集收益；
- k-center 的强弱结论曾被 batch/steps 配置改变，目前不能直接宣称它真正优于或劣于 ours；
- 小 synthetic 图上的 student 对 batch 和训练步数非常敏感，这是当前最重要的实验基础设施风险。

下列数字不能直接混用：

- 旧 frozen Gate 的 Cora/Citeseer 结果；
- 旧静态 external-baseline 结果；
- native TAGSAM 结果；
- arxiv temporal 与 arxiv class-incremental 结果；
- 不同 batch、student steps、seed 或评测 runner 下的结果。

所以目前不能把项目总结为 SOTA，也不能把旧 Gate 数字直接解释成 v3 统一协议下的最终性能。

## 6. 本次本地代码核验

- `code/v2_llm_fusion/*.py` 和 `code/frozen_v1_2026_09_09/*.py` 已通过 `python -m py_compile`；
- Cora runner 现在会记录 stage 0/1/2，而不是只记录最终 stage；
- Citeseer runner 的 append 配额和 raw reference steps 已与共享协议对齐；
- arxiv continual runner 现在可以通过 `--split-prefix` 显式选择 temporal 或 class-incremental split；
- arxiv recluster 的 seed 锁定改为锁定真实 farthest-point seed indices；
- scale-gate 的 95% bootstrap 上分位点已修正。

## 7. 远程重跑执行顺序

1. 将本地协议文件和当前 runner 同步到服务器 `tagsam` 环境。
2. 先做 Cora/Citeseer smoke，核对数据、MiniLM、预算、stage 行数和输出 metadata。
3. 在同一 v3 协议下重跑 `append_fixed/recluster/full_history` 和 raw reference。
4. 再重跑 ASM、jitter、scale-gate 和 Gate S2 关键消融。
5. 以同一 student batch/steps/seed 重新判断 k-center 是否更强。
6. 统一结果后再讨论 CCFA/WWW 级创新，优先寻找能同时改善历史保持、新类塑性和 batch 鲁棒性的机制，不把 D2/A2 结构直接搬回主线。

## 8. 远程封闭来源重跑已完成

2026-09-14 已在 `pc@REMOTE_SERVER_HOST` 的 `tagsam` 环境完成最终补丁同步与 Cora/Citeseer 重跑。结果记录在 [V3_CLOSED_CONTINUAL_RERUN_RESULTS_ZH_2026-09-14.md](V3_CLOSED_CONTINUAL_RERUN_RESULTS_ZH_2026-09-14.md)。

- Cora：pool counts `577/985/604`，budgets `28/78/108`；
- Citeseer：pool counts `898/1079/571`，budgets `44/98/127`；
- 两套结果均为 `test_source_sealed=true`、`test_accessed=false`、6 seeds、batch 16 和动态 student steps；
- `append_fixed` 最终阶段优于 `recluster`，full-history 仅作违规诊断参照；
- 旧版 test-source 未完全封闭的 Cora/Citeseer 数字降级为历史诊断，不进入 v3 主表。

## 9. 2026-09-15 统一执行状态

已新增 [后续完整工作计划](FOLLOWUP_WORKPLAN_V3_SCA_ZH_2026-09-14.md) 和统一 runner：

- `code/v2_llm_fusion/run_unified_v3_continual.py`：统一 GraphCLIP、TAGBench 和 arXiv 字典 schema 的 continual 入口；
- `code/v2_llm_fusion/run_v3_calibration.py`：dev-only 有限配置网格入口；
- `run_student_compatible_append.py`：新增 `bounded_refresh`、动态 rows、`mean/weighted/set_pool` 文本 operator；
- `run_external_baselines.py`：已能通过统一数据适配器处理 WikiCS、BookHis、BookChild、SportsFit。

正式小图结果已保存到 `results/formal_unified_*`、`results/formal_baseline_*`、`results/formal_text_*`、`results/formal_dynamic_*` 和 `results/formal_refresh_*`。当前证据显示：动态 rows 和 bounded refresh 值得继续；二者直接叠加会退化；pooling/set text operator 没有跨 Cora/Citeseer 的普适收益；k-center/kmeans-medoid 仍是强基线。大图 canonical continual 已完成，结果见 `CANONICAL_CONTINUAL_MAIN_TABLE_V3_ZH_2026-09-15.md`；外部 continual 方法已完成严格协议审计，但没有可直接横比的公开同协议 SOTA。

## 10. 2026-09-15 大图执行修正

- 大图文本 embedding 已改为 run-local cache，避免每个 arm/stage 重复调用 MiniLM；
- raw pool 达到阈值后，候选构造改用 GPU 分块 max-min/Lloyd，topology 改用分块 top-k；小图路径不变；
- arXiv runner 优先读取已冻结的 `ogbn_arxiv_full_classincr3_delta_t{1,2,3}.pt` 类别增量划分，不再使用临时 round-robin 划分；
- 已完成 arXiv、BookChild、SportsFit 的六 seed `append_fixed + SCA` canonical，SCA 记录独立 selection seeds、候选路径和每阶段 response 结果；
- 已完成三套大图 `recluster/full_history` 诊断，并将 `old/current` 分解和 stagewise cumulative F1 写入统一主表；
- 已完成三套大图 `random_real/k_center/herding/kmeans_medoid` matched continual baseline，统一结果见 `results/baseline_v3_2026_09_15_*`；
- 同协议外部方法审计见 [MATCHED_SOTA_COMPARISON_V3_ZH_2026-09-15.md](MATCHED_SOTA_COMPARISON_V3_ZH_2026-09-15.md)。
