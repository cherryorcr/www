# In-loop CaT-Q：外部基线与组件消融补充计划

## 冻结边界

主方法仍为 `uq_dm_edgefree`。本轮不改 Student、不扫描参数、不使用封存 test 选择版本。
主方法的有效配置来自 `run_unified_q_inloop_cat_20261007.config`，测试逐字段核对。
启用：可学习 Q、5% 受限 X 残差、in-loop CaT 风格 DM、多模态 GM、传播一致性、Q/R 正则和真实文本 ST 材料化。
关闭：CaT 预训练 anchor、旧 memory 更新、新增拓扑、boundary sketch、EM。

`-1e4` 只在分配 logits Z 中占位；实际 Q 跨类别元素严格为 0，且这些 logits 的梯度为 0。
ST 文本前向等于真实候选 embedding，反向使用软加权 embedding 的导数；不更新 MiniLM 或字符串。
ST 是有偏代理梯度，不保证 hard argmax 文本一定改善。

## 统一协议

- 数据：WikiCS、SportsFit、BookHis；同一 pool/report-dev/stage-ID hash。
- 构造访问：当前 raw delta + 本方法历史 synthetic memory，冻结旧前缀。
- 三个顺序：9101、9102、9103；构造 seed 271006；Student seeds 150/151/152。
- 主比较固定参考逐阶段 rows，不把它称为所有顺序均严格 5% fixed-rate。
- 相同 Prompt Student、MiniLM、2048 steps、batch 24、每阶段重新训练。
- 评价：report-dev；推理在完整原图上使用类别 prompt。不是严格未见类别 zero-shot。
- 报告 F1、Accuracy、阶段均值、Old/New F1、最差阶段和构造/训练成本。

## 外部基线表

只列外部方法适配和通用参考，不列 no-GM 等内部消融。

| 原方法/类别 | 当前名称 | 适配边界 |
|---|---|---|
| CaT | CaT-CCTP adapter | 当前 delta 的 CaT 风格 MMD 优化自由 X，附着 top-5 真实文本；冻结前缀 |
| GECC | GECC-hard TAG adapter | hop 传播与 hard Lloyd/restarts，附着真实文本；不是官方完整 fuzzy GECC |
| PUMA | PUMA-inspired TAG adapter | 当前标注 pool 内模拟 withheld/pseudo-label 子集，再凝聚；不是官方完整 PUMA |
| Random subset | Stratified random coreset | 相同类别配额选择实际节点与文本 |
| K-center | Stratified graph-text k-center | X/text 联合空间 farthest-first |

复用已有 9101/271006 参考 cell；新增 9102/9103 × 三数据集 × 四个非 CaT 基线，共 24 方法 cell。
CaT-CCTP、主方法已有 3 顺序 × 3 构造 seeds。新基线仅 1 构造 seed，三顺序表必须按相同子集配对，不能拿 9-cell 平均和 3-cell 平均直接配对。
官方 TAG 静态实现另外审计；可适配则加入同协议表，不可适配则单列官方/离线 reference，明确 raw 访问、Student、预算差异。任何自制近似都不得标为官方复现。

## 组件消融表

每个版本从头构造完整五阶段链，各有三 Student seeds；三图均执行同一配置。

| arm | 唯一改动/目的 |
|---|---|
| uq_dm_edgefree | 本批完整主方法参考，不与旧同 seed 重复当独立样本合并 |
| uq_no_dm | 去掉 CaT 分布匹配 |
| uq_no_prop | 去掉传播一致性 |
| uq_no_residual | R 始终为 0，只通过 Q 生成 X |
| uq_frozen_text | 固定为 Q0 的文本选择，X 仍随 Q 更新 |
| uq_detached_text | 当前 Q 仍硬选文本，但切断文本到 Q 的 ST 反向路径 |
| uq_no_text_init | 初始化 signature/score 不含文本相似度，保留后续 GM 文本分支 |
| uq_no_q_anchor | 去掉 Q-Q0 正则 |
| uq_no_x_anchor | 去掉 R 正则，仍保留 5% trust region |
| uq_dm_only | 只保留 DM；保留 Q/Sinkhorn、受限 R 和同一初始化/材料化，用于联合简化诊断 |

新增 30 方法 cell；此前 no-GM 的 9-cell 确认与 fixed-Q 对照单列复用证据，不冒充本批完整配对。
本轮尚不测试去掉 class mask、去掉预算可行性或读取历史 raw；这些会改变任务/安全边界。
单独删项不下降只说明未发现增益，不证明永远无效。联合简化版只是候选，不能按最好单 cell 自动晋升主方法。

## 存储与状态

远程根分区仅约 3.6 GB 可用；不删除既有数据。
本批 compact runner 在 RAM 中保留当前链，只保存 JSON 指标、配置、来源和 memory hash，不落盘完整 memory。
因此本批不支持部分 arm 断点恢复，也不能仅凭 hash 恢复模型输入；此限制必须在最终结果披露。
失败的部分 arm 会保留并从新目录重跑，不覆盖失败证据。

- 本地原回归测试：23 passed。
- 新增分配屏蔽/ST/config/消融测试：18 passed；远程 tagsam 同样通过。
- 五阶段 WikiCS 短 smoke 通过；4 Student steps/1 Q step 的数值不进入正式表。
- 正式基线与消融矩阵：运行中；WikiCS 十个消融已完成。
- 首次 WikiCS 去 Q-anchor 在全矩阵求和误差检查停止，未完成该 cell；旧日志保留，其余七个完整 arm 不重复跑。
- 后续审计保持类内 Sinkhorn 求解阈值 1e-4；仅全矩阵零填充求和另允许 8 个 float32 epsilon，并记录实际超出量，不改变优化过程。尾部三个 arm 在新目录完成。
- 额外三个 TAGSAM-response 适配 cell 已通过五阶段短 smoke，正式配置预注册：2048 teacher steps、100 outer × 10 inner、reference cap 256，固定官方 wBCELoss。它不是官方全流水线复现。
- 官方 TAGSAM 原生配置在目标三图上还缺输入格式转换和原生模型/teacher；官方 STAD 还依赖 Qwen 文本生成。未完成原生结果不能写成同协议数据。
