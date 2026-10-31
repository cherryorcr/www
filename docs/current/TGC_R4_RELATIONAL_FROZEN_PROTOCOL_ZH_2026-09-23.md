# TGC-R4 关系响应实验执行协议

日期：2026-09-23。关联：[设计审查](TGC_R4_EVIDENCE_REVIEW_AND_REDESIGN_ZH_2026-09-23.md)。

## 1. 范围与版本

- 本轮必做：正确性验证、teacher 支持图 2x2 诊断、shared-parent 小图完整候选矩阵、六 seed 正式 student、重载审计、成本与机制报告。
- 条件任务：主候选通过小图门槛后，才做独立确认、递归 rollout、rate/base+4、大图与 LP。未触发的任务不得写成完成。
- `20260923` 是首次 preflight 中止目录；`20260923b` 是采用放宽 GPU 容差的试运行，不能当严格训练路径对齐验收。
- 真实图诊断发现默认 CUDA foreach Adam 与原 functional single-tensor 算术不一致。局部 32 步最大参数误差为 0.0006003；native 重复、functional 重复均为零，single-tensor native 对 functional 约 3.35e-8。这是该图的算术路径差异证据，不是所有 GPU 噪声的归因。
- 正式 `20260923c` 使用独立 R4 foreach 可微实现，恢复 atol=2e-5、rtol=2e-4；不得只放宽断言。共同 student、历史 R3 模块不变。预检包括实际八个增量 cell 的 1/4/32 步和 CPU/GPU 有限差分。
- c 的优化参数、比较臂、门槛沿用 b 跑分前写入的 machine protocol；版本修复不依据 b 的 report F1 调参。b 已看过，c 的相同顺序属于开发条件，不是未见泛化。

## 2. 固定协议

| 项目 | 配置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 开发顺序 | order19341、order19342；实际类别分组写入 protocol/manifest |
| 保留确认顺序 | order19441、order19442；进入确认前还需检查与历史实验的实际类别分组重合 |
| 阶段 | 3；s0 各臂共用 C0，s1/s2 从相同未修订 C0 父记忆出发 |
| 预算 | 累计合法 pool 节点数的 5%，只增加 rows |
| 初始化 | dispersion 配额、joint_g5_c320、mean-X、concat 后冻结 MiniLM 重编码、frozen-old topology、cross degree=1 |
| teacher | seeds 8000/8001、128 步、batch64、Adam 0.001 |
| 支持诊断 | legacy/lifted topology × uniform/compact-ratio 无放回采样 |
| 正式候选支持 | 事前固定 lift_compact，不按 report 或 teacher 分数逐 cell 挑选 |
| inner student | scratch、batch16、Adam0.001、32 步，train seeds8100/8101 |
| 外层 | 16 步、Adam0.01、每行相对位移上限0.1、梯度范数裁剪5 |
| 旧关系约束 | 平均双向 KL≤0.001；非负 dual 初值1、更新率10；末端缩放1到0保证 train probe 可行 |
| validation | seeds8110/8111、独立 probe query；仅诊断，不选 outer checkpoint |
| report student | seeds100..105，步数max(50,min(2048,16×rows))；两层GCN、InfoNCE，全部从头训练 |
| 正式指标 | report-dev、all-prompt pooled Macro-F1；同步报告 seen/old/new、阶段均值 |

验证 query 与外层 train query 不重合，但 teacher 使用同一合法支持图，不能称为 teacher 未见样本。新端 softmax 候选包含旧、新两侧。旧端约束参照是未修订 C0 的配对 scratch student，不是旧 teacher。约束不保证真实历史分类无遗忘。

## 3. 完整小图矩阵

1. C0。
2. relation_new：主候选，只优化新 X。
3. random_new：同样行、逐行相同位移范数的随机方向。
4. stop：切断 student-to-X 梯度，必须与 C0 完全相同；按同 memory hash 复用评估。
5. infonce_new、infonce_old：同路径实例级对比目标。
6. relation_old：只优化最多旧行数10%的非保护旧 X，80%旧行作为保护集。
7. asm_new：同 teacher、horizon、Adam 的旧 ASM 双向 BCE/new-new 子矩阵控制，不是官方 TAGSAM 完整复现，也不只改变 KL/BCE 一个因素。
8. relation_new_free、relation_joint、relation_shuffled：无旧约束、有限旧新联合、打乱 teacher 对应的机制消融。

共 2数据集×2顺序×3阶段×11臂×6seed=792 个结果槽位；s0 共图和 stop 共图去重后应为 504 次真正训练。消融即使主候选未通过也执行用于诊断，但不据此事后更换主候选。

## 4. 访问与审计

- 只有当前 pool 的 class labels 参与 C0 来源分块；R4 teacher/优化不访问 report/test/classification labels。
- 离线 loader 物理读取完整 fixture，实验过程并非 OS 隔离。算法构造只将当前 delta 切片和旧合成 memory 传给 teacher/optimizer；不能把逻辑访问审计写成“从未物理加载全图”。
- 真实全图仅用于共同 transductive 评估；含未标注未来/holdout 的输入及边，此条件对全部方法一致。
- X 以外所有 memory 字段逐项不变；候选、teacher目标、支持图、决策、协议均哈希封存后才开始 report。
- 临时支持图、teacher、source assignment 只作离线审计，不是部署 memory，不供下一阶段算法读取。存储成本区分部署载荷与实验归档。
- 重载每个 checkpoint，独立 TP/FP/FN 复算所有 F1，核查配对 seed、rows、step、memory hash 与 no-op。

## 5. 晋升与停止

主候选 relation_new 的最终阶段四 cell 等权差值须同时满足：平均≥+0.5pp，至少3/4≥−1pp，最差≥−2pp，旧类最差≥−2pp，新类平均≥0pp，相对 matched-random 平均≥0pp，单阶段构造≤600秒。配对区间必须报告，过工程门槛不等于统计显著。

任何消融的事后最佳均值都不是已经确认的新主方法。主候选未通过时，完成本轮必做矩阵和报告后停止扩展；独立确认、递归、大图和 LP 保持未执行。若通过，另冻结独立 seed group 和共同 k-center/ASM 确认，不沿用开发 report 当测试。

不以 C0 毫秒级构造的任意倍数掩盖绝对成本。记录 teacher、四条件诊断、候选优化、正式 student、memory bytes、峰值显存，以及可复用研究成本与单个部署算法成本。
