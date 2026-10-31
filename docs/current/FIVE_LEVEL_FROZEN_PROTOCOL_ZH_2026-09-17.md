# 五层小图对比：追加实验冻结协议

2026-09-17。范围限定为 Cora / Citeseer，不启动大图实验。

## 复用与新增

1. 持续合成追加：复用 `small_comparison_audited_r2_2026_09_16` 的 v4。
2. 全历史重凝聚、旧合成记忆与新数据重聚类：复用同一已审计目录。
3. 新增静态全历史 TAGSAM 响应匹配适配：self-loop / semantic 两臂。
4. 完整数据等步数、8192 步及 selector-dev 最佳检查点：复用已审计目录。
5. 新增完整数据监督式 GCN 的有限网格搜索参考，独立于零样本主表。

## 共用边界

- 三阶段类别顺序：Cora `[0,1]/[2,3]/[4,5,6]`；Citeseer `[0,1]/[2,3]/[4,5]`。
- original ID 模 10：1 为 selector-dev，3 为 report-dev，2 为 test，其余为 pool。
- 本轮不评估 test 标签；项目历史开发使用过 test，不能声称最终盲测。
- 合成记忆预算为累计合法 pool 的 5% 向下取整：Cora 25/67/94，Citeseer 38/86/111。
- canonical 下游：GCN hidden/output 384、MiniLM 384、batch 16、Adam 0.001、InfoNCE、seeds 100..105，步数 `max(50,min(2048,16*K))`。
- 评估在真实全图上传播，属于 transductive；完整图输入不等于允许训练读取 held-out 标签。
- 本轮新增方法的超参数不依据 report-dev 结果调整。

## 静态 TAGSAM 适配

- 每阶段可读截至当前的全部合法原始 pool，独立重建全部合成行。
- 初始化复用全历史 joint/dispersion/concat 构造，合成 X 可优化，文本和拓扑固定。
- 官方仓库 SundayVHan/TAGSAM，提交 `d812f06fd16db3cd4d497804a8e504d2c738faa8`，直接提取未经改写的 `wBCELoss`。
- teacher 4096 步，seeds 9000/9001/9002；outer 500，inner 10，batch 16，reference cap 256。
- 合成 X：SGD lr=100、momentum=0.5、梯度裁剪5；可蒸馏 inner rates 初值0.005，outer lr=2e-6。
- 固定终点，无 report-dev 驱动的选择；每阶段六个 canonical student seeds。
- 这是同协议响应匹配适配，不是官方完整复现。没有官方 NPMI 文本选择器，未使用官方编码器、5000 outer、15 inner epochs 或蒸馏下游学习率，不能据此宣称超过官方 TAGSAM。

## 监督式参考

- 两层 GCN hidden384，输出为数据集全部类别，输入与图拓扑保持真实全图。
- 每阶段从头训练，仅对截至当前的合法 pool 节点计算分类交叉熵。
- Adam 固定网格：lr `{0.001,0.01}` × weight decay `{0,0.0005}`。
- 每阶段用独立 seeds 7001/7002 的 selector-dev 最佳 Macro-F1 均值选择配置；相同得分保留先枚举项。
- 再用 seeds 100..105 从头训练选定配置，每25步评价 selector-dev；至少512步，连续20次无提升后停止，上限8192步。最佳检查点的 tie 保留较早步。
- 搜索阶段完全不计算 report-dev；最终六个 seeds 仅使用 selector-dev 选检查点，然后报告 report-dev。
- 保存终点和最佳模型、训练 ID、哈希、选择曲线、实际步数和耗时。
- 称为“有限搜索监督式完整数据参考”，不是理论上界或穷尽调参的最优模型。full-batch 监督训练与 InfoNCE mini-batch 的计算不等价。

## 验收

验证 split 隔离、累计来源、预算、六 seeds × 三阶段完整性、检查点/JSON/参数哈希一致、选择曲线最大值与 selected step 一致。报告均值及 population std、阶段均值、旧类/新类 F1、训练与构造成本。监督结果另表；静态 TAGSAM 的适配限制需紧邻数字说明。
