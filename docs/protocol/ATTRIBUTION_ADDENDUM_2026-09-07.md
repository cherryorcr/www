# 原生 TAG v1：归因执行补充

日期：2026-09-07。继续执行用户授权的首版实现、可行性与持续算法验证。

## 已发现的评测问题与修正

旧长配置每128步检查，所有静态full/frozen学生都在第一个检查点被选中，不能视为峰值已充分观测。保留旧结果，追加每32步检查的512步独立重训；原始小图与模型种子不变。step0明确记录并参与选模，同时报告相对step0的训练收益。它是允许的保底候选，选中它意味着未证明训练有益。

新runner补齐了 `steps=0`、旧memory-only和当前delta-only诊断、逐阶段曲线。没有改变损失、标签权限或指标定义。各臂共享名义batch上限，尾批保留；25 roots产生20/5尾批，全量不一样，所以相同步数不等于完全相同样本暴露。报告examples_seen与选点，不宣称负例分布完全相同。

## 配对归因

`run_native_attribution.py` 从一个已完成运行的 stage0 导出小图开始，核对数据manifest、根预算、文本来源ID和stage。所有online分支复用完全同一份旧小图。旧raw只进入明确标识的full-history/offline参照。

1. static：同一旧artifact的full/random/coreset/frozen/ASM-selected/ASM-last每32步检查重训，区分早期选点与优化效果。
2. source：full-history、old-only、delta-only、old S+delta、同一S+delta仅按代表mass加权对比损失。后两者的teacher和再次凝聚均未参与，先检查替代阶段的效用变化。
3. recondense：offline frozen/ASM、union frozen/ASM、append frozen/ASM，以及只改变reference抽样的reference-quota探针。Union与探针共享teacher、初始X/文本/边、内层损失、训练预算与独立student种子。探针只把uniform reference替换为按已保留stage代表mass分配、受现有行数限制的无放回配额。

这些差值不是可相加的因果分解。旧/当前F1是在各阶段样本内统计，不能简单平均还原累计F1；跨阶段误分类带来的FP会改变累计指标。

reference-quota探针的根据是压缩造成的参考行数失衡：5% anchor的旧信息代表510个训练根，但只占合法source的25/874行。固定seed实际128个reference仅4行来自旧阶段。该观察证明参考覆盖发生变化，不证明它是下游性能主因；普通mass加权也不自动等于恢复完整旧图的训练响应。

## 后续有界验证登记

保持原生GCN/BERT模型、Cora固定七类候选与现有prepare_v2不变。新的共同静态配置为student512/teacher800/validation512、check_every32、link160、outer100/eval20、inner10/inner_lr0.005、outer_lr0.01、reference128。凝聚seeds44/45、比例1%/5%、student100/101，均为checkpoint开发结果。两种比例用于预算敏感性，两个凝聚seed用于核查单产物偶然性；不将两个student重复当成两个独立凝聚seed。

先完成上述归因，再决定是否把一个机制接入三阶段递归。若ASM仍只选回初始化，应如实报告静态优化门槛未通过，持续选择/配额探针可以运行，但不能冒充已验证的可学习凝聚贡献。首轮归因和外层学习率补充均只依据开发checkpoint确定运行配置，确认分区随后只做冻结候选的读取；不将本轮Cora命名为最终测试。

## 实现与可复现性

- `run_native_experiments.py`：通用静态/三阶段递归runner与独立导出重训。
- `run_native_attribution.py`：从同一旧小图进行配对归因，生成source/reference审计。
- `native_core.py`：ASM双层优化、step0、uniform/mass_stratified参考选择与审计。
- 新运行保留完整参数、源代码快照、输入与artifact SHA256；旧运行目录不覆盖。

截至本补充建立时，28项单元测试通过。数值结果另行生成归因报告，不回写九份早期冻结报告。

## 保留式原型登记

观察：5%两阶段Union初始化只保留旧25个文本代表中的14个；Append保留25个，且该开发seed的效果较高。1%中Append又低于Union，说明不能将该观察推广为普遍结论。参考配额探针在两比例下均选回初始化，尚未建立效用收益。

据此实现两个配对候选：`append_joint`从旧S与当前delta的紧凑初始化拼接后，以合法U=旧S+delta训练joint teacher并进行ASM，所有X可更新；`protected_joint`完全相同，但梯度只进入新增行的X，旧X/文本/内部边保持原样。两者共享每阶段delta初始化、源数据权限、名义预算与目标。与Append的区别包含joint teacher、joint inner训练和joint reference，不能将其差值只归于一个loss项。两候选相互比较才单独检验冻结旧X。

预算K0/K1/K2在5%下为25/67/94，在1%下为5/13/18。保留式原型只适用于累计比例带来的非递减预算，不解决固定容量下历史压缩淘汰问题。合成跨阶段边仍为空，此处的joint指跨阶段图文对比和响应匹配。

三阶段5%执行凝聚seeds44/45，另以1% seed44执行两阶段敏感性，均从各自已导出的stage0 ASM memory出发。添加冻结行精确不变与范数投影测试，27项测试通过。该原型目前是保留约束的工程基线，新颖性与增益均未成立。

复现审查发现早期attribution运行的run_config继承了静态arms字段；真实执行臂以results.json为准，脚本现已记录实际列表。旧目录不覆盖。旧运行仅记录当前artifact SHA，未与base结果登记的hash交叉比对；后续补做只读一致性审计，新增运行在加载时强制核对。

外层稳定性补充：固定5%静态协议，仅把ASM outer_lr从0.01改为0.001。seed44 checkpoint/confirmation为74.17/74.00，Frozen为75.64/76.36；seed45为80.87/74.68，Frozen为80.05/76.75。确认集没有收益，说明降低过冲不能单独解决student验证选点/凝聚效用不稳定。该运行仅作为开发诊断，不替换主表。

权限措辞：确认指标已经在冻结候选后读取，但不参与此前选模；确认标签仍位于eval文件，评估使用完整当阶段图的无标签消息传播，不称为物理隔离或严格节点归纳设置。当前mass加权在小批内归一化，不能解释为全局梯度比例恢复。

## 配对teacher补正

首轮preserve运行中，相同source/seed重新训练的teacher出现参数差异，例如5% seed45的stage2最大参数差0.00272。原因尚未完全确认，GPU数值非确定性是候选解释。即使配置相同，也不能声称这些teacher实例完全相同。

脚本改为按合法source七个数据字段的内容指纹缓存并复用同一个teacher实例，审计记录实际共享来源。旧结果保留为探索记录；主要方法对照以带sharedteacher后缀的新运行结果为准。若后续阶段两个方法的旧memory本身不同，就不共享teacher，属于完整递归方法比较；单阶段固定旧memory的对照才检验单个机制。
