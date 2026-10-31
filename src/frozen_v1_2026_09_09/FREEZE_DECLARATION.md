# 冻结声明 frozen_v1（2026-09-09）

本目录是 Gate 3'' test 读取前的方法快照。服务器镜像：`/home/pc/research/TAGSAM/graphclip_frozen_v1_2026_09_09/`（run_gate1_continual.py SHA256 前缀 `0542c41c6534ee6d`）。

## 冻结的方法（append-构造式 v1.1）

- 静态构造器：maxmin+Lloyd 联合 [X‖text] 聚类；每簇＝均值合成X + top-g=5 文本截断拼接融合（总预算 320 字符）+ 冻结 MiniLM 自编码 embedding + mass；拓扑＝融合文本 kNN，degree=1.0；
- 持续协议：旧 memory 原样冻结；新块配额 = max(每新类 15 行, 预算余量)；腾预算时旧行按 mass 升序删除（只删行）；新块 = 静态构造器(delta)；memory = 拼接；
- 预算 rate=0.05；student=2 层 GCN+文本投影+CLIP loss，steps=min(2048, 16×roots)，lr 1e-3，batch 24；student seeds 100–105。

## Gate 3'' 规则

- test 分区定义：`node_id % 10 == 2`（10%）。**该分区标签此前从未被读取**；其节点特征/文本曾作为凝聚源参与（whole-graph transductive、label-free），协议中声明；
- 每 artifact × 每 student 只评一次；无论结果如何如实报告；读取后本冻结版不再修改（后续调整在新版本号下进行）；
- 数据集：Cora + Citeseer（均为开发数据集，最终论文如需完全未接触数据集另行扩展）。

## c 规则在两数据集的落点

- Cora s2：3 新类 → 新块 45 行（旧 memory 87→76，按 mass 删行）；s2 memory 由冻结构造器从已存 s1 memory 确定性重建；
- Citeseer s2：2 新类 → 下限 30 ≤ 余量 33，不触发 → 使用已存 append s2 memory。
