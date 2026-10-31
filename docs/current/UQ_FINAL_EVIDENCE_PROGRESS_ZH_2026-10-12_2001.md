# Unified-Q-EF 进度记录（20:01）

## 论文证据层级已校正

- `paper_www/main.tex` 摘要不再把 ArXiv 归档 5.71% 单顺序结果（52.25/71.59、相对 GECC +1.16 F1）称为精确 5% 主结论。
- `paper_www/sections/06_results.tex` 单列 ArXiv 合法构造池真实 5% 的 sealed 九个配对 cell：Unified-Q-EF 52.91/71.15，GECC-hard-TAG 52.68/70.94，CaT-CCTP 34.98/60.41（Final Macro-F1/Accuracy，百分数）。相对 GECC 的 F1 配对差 +0.22 pp，95% CI [-0.34, 0.79]，4/9 正向；不能称稳定超越。
- 原 5.71% ArXiv、SportsFit/BookHis report-dev 表和 full-history retention 表仍保留，但已明确它们不是上述 sealed 确认。新增配对图 `paper_www/figures/arxiv_exact5_sealed_paired.png`。
- `paper_www/sections/07_ablation_discussion.tex` 补 BookHis 单顺序三构造种子归因：shared Q 46.55、independent Q 48.47 sealed F1，配对差 -1.92 pp，0/3 正向；不能以共享 Q 宣称精度因果增益。
- 拓扑限制段已按现有固定 link probe 更新；Unified-Q-T 仍是扩展，不是训练式 GAE/GNN 结构任务的确证。

## 远程队列

- GPU0 的 GECC-soft BookHis 与 GPU1 的 Q 控制组仍在跑；约 29 GB 磁盘可用，没有新的完整比较可汇入主表。
- 原 `run_uq_remaining_confirmation_batch_20261012.sh`（PID 124389）等待 GPU1 控制队列（PID 110886），之后跑 BookHis/SportsFit 的顺序 9101、三个构造种子。
- 新 `tools/run_uq_remaining_orders_20261012.sh` 已通过远程 `bash -n`，以 PID 125140 等待上一队列**完整成功**，再跑这两个数据集的顺序 9102/9103、三个构造种子。保留磁盘/GPU 护栏、完整结果复用和部分输出停止规则；不会覆盖 sealed 结果。

## 验证与边界

- `git -C paper_www diff --check` 通过。内置 LaTeX 编译器连续三次在预览环境准备阶段返回 `windows sandbox: helper_unknown_error: setup refresh had errors`，没有得到 TeX 源码诊断；不能据此声称 PDF 已成功编译。
- 尚未把进行中的控制组、GECC-soft 或后续顺序写入终点表。SportsFit/BookHis 的 sealed 配对统计必须等九个完整 cell 后重新独立汇总；不能用旧 report-dev 值替代。
