# v4 主线小图验收报告

日期：2026-09-15  
协议：unified v3、fixed-rate 5%、3-stage、batch=16、canonical student seeds 100--105、test 封存

## 1. 本次完成内容

已将 v4 主线实现为一个可运行 preset：

```text
--v4-mainline
```

该 preset 固定：

- fixed-rate budget；
- matched keep/new quota；
- dispersion 作为稳定 allocation prior；
- selector seeds 300/301/302，与 canonical student seeds 分离；
- frozen-old topology；
- synthetic-only append；
- 多 horizon response curve；
- response LCB 风险下界；
- worst-horizon gain 门禁；
- selector 成本和实际 rows 写入 manifest。

默认不启用 reserve extra budget。reserve 通过 `--v4-enable-reserve` 单独开启。

## 2. matching baseline 与 v4 结果

两组结果均使用同一个最新 runner，避免与历史 runner 的 allocation 逻辑混淆。

| 数据集 | 当前 dispersion baseline | v4 主线 | 差值 |
|---|---:|---:|---:|
| Cora | 49.85 | 49.85 | +0.00 |
| Citeseer | 59.17 | 60.77 | +1.59 |

阶段结果：

| 数据集 | s0 | s1 | s2 |
|---|---:|---:|---:|
| Cora v4 | 58.11 | 51.29 | 49.85 |
| Citeseer v4 | 64.52 | 69.18 | 60.77 |

## 3. 结果含义

### Cora

所有候选 transfer 的最差 horizon gain 都没有稳定为正，因此 v4 自动回退到 dispersion allocation。最终结果与 baseline 完全一致。

这证明 v4 的安全回退机制工作正常，短程 selector 没有被强行写入主 memory。

### Citeseer

stage 1 的一次 class-row transfer 在各 response horizon 上仍保持轻微正 gain，因此被接受，最终从 `24/24` 调整为 `25/23`，s2 cumulative Macro-F1 从 59.17 提升到 60.77。

该提升目前只能视为一个小图正例，因为只覆盖一个阶段、一个类别顺序和一组 selection seed。

## 4. 代码行为

v4 selector 现在遵循：

1. 先用 dispersion 生成 base allocation；
2. 枚举 donor/recipient transfer；
3. 计算多 seed、多 batch、多 horizon response；
4. 使用 LCB 判断整体风险；
5. 使用最差 horizon gain 判断 ranking 是否一致；
6. 任一门禁失败就保留 base allocation；
7. selector runs、optimizer steps、response score 和 lcb 保存到结果 manifest。

reserve budget 进一步支持 0/2/4/8 rows、LCB 和 full-step terminal confirmation，但默认关闭，避免将未经多数据集确认的扩容机制写入主方法。

## 5. 当前验收结论

- [x] v4 主线 runner 可运行；
- [x] Cora/Citeseer 小图完成 canonical 3-stage 重跑；
- [x] Cora 不稳定 transfer 自动回退；
- [x] Citeseer 出现一个固定预算 response 增益；
- [x] selector 成本和风险审计进入 manifest；
- [ ] 多 selection seed group；
- [ ] 1%/10% rate；
- [ ] base+4 和额外类别顺序；
- [ ] 大图主表；
- [ ] response-gain reserve 的正式主表验证。

## 6. 当前方法定位

当前可以把 v4 描述为：

> 一个以 dispersion 为稳定先验、以多 horizon student response 校准固定 memory 分配、并在 selector 不确定时自动回退的 synthetic-only continual append 方法。

当前还不能宣称 response selector 跨数据集稳定优于 dispersion，也不能宣称 reserve extra budget 已经有效。

结果文件：

- `results/v4_baseline_current_2026_09_15/cora_results.json`；
- `results/v4_baseline_current_2026_09_15/citeseer_results.json`；
- `results/v4_mainline_safe_2026_09_15/cora_results.json`；
- `results/v4_mainline_safe_2026_09_15/citeseer_results.json`。
