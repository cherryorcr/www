# V2: LLM文本凝聚模块 - 使用指南

## 概述

这是frozen_v1方法的v2升级版本,核心改进是用**LLM生成式摘要**替代简单的文本拼接。

### 方法对比

| 维度 | V1 (Baseline) | V2 (LLM Fusion) |
|---|---|---|
| 文本选择 | 距离排序top-g | 距离排序top-g (相同) |
| 文本融合 | 截断拼接 `" ".join(texts[:budget])` | LLM生成200词摘要 |
| 重新编码 | 冻结MiniLM | 冻结MiniLM (相同) |
| X初始化 | 簇均值 | 簇均值 (相同) |
| 拓扑 | 文本kNN | 文本kNN (相同) |
| **预期收益** | 基线 | **塑性提升 + 方法novelty** |

---

## 目录结构

```
code/v2_llm_fusion/
├── llm_text_fusion.py           # LLM融合核心模块
├── run_v2_comparison.py         # GraphCLIP v2实现
├── run_v1_v2_comparison.sh      # 实验运行脚本
└── README.md                    # 本文档
```

---

## 环境准备

### 1. 安装依赖

```bash
# 基础依赖
pip install torch transformers scikit-learn numpy --break-system-packages

# LLM客户端 (根据需要选择)
pip install openai anthropic requests --break-system-packages
```

### 2. 配置API密钥

根据要使用的LLM后端设置环境变量:

```bash
# OpenAI (GPT-4o, GPT-4-turbo)
export OPENAI_API_KEY="sk-..."

# Anthropic (Claude-3.5-Sonnet, Claude-3-Opus)
export ANTHROPIC_API_KEY="sk-ant-..."

# vLLM (本地Llama-3.1-70B)
export VLLM_ENDPOINT="http://localhost:8000/v1"
```

---

## 快速开始

### 方法1: 使用实验脚本 (推荐)

```bash
cd /home/pc/research/TAG_condensation/code/v2_llm_fusion

# 运行完整对比实验 (v1 + v2 GPT-4o + v2 Claude)
bash run_v1_v2_comparison.sh
```

这会自动运行:
1. V1 baseline (concatenation)
2. V2 with GPT-4o
3. V2 with Claude-3.5-Sonnet
4. 生成对比报告

结果保存在 `/home/pc/research/problem_validation_results/v2_llm_fusion_YYYY_MM_DD/`

### 方法2: 手动运行单个实验

```bash
# V1 Baseline
python run_v2_comparison.py \
    --data /home/pc/research/GraphCLIP/processed_data/cora.pt \
    --minilm /home/pc/models/all-MiniLM-L6-v2 \
    --output ./results/v1_baseline \
    --device cuda:0 \
    --method v1_concat \
    --rate 0.05 \
    --fuse-g 5 \
    --degree 1.0 \
    --student-steps 2048

# V2 LLM Fusion (GPT-4o)
python run_v2_comparison.py \
    --data /home/pc/research/GraphCLIP/processed_data/cora.pt \
    --minilm /home/pc/models/all-MiniLM-L6-v2 \
    --output ./results/v2_gpt4o \
    --device cuda:0 \
    --method v2_llm \
    --llm-backend gpt4o \
    --target-words 200 \
    --rate 0.05 \
    --fuse-g 5 \
    --degree 1.0 \
    --student-steps 2048
```

---

## 核心参数说明

### 凝聚参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--rate` | 0.05 | 凝聚预算比例 (5% = 135节点 for Cora) |
| `--fuse-g` | 5 | 每簇选择top-g个文本 |
| `--degree` | 1.0 | 拓扑平均度数 |

### V2特有参数

| 参数 | 可选值 | 说明 |
|---|---|---|
| `--llm-backend` | `gpt4o`, `gpt4-turbo`, `claude-sonnet`, `claude-opus`, `llama70b` | LLM后端选择 |
| `--target-words` | 200 | LLM生成摘要的目标词数 |

### Student训练参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--student-seeds` | 100-105 | 6个独立student seeds |
| `--student-steps` | 2048 | 训练步数 (大图可能需要调整) |

---

## 实验设计建议

### 实验1: V1 vs V2对比 (主实验)

**目标**: 验证LLM融合能否提升塑性

```bash
# 固定: rate=0.05, g=5, deg=1.0, steps=2048
# 变量: method (v1_concat vs v2_llm)

# Arms:
1. v1_concat (baseline)
2. v2_llm (gpt4o)
3. v2_llm (claude-sonnet)
```

**预期结果**:
- Dev Macro-F1: v2 > v1 (期望+3~5pp)
- 特别是Stage-2 current指标提升

### 实验2: LLM后端消融

**目标**: 比较不同LLM的融合质量

```bash
# Arms:
1. v2_llm (gpt4o)
2. v2_llm (gpt4-turbo)
3. v2_llm (claude-sonnet)
4. v2_llm (claude-opus)
5. v2_llm (llama70b)  # 如果有本地vLLM
```

### 实验3: 摘要长度扫描

**目标**: 找到最优target_words

```bash
# 固定: llm-backend=gpt4o
# 变量: target-words in {100, 150, 200, 250, 300}
```

### 实验4: 持续学习场景 (Gate 1'' style)

**目标**: 验证v2在持续场景下的优势

需要修改代码支持stage 0→1→2,暂时在单阶段验证。

---

## 结果分析

### 查看结果

```bash
# 查看单个实验结果
cat ./results/v2_gpt4o/results.json

# 对比多个实验
python -c "
import json
from pathlib import Path

for p in Path('./results').glob('*/results.json'):
    with open(p) as f:
        data = json.load(f)
    print(f\"{p.parent.name}: {data['dev_macro_f1_mean']:.2f} ± {data['dev_macro_f1_std']:.2f}\")
"
```

### 关键指标

1. **Dev Macro-F1 (主要)**: 6个student均值±标准差
2. **文本质量 (定性)**: 查看 `syn_v2_llm.pt` 中的 `texts` 字段
3. **计算成本**: LLM调用次数 = k (预算行数)

---

## LLM Prompt模板

当前使用的prompt (可在 `llm_text_fusion.py` 中自定义):

```
You are a research assistant tasked with creating a coherent, 
information-rich abstract by fusing multiple related research paper abstracts.

**Domain**: machine learning research

**Task**: Below are {g} abstracts from Cluster {i}. Generate a single 
{target_words}-word fused abstract that:

1. **Preserves key technical terms** (algorithms, architectures, datasets, metrics)
2. **Maintains semantic coherence** (reads like a natural abstract, not a list)
3. **Covers main topics** from all input abstracts
4. **Avoids redundancy** (consolidate overlapping concepts)
5. **Uses precise language** (no vague phrases like "various methods")

**Input Abstracts**:
1. {text_1}
2. {text_2}
...

**Output**: Generate ONLY the fused abstract, no preamble.
```

**优化方向**:
- 调整temperature (当前0.3)
- 增加Few-shot examples
- 针对不同domain定制prompt

---

## 常见问题

### Q1: LLM调用失败怎么办?

A: 代码内置了fallback机制,会回退到v1的简单拼接。检查:
1. API密钥是否正确
2. 网络连接是否正常
3. API配额是否充足

### Q2: 实验需要多长时间?

A: 时间分解 (Cora, k=135):
- **凝聚阶段**: 
  - V1: <1分钟 (确定性)
  - V2: ~2-3分钟 (135次LLM调用,每次0.5s间隔)
- **Student训练**: ~10分钟/seed × 6 = 1小时
- **总计**: V1约1小时, V2约1.5小时

### Q3: 成本估算?

A: 以GPT-4o为例 (Cora k=135):
- Input tokens: ~135 clusters × 5 texts × 150 tokens/text ≈ 100k tokens
- Output tokens: ~135 clusters × 200 words ≈ 27k tokens
- **成本**: ~$1.5 USD/次实验

Claude-3.5-Sonnet更便宜: ~$0.75 USD

### Q4: 如何加速实验?

A: 
1. 使用本地vLLM (Llama-3.1-70B)
2. 减少student seeds (6→3)
3. 减少student steps (2048→512,适合smoke test)
4. 使用更小的预算rate

### Q5: V2如何扩展到持续学习?

A: 当前代码是静态凝聚。持续版本需要:
1. 分阶段凝聚 (stage 0/1/2)
2. Append算子 (旧memory冻结 + 新delta独立凝聚)
3. 修改 `run_v2_comparison.py` 支持continual模式

暂时先在静态场景验证LLM融合效果。

---

## 下一步工作

### 短期 (1-2周)

1. ✅ 实现LLM文本融合模块
2. ✅ 集成到GraphCLIP构造式凝聚
3. ⏳ **运行实验1: V1 vs V2对比**
4. ⏳ **分析结果,确认塑性提升**

### 中期 (3-4周)

5. ⏳ 实验2: LLM后端消融
6. ⏳ 实验3: 摘要长度扫描
7. ⏳ 扩展到持续学习场景 (Gate 1'')
8. ⏳ 撰写v2章节加入论文

### 长期 (可选)

9. Prompt engineering优化
10. 可学习文本生成器 (RL微调)
11. 多模态扩展 (图像-图)

---

## 引用与参考

### 相关论文

- **TAGSAM** (NeurIPS 2023): 首个TAG凝聚方法
- **GraphCLIP** (NeurIPS 2023): TAG领域SOTA
- **Frozen_v1** (本项目): 构造式持续TAG凝聚

### 联系方式

- 项目负责人: [Your Name]
- 服务器: `REMOTE_SERVER_HOST`
- 数据路径: `/home/pc/research/problem_validation_results/`

---

## License

本项目代码遵循MIT License。
