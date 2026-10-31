# 环境与复现说明

## 已验证运行环境

| 项目 | 版本 |
|---|---|
| OS | Linux 远程服务器 |
| Python | 3.11.15 |
| PyTorch | 2.4.1+cu121 |
| CUDA runtime | 12.1（由 PyTorch build 标识） |
| GPU | 2 × NVIDIA GeForce RTX 4090，49140 MiB |
| NVIDIA driver | 580.173.02 |
| NumPy | 2.1.2 |
| scikit-learn | 1.6.1 |
| Transformers | 4.49.0 |
| huggingface-hub | 0.36.2 |

远程解释器原路径：

```text
/home/pc/miniconda3/envs/tagsam/bin/python
```

## 依赖文件

- `requirements-core.txt`：当前 native 训练与数据准备所需的核心直接依赖；
- `requirements-remote-freeze.txt`：实际已验证远程 conda 环境的完整 `pip freeze`；
- `requirements-full-source.txt`：原 TAGSAM 仓库保存的完整依赖锁定。

如果直接使用 `data/prepared_v2`，训练和评估主要依赖 PyTorch、NumPy 和 scikit-learn。只有重新计算 BERT CLS 时才需要 Transformers 和 Hugging Face。

带 `+cu121` 的 PyTorch wheel 需要使用匹配的 PyTorch CUDA wheel index 或 conda channel，普通 PyPI 环境不一定能直接解析该构建标识。

## 单元测试

```bash
cd code/native_tag_v1
python -m unittest discover -s . -p 'test_*.py'
```

整理后已从本包的 `code/native_tag_v1` 再次运行，结果为 `38/38 passed`。本地整理验收环境是 Windows、Python 3.12.10、PyTorch 2.9.1+cpu、NumPy 2.3.1、scikit-learn 1.7.2；这只验证代码和接口，不重新验证上表 Linux/RTX 4090 环境中的 GPU 数值。

## 静态运行示例

从 `code/native_tag_v1` 执行：

```bash
python run_native_experiments.py \
  --prepared ../../data/prepared_v2 \
  --output ../../work_runs/static_r05_c44 \
  --phase static --stages 1 --rate .05 --seed 44 \
  --model-seeds 100 101 --arms full random coreset frozen offline_asm \
  --device cuda:0 --student-steps 512 --teacher-steps 800 \
  --validation-steps 512 --check-every 32 --link-steps 160 \
  --outer-steps 100 --outer-eval-every 20 --inner-steps 10 \
  --inner-lr .005 --outer-lr .01 --reference-size 128
```

Runner 要求输出目录此前不存在，避免覆盖不可变实验记录。

## 数据和评测边界

- 当前 Cora test/confirmation/future 已访问，只能用于开发；
- prepared eval 文件包含标签，凝聚和训练函数只接收经过清理的无标签 graph dictionary；
- 运行新的最终实验前需要选择未被当前开发访问的数据集和留出。
