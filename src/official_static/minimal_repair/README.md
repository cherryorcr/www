# TAGSAM
## Preparation
1. Create a virtual environment
```shell
conda create -n TAGSAM python=3.11
conda activate TAGSAM
```
2. Install pytorch manually
```shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
```
3. Install requirements
```shell
pip install -r requirements.txt
```
4. Login to wandb
```shell
wandb login
```
All **results** will be logged to wandb.   
If you do not want to use wandb, you can set WANDB_MODE to disabled in the config file.
```shell
export WANDB_MODE=disabled
```

## Pre-process
You first need to train a **teacher/expert model** on the original TAG. This process is generally referred to as the **buffer**.
```shell
python buffer.py --dataset_name computer
```

## Condensation
Then you can **condense/distill** TAG into a **smaller** one.
```shell
python distill.py --dataset_name computer --syn_size 200
```

## Evaluation
You can automatically perform **asynchronous evaluation** during the condensation process if you set **async_eval** to True.  
>Note that you need to ensure the setting **eval_gpu** is correct; otherwise, it may lead to issues such as the GPU not being available, reduced efficiency, and memory overflow (when **gpu** and **eval_gpu** are the same).  

Additionally, you can also manually trigger the evaluation if needed.
```shell
python eval.py --dataset_name computer --syn_size 200
```