# Change Detection Project with InternViT & RL

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---
## 📖 项目概述 (Project Overview)

本项目实现了一个高精度的多标签变化检测框架，专为处理复杂的街景影像（SVI）变化检测任务而设计。系统采用“Sensing-Reasoning”交替的 Transformer 架构，结合强大的 InternViT 视觉编码器，实现对双时相图像的深度特征融合与理解。

此外，项目引入了 **强化学习 (PPO 算法)** 模块，用于在预训练模型的基础上进一步微调策略，优化模型的推理过程（如动态 Query 修正与推理终止决策）。

### 核心特性
* **多任务学习 (Multitask Learning)**: 同时处理 Road, Building, Greenery, Infrastructure 四大类共 31 个子任务的变化检测.
* **Sensing-Reasoning 架构**: 独特的 Transformer 结构，交替进行视觉感知 (Sensing, Cross-Attn) 与 逻辑推理 (Reasoning, Self-Attn).
* **深度监督 (Deep Supervision)**: 在多个推理层级挂载分类头，提升中间层特征的判别能力.
* **强化学习微调 (RL Fine-tuning)**: 集成 PPO Agent，支持联合优化分类头与策略网络，具备动态修正 Query 的能力.
* **InternViT 主干**: 利用 InternViT-300M 强大的视觉特征提取能力，支持 Pixel Unshuffle 和 MLP 投影.

---

## 🏗️ 技术架构 (Technical Architecture)

### 核心组件
* **Vision Encoder**: 基于 `InternViT-300M-448px-V2_5`，支持冻结参数与层选择.
* **Fusion Transformer**: 包含 `FusionTransformerBlock2`，支持 FlashAttention 加速。
* **RL Agent**: 基于 PPO (Proximal Policy Optimization) 的 Actor-Critic 网络，输出连续动作 (Correction) 和 离散动作 (Stop).

### 模型流向图
```mermaid
graph TD
    subgraph Input
        IMG1[Image T1]
        IMG2[Image T2]
    end

    subgraph "Vision Backbone (InternViT)"
        VE[Vision Encoder]
    end

    subgraph "Assembled Fusion Model"
        QG[Query Generator]
        PE[Positional Embedder]
        
        subgraph "Transformer Layers (Alternating)"
            SL["Sensing Layer (Even)"]
            RL["Reasoning Layer (Odd)"]
            SL -- Query Interaction --> RL
            VE -- Visual Feat --> SL
        end
        
        Heads[Multitask Classifiers]
        RL -.-> |Deep Supervision| Heads
    end

    IMG1 & IMG2 --> VE
    VE --> SL
    QG --> SL
    Heads --> Output[Change Predictions]

```

---

## 📂 项目结构 (Project Structure)

```text
.
├── configs/                # 配置文件
│   ├── defaults.yaml       # 全局默认配置 (数据路径, 模型参数, Loss权重)
│   ├── rl_stage.yaml       # RL 阶段特定配置
│   └── warmup_stage.yaml   # 预热训练配置
├── dataset/                # 数据加载模块
│   ├── dataset.py          # SVIPairsDataset 定义，处理图像切片与CSV读取
│   ├── dataloader.py       # Dataloader 构建逻辑
│   └── transforms.py       # 图像增强与预处理
├── models/                 # 模型定义
│   ├── model.py            # 主模型 AssembledFusionModel
│   ├── vision/             # 视觉主干 (backbone.py)
│   ├── transformer/        # Transformer 块与 Attention 实现
│   ├── heads/              # 多任务分类头
│   └── position_embedding_v2.py # 位置编码
├── rl/                     # 强化学习模块
│   ├── agent.py            # PPOAgent (Actor-Critic, Update Logic)
│   ├── env.py              # RL 环境封装
│   ├── buffer.py           # Rollout Buffer
│   ├── networks.py         # ActorCriticNetwork 定义
│   └── rewards.py          # 奖励函数计算
├── scripts/                # 运行脚本
│   ├── DL train.py         # 深度学习(监督)训练入口
│   ├── RL train.py         # 强化学习训练入口
│   └── inference.py        # 推理脚本
├── trainer/                # 训练器逻辑
│   ├── base_trainer.py     # 基础监督学习训练器
│   └── rl_trainer.py       # PPO 训练器
└── utils/                  # 工具库 (Logger, Config, Loss)

```

---

## ⚙️ 安装与依赖 (Installation)

本项目依赖 Python 3.10+ 和 PyTorch 2.0+。

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n change_det python=3.10
conda activate change_det

# 安装 PyTorch (根据你的 CUDA 版本选择)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

```

### 2. 安装依赖

请确保安装以下核心库：

```bash
pip install transformers numpy pandas pyyaml pillow scikit-learn
# 如果使用 FlashAttention
pip install flash-attn --no-build-isolation

```

---

## 🚀 快速开始 (Quick Start)

### 1. 数据准备

请在 `configs/defaults.yaml` 中配置数据路径。数据应包含：

* **图像文件夹**: 存放 T1 和 T2 时刻的图片。
* **CSV 文件**: 包含文件名索引 (`OID_`, `name_15`, `name_19`) 和 标签列 (`A01_01_label` 等).

**CSV 格式示例:**
| OID_ | name_15 | name_19 | A01_01_label | ... |
|------|---------|---------|--------------|-----|
| 1001 | img_a   | img_b   | 1            | ... |

### 2. 监督训练 (Warmup / DL Stage)

使用 `DL train.py` 进行基础模型的监督训练。

```bash
# 确保在项目根目录下
export PYTHONPATH=$PYTHONPATH:.

# 运行训练
python scripts/DL\ train.py

```

*配置调整*: 修改 `configs/defaults.yaml` 中的 `train` 部分参数 (如 `lr`, `batch_size`).

### 3. 强化学习微调 (RL Stage)

在监督训练完成后，加载预训练权重进行 RL 微调。

```bash
# 运行 RL 训练
python scripts/RL\ train.py

```

*注意*: 需在 `configs/defaults.yaml` 的 `rl` 部分指定 `pre_model_path` 为预训练好的模型路径 (e.g., `./results/checkpoints/model_best.pth`).

---

## 📊 模型输入与输出

* **Input**:
* `pixel_values_t1`: [Batch, N_patches, 3, 448, 448] (经过 InternViT processor 处理)
* `pixel_values_t2`: [Batch, N_patches, 3, 448, 448]


* **Output**:
* `all_results`: 字典，包含不同 Reasoning 层的分类结果。
* Key: `ClassifyLayer_{i}`
* Value: Logits [Batch, Num_Tasks, 2].



## 📜 许可证 (License)

MIT License

---

*Created by Project Team*

```

```