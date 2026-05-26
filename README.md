# Real-Time Infinite-Length Streaming Text-Image to Audio-Video Generation

本系统旨在实现高质量、实时且支持无限长度流式输出的音视频生成。以下是项目的核心目录结构、模块说明及安装指南。

---

## 📁 目录结构概览

```text
.
├── asserts/               # 资源文件
│   ├── examples/          # 示例文件
│   └── testset/           # 测试数据集 (如 testdata.yaml)
├── logs/                  # 日志目录 (按日期组织)
├── packages/              # 核心功能组件库
│   ├── ltx-core/          # 基础模型结构 (Transformer/VAE/Encoder)
│   ├── ltx-distillation/  # 蒸馏训练与推理模块 (ODE/DMD)
│   ├── ltx-causal/        # 因果注意力与掩码处理
│   └── ltx-pipelines/     # 训练与推理流水线
├── scripts/               # 运行脚本
│   ├── train/             # 训练入口及脚本
│   └── test/              # 推理与测试入口
└── README.md
```

## 📦 核心模块详解

### 1. `ltx-core` (模型底层结构)
负责定义生成系统的骨架，包含：
* **Transformer 实现**: `causal_model.py` 构建完整结构。
* **注意力机制**: `causal_attention.py` 实现因果自注意力、A2V/V2A 交叉注意力，支持 **KV-Cache** 优化。
* **模块化设计**: `causal_transformer.py` 实现通用的 Transformer Block。

### 2. `ltx-distillation` (蒸馏与训练优化)
专注于通过 ODE 和 DMD 技术提升模型性能与速度：
* **Inference**: 训练与推理的 Pipeline 逻辑。
* **Models**: 包含 ODE/DMD 训练时的 Loss 计算模块。
* **Trainer**: 负责训练调度与过程管理。

### 3. `ltx-causal` (因果逻辑支持)
提供流式生成所需的辅助工具：
* **attention/mask_builder.py**: 为 SDPA 构建 Attention Mask，支持 ODE 训练及音视频块处理。

### 4. `ltx-pipelines` (流水线集成)
实现具体的业务逻辑流：
* **ode_sample_generation_one_stage_multi_gpus.py**: 支持多显卡并行的 ODE 样本生成。
* **a2vid_one_stage_causal.py**: 一阶段因果 AI 视频生成 Pipeline。

---

## 🚀 快速开始

### 1. 环境准备与安装
在使用本系统前，需要以可编辑模式（editable mode）重新安装各个本地模块，以确保依赖关系正确：

```bash
# 依次安装各个核心组件
pip install -e packages/ltx-core
pip install -e packages/ltx-distillation
pip install -e packages/ltx-causal
pip install -e packages/ltx-pipelines
pip install -e packages/ltx-trainer
```

### 2. 运行脚本
* **训练**: 进入 `scripts/train/` 运行相关 `.sh` 脚本或调用 `LTX_train.py`。
* **推理**: 直接使用 `scripts/test/infer.py` 作为总入口进行生成测试。

---

## 📝 其他备注
* **日志**: 调试过程中请查看 `logs/` 目录下对应日期的日志文件。
* **数据**: 测试用的短 Prompt 位于 `/gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/assets/testset/testdata2_shortprompt.yaml`，。

* **研发云**: 河南H100不通研发云，本地可通过git bundle打包下载下来再同步到研发云。
* **合作**: 先在自己的branch上开发，如果验证有效可考虑提MR合并进main。

---

## 🌿 `dynamic` 分支说明

详见 [dynamic.md](dynamic.md)。
