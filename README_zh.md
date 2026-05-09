# PKI: 个人知识内化系统

**基于极小模型的本地化“第二大脑”实现方案。**

PKI 是一个纯 Rust 开发的知识内化系统。它通过 LoRA 微调技术，将你的非结构化文档（PDF、Markdown 等）转化为极小大模型（Qwen3-0.6B）的内在权重。

## 🌟 核心特性

- **真正的内化**: 不同于 RAG（检索增强生成），PKI 将知识编码进模型参数，支持跨文档推理和更强的逻辑关联。
- **极致性能**: 基于 Rust 编写，利用 `candle` 推理框架和 KV Cache 优化。冷启动 < 200ms，内存占用 < 800MB。
- **隐私至上**: 100% 本地运行。数据不出设备，推理过程完全脱网。
- **极简分发**: 追求单执行文件体验，无复杂的 Python 环境依赖方案（推理端）。

## 🏗 模块组成

- **`pki-ingest`**: 高性能文档解析与语义分块。
- **`pki-trainer`**: 训练编排层，通过 `uv` 管理虚拟环境并调用 Unsloth 进行高效微调。
- **`pki-engine`**: 基于 Rust 的高性能推理引擎，已实现硬件加速指令集适配。
- **`pki-cli`**: 统一的命令行交互入口。

## ⚡ 快速开始

### 前置要求
- [Rust](https://www.rust-lang.org/) (最新稳定版)
- [uv](https://github.com/astral/uv) (用于管理训练所需的 Python 环境)
- NVIDIA GPU (可选，用于加速真实训练)

### 运行全流程示例
使用硬件加速指令集运行端到端模拟流水线：

```powershell
# Windows (PowerShell)
$env:RUSTFLAGS="-C target-cpu=native"; cargo run --release -p pki-cli -- pipeline --file test.md --query "什么是知识内化？"
```

## 🗺 路线图

- [x] **v0.2.0**: 端到端流水线验证 (POC)。
- [x] **推理优化**: 实现 KV Cache 支持与 SIMD 指令集加速。
- [ ] **真实 QA 生成**: 接入本地模型实现高质量训练集自动构建。
- [ ] **权重自动合并**: 训练后自动合并 LoRA 权重，消除推理延迟。
- [ ] **TUI 交互界面**: 引入 `ratatui` 构建极客风格的终端管理面板。
