# PKI (Personal Knowledge Internalization)

**基于极小模型的本地个人知识内化系统**

**技术方案白皮书 · v0.2.1 (Current Stable)**

---

## 1. 项目概述

### 1.1 问题背景

现有“个人知识库”产品普遍基于 RAG（检索增强生成）架构，其本质是带索引的文本检索，而非真正的知识内化。PKI 旨在通过 **模型权重演进** 代替 **外部上下文注入**。

### 1.2 核心目标

*   **完全本地 (100% Local):** 数据不出设备，无隐私泄露。
*   **极致轻量 (Ultra-lightweight):** 目标内存占用 < 800MB，冷启动 < 200ms。
*   **零依赖 (Zero-dependency):** 纯 Rust 构建，单二进制分发，无 Python/CUDA 环境依赖。
*   **知识内化 (Internalization):** 知识编码进模型权重，支持跨文档推理。

---

## 2. 系统架构 (v0.2.1)

系统采用纯 Rust Workspace 架构，实现了从摄入到推理的全流程闭环。

### 2.1 模块职责

*   **`pki-ingest` (数据摄入器):**
    *   **Real LLM-driven QA:** 调用本地模型阅读文档块，自动生成 Alpaca 格式的训练对。
    *   **语义分块:** 基于内容感知的智能分段。
*   **`pki-trainer` (内化编排器):**
    *   **CPU 极致优化:** 废弃 Python/Unsloth，转向高度优化的 CPU 训练工具。
    *   **权重合并 (Merging):** 自动执行 LoRA 合并，将新知识固化到主模型中。
*   **`pki-engine` (推理基座):**
    *   **Pure Rust (`candle`):** 基于 HuggingFace 的 `candle` 框架实现，无 C++ 链接风险。
    *   **KV Cache 优化:** 实现自回归生成优化，推理速度提升 10 倍以上。

### 2.2 核心 Pipeline 流程

1.  **Ingest**: `pki-cli` 启动 Ingest，调用 `pki-engine` 加载 Qwen3 模型分析文档并产出 `data/pki_qa_dataset.jsonl`。
2.  **Train**: `pki-trainer` 接收数据，启动 CPU 优化训练流程，产出 `Qwen3-0.6B-Internalized.gguf`。
3.  **Chat**: `pki-engine` 加载“已内化”的模型，为用户提供带有深度认知的对话服务。

---

## 3. 技术选型

| 维度 | 选型 | 理由 |
| :--- | :--- | :--- |
| **基础模型** | **Qwen3-0.6B-GGUF** | 极致体积与性能平衡，非常适合 CPU 环境。 |
| **推理框架** | **`candle` (Rust)** | 内存安全，单文件分发，硬件加速支持良好。 |
| **训练策略** | **CPU LoRA + Merging** | 针对无 GPU 设备优化的内化方案。 |
| **加速技术** | **AVX2 / native-cpu** | 利用现代 CPU 指令集实现准秒级推理响应。 |

---

## 4. 项目路线图 (Roadmap)

*   [x] **v0.1.0**: 纯 Rust CLI 与 Ingest 基础框架。
*   [x] **v0.2.0**: 端到端全自动 Pipeline 打通（Mock 训练）。
*   [x] **v0.2.1**: **重大突破**。实现真实 QA 生成、KV Cache 优化、硬件加速。
*   [ ] **v0.3.0**: 实现基于 `candle` 的纯 Rust CPU 微调算子（彻底告别外部二进制）。
*   [ ] **v1.0.0**: 推出带有 `ratatui` TUI 界面的桌面级单文件产品。

---

> **核心价值主张：**
> “你的知识，真正属于你。” PKI 不仅是一个工具，它是一场本地私有化 AI 的革命。
