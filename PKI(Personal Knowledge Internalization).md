# PKI (Personal Knowledge Internalization)

**技术方案白皮书 · v0.2.4 (The Pragmatic Edition)**

---

## 1. 核心重新定位

PKI 项目确立了 **“混合架构，本地优先”** 的务实路线。我们认为知识内化是一个严谨的工程过程：**用 LLM 把文档蒸馏成高质量 QA 对，通过 LoRA 微调把知识写进权重。**

---

## 2. 核心模块与交付状态

### 2.1 推理引擎 (`pki-engine`) - [100% 可交付]
*   **架构**: 基于 `candle` + GGUF + AVX2 加速。
*   **性能**: Qwen3-0.6B Q4 量化版占用 ~400MB 内存，支持 KV Cache，响应无延迟。
*   **形态**: 纯 Rust 编译，单二进制文件，无 Python 环境依赖。

### 2.2 数据提取器 (`pki-ingest`) - [100% 可交付]
*   **逻辑**: 驱动本地量化模型阅读文档，自动生成符合 Alpaca 格式的语义 QA 对。
*   **产出**: 标准化的 `pki_qa_dataset.jsonl` 训练集。

### 2.3 知识内化器 (`pki-trainer`) - [需特定硬件]
*   **工具链**: 集成 HuggingFace PEFT 与 Unsloth 训练框架。
*   **起点**: 必须使用 `Qwen/Qwen3-0.6B` 原始 BF16 权重。
*   **硬件门槛**: 至少 8GB 显存的 GPU 或 Apple Silicon (M1/M2/M3)。
*   **职责**: 负责一次性的“内化”计算，并产出合并后的 GGUF 模型。

---

## 3. 完整系统工作流

```text
文档输入 (Markdown/PDF)
  ↓ [pki-ingest] (Rust)
  生成真实 QA 数据集 (pki_qa_dataset.jsonl)
  ↓ [pki-trainer] (Python, 一次性离线内化)
  LoRA 微调 (BF16) → 合并权重 → 量化 → .gguf
  ↓ [pki-engine] (Rust)
  本地极速推理，永久拥有该知识
```

---

## 4. 发展规划 (Roadmap)

*   **v0.2.4**: 推理与 Ingest 侧完全打通。
*   **v0.3.0**: 训练流程标准化，提供傻瓜式的 Python 脚本。
*   **v1.0.0**: 实现 `ratatui` TUI 交互，完成桌面级极客工具化。
*   **v2.0.0**: 探索原生 CPU 训练算子（长期技术储备）。

---

> **结语**：PKI 不承诺魔法，我们只提供将知识固化为本地智能的硬核路径。
