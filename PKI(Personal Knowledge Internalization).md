# PKI (Personal Knowledge Internalization)

**基于极小模型的本地个人知识内化系统**

**技术方案白皮书 · v0.2.3 (Engineering Truth Edition)**

---

## 1. 核心定义与架构边界

PKI 项目严格遵守 LLM 训练与推理的物理规律，确立了以下职责边界，拒绝任何形式的“技术粉饰”。

### 1.1 职责分离原则

*   **训练侧 (Training - The Forge):** 
    *   **精度要求**: 必须使用 BF16/FP32 原始权重（Safetensors）。
    *   **资源需求**: 0.6B 模型 LoRA 微调需 **10GB+ RAM/VRAM**（包含梯度、优化器状态）。
    *   **环境依赖**: 深度绑定 Python 生态（PyTorch, PEFT, Unsloth）。
*   **推理侧 (Inference - The Execution):** 
    *   **效率要求**: 使用 GGUF 量化模型（Q8_0/Q4_K_M）。
    *   **资源需求**: < 800MB RAM。
    *   **环境依赖**: 纯 Rust 实现 (`candle`)，秒级冷启动。

---

## 2. 诚实的系统流水线 (Corrected Pipeline)

1.  **数据提取 (`pki-ingest` / Rust)**: 高性能解析文档，驱动本地量化模型生成 Alpaca 问答对。
2.  **内化微调 (`pki-trainer` / Python)**: 
    *   加载 BF16 原始权重。
    *   进行真实的梯度下降（CPU 或 GPU）。**警告：CPU 微调极其缓慢，单步可能需 10min+。**
3.  **量化鸿沟 (Quantization Bridge)**: 
    *   将微调后的 FP32 模型导出。
    *   **手动/自动化**调用 `llama.cpp` 进行权重合并与重新量化。
4.  **消费知识 (`pki-engine` / Rust)**: 加载重新量化后的“知识增强版” GGUF 进行推理。

---

## 3. 硬件分级与预期表现

| 硬件配置 | 训练阶段表现 | 推理阶段表现 |
| :--- | :--- | :--- |
| **NVIDIA GPU (8GB+ VRAM)** | **生产级**：数分钟完成内化。 | 极致极速。 |
| **高端 PC (32GB+ RAM)** | **开发者验证级**：通宵运行可完成小规模内化。 | 极速。 |
| **普通办公本 (8GB RAM)** | **不可用**：内存溢出或系统卡死。 | 良好（使用 Base 模型）。 |

---

## 4. 结论

PKI 不是魔法。它是一个将数据“刻录”进权重的精密过程。我们优先保证工程的**真实性**与**可靠性**，而非虚假的“全能性”。
