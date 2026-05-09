# PKI (Personal Knowledge Internalization)

**基于极小模型的本地个人知识内化系统**

**技术方案白皮书 · v0.2.2 (Corrected Architecture)**

---

## 1. 核心定义与架构演进

### 1.1 职责分离原则

基于对 LLM 训练与推理底层逻辑的深度理解，PKI 放弃了“全量 Rust 训练”的幻想，确立了以下职责边界：

*   **训练侧 (Training Side):** 必须使用 Python 生态（PyTorch/Unsloth）。必须从 **BF16 原始权重** 开始。
*   **推理侧 (Inference Side):** 坚持使用 Rust 生态 (`candle`)。使用 **GGUF 量化模型** 以保证本地极致性能。

### 1.2 技术红线 (Constraints)

*   **GGUF 不可训练:** 它是推理专用的量化格式，权重不连续且不可微，无法进行任何形式的 Fine-tuning。
*   **训练资源预估:** LoRA 微调 Qwen3-0.6B 时，权重+梯度+优化器状态合计需 **5GB+** 显存/内存。800MB 仅适用于量化后的推理。

---

## 2. 系统流水线 (Corrected Pipeline)

1.  **数据提取 (`pki-ingest` / Rust):** 解析本地文档，利用本地 GGUF 模型生成高质量的 Alpaca 格式 QA 数据集。
2.  **内化训练 (`pki-trainer` / Python):** 
    *   自动加载 `Qwen/Qwen3-0.6B` 的 BF16 原始权重。
    *   使用 Unsloth 在 Python 进程中进行高效 LoRA 微调。
3.  **模型导出 (Bridge / Python & C++):**
    *   执行 `merge_and_unload` 合并 LoRA 适配器。
    *   通过 `llama.cpp` 的 `quantize` 工具将合并后的 BF16 模型转化为 GGUF。
4.  **知识消费 (`pki-engine` / Rust):** 加载这个全新的、包含了内化知识的 GGUF 模型进行极速对话。

---

## 3. 核心优势

*   **知识闭环:** 虽然训练借用了 Python，但最终产出的模型是完全内化的，不再依赖 RAG 检索。
*   **极致推理:** Rust 侧的 `pki-engine` 依然保持着秒开、轻量、硬件加速的优势。
*   **数据主权:** 所有数据处理（解析、QA 生成、训练）均可在用户本地完成。

---

## 4. 挑战与对策

| 风险点 | 描述 | 对策 |
| :--- | :--- | :--- |
| **训练硬件门槛** | 5GB+ 的资源要求对低端设备仍有压力 | 实现“静默训练”模式，利用空闲时间进行 CPU 缓慢微调；或支持远程 GPU 卸载。 |
| **量化精度损失** | 训练后量化可能导致知识微弱模糊 | 采用 Q8_0 等高位量化格式以平衡知识留存与推理速度。 |

---

> **核心价值主张：**
> PKI 致力于让 AI 真正“读懂”你的数据，而不是仅仅在对话时把它翻出来看一眼。
