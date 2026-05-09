# PKI: Personal Knowledge Internalization

**Hybrid High-Performance Knowledge Internalization System**

PKI is a project dedicated to exploring true knowledge internalization through model weight evolution. **Warning: This is a heavyweight engineering task.**

## 🚦 Brutal Reality Check

Before running this project, you must understand the hardware and time costs associated with "Internalization" (Fine-tuning):

1.  **Training is NOT Inference:** While inference on a quantized Qwen3-0.6B takes <800MB RAM, **Training** (Fine-tuning) starts from BF16/FP32 weights and requires **10GB+ RAM/VRAM**.
2.  **No GGUF Fine-tuning:** You cannot train on GGUF files. Period. Training starts from raw Safetensors.
3.  **CPU Training is Extremely Slow:** Running a LoRA fine-tuning on a standard CPU for a 0.6B model is **not a real-time task**. 10 training steps can take **hours**, not minutes.
4.  **No Mocking / No Placeholders:** We do not use "dummy" files to fake a completed pipeline. If the hardware fails or the process is too slow, the system will report it as a failure.

## 🏗 System Architecture

PKI employs a "Separation of Concerns" architecture:

### Phase 1: Ingest (Rust/GGUF)
- **Goal:** Parse documents and generate a high-quality QA dataset.
- **Hardware:** Runs on any modern CPU. Memory < 800MB.

### Phase 2: Internalization (Python/PyTorch/BF16)
- **Goal:** Real LoRA fine-tuning to encode knowledge into weights.
- **Hardware:** **GPU with 8GB+ VRAM highly recommended.**
- **CPU Mode:** Possible but considered "Developer-only/Overnight" mode. Requires 10GB+ RAM and hours of execution time.

### Phase 3: Integration & Quantization (The Gap)
- **Goal:** Merge LoRA weights and quantize back to GGUF for the Rust engine.
- **Status:** Requires a functional `llama.cpp` toolchain or a successful Python-based export.

### Phase 4: Inference (Rust/GGUF)
- **Goal:** Fast, lightweight chat using the newly internalized model.

## 📉 Project Status: Research & Development
Current version (v0.2.2) is a proof-of-concept. It focuses on getting the pipeline structure right, but the **Internalization Step** is a high-cost operation that requires professional-grade hardware to see real results in real-time.
