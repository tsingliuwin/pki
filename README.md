# PKI: Personal Knowledge Internalization

**Hybrid High-Performance Knowledge Internalization System**

PKI is a research project designed to bridge the gap between heavy-duty model fine-tuning and lightweight local inference. It employs a "Separation of Concerns" architecture: high-precision training in Python and ultra-fast quantized inference in Rust.

## 🚀 The PKI Philosophy

Unlike RAG (Retrieval-Augmented Generation), which injects context into a prompt, PKI aims to **internalize** knowledge into model weights. This requires a rigorous multi-stage pipeline.

## 🏗 Architecture & Workflow

PKI operates on a clear boundary between training and runtime:

### 1. Training Phase (Heavyweight - Python/PyTorch)
- **Engine:** PyTorch + Unsloth / PEFT.
- **Weights:** Starts from **BF16 Original Weights** (not GGUF).
- **Process:** LoRA Fine-tuning on user-generated datasets.
- **Memory:** Requires ~5GB+ VRAM/RAM (not suitable for low-end mobile devices).

### 2. Integration Phase (The Bridge)
- **Merge:** Merging LoRA adapters back into the base BF16 weights.
- **Quantization:** Converting the internalized model into **GGUF (Q8_0/Q4_K_M)** using `llama.cpp`.

### 3. Inference Phase (Lightweight - Rust/Candle)
- **Engine:** Pure Rust (`candle`) based inference.
- **Weights:** Optimized **GGUF** quantized models.
- **Performance:** Cold start < 200ms, Memory footprint < 800MB.

## 🛠 Project Modules

- **`pki-ingest`**: Rust-native high-performance document parsing and QA dataset generation.
- **`pki-trainer`**: Orchestrates the Python fine-tuning environment and export pipeline.
- **`pki-engine`**: High-efficiency inference base optimized for GGUF.
- **`pki-cli`**: Unified command-line interface for the end-to-end pipeline.

## 🚦 Reality Check (Must Read)

1. **No GGUF Fine-tuning:** Quantized formats are for inference only. You cannot calculate gradients on discrete weights.
2. **Candle is for Inference:** Rust's `candle` framework is excellent for runtime but currently unsuitable for complex stable fine-tuning.
3. **Hardware Split:** You need a decent machine (GPU recommended) for the **Internalization (Training)** step, but the **Usage (Inference)** step can run on any potato.

## 📈 Roadmap

- [x] **v0.2.1**: End-to-end pipeline orchestration.
- [x] **Inference Optimization**: KV Cache support and SIMD acceleration in Rust.
- [ ] **Automated Quantization Pipeline**: Seamless integration with `llama.cpp` for GGUF export.
- [ ] **Remote Trainer Support**: Offload the heavy training to a GPU box while keeping inference local.
