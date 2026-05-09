# PKI: Personal Knowledge Internalization

**Build your second brain with lightweight local models.**

PKI is a high-performance system designed to transform your documents into the internal weights of a tiny LLM (Qwen3-0.6B). It focuses on **weight evolution** rather than prompt-based context injection (RAG).

## 🚀 The PKI Philosophy
> "One-time offline training cost, lifetime local intelligence."

PKI distills your documents into high-quality QA pairs and "burns" that knowledge into a model via LoRA fine-tuning. The result is a single GGUF file that *knows* your data, running entirely on your CPU.

## 🏗 Workflow: Separation of Concerns

### 1. Ingest (Rust-native) - [100% Ready]
- **Tool**: `pki-ingest`
- **Process**: Semantic chunking + LLM-driven QA generation.
- **Hardware**: Any modern CPU.

### 2. Internalization (Python/PyTorch) - [One-time Step]
- **Tool**: `pki-trainer` (Orchestrating HF PEFT / Unsloth)
- **Starting Point**: Raw BF16 Safetensors (1.15GB).
- **Hardware**: **NVIDIA GPU (8GB+ VRAM)** or **Apple Silicon M-series**.
- **Output**: Merged weights + Quantized GGUF.

### 3. Inference (Rust-native) - [100% Ready]
- **Tool**: `pki-engine` (Pure Rust/Candle)
- **Hardware**: Any CPU (AVX2/SIMD accelerated).
- **Footprint**: < 800MB RAM, < 200ms cold start.

## 🚦 Roadmap

| Version | Milestone | Status |
| :--- | :--- | :--- |
| **v0.2.x** | Pipeline POC + KV Cache + Auto-QA | ✅ Done |
| **v0.3.0** | Professional documentation & Stable Python scripts | 🏗 In Progress |
| **v1.0.0** | `ratatui` TUI + Single Binary distribution | 📅 Planned |
| **v2.0.0** | Native CPU Fine-tuning (When tech matures) | 🔭 Future |

## 🛡 Privacy & Sovereignty
Everything stays local. Inference requires no internet. Your knowledge is truly yours, encoded into a model that only you own.
