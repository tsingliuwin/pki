# PKI: Personal Knowledge Internalization

**Build your second brain with extremely lightweight local models.**

PKI is a Rust-native system designed to transform your unstructured data (PDFs, Markdown, etc.) into the internal weights of a tiny LLM (Qwen3-0.6B) via LoRA fine-tuning. Unlike RAG, PKI aims for true knowledge internalization.

## 🚀 Why PKI?

- **True Internalization:** Knowledge is encoded into model weights, enabling cross-document reasoning and logical consistency.
- **Rust-Powered Performance:** Built with pure Rust (`candle` & `tokio`). Cold start in <200ms with minimal memory footprint (<800MB).
- **Privacy First:** 100% local. No data leaves your machine. No internet required for inference.
- **Zero-Dependency Inference:** Single binary execution (Proof-of-Concept).

## 🛠 Architecture

- **`pki-ingest`**: High-performance document parsing (PDF/MD) and semantic chunking.
- **`pki-trainer`**: Orchestrates QLoRA training using Unsloth (via a managed Python environment).
- **`pki-engine`**: A high-efficiency inference engine based on `candle-transformers` with KV Cache optimization.
- **`pki-cli`**: Unified command-line interface for the entire pipeline.

## 🚦 Quick Start

### Prerequisites
- [Rust](https://www.rust-lang.org/) (latest stable)
- [uv](https://github.com/astral/uv) (for Python environment management)
- NVIDIA GPU (Optional, for real training acceleration)

### Run the Pipeline
To experience the end-to-end mock pipeline with hardware acceleration:

```powershell
# Windows (PowerShell)
$env:RUSTFLAGS="-C target-cpu=native"; cargo run --release -p pki-cli -- pipeline --file test.md --query "What is Knowledge Internalization?"
```

## 📈 Roadmap

- [x] **v0.2.0**: End-to-end Pipeline POC (Rust CLI + Python Trainer).
- [x] **Inference Optimization**: KV Cache support and SIMD acceleration.
- [ ] **Real QA Generation**: Replace mock data with real LLM-generated QA pairs.
- [ ] **On-device LoRA Merging**: Seamlessly merge adapters for zero-latency switching.
- [ ] **TUI Dashboard**: A professional terminal UI for monitoring training and chat.
