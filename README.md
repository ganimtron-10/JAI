# JAI (Just another AI Inference Engine)

**JAI** is a minimalist, LLM inference engine written from scratch in **Go**. It runs modern Transformer models (SmolLM2, Tinyllama, Qwen) using raw linear algebra and memory-mapped I/O, bypassing heavy deep-learning frameworks.

## 🚀 Key Features
* **Zero-Framework Inference:** Core transformer logic (RoPE, RMSNorm, SwiGLU, Softmax) implemented in pure Go.
* **Concurrent Weight Loading:** Uses `mmap` for zero-copy memory access and parallelized conversion of Safetensor weights.
* **Stateful KV Caching:** Manages context for multi-turn conversations.
* **Advanced Sampling:** Full support for Temperature, Top-K, Top-P, and Repetition Penalty.
* **Automated Setup:** Built-in model downloader for small-language models (SLMs).

---

## 🏃 Quick Start

### 1. Clone and Install Dependencies
```bash
git clone https://github.com/yourusername/jai.git
cd jai
go get github.com/sugarme/tokenizer
go get gonum.org/v1/gonum/blas
```

### 2. Download Models
Use the provided interactive script to fetch optimized weights from HuggingFace (SmolLM2, Qwen2.5, or TinyLlama):
```bash
chmod +x download.sh
./download.sh
```

### 3. Run Inference
```bash
go run main.go
```

---

## 📂 Project Structure
* `main.go`: The core inference engine, sampler, and REPL logic.
* `download.sh`: A utility to automate model fetching and directory structuring.
* `models/`: Directory containing model weights and configurations.
