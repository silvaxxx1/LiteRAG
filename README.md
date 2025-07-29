# ⚡ LiteRAG — Local RAG Pipeline, Production-Ready from Your Own Hardware

> 🧠 **Based on:** [HandsOnLLMs (by silvaxxx1)](https://github.com/silvaxxx1/HandsOnLLMs)
> 🎯 **Goal:** Transform the original Local RAG prototype into a **scalable, user-ready, robust RAG pipeline** — no cloud, no hassle, no expertise required.

---

## 🔄 From HandsOnLLMs → LiteRAG

This project is **not a fork of someone else's idea** — it's a **continuation and evolution of my own work** from the [HandsOnLLMs](https://github.com/silvaxxx1/HandsOnLLMs) repo.

In that repo, the `Local RAG/` subproject introduced a lightweight RAG system for local experimentation. **LiteRAG now productizes and scales that idea** by:

* Adding a full GUI for end users (Gradio / Streamlit)
* Creating auto-installers and Docker support
* Improving modularity and configurability
* Optimizing for performance on local hardware (CPU and GPU)
* Preserving CLI pipelines for developers and contributors

---

## 🧭 What Is LiteRAG?

A **modular Retrieval-Augmented Generation (RAG)** pipeline that runs entirely offline — designed for **scalability, reliability, and ease of use**.

* 🧩 Modular architecture (easy to swap embedding or LLMs)
* 💻 100% local execution (no cloud, no vendor lock-in)
* 🧠 Small but capable models (like TinyLlama) for real-time CPU inference
* 📊 Efficient FAISS-based retrieval
* 🖥️ Gradio UI for anyone to use — no code required

---

## 📦 Folder Structure

```bash
LiteRAG/
├── app.py                   # Main GUI entrypoint (for end users)
├── installer/               # Auto-start scripts for Windows, macOS, Linux
├── Dockerfile               # Containerized setup
├── models/                  # Local LLMs (GGUF)
├── src/
│   ├── assets/              # PDFs, chunked text, vector files
│   ├── config/              # YAML configurations
│   ├── data/                # Document loaders & chunkers
│   ├── embedding/           # Embedding model logic
│   ├── inference/           # Local LLM inference (llama.cpp, etc.)
│   ├── retrieval/           # Vector search (FAISS)
│   ├── utils/               # File I/O, logging, helpers
│   └── rag_pipeline.py      # CLI pipeline for developers
└── README.md
```

---

## 🖥️ How Users Interact (Simple, No Code)

End users don’t need to touch the terminal.

> ✅ Just run one script:

```bash
bash installer/start.sh
```

Then:

* Upload your documents (PDF, TXT, etc.)
* Ask questions in chat
* All answers are generated **locally** using embedded knowledge from your own files

---

## ⚙️ How Developers Interact (Powerful, Modular CLI)

All pipeline stages are still available for advanced users:

```bash
# Chunk & preprocess docs
python src/rag_pipeline.py data --url http://example.com/file.pdf

# Embed with a specific model
python src/rag_pipeline.py embed --model-key sentence-transformers/all-MiniLM-L6-v2

# Run TinyLlama inference
python src/rag_pipeline.py inf --device cpu

# Full end-to-end RAG pipeline
python src/rag_pipeline.py all --url ... --chunk-size 512 --model-key ... --device cpu
```

> These CLI components are **preserved from the original HandsOnLLMs design**, now made more robust and configurable.

---

## 🔧 Tech Stack

| Component     | Tool / Model                                    |
| ------------- | ----------------------------------------------- |
| Embeddings    | Sentence Transformers (MiniLM, BGE, Instructor) |
| Vector Search | FAISS (default), easily swappable               |
| LLM Inference | TinyLlama (GGUF) + llama.cpp                    |
| Runtime       | `llama-cpp-python`, `transformers`, YAML-based  |
| UI            | Gradio (or Streamlit, user-selectable)          |
| Deployment    | Bash scripts, Docker, cross-platform support    |

---

## 📈 Roadmap

| Status | Feature                                       |
| ------ | --------------------------------------------- |
| ✅      | Modular local pipeline (data, embedding, LLM) |
| ✅      | Developer CLI preserved from HandsOnLLMs      |
| ✅      | Quantized TinyLlama inference via llama.cpp   |
| ✅      | Gradio-based GUI for end users                |
| 🔜     | Windows/macOS GUI installers                  |
| 🔜     | GPU support fallback                          |
| 🔜     | Multi-format document support                 |
| 🔜     | Vector DB backends: ChromaDB, Qdrant          |
| 🔜     | Settings UI, Model/Embedding switcher         |

---

## 🙏 Credits & License

* Original repo: [HandsOnLLMs](https://github.com/silvaxxx1/HandsOnLLMs)
* License: MIT © 2025 [@silvaxxx1](https://github.com/silvaxxx1)
* Thanks to the open-source community for tools like `sentence-transformers`, `FAISS`, `llama.cpp`, and `TinyLlama`.

---

## 💬 TL;DR

**LiteRAG** = hands-on meets hands-off.

* For developers: full control via CLI
* For users: one-click document QA chat
* For everyone: free, local, fast

---
