# 🖥️ LiteRAG — Local RAG Desktop App, Production-Ready on Your Own Hardware

> **Goal**: Transform the original Local RAG prototype from my HandsOnLLMs repo into a **scalable, user-ready desktop application** — running entirely offline, with a simple native UI for Linux PCs and laptops. No cloud. No hassle. No ML expertise required.
> **Built for me**: I designed LiteRAG because I personally needed a **fast, private, production-ready RAG app** I could run on my own machines every day — and now I’m making it available for others who want the same.

---

## 📝 Why I Built This

I work with a lot of private datasets — documents, research papers, code, notes — and I wanted a **single, reliable desktop app** that could:

* Search my files semantically, not just by keyword.
* Summarize and answer questions in natural language.
* Run **completely offline** without risking data leaks.
* Be **fast enough** to run on my laptop without a huge GPU.
* Have a **clean, native UI** instead of a clunky browser tab.
* Let me swap models or vector stores easily when I need to experiment.

There was nothing out there that was **both personal and production-ready** — so I built LiteRAG **first for myself**.
The result: my daily driver for research, work, and personal knowledge management. Now, it’s yours too.

---

## 🔄 From *HandsOnLLMs* → **LiteRAG**

LiteRAG is **not a fork** of someone else’s work — it’s the evolution of my own HandsOnLLMs **Local RAG** subproject.

* **Then**: CLI-only prototype for experimentation.
* **Now**: A full **native desktop app** with a local backend, embedded vector DB, and GPU/CPU-optimized models.

LiteRAG upgrades the original concept by:

* Adding a **native PySide6 UI** for desktop users.
* Running a **FastAPI backend** locally for RAG processing.
* Embedding a **lightweight vector DB** (Qdrant or Chroma) — no FAISS installation headaches.
* Optimizing for **real-time CPU & GPU inference**.
* Providing **one-click installers** (`.AppImage` / `.deb`) for Linux.
* Preserving **CLI pipelines** for developers.

---

## 🧩 What Is LiteRAG?

A **modular Retrieval-Augmented Generation (RAG) desktop application** that runs entirely offline. Designed for **portability, reliability, and ease of use**.

* 🔌 **Modular architecture** — swap embedding models or LLMs with config changes.
* 🛡 **100% local execution** — no cloud, no vendor lock-in.
* ⚡ **Small but capable models** — Mistral 7B, Gemma, TinyLlama (quantized) for real-time inference.
* 📂 **Embedded vector DB** — Qdrant or Chroma for fast retrieval.
* 🖥 **Native Linux UI** — PySide6 for smooth, non-web, OS-integrated experience.
* 👤 **Personal-first design** — built to be my own daily driver before sharing with others.

---

## 📁 Folder Structure

```bash
LiteRAG/
├── ui/                # PySide6 UI components
├── backend/           # FastAPI RAG backend
│   ├── embeddings/    # Embedding model logic
│   ├── inference/     # llama.cpp, Ollama integration
│   ├── retrieval/     # Qdrant/Chroma integration
│   ├── utils/         # Parsing, chunking, logging
│   └── config/        # YAML configs
├── models/            # Local LLMs & embedding models
├── installer/         # Auto-start scripts & packaging
├── vector_store/      # Local Qdrant/Chroma data
├── main.py            # App entry point
└── README.md
```

---

## 🎯 How Users Interact (No Terminal Needed)

End users run LiteRAG like any other Linux app.

1. Download `.AppImage` or `.deb`.
2. Double-click to launch.
3. Drag & drop documents (PDF, TXT, DOCX) into the app.
4. Ask questions in the chat window.
5. Get answers **entirely offline** using your local models.

---

## 🛠 How Developers Interact (CLI Power Mode)

CLI pipeline remains available for advanced control:

```bash
# Chunk & preprocess docs
python backend/rag_pipeline.py data --file mydoc.pdf

# Embed with specific model
python backend/rag_pipeline.py embed --model-key bge-small-en-v1.5

# Run local inference
python backend/rag_pipeline.py inf --device cpu

# Full RAG
python backend/rag_pipeline.py all --file mydoc.pdf --chunk-size 512 --model-key bge-small-en-v1.5 --device cpu
```

---

## 🧪 Tech Stack

| Component       | Tool / Model                     |
| --------------- | -------------------------------- |
| **UI**          | PySide6 (Qt for Python)          |
| **Backend**     | FastAPI (local server)           |
| **Embeddings**  | BGE-small, MiniLM, Instructor-XL |
| **Vector DB**   | Qdrant (embedded) / Chroma       |
| **LLM Runtime** | llama.cpp / Ollama               |
| **Doc Parsing** | pymupdf, unstructured            |
| **Packaging**   | PyInstaller + AppImageKit        |
| **Monitoring**  | (Optional) Prometheus + Grafana  |

---

## 📜 Credits & License

* Original repo: **HandsOnLLMs**
* License: MIT © 2025 @silvaxxx1
* Thanks to the open-source community: sentence-transformers, Qdrant, Chroma, llama.cpp, Mistral, Gemma.

---

## 💡 TL;DR

LiteRAG = **Local RAG, done right**.
Built first for **me**, to be my **own daily driver**, and now released so others can enjoy a truly **offline, private, production-ready RAG**.

* For developers: **full CLI control**.
* For end users: **one-click, offline document QA**.
* For everyone: **fast, free, and private**.

---