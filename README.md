# 🖥️ LiteRAG — Local RAG Desktop App

> **Production-ready, offline-first personal knowledge assistant** running entirely on your own hardware. No cloud. No API keys. No ML expertise required.

LiteRAG transforms your documents into a searchable, queryable knowledge library with a native desktop interface. Create projects for different topics, drag & drop documents, and ask questions in natural language — all running locally on your Linux PC or laptop.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📁 **Project-Based** | Create separate projects for Work, Research, Personal — keep contexts clean |
| 🖱️ **Drag & Drop** | Add documents instantly — no terminal, no commands |
| 🔍 **Semantic Search** | Find information by meaning, not just keywords |
| 💬 **Natural Language Q&A** | Ask questions, get answers with page citations |
| 🔌 **Fully Offline** | 100% local execution — your data never leaves your machine |
| ⚡ **CPU Optimized** | Runs smoothly on laptops without GPU |
| 🗑️ **Clean Deletion** | Remove projects and all associated files with one click |
| 🎨 **Native UI** | Built with PySide6 — feels like a real desktop app |

---

## 🚀 Quick Start

### Install

```bash
# Download the AppImage or .deb from Releases
# Make executable (AppImage only)
chmod +x LiteRAG-*.AppImage

# Run
./LiteRAG-*.AppImage
```

### First Use

1. **Create a project** — Name it (e.g., "Research Papers")
2. **Drag & drop** your PDFs or documents
3. **Wait for processing** — documents are chunked and embedded
4. **Start asking questions** — answers with source citations appear instantly

---

## 🎯 How It Works

```
Drag & Drop → Project → Chunk & Embed → Vector Store → LLM → Answer
```

1. **Drag & Drop** — Add PDFs, text files, or markdown documents to any project
2. **SQLite** — Stores project metadata and document information
3. **Text Chunking** — Splits documents into meaningful sentence windows
4. **Embeddings** — Converts text to vectors using lightweight models (BGE-small, MiniLM)
5. **FAISS** — Enables fast semantic search across project documents
6. **Local LLM** — TinyLlama/Mistral generates answers using retrieved context
7. **Clean Deletion** — Remove projects and all associated data instantly

---

## 📁 Project Structure

```
LiteRAG/
├── app/                    # PySide6 UI components
│   ├── ui/                 # Main window, project view, chat panel
│   ├── controller.py       # Main app logic (no HTTP)
│   ├── projects.py         # Project CRUD operations
│   └── config.py           # User preferences
├── core/                   # RAG core (no UI dependencies)
│   ├── document_loader.py  # PDF, txt, markdown parsing
│   ├── chunker.py          # Sentence-based text splitting
│   ├── embedder.py         # SentenceTransformer wrapper
│   ├── vector_store.py     # FAISS index per project
│   ├── retriever.py        # Similarity search logic
│   └── llm_engine.py       # llama.cpp inference
├── data/                   # User data (auto-created)
│   ├── projects/           # Each project folder
│   │   └── {project_id}/
│   │       ├── documents/  # Original uploaded files
│   │       ├── chunks.json # Text chunks with metadata
│   │       ├── vectors.faiss
│   │       └── metadata.db
│   ├── models/             # Downloaded GGUF models
│   └── config.yaml
├── scripts/                # Build & packaging
│   ├── build_appimage.sh
│   └── create_deb.sh
├── main.py                 # Application entry point
└── README.md
```

---

## 🛠️ For Developers

### Run from Source

```bash
# Clone the repository
git clone https://github.com/silvaxxx1/LiteRAG
cd LiteRAG

# Install dependencies
pip install -r requirements.txt

# Download a model (optional, app will prompt)
# TinyLlama: 750MB, runs well on CPU
# Mistral 7B: 4GB, requires more RAM

# Run the app
python main.py
```

### CLI Mode (Power Users)

```bash
# Create a project
python main.py project create --name "Research"

# Add documents
python main.py project add --project-id 1 --files *.pdf

# Ask a question
python main.py ask --project-id 1 --question "What is backpropagation?"

# Delete project
python main.py project delete --project-id 1
```

### Configuration

Edit `~/.config/literag/config.yaml`:

```yaml
# Model settings
llm_model: "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
embed_model: "all-MiniLM-L6-v2"

# Processing
chunk_size: 10          # sentences per chunk
chunk_overlap: 0.2      # 20% overlap between chunks
top_k: 5                # retrieved chunks per query

# Hardware
n_threads: 8            # CPU threads for inference
device: "cpu"           # "cpu" or "cuda"
```

---

## 🧪 Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| **UI** | PySide6 | Native Qt6 — fast, modern, cross-platform |
| **Vector Store** | FAISS | Lightweight, CPU/GPU, no separate service |
| **Embeddings** | SentenceTransformers | BGE-small, MiniLM, Instructor-XL |
| **LLM Runtime** | llama.cpp | Fast CPU inference, quantized models |
| **Document Parsing** | PyMuPDF + Unstructured | Clean text extraction from PDFs |
| **Metadata** | SQLite | Embedded, ACID, zero config |
| **Packaging** | PyInstaller + AppImageKit | Single executable, easy distribution |

---

## 🎯 Roadmap

### Phase 1 — MVP (Current)
- [x] Project-based document organization
- [x] Drag & drop PDF ingestion
- [x] FAISS vector store per project
- [x] Native PySide6 UI
- [x] CPU-only inference

### Phase 2 — Enhanced UX
- [ ] Multiple file types (txt, markdown, EPUB)
- [ ] Progress indicators during processing
- [ ] Document preview within app
- [ ] Export conversations to markdown

### Phase 3 — Advanced Features
- [ ] GPU acceleration auto-detection
- [ ] Hybrid search (BM25 + vectors)
- [ ] Conversation memory
- [ ] Model management (download from within app)

### Phase 4 — Distribution
- [ ] Windows & Mac builds
- [ ] Auto-update system
- [ ] App Store distribution

---

## 🤔 Why LiteRAG?

Most RAG solutions are either:
- **Cloud-based** — your data leaves your machine
- **Frameworks** — require coding knowledge
- **CLI-only** — intimidating for non-technical users

LiteRAG is different:
- ✅ **Native desktop app** — install and use
- ✅ **100% local** — privacy guaranteed
- ✅ **Project-based** — organize by topic
- ✅ **Drag & drop** — no terminal needed
- ✅ **Transparent** — open source, hackable

---

## 📜 License

MIT © 2025 @silvaxxx1

Built as part of the [HandsOnLLMs](https://github.com/silvaxxx1/HandsOnLLMs) project — learning LLMs by building them.

---

## 🙏 Acknowledgements

- [SentenceTransformers](https://www.sbert.net/) — Embedding models
- [FAISS](https://github.com/facebookresearch/faiss) — Vector search
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Local LLM inference
- [PyMuPDF](https://pymupdf.readthedocs.io/) — PDF processing
- [PySide6](https://doc.qt.io/qtforpython/) — Qt6 Python bindings

---

**Start building your personal knowledge library today — offline, private, and always available.** 🚀
