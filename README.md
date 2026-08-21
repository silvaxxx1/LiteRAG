# ⚡ FlashRAG

> **Efficient local RAG for consumer hardware.**

FlashRAG is an **offline-first, CPU-first Retrieval-Augmented Generation (RAG) system** for building searchable personal knowledge libraries on consumer hardware.

It is designed around a simple constraint:

> **How far can we push local RAG performance when RAM, storage, CPU, and latency are limited?**

FlashRAG combines:

* efficient document ingestion
* local embeddings
* compressed vector retrieval
* hybrid BM25 + semantic search
* optional reranking
* local LLM inference
* incremental indexing
* source-level citations
* reproducible performance benchmarks

No cloud API is required.

No external vector database is required.

A GPU is optional.

---

## Project Status

🚧 **Active development**

FlashRAG is being developed as both:

1. a **production-oriented local RAG application**, and
2. an **AI systems engineering project** for studying efficient retrieval, quantization, memory usage, CPU performance, and local inference.

The initial evaluation corpus is a large personal technical book library.

---

# Why FlashRAG?

Most RAG systems optimize primarily for developer convenience.

FlashRAG optimizes for **resource efficiency**.

| Objective              | FlashRAG       |
| ---------------------- | -------------- |
| Cloud dependency       | None           |
| External services      | None required  |
| GPU requirement        | No             |
| Vector database server | No             |
| Local documents        | Yes            |
| Incremental indexing   | Yes            |
| Compressed retrieval   | Yes            |
| Hybrid retrieval       | Yes            |
| Local generation       | Yes            |
| Retrieval benchmarks   | Yes            |
| Consumer hardware      | Primary target |

The goal is not to build another RAG wrapper.

The goal is to build a **small, measurable, efficient local retrieval-and-generation system**.

---

# Architecture

```text
                         ┌─────────────────────┐
                         │      FlashRAG       │
                         └──────────┬──────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
         INGESTION                                      QUERY
              │                                           │
          Documents                                   User Query
              │                                           │
              ▼                                           ▼
          Extraction                               Query Embedding
              │                                           │
              ▼                                  ┌────────┴────────┐
           Chunking                               │                 │
              │                                   ▼                 ▼
              ▼                              Vector Search       BM25
          Embedding                               │                 │
              │                                   │                 │
              ▼                                   └───────┬─────────┘
        Quantization                                      │
              │                                         Fusion
              ▼                                            │
      Compressed Index                                     ▼
              │                                         Reranker
              │                                            │
              └────────────────────────────────────────────┤
                                                           ▼
                                                   Context Builder
                                                           │
                                                           ▼
                                                     Local LLM
                                                           │
                                                           ▼
                                                 Answer + Citations
```

The system is deliberately modular.

The desktop application does not contain the RAG logic.

---

# Core Components

## 1. Document Ingestion

Supported formats:

* PDF
* EPUB
* Markdown
* TXT

Pipeline:

```text
Document
   ↓
Parser
   ↓
Structure extraction
   ↓
Chunking
   ↓
Metadata
   ↓
Embedding
   ↓
Quantization
   ↓
Index
```

Each document receives a stable identity.

Each chunk stores enough metadata to reconstruct its original source.

Example:

```json
{
  "document_id": "book_001",
  "chunk_id": "book_001_00421",
  "title": "Deep Learning",
  "author": "Ian Goodfellow",
  "page": 142,
  "content_hash": "..."
}
```

---

# 2. Incremental Indexing

FlashRAG does not rebuild the entire knowledge base when a new document is added.

```text
New document
     ↓
File hash
     ↓
Already indexed?
   /       \
 yes        no
  │          │
 skip       parse
             ↓
           chunk
             ↓
          embed
             ↓
        quantize
             ↓
           index
```

Document changes are detected through content/file hashes.

This allows large libraries to grow without repeatedly processing existing documents.

Index metadata also records:

* embedding model
* embedding dimension
* chunking configuration
* quantization configuration
* index version
* pipeline version

This prevents incompatible vectors from silently entering an existing index.

---

# 3. Vector Retrieval

FlashRAG exposes a common vector-index interface:

```python
class VectorIndex:
    def add(self, vectors, metadata):
        ...

    def search(self, query_vector, k):
        ...

    def delete(self, ids):
        ...

    def save(self, path):
        ...

    def load(self, path):
        ...
```

Initial backends:

```text
VectorIndex
    │
    ├── FAISS
    │
    └── TurboVec
```

### FAISS

FAISS is the initial reference implementation and baseline.

### TurboVec / TurboQuant

TurboVec is investigated as a compressed vector retrieval backend based on TurboQuant-style quantization.

The objective is to study the trade-off between:

```text
Vector quality
       ↕
Memory
       ↕
Storage
       ↕
Search latency
```

TurboVec is therefore treated as a **retrieval backend**, not as the definition of the entire FlashRAG architecture.

---

# 4. Hybrid Retrieval

Semantic similarity is not sufficient for every query.

Technical books frequently contain exact terminology, identifiers, equations, names, and section titles.

FlashRAG therefore supports hybrid retrieval:

```text
                    Query
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
        Vector Search        BM25
             │                 │
          Top-K               Top-K
             │                 │
             └────────┬────────┘
                      ▼
                    Fusion
                      │
                      ▼
                  Reranking
                      │
                      ▼
                 Final Top-K
```

This allows FlashRAG to combine:

* semantic relevance
* lexical relevance
* optional cross-encoder/reranker relevance

---

# 5. Reranking

The first-stage retriever should optimize for **high recall**.

The reranker can then optimize for **precision**.

Example:

```text
Query
  ↓
TurboVec / FAISS
  ↓
Top 20
  ↓
Reranker
  ↓
Top 5
  ↓
Context builder
```

Reranking is optional because it increases CPU cost.

FlashRAG should allow users to explicitly choose the quality/latency trade-off.

---

# 6. Local LLM Inference

FlashRAG performs generation locally.

Primary runtime:

**llama.cpp**

Model format:

**GGUF**

Example configuration:

```yaml
generation:
  runtime: llama.cpp
  model: models/qwen.gguf
  context_size: 8192
  temperature: 0.1
  threads: 8
```

The LLM runtime is independent of the retrieval backend.

Therefore:

```text
TurboVec + llama.cpp
FAISS + llama.cpp
BM25 + llama.cpp
```

are all valid configurations.

---

# 7. Citation-Aware Generation

Retrieved chunks retain their document metadata.

The generation layer receives structured context:

```text
[Source 1]
Book: Deep Learning
Author: Ian Goodfellow
Page: 142

<chunk text>


[Source 2]
Book: ...
Page: ...
```

The final response can reference the underlying document and page.

The objective is not simply:

> "Generate an answer."

It is:

> **Generate an answer grounded in identifiable local sources.**

---

# Storage Architecture

FlashRAG intentionally avoids requiring a separate database server.

```text
data/
├── documents/
│   ├── original files
│   └── extracted content
│
├── indexes/
│   ├── vector/
│   └── lexical/
│
├── metadata/
│   └── flashrag.db
│
├── models/
│   ├── embeddings/
│   └── llm/
│
└── cache/
```

SQLite stores metadata.

The filesystem stores documents and model/index artifacts.

The vector index is stored separately from metadata.

---

# Configuration

FlashRAG uses a versioned configuration file.

Example:

```yaml
embedding:
  model: BAAI/bge-small-en-v1.5
  normalize: true
  batch_size: 32

chunking:
  strategy: sentence
  chunk_size: 10
  overlap: 2

retrieval:
  vector_backend: turbovec
  metric: cosine
  top_k: 20

hybrid:
  enabled: true
  bm25_weight: 0.3
  vector_weight: 0.7

reranking:
  enabled: false
  top_k: 5

generation:
  runtime: llama.cpp
  model: models/model.gguf
  context_size: 8192
  temperature: 0.1

hardware:
  device: cpu
  threads: 8
```

Configuration is part of the experiment metadata so benchmarks can be reproduced.

---

# Hardware Targets

FlashRAG is designed primarily for:

### Minimum target

```text
CPU: 4+ cores
RAM: 8 GB
Storage: SSD recommended
GPU: optional
```

### Recommended

```text
CPU: 8+ cores
RAM: 16–32 GB
Storage: NVMe SSD
GPU: optional
```

### Long-term target

FlashRAG should remain useful on:

* laptops
* desktops
* CPU-only workstations
* integrated-GPU systems
* ARM systems
* consumer NVIDIA GPUs

Hardware-specific optimizations may include:

* AVX2
* AVX-512
* ARM NEON
* multithreading
* cache-aware memory access

---

# Performance Engineering

Performance is a first-class feature.

FlashRAG measures five major areas.

## Retrieval Quality

```text
Recall@1
Recall@5
Recall@10
MRR
NDCG@10
```

## Retrieval Performance

```text
p50 latency
p95 latency
p99 latency
queries/sec
```

## Memory

```text
Embedding memory
Index memory
Peak RSS
Runtime memory
```

## Storage

```text
Raw vector size
Compressed vector size
Index size
Metadata size
Total corpus footprint
```

## Ingestion

```text
Documents/sec
Pages/sec
Chunks/sec
Embeddings/sec
Index build time
```

---

# Benchmark Methodology

FlashRAG separates evaluation into three levels.

## Benchmark A — Retrieval

```text
Query
  ↓
Embedding
  ↓
Index Search
  ↓
Top-K
```

Measures:

* Recall@K
* MRR
* NDCG
* latency
* RAM

This isolates the retrieval system.

---

## Benchmark B — RAG

```text
Query
  ↓
Retrieval
  ↓
Reranking
  ↓
Context construction
  ↓
LLM
  ↓
Answer
```

Measures:

* answer quality
* citation correctness
* retrieval quality
* TTFT
* tokens/sec
* peak memory

---

## Benchmark C — End-to-End

```text
Document
  ↓
Ingestion
  ↓
Index
  ↓
Query
  ↓
Answer
```

Measures:

* total indexing time
* storage footprint
* query latency
* generation latency
* memory consumption
* answer/citation quality

This prevents a retrieval optimization from being incorrectly evaluated as an end-to-end optimization.

---

# Evaluation Dataset

The primary development corpus is a personal technical book library.

A separate evaluation set will contain manually or programmatically constructed questions with known relevant chunks.

Example:

```json
{
  "question": "What is the difference between RAG and fine-tuning?",
  "relevant_chunks": [
    "book_042:chunk_1832",
    "book_017:chunk_0912"
  ]
}
```

The evaluation dataset should be versioned separately from the application.

This allows:

```text
FAISS
   vs
TurboVec
   vs
Hybrid
```

to be evaluated on exactly the same questions.

---

# Example Benchmark

Benchmark results will eventually look like:

```text
┌────────────────────┬──────────┬─────────┬──────────┐
│ Backend             │ Recall10 │ RAM     │ p95      │
├────────────────────┼──────────┼─────────┼──────────┤
│ FAISS               │    ...   │   ...   │   ...    │
│ TurboVec            │    ...   │   ...   │   ...    │
│ Hybrid              │    ...   │   ...   │   ...    │
└────────────────────┴──────────┴─────────┴──────────┘
```

**Numbers will only be published after reproducible local benchmarks.**

No performance claims are made here without measurements.

---

# Desktop Application

The desktop application is a client of the FlashRAG engine.

```text
┌──────────────────────────────┐
│          PySide6             │
│                              │
│  Projects                    │
│  Documents                   │
│  Search                      │
│  Chat                        │
│  Citations                   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       FlashRAG Engine        │
├──────────────────────────────┤
│ Ingestion                    │
│ Retrieval                    │
│ Reranking                    │
│ Context                      │
│ Generation                   │
└──────────────────────────────┘
```

The engine must remain usable without the graphical interface.

---

# CLI

FlashRAG will provide a CLI for development, automation, and benchmarking.

Example:

```bash
# Create a library
flashrag library create "AI Books"

# Index documents
flashrag index add ./books --library ai-books

# Search
flashrag search "What is attention?"

# Ask
flashrag ask "Explain the difference between RAG and fine-tuning."

# Benchmark
flashrag benchmark retrieval

# Show index information
flashrag index info
```

The CLI is also useful for reproducible experiments.

---

# Project Structure

```text
FlashRAG/
│
├── app/
│   └── desktop/
│       └── ...                 # PySide6 UI
│
├── flashrag/
│   ├── ingestion/
│   │   ├── parsers/
│   │   ├── chunking/
│   │   └── pipeline.py
│   │
│   ├── embeddings/
│   │   ├── base.py
│   │   └── sentence_transformer.py
│   │
│   ├── quantization/
│   │   └── ...
│   │
│   ├── retrieval/
│   │   ├── base.py
│   │   ├── hybrid.py
│   │   ├── fusion.py
│   │   └── reranker.py
│   │
│   ├── indexes/
│   │   ├── faiss.py
│   │   └── turbovec.py
│   │
│   ├── storage/
│   │   ├── sqlite.py
│   │   └── filesystem.py
│   │
│   ├── generation/
│   │   └── llama_cpp.py
│   │
│   └── pipeline.py
│
├── benchmarks/
│   ├── retrieval/
│   ├── rag/
│   ├── end_to_end/
│   └── reports/
│
├── evaluation/
│   ├── questions/
│   └── datasets/
│
├── tests/
│
├── scripts/
│
├── configs/
│
├── docs/
│
├── main.py
├── pyproject.toml
└── README.md
```

---

# Installation

## Requirements

```text
Python 3.11+
Linux recommended for development
8 GB+ RAM
SSD recommended
```

Clone the repository:

```bash
git clone https://github.com/silvaxxx1/FlashRAG.git
cd FlashRAG
```

Create the environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -e ".[dev]"
```

Verify:

```bash
flashrag --help
```

---

# Quick Start

Create a library:

```bash
flashrag library create "Research"
```

Index documents:

```bash
flashrag index add ./books --library Research
```

Search:

```bash
flashrag search "What is self-attention?"
```

Ask a question:

```bash
flashrag ask "Explain self-attention and cite the relevant sources."
```

Launch the desktop application:

```bash
flashrag gui
```

---

# Development

Run tests:

```bash
pytest
```

Run formatting:

```bash
ruff format .
```

Run linting:

```bash
ruff check .
```

Run benchmarks:

```bash
flashrag benchmark retrieval
```

Run the full benchmark suite:

```bash
flashrag benchmark all
```

---

# Testing Strategy

FlashRAG uses several test layers.

### Unit tests

Test:

* parsers
* chunking
* hashing
* metadata
* embeddings
* retrieval
* fusion
* context construction

### Integration tests

Test:

```text
Document
 → Index
 → Search
 → Retrieval
 → Generation
```

### Regression tests

Prevent changes from silently degrading:

* retrieval quality
* latency
* memory usage
* citation accuracy

### Benchmark tests

Performance measurements are stored with configuration metadata.

---

# Data Integrity

FlashRAG should never silently mix incompatible indexes.

Index metadata includes:

```text
embedding_model
embedding_dimension
embedding_version
chunking_version
quantization_version
index_backend
index_version
```

If an incompatible configuration is detected, FlashRAG should require explicit migration or rebuilding.

---

# Privacy

FlashRAG is designed for local execution.

By default:

```text
Documents        → local
Embeddings       → local
Indexes          → local
Metadata         → local
LLM inference    → local
Queries          → local
```

No cloud API is required.

Network access may still be used explicitly for actions such as downloading models or application updates. Such actions should be clearly separated from normal local RAG execution.

---

# Design Principles

## 1. Local first

External infrastructure should not be required.

## 2. CPU first

The system should remain useful without a dedicated GPU.

## 3. Measure before optimizing

Every major optimization should have a benchmark.

## 4. Memory is a constraint

Latency alone is not sufficient.

## 5. Modular backends

Retrieval and inference components should be replaceable.

## 6. Incremental by default

Large libraries should not require complete re-indexing for small changes.

## 7. Reproducible experiments

Configuration, datasets, hardware, and software versions should be recorded.

## 8. Optimize the complete pipeline

```text
Storage
   ↓
Parsing
   ↓
Chunking
   ↓
Embedding
   ↓
Quantization
   ↓
Retrieval
   ↓
Reranking
   ↓
Generation
```

---

# Roadmap

## Phase 1 — Core

* [ ] Repository restructuring
* [ ] Configuration system
* [ ] Document abstraction
* [ ] Stable document/chunk IDs
* [ ] Content hashing
* [ ] SQLite metadata layer
* [ ] Incremental indexing
* [ ] Embedding abstraction
* [ ] VectorIndex interface
* [ ] FAISS backend

## Phase 2 — Efficient Retrieval

* [ ] TurboVec integration
* [ ] TurboQuant experiments
* [ ] Quantized vector storage
* [ ] Memory profiling
* [ ] CPU profiling
* [ ] SIMD-aware optimization
* [ ] FAISS vs TurboVec benchmark

## Phase 3 — Retrieval Quality

* [ ] BM25
* [ ] Hybrid retrieval
* [ ] Reciprocal Rank Fusion
* [ ] Reranking
* [ ] Evaluation dataset
* [ ] Recall/NDCG benchmark suite

## Phase 4 — Local Generation

* [ ] llama.cpp integration
* [ ] GGUF model support
* [ ] Streaming
* [ ] Context management
* [ ] Citation-aware generation
* [ ] Generation benchmarks

## Phase 5 — Desktop

* [ ] PySide6 application
* [ ] Project/library management
* [ ] Drag & drop
* [ ] Search
* [ ] Chat
* [ ] Citation navigation
* [ ] Document preview
* [ ] Index progress monitoring

## Phase 6 — Distribution

* [ ] Linux AppImage
* [ ] Debian package
* [ ] Windows
* [ ] macOS
* [ ] Model management
* [ ] Automatic updates

---

# Long-Term Architecture

```text
                         FlashRAG
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   Ingestion            Retrieval           Generation
        │                   │                   │
   PDF / EPUB          FAISS              llama.cpp
   Markdown            TurboVec           GGUF
   TXT                 BM25
                       Reranker
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                     Benchmark Engine
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
            Quality        Speed         Memory
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                   Consumer Hardware
```

The long-term goal is to build a retrieval engine that can provide strong RAG quality while remaining practical on ordinary personal computers.

---

# What FlashRAG Is Not

FlashRAG is not intended to be:

* a LangChain wrapper
* a cloud RAG service
* a hosted vector database
* a Kubernetes platform
* a GPU-only application
* a collection of unrelated AI libraries

The core system should remain small enough to understand, benchmark, profile, and optimize.

---

# Technology Stack

| Layer                   | Technology                  |
| ----------------------- | --------------------------- |
| Core                    | Python                      |
| Systems optimization    | Rust / C++                  |
| UI                      | PySide6                     |
| Metadata                | SQLite                      |
| Embeddings              | Sentence Transformers       |
| Vector baseline         | FAISS                       |
| Efficient vector search | TurboVec / TurboQuant       |
| Lexical search          | BM25                        |
| Reranking               | Local reranker              |
| LLM runtime             | llama.cpp                   |
| Model format            | GGUF                        |
| Documents               | PDF / EPUB / Markdown / TXT |
| Testing                 | pytest                      |
| Formatting              | Ruff                        |
| Packaging               | PyInstaller / AppImage      |

---

# Contributing

Contributions are welcome once the core architecture stabilizes.

Before submitting a performance optimization, include measurements where possible:

```text
Hardware
Software versions
Dataset
Configuration
Baseline
New result
Memory
Latency
Accuracy
```

Performance changes without reproducible measurements are difficult to evaluate.

---

# License

MIT

---

# Acknowledgements

FlashRAG builds on the work of the open-source and research communities, including projects and technologies such as:

* FAISS
* TurboQuant / TurboVec
* Sentence Transformers
* llama.cpp
* PyMuPDF
* SQLite
* PySide6

---

# Vision

Personal knowledge should not require a cloud service.

FlashRAG aims to make it practical to keep a large technical library on a personal computer, search it semantically, and reason over it locally.

```text
Your books
    ↓
Your machine
    ↓
Your index
    ↓
Your retrieval
    ↓
Your model
    ↓
Your answers
```

**Local knowledge. Efficient retrieval. Consumer hardware.**
