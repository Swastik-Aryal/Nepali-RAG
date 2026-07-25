# Nepali-RAG

**The all-in-one Retrieval-Augmented Generation framework purpose-built for the Nepali documents, integratable into any LLM environment.**

## Features

- **✦ Devanagari-native pipeline**: Unicode NFC normalization, Nepali sentence splitting and Devanagari script-aware chunking built in.
- **Evaluations**: All embedding + reranker combinations [evaluated](#evaluation-highlights) across 5 RAG metrics on Nepali and English QnA pairs. [Full results](#14-evaluation-results) | [Evaluation docs](https://github.com/Swastik-Aryal/Nepali-RAG/blob/main/nepali_rag/evals/README.md)
- **Hybrid retrieval + reranking**: Dense + Sparse (BM25) fused via Reciprocal Rank Fusion, then cross-encoder reranked for precision.
- **LLM tool-calling mode**: The model autonomously decides when document retrieval is needed; greetings, follow-ups, and translations skip the vector store entirely.
- **Any LLM provider, one interface**: Ollama, vLLM, OpenAI, Gemini, Anthropic, Hugging Face, plus any OpenAI-compatible API (Groq, DeepSeek, Together, etc.).
- **9 document formats**: PDF (OCR), DOCX, PPTX, TXT, Markdown, HTML, EPUB, RTF, ODT with single and batch ingestion.
- **MCP server**: Expose ingestion and retrieval as tools for AI agents (Claude Desktop, Cursor, VS Code, etc.).
- **Python library, CLI, or FastAPI server**: Three interfaces, fully configurable via `.env`, code, or CLI flags.

**[Jump to docs](#contents)** OR **[Quick start](#3-quick-start)**

## Evaluation Highlights

| Rank | Embedding | Reranker | Ans. Relevancy | Faithfulness | Ctx. Recall | Ctx. Precision | Ctx. Relevancy | Retrieval Avg |
|---|---|---|---|---|---|---|---|---|
| **1** | **`jina-embeddings-v5-text-nano-retrieval`** | **`bge-reranker-v2-m3`** | **1.000** | 0.892 | **0.988** | 0.886 | 0.393 | **0.756** |
| 2 | `multilingual-e5-base` | `bge-reranker-v2-m3` | 0.985 | 0.908 | 0.975 | **0.929** | 0.355 | 0.753 |
| 3 | `multilingual-e5-large-instruct` | `bge-reranker-v2-m3` | 0.959 | 0.922 | 0.975 | 0.926 | 0.350 | 0.750 |
| 4 | `bge-m3` | `bge-reranker-v2-m3` | 0.962 | 0.942 | 0.975 | 0.928 | 0.328 | 0.744 |
| 5 | `multilingual-e5-small` | `bge-reranker-v2-m3` | 0.969 | 0.941 | 0.975 | 0.922 | 0.330 | 0.742 |

For the full combination results, see [Evaluation Results](#14-evaluation-results). For methodology and running your own evaluations, see [Evaluation Documentation](https://github.com/Swastik-Aryal/Nepali-RAG/blob/main/nepali_rag/evals/README.md).

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
   - [3.1 Ingest a document](#31-ingest-a-document)
   - [3.2 Query your documents](#32-query-your-documents)
4. [FastAPI Server](#4-fastapi-server)
5. [MCP Server](#5-mcp-server)
6. [Configuration](#6-configuration)
7. [CLI Reference](#7-cli-reference)
8. [Supported Document Formats](#8-supported-document-formats)
9. [LLM Providers](#9-llm-providers)
10. [Embedding Models](#10-embedding-models)
11. [Rerankers](#11-rerankers)
12. [Project Structure](#12-project-structure)
13. [Component Details](#13-component-details)
    - [i. Architecture](#i-architecture)
    - [ii. Document Ingestion](#ii-document-ingestion-nepali_ragingestion)
    - [iii. Text Preprocessing](#iii-text-preprocessing-nepali_ragpreprocessing)
    - [iv. Chunking](#iv-chunking-nepali_ragchunking)
    - [v. Embeddings](#v-embeddings-nepali_ragembeddings)
    - [vi. Vector Store](#vi-vector-store-nepali_ragvectorstore)
    - [vii. Sparse BM25 Search](#vii-sparse-bm25-search)
    - [viii. Retrieval](#viii-retrieval-nepali_ragretrieval)
    - [ix. RAG Chain](#ix-rag-chain-nepali_ragrag)
    - [x. Tool-Calling Mode](#x-tool-calling-mode)
    - [xi. LLM Integration](#xi-llm-integration-nepali_ragllm)
    - [xii. Document Pipeline](#xii-document-pipeline-nepali_ragpipeline)
    - [xiii. RAG Configuration](#xiii-rag-configuration-nepali_ragconfig)
14. [Evaluation Results](#14-evaluation-results)
15. [References](#15-references)
16. [License](#16-license)

## 1. Prerequisites

- **Python 3.10+**
- **Tesseract OCR** with Nepali language data (`nep` trained data)
  - Linux: `sudo apt install tesseract-ocr tesseract-ocr-nep`
  - macOS: `brew install tesseract` then download `nep.traineddata`
  - Windows: install from [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki) and add Nepali language data from [here](https://github.com/tesseract-ocr/tessdata/blob/main/nep.traineddata)
- **Qdrant** vector database instance (cloud)
  - Create a cluster at [cloud.qdrant.io](https://cloud.qdrant.io) - you'll need the URL and API key
- **LLM backend** (at least one):
  - Ollama (easiest): `ollama pull gemma4` or any other model of choice (would recommend a model that supports tool-calling)
  - vLLM server for OpenAI-compatible API
  - OpenAI API key
  - Any OpenAI-compatible provider (Groq, DeepSeek, Together AI, Fireworks, Mistral, etc.) - just needs an API key and base URL
  - Google Gemini API key
  - Hugging-face API key

## 2. Installation

```bash
git clone https://github.com/Swastik-Aryal/Nepali-RAG.git
cd Nepali-RAG
pip install -e . --config-settings editable_mode=compat
```

This installs all dependencies (document parsers, embeddings, vector store, LLM providers, API server).

> [!TIP]
> **GPU acceleration:** The default install pulls CPU-only PyTorch. If you have an NVIDIA GPU, install the CUDA build **before or after** `pip install -e .`:

```bash
# Example for CUDA 12.1 (check your version with: nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Environment File (Optional but Recommended)

For the Python usage (less control snippets), the library reads settings from a `.env` file. Copy the template:

```bash
cp .env.example .env
```

> [!IMPORTANT]
> For Python scripts, the `.env` file is loaded from your **current working directory** (CWD). For the API server, you can point to any `.env` file with `nepali-rag serve --env-file /path/to/.env`.

At minimum, set your Qdrant connection:

```env
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

> [!NOTE]
> If `.env` file is absent, the library will use the default values set in [config/settings.py](https://github.com/Swastik-Aryal/Nepali-RAG/blob/main/nepali_rag/config/settings.py) (also viewable in `.env.example`).

> [!NOTE]
> The `.env` file is **not required** for CLI usage. All parameters are passed directly as command-line arguments.

## 3. Quick Start

### 3.1 Ingest a document

**Python (less control, requires `.env` file):**

```python
from nepali_rag import DocumentPipeline

pipeline = DocumentPipeline()
result = pipeline.ingest("path/to/document.pdf")
print(result)
# {'collection_name': 'doc_a1b2c3d4e5f6', 'filename': '...', 'pages_extracted': 12, 'chunks_created': 45, 'chunks_indexed': 45}
```

**Python (more control, no `.env` needed, all params in code):**

```python
from nepali_rag import DocumentPipeline, NepaliEmbeddingModel, QdrantStore

embedding_model = NepaliEmbeddingModel(model_name = "intfloat/multilingual-e5-large-instruct")
store = QdrantStore(
    vector_size=embedding_model.dimension,  
    collection_name="my_collection",
    url="https://your-cluster.cloud.qdrant.io",
    api_key="your-qdrant-api-key",
)

pipeline = DocumentPipeline(
    embedding_model=embedding_model,
    vector_store=store,
    chunk_size=256,
)

# Single document
result = pipeline.ingest("path/to/document.pdf")
print(result)
# {'collection_name': 'my_collection', 'filename': '...', 'pages_extracted': 12, 'chunks_created': 45, 'chunks_indexed': 45}

# Multiple documents (batch ingestion into the same collection)
results = pipeline.ingest_batch([
    "path/to/document1.pdf",
    "path/to/document2.pdf",
    "path/to/notes.docx",
])
for r in results:
    print(f"{r['filename']}: {r['chunks_indexed']} chunks indexed")
```

**Terminal (no `.env` needed, all params as arguments):**

```bash
nepali-rag ingest doc1.pdf doc2.pdf notes.docx \
  --qdrant-url https://your-cluster.cloud.qdrant.io \
  --qdrant-api-key your-qdrant-api-key \
  -c my_collection \
  --embedding intfloat/multilingual-e5-large \
  --chunk-size 256 \
  -v
```

---

### 3.2 Query your documents

**Python (less control, requires `.env` file):**

```python
from nepali_rag import Retriever, NepaliEmbeddingModel, QdrantStore, RAGChain, get_llm

embedding_model = NepaliEmbeddingModel()
store = QdrantStore(vector_size=embedding_model.dimension, collection_name="my_collection") # Name of collection where chunks are stored (Defaults to .env/settings.py value)
retriever = Retriever(embedding_model=embedding_model,store=store)
llm = get_llm()
chain = RAGChain(retriever=retriever, llm=llm)

result = chain.run(question="नेपालको संविधानमा के लेखिएको छ?")
print(result["answer"])
```

**Python (more control, no `.env` needed, all params in code):**

```python
from nepali_rag import (
    Retriever, Reranker, QdrantStore,
    NepaliEmbeddingModel, RAGChain, get_llm,
)

embedding_model = NepaliEmbeddingModel(model_name = "intfloat/multilingual-e5-large-instruct")
reranker = Reranker(model_name = "BAAI/bge-reranker-v2-m3")

store = QdrantStore(
  vector_size=embedding_model.dimension, # Dimensionality of the dense embedding vectors
  collection_name="my_collection", # Name of collection where chunks are stored (Defaults to .env/settings.py value)
  url="https://your-cluster.cloud.qdrant.io",
  api_key="your-qdrant-api-key"
)

retriever = Retriever(embedding_model=embedding_model, reranker=reranker, store=store)
llm = get_llm(provider="gemini", model="gemini-2.5-flash", api_key="your-google-api-key") # api_key not used if provider = "ollama" OR "vllm", set base_url in that case 
chain = RAGChain(retriever=retriever, llm=llm, use_tools=True)

# First question
result = chain.run(question="नेपालको संविधानमा के लेखिएको छ?", top_k_retrieval=20, top_k_rerank=5)
print(result["answer"])

# Follow-up with conversation history (only the current question is used for retrieval)
history = [
    {"role": "user", "content": "नेपालको संविधानमा के लेखिएको छ?"},
    {"role": "assistant", "content": result["answer"]},
]
follow_up = chain.run(
    question="त्यसको बारेमा थप विस्तारमा बताउनुहोस्",
    conversation_history=history,
    top_k_retrieval=20,
    top_k_rerank=5
)
print(follow_up["answer"])
```

**Conversation history format:**

```python
# A list of previous turns (do NOT include the current question here)
conversation_history = [
    {"role": "user", "content": "नेपालको राजधानी कहाँ हो?"},
    {"role": "assistant", "content": "नेपालको राजधानी काठमाडौं हो।"},
    {"role": "user", "content": "त्यहाँको जनसंख्या कति हो?"},
    {"role": "assistant", "content": "काठमाडौं उपत्यकाको जनसंख्या लगभग ३० लाख छ।"},
]
```

**Terminal (no `.env` needed, all params as arguments):**

```bash
nepali-rag query "नेपालको संविधानमा के लेखिएको छ?" \
  --qdrant-url https://your-cluster.cloud.qdrant.io \
  --qdrant-api-key your-qdrant-api-key \
  -c my_collection \
  --provider ollama \
  --model gemma4 \
  --top-k 20 \
  --top-r 5 \
  --temperature 0.3 \
  --embedding intfloat/multilingual-e5-large \
  --reranker BAAI/bge-reranker-v2-m3 \
  -v
```

With conversation history (pass a JSON file or inline JSON):

```bash
# Using a JSON file
nepali-rag query "त्यसको बारेमा थप विस्तारमा बताउनुहोस्" \
  --qdrant-url https://your-cluster.cloud.qdrant.io \
  --qdrant-api-key your-qdrant-api-key \
  -c my_collection \
  --provider ollama \
  --model gemma4 \
  --history history.json \
  -v

# Using inline JSON
nepali-rag query "Can you translate that to English?" \
  --qdrant-url https://your-cluster.cloud.qdrant.io \
  --qdrant-api-key your-qdrant-api-key \
  -c my_collection \
  --provider ollama \
  --model gemma4 \
  --history '[{"role":"user","content":"नेपालको राजधानी कहाँ हो?"},{"role":"assistant","content":"नेपालको राजधानी काठमाडौं हो।"}]' \
  -v
```

> [!TIP]
> By default, tool-calling mode is enabled and the LLM decides when retrieval is needed (recommended). Use `--disable-tools` to force retrieval on every query (always-retrieve mode).

For providers that need an API key (openai, gemini, hf_cloud):

```bash
nepali-rag query "नेपालको संविधानमा के लेखिएको छ?" \
  --qdrant-url https://your-cluster.cloud.qdrant.io \
  --qdrant-api-key your-qdrant-api-key \
  -c my_collection \
  --provider gemini \
  --model gemini-2.5-flash \
  --api-key your-google-api-key \
  -v
```

## 4. FastAPI Server

```bash
# Uses .env from CWD by default
nepali-rag serve --host 0.0.0.0 --port 8080 --reload -v

# Point to a .env file anywhere on disk
nepali-rag serve --env-file /path/to/.env --port 8080 -v
```

The API will be available at `http://localhost:8080`. Interactive docs at `http://localhost:8080/docs`.

> [!NOTE]
> The API server reads configuration from a `.env` file (Qdrant credentials, LLM provider, etc.). Use `--env-file` to specify the path if it's not in your current directory.

Full API server documentation here: [API SERVER DOCUMENTATION](https://github.com/Swastik-Aryal/Nepali-RAG/blob/main/nepali_rag/api/README.md)

## 5. MCP Server

Nepali-RAG ships with an MCP (Model Context Protocol) server that exposes ingestion and retrieval as tools for AI agents. AI coding assistants (Claude Desktop, Cursor, Windsurf, Claude Code, VS Code, etc.) can call these tools directly without adopting the full Nepali-RAG framework.

The server uses **Streamable HTTP transport** and listens at `http://localhost:8000/mcp` by default.

### Starting the Server

```bash
# Eager mode (default) -- embedding model and reranker loaded at startup
nepali-rag mcp
```

### Tools

- **`ingest_document`** - Ingest one or more document files into a Qdrant collection for retrieval.
- **`retrieve`** - Search a Qdrant collection and return reranked results.

Full documentation including input/output schemas, storage guide, and configuration examples for all supported clients here: [MCP SERVER DOCUMENTATION](https://github.com/Swastik-Aryal/Nepali-RAG/blob/main/nepali_rag/mcp/README.md)

## 6. Configuration

Default settings in [config/settings.py](https://github.com/Swastik-Aryal/Nepali-RAG/blob/main/nepali_rag/config/settings.py). Can be overridden/managed through a `.env` file. See [`.env.example`](https://github.com/Swastik-Aryal/Nepali-RAG/blob/main/.env.example) for the full list.

| Setting                 | Default                              | Purpose                                                                         |
| ----------------------- | ------------------------------------ | ------------------------------------------------------------------------------- |
| `QDRANT_URL`          | *(empty)*                          | Vector database connection URL                                                  |
| `QDRANT_API_KEY`      | *(empty)*                          | Qdrant Cloud authentication key                                                 |
| `COLLECTION_NAME`     | `nepali_docs`                      | Default Qdrant collection                                                       |
| `EMBEDDING_MODEL`     | `intfloat/multilingual-e5-large`   | Embedding model name                                                            |
| `RERANKER_MODEL`      | `BAAI/bge-reranker-v2-m3`          | Reranker model name                                                             |
| `LLM_PROVIDER`        | `ollama`                           | LLM backend:`vllm`, `openai`, `gemini`, `ollama`, `hf`, `hf_cloud`, `anthropic` |
| `VLLM_MODEL`          | `google/gemma-4-E4B-it`            | Model for vLLM provider                                                         |
| `VLLM_BASE_URL`       | `http://localhost:8000/v1`         | vLLM server endpoint                                                            |
| `OPENAI_API_KEY`      | *(empty)*                          | OpenAI API key (or key for any OpenAI-compatible provider)                      |
| `OPENAI_BASE_URL`     | *(empty)*                          | Custom base URL for OpenAI-compatible providers (e.g. Groq, DeepSeek)           |
| `OPENAI_MODEL`        | `gpt-4o-mini`                      | Model for OpenAI / OpenAI-compatible provider                                   |
| `GOOGLE_API_KEY`      | *(empty)*                          | Google API key for Gemini                                                       |
| `GEMINI_MODEL`        | `gemini-2.5-flash`                 | Model for Gemini provider                                                       |
| `OLLAMA_BASE_URL`     | `http://localhost:11434`           | Ollama server endpoint                                                          |
| `OLLAMA_MODEL`        | `gemma4`                           | Model for Ollama provider                                                       |
| `HF_MODEL_NAME`       | `Qwen/Qwen3.6-35B-A3B`             | Model for local or cloud Hugging Face inference                                 |
| `HF_BASE_URL`         | `https://router.huggingface.co/v1` | Hugging Face cloud inference endpoint                                           |
| `HF_TOKEN`            | *(empty)*                          | Hugging Face cloud API token                                                    |
| `ANTHROPIC_API_KEY`   | *(empty)*                          | Anthropic API key for Claude                                                    |
| `ANTHROPIC_MODEL`     | `claude-opus-4-6`                  | Model for Anthropic provider                                                    |
| `TESSERACT_LANGUAGES` | `nep`                              | OCR language configuration                                                      |
| `CHUNK_SIZE`          | `256`                              | Maximum chunk size in tokens                                                    |
| `TOP_K_RETRIEVAL`     | `20`                               | Hybrid search candidate count                                                   |
| `TOP_K_RERANK`        | `5`                                | Final reranked result count                                                     |
| `DENSE_WEIGHT`        | `0.7`                              | Dense score weight in hybrid search                                             |
| `SPARSE_WEIGHT`       | `0.3`                              | Sparse score weight in hybrid search                                            |
| `LLM_TEMPERATURE`     | `0.3`                              | LLM generation temperature                                                      |
| `USE_TOOLS`           | `true`                             | Enable tool-calling mode                                                        |

## 7. CLI Reference

```
nepali-rag ingest <file> [<file> ...]
    --qdrant-url URL          Qdrant instance URL (default: $QDRANT_URL)
    --qdrant-api-key KEY      Qdrant API key (default: $QDRANT_API_KEY)
    -c, --collection NAME     collection name (default: nepali_docs)
    --chunk-size N            token chunk size (default: 256)
    --embedding <name>        embedding model (default: intfloat/multilingual-e5-large)
    -v                        enable debug logging

nepali-rag query "<question>"
    --qdrant-url URL          Qdrant instance URL (default: $QDRANT_URL)
    --qdrant-api-key KEY      Qdrant API key (default: $QDRANT_API_KEY)
    -c, --collection NAME     collection name (default: nepali_docs)
    --provider PROVIDER       LLM provider (default: ollama)
    --model MODEL             LLM model name (default: gemma4)
    --api-key KEY             API key for LLM provider (default: $LLM_API_KEY)
    --llm-base-url URL        base URL for LLM server (default: $LLM_BASE_URL)
    --temperature FLOAT       LLM temperature (default: 0.3)
    --top-k N                 retrieval candidates (default: 20)
    --top-r N                 rerank candidates (default: 5)
    --history FILE_OR_JSON    conversation history (JSON file path or inline JSON string)
    --test                    test mode (no LLM, returns raw prompt)
    --disable-tools           disable tool-calling mode (forces retrieval on every query)
    --embedding <name>        embedding model (default: intfloat/multilingual-e5-large)
    --reranker <name>         reranker model (default: BAAI/bge-reranker-v2-m3)
    -v                        enable debug logging

nepali-rag serve
    --env-file PATH           path to .env file (default: .env in CWD)
    --host HOST               bind address (default: 0.0.0.0)
    --port PORT               port (default: 8080)
    --reload                  auto-reload for development
    -v                        enable debug logging

nepali-rag mcp
    --no-init                 (Not recommended)skip eager model loading (load on first tool call)
    -v                        enable debug logging

```

> [!TIP]
> Qdrant credentials (`--qdrant-url`, `--qdrant-api-key`) and LLM credentials (`--api-key`, `--llm-base-url`) all fall back to environment variables when not passed as flags. Export `QDRANT_URL`, `QDRANT_API_KEY`, `LLM_API_KEY`, and `LLM_BASE_URL` in your shell to avoid repeating them on every command.
For evaluation guide visit [Evaluation Documentation](https://github.com/Swastik-Aryal/Nepali-RAG/blob/main/nepali_rag/evals/README.md).
## 8. Supported Document Formats

| Format                    | Extensions          | Method                     |
| ------------------------- | ------------------- | -------------------------- |
| PDF (digital and scanned) | `.pdf`            | Tesseract OCR on all pages |
| Plain Text                | `.txt`            | Direct read                |
| Markdown                  | `.md`             | Direct read                |
| Word                      | `.docx`           | python-docx                |
| PowerPoint                | `.pptx`           | python-pptx                |
| RTF                       | `.rtf`            | striprtf                   |
| HTML                      | `.html`, `.htm` | BeautifulSoup              |
| EPUB                      | `.epub`           | EbookLib                   |
| OpenDocument              | `.odt`            | odfpy                      |

> [!IMPORTANT]
> **PDF OCR is optimized for Nepali (Devanagari) text only.** The OCR pipeline uses Tesseract with `nep` trained data. English-only or primarily English PDFs will produce garbled or incorrect characters. For English content, use non-PDF formats (`.docx`, `.txt`, `.md`, etc.) which rely on direct text extraction instead of OCR.

## 9. LLM Providers

The system supports seven LLM backends through a unified `get_llm()` factory. Switch providers by changing the `LLM_PROVIDER` environment variable or passing `--provider` to the CLI.

| Provider                | Setup                                                    | Model                                                                               |
| ----------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **`ollama`**    | `ollama pull gemma4`                                   | Set `model = "gemma4"` OR any other model but in the ollama format.               |
| **`vllm`**      | Start vLLM server with `--model google/gemma-4-E4B-it` | Set `model = "google/gemma-4-E4B-it"` OR any other model but in the vllm format.  |
| **`openai`**    | Set `OPENAI_API_KEY` or pass `--api-key`             | Set `model = "gpt-4o-mini"` OR any other model based on the OpenAI API docs.      |
| **`gemini`**    | Set `GOOGLE_API_KEY` or pass `--api-key`             | Set `model = "gemini-2.5-flash"` OR any other model based on the Google API docs. |
| **`hf`**        | Install model dependencies; runs locally             | Set `model = "Qwen/Qwen3.6-35B-A3B"` OR any other chat model repository.          |
| **`hf_cloud`**  | Set `HF_TOKEN` or pass `--api-key`                   | Set `model = "Qwen/Qwen3.6-35B-A3B"` OR any cloud-supported model repository.     |
| **`anthropic`** | Set `ANTHROPIC_API_KEY` or pass `--api-key`          | Set `model = "claude-opus-4-6"` OR any Anthropic Claude model.                    |

When no LLM is available, the system operates in **test mode**: the full retrieval pipeline runs but returns the formatted prompt instead of a generated answer.

### OpenAI-Compatible Providers

The `openai` provider also works with **any third-party API that follows the OpenAI format** - just set `OPENAI_BASE_URL` to point at the provider's endpoint. This includes:

| Provider              | Base URL                                  | Example Model                              |
| --------------------- | ----------------------------------------- | ------------------------------------------ |
| **Groq**        | `https://api.groq.com/openai/v1`        | `llama-3.3-70b-versatile`                |
| **DeepSeek**    | `https://api.deepseek.com/v1`           | `deepseek-chat`                          |
| **Together AI** | `https://api.together.xyz/v1`           | `meta-llama/Llama-3-70b`                 |
| **Fireworks**   | `https://api.fireworks.ai/inference/v1` | `accounts/fireworks/models/llama-v3-70b` |
| **Mistral**     | `https://api.mistral.ai/v1`             | `mistral-large-latest`                   |
| **OpenRouter**  | `https://openrouter.ai/api/v1`          | `openai/gpt-4o`                          |

**`.env` example for Groq:**

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=gsk_your-groq-key-here
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.3-70b-versatile
```

**Python:**

```python
from nepali_rag import get_llm

llm = get_llm(
    "openai",
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_...",
    model="llama-3.3-70b-versatile",
)
```

**CLI:**

```bash
nepali-rag query "नेपालको संविधानमा के लेखिएको छ?" \
  --qdrant-url https://your-cluster.cloud.qdrant.io \
  --qdrant-api-key your-qdrant-api-key \
  -c my_collection \
  --provider openai \
  --model llama-3.3-70b-versatile \
  --api-key gsk_your-groq-key \
  --llm-base-url https://api.groq.com/openai/v1
```

> [!WARNING]
> The base URL should point to the `/v1` endpoint (not `/v1/chat/completions`). The client appends `/chat/completions` automatically.

## 10. Embedding Models

The system supports multiple embedding model families through the `NepaliEmbeddingModel` wrapper. 

Switch models by setting the `EMBEDDING_MODEL` environment variable in your `.env` file or by passing `--embedding <model_name>` in the CLI.

The wrapper automatically handles prefix/prompt strategies per model family.


| Model | Parameters | Max Tokens | Embedding Dimensions |
| --- | --- | --- | --- |
| `BAAI/bge-m3` | 568M | 8192 | 1024 |
| `intfloat/multilingual-e5-small` | 118M | 512 | 384 |
| `intfloat/multilingual-e5-base` | 278M | 512 | 768 |
| `intfloat/multilingual-e5-large` | 560M | 512 | 1024 |
| `intfloat/multilingual-e5-large-instruct` | 560M | 512 | 1024 |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | 32768 | 1024 |
| `Qwen/Qwen3-Embedding-4B` | 4B | 32768 | 2560 |
| `Qwen/Qwen3-Embedding-8B` | 8B | 32768 | 4096 |
| `jinaai/jina-embeddings-v3` | 572M | 8192 | 1024 |
| `jinaai/jina-embeddings-v5-text-nano-retrieval` | 239M | 8192 | 768 |
| `jinaai/jina-embeddings-v5-text-small-retrieval` | 677M | 8192 | 1024 |
| `Alibaba-NLP/gte-multilingual-base` | 305M | 8192 | 768 |

> [!NOTE]
> **Default:** `intfloat/multilingual-e5-large` (1024-dim). See [Component Details - Embeddings](#v-embeddings-nepali_ragembeddings) for rationale.

## 11. Rerankers

The retrieval pipeline uses a ***cross-encoder*** reranker to re-score candidates after hybrid search. 

Switch models by setting the `RERANKER_MODEL` environment variable in your `.env` file or by passing `--reranker <model_name>` in the CLI.


### Recommended Rerankers

| Model | Parameters | Max Tokens |
| --- | --- | --- |
| `BAAI/bge-reranker-v2-m3` | 568M | 8192 |
| `Qwen/Qwen3-Reranker-0.6B` | 0.6B | 32768 |
| `BAAI/bge-reranker-v2-gemma` | 2.6B | 8192 |

> [!NOTE]
> **Default:** `BAAI/bge-reranker-v2-m3` - lightweight, fast, and strong multilingual relevance scoring. See [Component Details - Retrieval](#viii-retrieval-nepali_ragretrieval) for rationale.

## 12. Project Structure

```
nepali-rag/
├── nepali_rag/                  # Core package
│   ├── __init__.py              # Public API exports and __version__
│   ├── config/settings.py       # Centralized settings (pydantic-settings)
│   ├── ingestion/               # Document parsers (OCR, docx, pptx, etc.)
│   ├── preprocessing/           # Unicode NFC normalization, Nepali detection
│   ├── chunking/                # Nepali-aware recursive character splitter (token-sized)
│   ├── embeddings/              # Embedding model wrappers
│   ├── vectorstore/             # Qdrant dense + sparse hybrid store
│   ├── retrieval/               # Hybrid retrieval + cross-encoder reranking
│   ├── rag/                     # RAG chain (always-retrieve + tool-calling)
│   ├── llm/                     # Multi-provider LLM factory
│   ├── pipeline/                # Full document ingestion orchestrator
│   ├── evals/                   # Evaluation config, scripts
│   ├── mcp/                     # MCP server (streamable-http transport, 2 tools)
│   │   └── server.py            # Tool definitions, model lifecycle, entry point
│   ├── data/                    # Test data and results
│   │   ├── chunking_tests/      # Documents and result JSONs for chunking tests
│   │   └── evaluation_tests/    # Eval dataset and eval result JSONs
│   └── api/                     # FastAPI HTTP layer
│       ├── main.py              # App setup, lifespan, CORS
│       ├── routes.py            # Upload, query, health, collection endpoints
│       ├── schemas.py           # Pydantic request/response models
│       └── conversations/       # Server-side conversation persistence
├── cli.py                       # CLI entry point (ingest, query, serve)
├── pyproject.toml               # Package metadata and dependencies
├── .env.example                 # Environment variable template
├── LICENSE                      # MIT License
└── README.md
```

## 13. Component Details

### i. Architecture

![architecture](https://raw.githubusercontent.com/Swastik-Aryal/Nepali-RAG/main/nepali_rag/data/architecture/architecture.svg)

### ii. Document Ingestion (`nepali_rag/ingestion/`)

Handles loading and text extraction from 9 document formats. PDFs are rendered page-by-page to 300 DPI images using **PyMuPDF** (`fitz`) and processed through **Tesseract OCR** with Nepali (`nep`) language data. Other formats use dedicated parsers:

| Library                                   | Formats                      |
| ----------------------------------------- | ---------------------------- |
| **pytesseract** + **PyMuPDF** | `.pdf` (all pages via OCR) |
| **python-docx**                     | `.docx`                    |
| **python-pptx**                     | `.pptx`                    |
| **striprtf**                        | `.rtf`                     |
| **BeautifulSoup**                   | `.html`, `.htm`          |
| **EbookLib**                        | `.epub`                    |
| **odfpy**                           | `.odt`                     |
| Built-in                                  | `.txt`, `.md`  |

Every page is returned as a `PageResult` dataclass carrying the extracted text, source path, page number, and extraction method.

---

### iii. Text Preprocessing (`nepali_rag/preprocessing/`)

Normalizes raw extracted text for consistent downstream processing:

1. **Unicode NFC normalization** - ensures composed Devanagari forms (e.g., combining sequences are collapsed into single codepoints).
2. **Zero-width character removal** - strips `U+200B` (zero-width space), `U+200C`/`U+200D` (joiners), and `U+FEFF` (BOM).
3. **Whitespace normalization** - collapses runs of spaces/tabs to a single space, and 3+ newlines to a double newline.
4. **Nepali detection** - `is_nepali_text()` classifies text as primarily Devanagari if >60% of non-whitespace characters fall in the `U+0900`–`U+097F` range.

---

### iv. Chunking (`nepali_rag/chunking/`)

Uses LangChain's **`RecursiveCharacterTextSplitter`** with a Nepali-aware separator hierarchy and token-based sizing. Pages are concatenated before splitting so chunks can flow across page boundaries.

- **Separator hierarchy**: the splitter tries, in order, paragraph breaks (`\n\n`), line breaks (`\n`), Nepali sentence delimiters (purna viram `।`, double danda `॥`), other terminators (`?`, `!`), spaces, and finally raw characters. This keeps splitting language-agnostic while still respecting Devanagari sentence boundaries. The full stop `.` is intentionally **excluded** because in Nepali it mostly appears inside numbers/abbreviations, where splitting on it fragments text badly.
- **Token-based sizing**: chunk size and overlap are measured with the **HuggingFace `AutoTokenizer`** for the configured embedding model, so chunks never exceed the model's context window. Default maximum chunk size is **256 tokens**.
- **Overlap**: a configurable percentage of `chunk_size` (default **15%**, via `OVERLAP_RATIO`) is repeated between consecutive chunks to preserve context across boundaries. Every chunk carrying overlap from its predecessor is flagged via `has_overlap`.
- **Multi-page chunks**: `page_number` metadata joins pages with `&` (e.g., `"3&4"`) when a chunk spans pages.
- **Chunk metadata**: each chunk carries `source`, `page_number`, `chunk_index`, `token_count`, and `has_overlap`.
- **Test data**: Sample documents and chunking results obtained during testing JSONs are in [`nepali_rag/data/chunking_tests/`](https://github.com/Swastik-Aryal/Nepali-RAG/tree/main/nepali_rag/data/chunking_tests/).

---

### v. Embeddings (`nepali_rag/embeddings/`)

- **Library**: **sentence-transformers** (`SentenceTransformer`). All embeddings are L2-normalized at encode time (`normalize_embeddings=True`).
- **Prefix/prompt handling**: The wrapper auto-applies the correct prefix strategy per model family (e.g., `"passage: "`/`"query: "` for E5, task instructions for Qwen3).
- **Default model**: [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) (1024-dim).

**Supported model families and selection rationale:**

**1. BGE-M3** [[2]](#ref-2)[[3]](#ref-3)[[7]](#ref-7)

- A 2026 preprint [[7]](#ref-7) evaluating seven multilingual and three monolingual models on Nepali legal document retrieval found that `BGE-M3` achieved the highest scores across all metrics and all evaluation settings: *Recall@10 = 0.9233, Precision@1 = 0.7400, and MRR@10 = 0.8300*
- The model uses *XLM-RoBERTa* as its backbone, pretrained on CommonCrawl data covering 100+ languages including Nepali (Devanagari script), giving it a strong foundation for Nepali text understanding out of the box. 
- In the BGE-M3 paper [[2]](#ref-2), `BGE-M3` achieved state-of-the-art results on multilingual (MIRACL) and cross-lingual (MKQA) benchmarks, outperforming prior models like `multilingual-e5-large` and `E5-mistral-7b` across 18 languages.

| Model | Parameters | Max Tokens | Embedding Dimensions |
| --- | --- | --- | --- |
| `BAAI/bge-m3` | 568M | 8192 | 1024 |

**2. Multilingual E5 Series** [[1]](#ref-1)[[7]](#ref-7)[[8]](#ref-8)

- On the MMTEB Indic language subset (23 datasets), `multilingual-e5-large-instruct` ranks #1 with an average score of 70.2, including 84.9 on retrieval tasks and 67.0 on classification (2025) [[8]](#ref-8).
- On the Nepali legal retrieval benchmark [[7]](#ref-7), `multilingual-e5-large-instruct` achieved *Recall@10 = 0.837, Precision@1 = 0.580, and MRR@10 = 0.663*, solidly in the top tier though trailing `BGE-M3` and `jina-v3`. 
- In a separate study on Nepali FAQ retrieval [[1]](#ref-1), finetuned E5 variants (small, base, large) were evaluated against SBERT-based Nepali models and BM25; E5-large achieved the highest *Recall@10 (0.8902)* and competitive MRR scores, outperforming all other approaches.
- The series also uses an XLM-RoBERTa backbone and remains lightweight (560M for the large variant).

| Model | Parameters | Max Tokens | Embedding Dimensions |
| --- | --- | --- | --- |
| `intfloat/multilingual-e5-small` | 118M | 512 | 384 |
| `intfloat/multilingual-e5-base` | 278M | 512 | 768 |
| `intfloat/multilingual-e5-large` | 560M | 512 | 1024 |
| `intfloat/multilingual-e5-large-instruct` | 560M | 512 | 1024 |

**3. Jina Embeddings v3/v5** [[7]](#ref-7)[[9]](#ref-9)[[10]](#ref-10)

- `jina-embeddings-v3` ranks as the second-best performing model for Nepali retrieval, achieving *Recall@10 = 0.923* (tied with `BGE-M3`) and *MRR@10 = 0.817* in the Nepali legal document retrieval benchmark [[7]](#ref-7). 
- On the original MTEB benchmark, `jina-embeddings-v3` achieves *65.52* on English tasks and *64.44* on multilingual tasks, outperforming `multilingual-e5-large` across all multilingual tasks except reranking [[9]](#ref-9). 
- For Nepali specifically, the *Task-Targeted Embedding Distillation* paper [[10]](#ref-10) reports `jina-v5-text-small` scoring *99.5* on Nepali Classification within MMTEB. `jina-embeddings-v5-text-small` (677M parameters) achieves *67.0* on MMTEB, the highest score among sub-1B models [[10]](#ref-10).

| Model | Parameters | Max Tokens | Embedding Dimensions |
| --- | --- | --- | --- |
| `jinaai/jina-embeddings-v3` | 572M | 8192 | 1024 |
| `jinaai/jina-embeddings-v5-text-nano-retrieval` | 239M | 8192 | 768 |
| `jinaai/jina-embeddings-v5-text-small-retrieval` | 677M | 8192 | 1024 |

> [!NOTE]
> Jina v5 bare names are automatically resolved to the `-retrieval` variant.

**4. Qwen3 Embedding Series** [[6]](#ref-6)[[7]](#ref-7)[[10]](#ref-10)

- The flagship `Qwen3-Embedding-8B` model achieves *70.58* on MTEB Multilingual, surpassing Google's `GeminiEmbedding` and establishing a new state-of-the-art among open-source models [[6]](#ref-6). 
- In the *Task-Targeted Embedding Distillation* paper, `Qwen3-Embedding-4B` scores *97.3* on Nepali News Classification in MMTEB, the highest score of any model evaluated [[10]](#ref-10). 
- In the Nepali legal document retrieval benchmark [[7]](#ref-7), `Qwen3-Embedding-0.6B` achieved only *Recall@10 = 0.757, Precision@1 = 0.340, and MRR@10 = 0.492*, significantly trailing `BGE-M3`, `jina-v3`, and `multilingual-e5-large` (the larger 4B and 8B variants were not tested in that benchmark).

| Model | Parameters | Max Tokens | Embedding Dimensions |
| --- | --- | --- | --- |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | 32768 | 1024 |
| `Qwen/Qwen3-Embedding-4B` | 4B | 32768 | 2560 |
| `Qwen/Qwen3-Embedding-8B` | 8B | 32768 | 4096 |

**5. GTE-Multilingual** [[7]](#ref-7)

- The model supports 8,192 tokens and achieves state-of-the-art results among encoder-only models of similar size on multilingual retrieval tasks. 
- In the Nepali legal retrieval benchmark [[7]](#ref-7), GTE achieved *Recall@10 = 0.857, Precision@1 = 0.600, and MRR@10 = 0.694*, placing it third among multilingual models behind only `BGE-M3` and `jina-v3`.

| Model | Parameters | Max Tokens | Embedding Dimensions |
| --- | --- | --- | --- |
| `Alibaba-NLP/gte-multilingual-base` | 305M | 8192 | 768 |

---

### vi. Vector Store (`nepali_rag/vectorstore/`)

- **Database**: **Qdrant** (Qdrant Cloud).
- **Dense vector storage**: Each chunk is stored with a **dense vector** (Cosine vector corresponding to the output dimension of the embedding model).
- **Library**: `qdrant-client`.

> [!IMPORTANT]
> Current setup (except MCP) only supports *Qdrant Cloud*. For local/docker Qdrant setups use:

```bash
from nepali_rag import QdrantStore
from qdrant_client import QdrantClient

local_client = QdrantClient("CONFIGURE DOCKER PORT/LOCAL TYPE HERE") # for custom local client
qdrant_store = QdrantStore(vector_size=EMBEDDING_MODEL_DIMENSION,collection_name="SOME_COLLECTION_NAME", client=local_client)

```

---

### vii. Sparse BM25 Search

- **Library**: [`bm25s`](https://github.com/xhluca/bm25s).
- **Approach**: BM25 scoring is performed **client-side at query time** rather than storing pre-computed sparse vectors in Qdrant. All chunk texts are scrolled from Qdrant, cleaned (punctuation/newlines stripped, whitespace collapsed), tokenized with `token_pattern=r"\S+"` (no stemmer needed since Nepali is whitespace-delimited), and scored in-memory.
- **Why not Qdrant-native sparse vectors?** Standard sparse encoders (e.g., SPLADE) lack Devanagari vocabulary coverage, so client-side BM25 over raw text gives accurate keyword matching without needing a language-specific sparse encoder.

**English query handling:** `search_hybrid()` detects non-Nepali queries via `is_nepali_text()` (checks if >60% of characters are Devanagari). For English/non-Nepali queries, only dense (semantic) search runs, as this BM25 implementation doesnot support cross-lingual matching.

---

### viii. Retrieval (`nepali_rag/retrieval/`)

Two-stage retrieval pipeline:

1. **Stage 1 - Hybrid search**: Embeds the query, runs both dense and sparse(BM25) search against Qdrant, and fuses results using **weighted Reciprocal Rank Fusion (RRF)** (default: top 20 candidates).
2. **Stage 2 - Cross-encoder reranking**: Scores each `(query, passage)` pair with a cross-encoder and returns the top results (default: top 5).

- **Default Reranker model**: [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) - a multilingual cross-encoder reranker from the BGE family, providing strong relevance scoring across 100+ languages including Nepali.
- **Library**: **sentence-transformers** (`CrossEncoder`).

**Why this model:** The reranker is built on BGE-M3 [[2]](#ref-2)[[3]](#ref-3), which uses **XLM-RoBERTa** as its backbone, a model pretrained on CommonCrawl data covering 100+ languages including Nepali (Devanagari script). This gives it a strong foundation for Nepali text understanding out of the box. In the BGE-M3 paper [[2]](#ref-2), BGE-M3 achieved state-of-the-art results on multilingual (MIRACL) and cross-lingual (MKQA) benchmarks, outperforming prior models like mE5-large and E5-mistral-7b across 18 languages. The `bge-reranker-v2-m3` variant inherits this multilingual capability while being lightweight (0.6B params), fast to infer, and easy to deploy, the best performance-to-efficiency balance for multilingual reranking.

**Other strong reranker options:**

- **Qwen3-Reranker-0.6B** [[6]](#ref-6) - Matches BGE-M3's parameter count (0.6B) but scores slightly lower on the MTEB multilingual reranking subtask (61.41 vs 62.79). Choose this if you are already using the Qwen3 ecosystem and prefer model-family consistency across embedding and reranking.
- **bge-reranker-v2-gemma** [[5]](#ref-5) - A 3B LLM-based reranker built on Google's Gemma-2B. Delivers stronger ranking accuracy at the cost of higher memory and slower inference. Choose this when throughput is not the bottleneck and maximum ranking precision is the priority.

---

### ix. RAG Chain (`nepali_rag/rag/`)

Connects retrieval to generation with three operating modes:

| Mode                                        | Behavior                                                                                                                                                                |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Always-retrieve** (default)         | Every user message triggers the full retrieval pipeline. Retrieved context is injected into the prompt.                                                                 |
| **Tool-calling** (`use_tools=True`) | The LLM receives a `retrieve_documents` tool definition and autonomously decides whether to call it. Follow-ups, translations, and greetings skip retrieval entirely. |
| **Test mode** (`llm=None`)          | The full retrieval pipeline runs but returns the formatted prompt instead of an LLM-generated answer.                                                                   |

- **Framework**: **LangChain** (`langchain-core` messages: `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`).
- **Prompt design**: Bilingual system prompt instructs the LLM to prioritize retrieved context, answer in the user's language, and avoid fabrication when context is insufficient.
- **Conversation support**: Optional multi-turn history is prepended to the LLM messages for continuity but is not used for retrieval.

---

### x. Tool-Calling Mode

By default, the RAG chain operates in **tool-calling mode**, the LLM receives a `retrieve_documents` tool definition and autonomously decides whether retrieval is necessary for each message.

**How it works:**

1. The user sends a message (e.g., a question about Nepali law).
2. The LLM evaluates whether it needs to search documents to answer.
3. If yes → it calls `retrieve_documents`, the retrieval pipeline runs, and the LLM generates an answer grounded in the retrieved context.
4. If no → the LLM responds directly from conversation context (e.g., for greetings, follow-up clarifications, translations of a previous answer).

---

### xi. LLM Integration (`nepali_rag/llm/`)

A factory-based architecture (`get_llm()`) that returns a **LangChain `BaseChatModel`** for any supported provider. Provider-specific logic is isolated in individual modules:

| Provider                                           | LangChain Class                             | Default Model             |
| -------------------------------------------------- | ------------------------------------------- | ------------------------- |
| **Ollama**                                   | `ChatOllama`                              | `gemma4`                |
| **vLLM**                                     | `ChatOpenAI` (OpenAI-compatible endpoint) | `google/gemma-4-E4B-it` |
| **OpenAI**                                   | `ChatOpenAI`                              | `gpt-4o-mini`           |
| **OpenAI-compatible** (Groq, DeepSeek, etc.) | `ChatOpenAI` (custom `base_url`)        | *(varies by provider)*  |
| **Gemini**                                   | `ChatGoogleGenerativeAI`                  | `gemini-2.5-flash`      |
| **Hugging Face (HF)**                        | `ChatOpenAI` (OpenAI-compatible endpoint) | `Qwen/Qwen3.6-35B-A3B`  |
| **Anthropic**                                | `ChatAnthropic`                           | `claude-opus-4-6`       |

All providers expose the same interface - callers use `get_llm()` and never import provider modules directly.

---

### xii. Document Pipeline (`nepali_rag/pipeline/`)

The `DocumentPipeline` class orchestrates the full ingestion workflow in a single call:

```
load_document() → normalize_text() → chunk_pages() → embed_documents() → insert_chunks()
```

- Auto-generates a unique collection name (e.g., `doc_a1b2c3d4e5f6`) if none is provided.
- Supports batch ingestion of multiple files via `ingest_batch()`.
- Returns a summary dict with `collection_name`, `filename`, `pages_extracted`, `chunks_created`, and `chunks_indexed`.

---

### xiii. RAG Configuration (`nepali_rag/config/`)

All settings are managed through **pydantic-settings** (`BaseSettings`), loaded from environment variables or a `.env` file. This provides type validation, defaults, and a single source of truth for the entire system.

## 14. Evaluation Results

30 embedding + reranker combinations evaluated across 5 [DeepEval](https://docs.confident-ai.com/docs/metrics-introduction) RAG metrics. Sorted by **Retrieval Avg** (average of Contextual Recall, Precision, and Relevancy).

**Benchmark setup:**
- **Dataset**: 8 document samples, 10 QnA pairs each (English + Nepali)
- **RAG LLM**: `gemma4:31b` (Google Gemini API)
- **Judge LLM**: `Qwen3-235B-A22B-Instruct-2507`
- **Skipped**: `Qwen3-Embedding-4B` and `Qwen3-Embedding-8B` (hardware limitations)

| Embedding | Reranker | Ans. Relevancy | Faithfulness | Ctx. Recall | Ctx. Precision | Ctx. Relevancy | Retrieval Avg |
|---|---|---|---|---|---|---|---|
| **`jina-embeddings-v5-text-nano-retrieval`** | **`bge-reranker-v2-m3`** | **1.000** | 0.892 | **0.988** | 0.886 | 0.393 | **0.756** |
| `multilingual-e5-base` | `bge-reranker-v2-m3` | 0.985 | 0.908 | 0.975 | **0.929** | 0.355 | 0.753 |
| `multilingual-e5-large-instruct` | `bge-reranker-v2-m3` | 0.959 | 0.922 | 0.975 | 0.926 | 0.350 | 0.750 |
| `bge-m3` | `bge-reranker-v2-m3` | 0.962 | 0.942 | 0.975 | 0.928 | 0.328 | 0.744 |
| `multilingual-e5-small` | `bge-reranker-v2-m3` | 0.969 | 0.941 | 0.975 | 0.922 | 0.330 | 0.742 |
| `gte-multilingual-base` | `bge-reranker-v2-m3` | 0.977 | 0.971 | 0.975 | 0.931 | 0.318 | 0.741 |
| `jina-embeddings-v3` | `bge-reranker-v2-m3` | 0.973 | 0.933 | 0.975 | 0.919 | 0.329 | 0.741 |
| `multilingual-e5-large` | `bge-reranker-v2-m3` | 0.965 | 0.941 | 0.975 | 0.917 | 0.328 | 0.740 |
| `jina-embeddings-v5-text-small-retrieval` | `bge-reranker-v2-m3` | 0.954 | 0.919 | 0.906 | 0.816 | 0.433 | 0.718 |
| `multilingual-e5-large` | `Qwen3-Reranker-0.6B` | 0.973 | 0.947 | 0.950 | 0.820 | 0.335 | 0.702 |
| `multilingual-e5-small` | `Qwen3-Reranker-0.6B` | 0.984 | 0.929 | 0.963 | 0.824 | 0.316 | 0.701 |
| `multilingual-e5-base` | `Qwen3-Reranker-0.6B` | 0.974 | 0.947 | 0.956 | 0.823 | 0.312 | 0.697 |
| `gte-multilingual-base` | `Qwen3-Reranker-0.6B` | 0.970 | 0.940 | 0.963 | 0.819 | 0.308 | 0.697 |
| `multilingual-e5-large-instruct` | `Qwen3-Reranker-0.6B` | 0.989 | 0.959 | 0.963 | 0.805 | 0.321 | 0.696 |
| `qwen3-embedding-0.6b` | `bge-reranker-v2-m3` | 0.982 | 0.964 | 0.881 | 0.780 | 0.423 | 0.695 |
| `jina-embeddings-v5-text-nano-retrieval` | `Qwen3-Reranker-0.6B` | 0.973 | 0.919 | 0.898 | 0.790 | 0.385 | 0.691 |
| `jina-embeddings-v3` | `Qwen3-Reranker-0.6B` | 0.956 | 0.963 | 0.950 | 0.804 | 0.319 | 0.691 |
| `bge-m3` | `Qwen3-Reranker-0.6B` | 0.970 | 0.959 | 0.950 | 0.797 | 0.314 | 0.687 |
| `multilingual-e5-large` | `bge-reranker-v2-gemma` | 0.925 | 0.884 | 0.875 | 0.757 | 0.304 | 0.645 |
| `qwen3-embedding-0.6b` | `Qwen3-Reranker-0.6B` | 0.968 | 0.959 | 0.806 | 0.737 | 0.375 | 0.640 |
| `multilingual-e5-large-instruct` | `bge-reranker-v2-gemma` | 0.950 | 0.942 | 0.887 | 0.723 | 0.283 | 0.631 |
| `jina-embeddings-v5-text-small-retrieval` | `Qwen3-Reranker-0.6B` | 0.939 | 0.948 | 0.813 | 0.689 | 0.389 | 0.630 |
| `multilingual-e5-base` | `bge-reranker-v2-gemma` | 0.946 | 0.953 | 0.888 | 0.707 | 0.286 | 0.627 |
| `gte-multilingual-base` | `bge-reranker-v2-gemma` | 0.963 | 0.933 | 0.900 | 0.692 | 0.285 | 0.625 |
| `jina-embeddings-v3` | `bge-reranker-v2-gemma` | 0.923 | 0.896 | 0.850 | 0.732 | 0.269 | 0.617 |
| `multilingual-e5-small` | `bge-reranker-v2-gemma` | 0.943 | 0.932 | 0.862 | 0.700 | 0.273 | 0.612 |
| `jina-embeddings-v5-text-nano-retrieval` | `bge-reranker-v2-gemma` | 0.934 | 0.932 | 0.860 | 0.604 | 0.328 | 0.597 |
| `jina-embeddings-v5-text-small-retrieval` | `bge-reranker-v2-gemma` | 0.936 | **0.983** | 0.819 | 0.591 | 0.342 | 0.584 |
| `qwen3-embedding-0.6b` | `bge-reranker-v2-gemma` | 0.935 | 0.968 | 0.742 | 0.576 | 0.307 | 0.542 |
| `bge-m3` | `bge-reranker-v2-gemma` | 0.953 | 0.963 | 0.738 | 0.500 | 0.253 | 0.497 |

> [!TIP]
> **Best per metric:** Ans. Relevancy: `jina-v5-nano` + `bge-m3` (1.000) | Faithfulness: `jina-v5-small` + `bge-gemma` (0.983) | Ctx. Recall: `jina-v5-nano` + `bge-m3` (0.988) | Ctx. Precision: `gte-multilingual-base` + `bge-m3` (0.931) | Ctx. Relevancy: `jina-v5-small` + `bge-m3` (**0.433**)

For evaluation methodology and how to run your own benchmarks, see [Evaluation Documentation](https://github.com/Swastik-Aryal/Nepali-RAG/blob/main/nepali_rag/evals/README.md).

## 15. References

<a id="ref-1"></a> 1. Limbu Begha, F., Acharya, P., & Bal, B. K. (2026). *Nepali Passport Question Answering: A Low-Resource Dataset for Public Service Applications*. arXiv:2603.13320. [https://arxiv.org/html/2603.13320v1](https://arxiv.org/html/2603.13320v1)

<a id="ref-2"></a> 2. Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. arXiv:2402.03216. [https://arxiv.org/html/2402.03216v5](https://arxiv.org/html/2402.03216v5)

<a id="ref-3"></a> 3. BAAI. *BAAI/bge-m3*. Hugging Face. [https://huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)

<a id="ref-4"></a> 4. BAAI. *BAAI/bge-reranker-v2-m3*. Hugging Face. [https://huggingface.co/BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

<a id="ref-5"></a> 5. BAAI. *BAAI/bge-reranker-v2-gemma*. Hugging Face. [https://huggingface.co/BAAI/bge-reranker-v2-gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma)

<a id="ref-6"></a> 6. Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu, D., Lin, J., Huang, F., & Zhou, J. (2025). *Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models*. arXiv:2506.05176. [https://arxiv.org/pdf/2506.05176](https://arxiv.org/pdf/2506.05176)

<a id="ref-7"></a> 7. Aryal, S. (2025). *Multilingual Embedding Models for Nepali Legal Document Retrieval*. Preprints. [https://www.preprints.org/manuscript/202606.0033](https://www.preprints.org/manuscript/202606.0033)

<a id="ref-8"></a> 8. Enevoldsen, K. et al. (2025). *MMTEB: Massive Multilingual Text Embedding Benchmark*. arXiv:2502.13595. [https://arxiv.org/html/2502.13595v3](https://arxiv.org/html/2502.13595v3)

<a id="ref-9"></a> 9. Sturua, N. et al. (2024). *jina-embeddings-v3: Multilingual Embeddings With Task LoRA*. arXiv:2409.10173. [https://arxiv.org/abs/2409.10173](https://arxiv.org/abs/2409.10173)

<a id="ref-10"></a> 10. Mohr, I. et al. (2025). *Task-Targeted Embedding Distillation for Multi-Task Retrieval*. arXiv:2602.15547. [https://arxiv.org/html/2602.15547v1](https://arxiv.org/html/2602.15547v1)

## 16. License

This project is licensed under the MIT License. See [LICENSE](https://github.com/Swastik-Aryal/Nepali-RAG/blob/main/LICENSE) for details.
