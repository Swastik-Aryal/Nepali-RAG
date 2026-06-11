# Nepali-RAG

Retrieval Augmented Generation framework purpose-built for the Nepali language documents, articles, books etc.

Plugable into any chatbot, search interface, or LLM wrapper to give your application grounded, citation-backed answers from Nepali documents. Supports **LLM tool-calling**, the model autonomously decides when document retrieval is needed..

**Why this exists:** Most RAG tooling assumes Latin-script languages. Nepali RAG handles the specific challenges of Devanagari text end-to-end.

## Prerequisites

- **Python 3.10+**
- **Tesseract OCR** with Nepali language data (`nep` trained data)
  - Linux: `sudo apt install tesseract-ocr tesseract-ocr-nep`
  - macOS: `brew install tesseract` then download `nep.traineddata`
  - Windows: install from [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki) and add Nepali language data from [here](https://github.com/tesseract-ocr/tessdata/blob/main/nep.traineddata)
- **Qdrant** vector database instance (cloud)
  - Create a cluster at [cloud.qdrant.io](https://cloud.qdrant.io) — you'll need the URL and API key
- **LLM backend** (at least one):
  - Ollama (easiest): `ollama pull gemma3:4b` or any other model of choice (would recommend a model that supports tool-calling)
  - vLLM server for OpenAI-compatible API
  - OpenAI API key
  - Google Gemini API key
  - Hugging-face API key

## Installation

```bash
git clone https://github.com/Swastik-Aryal/Nepali-RAG.git
cd Nepali-RAG
pip install -e .
```

This installs all dependencies (document parsers, embeddings, vector store, LLM providers, API server).

**GPU acceleration (recommended):** The default install pulls the CPU-only PyTorch build from PyPI. If you have an NVIDIA GPU, install the CUDA version of PyTorch **before or after** running `pip install -e .`:

```bash
# Example for CUDA 12.1 (check your version with: nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Environment File (Optional but Recommended)

For the Python usage (less control snippets), the library reads settings from a `.env` file. Copy the template:

```bash
cp .env.example .env
```

> **Important:** For Python scripts, the `.env` file is loaded from your **current working directory** (CWD). For the API server, you can point to any `.env` file with `nepali-rag serve --env-file /path/to/.env`.

At minimum, set your Qdrant connection:

```env
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```
> **Note:** If `.env` file is absent, the library will use the default values set in the [config/settings.py](./nepali-rag/config/settings.py)(can also be viewed in `env.example`) file.

> **Note:** The `.env` file is **not required** for terminal (CLI) usage. All parameters are passed directly as command-line arguments.

## Quick Start

### 1. Ingest a document

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

embedding_model = NepaliEmbeddingModel()
store = QdrantStore(
  collection_name="my_collection",
  url="https://your-cluster.cloud.qdrant.io",
  api_key="your-qdrant-api-key",
)

pipeline = DocumentPipeline(
    embedding_model=embedding_model,
    vector_store=store,
    chunk_size=256,
    chunk_overlap_sentences=2,
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
  --chunk-size 256 \
  --chunk-overlap 2 \
  -v
```

### 2. Query your documents

**Python (less control, requires `.env` file):**

```python
from nepali_rag import Retriever, QdrantStore, RAGChain, get_llm

store = QdrantStore(collection_name="my_collection")
retriever = Retriever(store=store)
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

store = QdrantStore(
  collection_name="my_collection", 
  url="https://your-cluster.cloud.qdrant.io",
  api_key="your-qdrant-api-key"
)

embedding_model = NepaliEmbeddingModel()
reranker = Reranker()

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
  --model gemma3:4b \
  --top-k 20 \
  --top-r 5 \
  --temperature 0.3 \
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
  --model gemma3:4b \
  --history history.json \
  -v

# Using inline JSON
nepali-rag query "Can you translate that to English?" \
  --qdrant-url https://your-cluster.cloud.qdrant.io \
  --qdrant-api-key your-qdrant-api-key \
  -c my_collection \
  --provider ollama \
  --model gemma3:4b \
  --history '[{"role":"user","content":"नेपालको राजधानी कहाँ हो?"},{"role":"assistant","content":"नेपालको राजधानी काठमाडौं हो।"}]' \
  -v
```

> **Tip:** By default, tool-calling mode is enabled, the LLM decides when retrieval is needed (recommended). Use `--disable-tools` to force retrieval on every query (always-retrieve mode).

For providers that need an API key (openai, gemini, hf):

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

### 3. Start the API server

**Terminal:**

```bash
# Uses .env from CWD by default
nepali-rag serve --host 0.0.0.0 --port 8080 --reload -v

# Point to a .env file anywhere on disk
nepali-rag serve --env-file /path/to/.env --port 8080 -v
```

The API will be available at `http://localhost:8080`. Interactive docs at `http://localhost:8080/docs`.

> **Note:** The API server reads configuration from a `.env` file (Qdrant credentials, LLM provider, etc.). Use `--env-file` to specify the path if it's not in your current directory.

## Configuration

All settings are managed through environment variables or a `.env` file. See [`.env.example`](.env.example) for the full list.

| Setting                     | Default                                        | Purpose                                                          |
| --------------------------- | ---------------------------------------------- | ---------------------------------------------------------------- |
| `QDRANT_URL`              | *(empty)*                                    | Vector database connection URL                                   |
| `QDRANT_API_KEY`          | *(empty)*                                    | Qdrant Cloud authentication key                                  |
| `COLLECTION_NAME`         | `nepali_docs`                                | Default Qdrant collection                                        |
| `EMBEDDING_MODEL`         | `intfloat/multilingual-e5-large`             | Embedding model name                                             |
| `RERANKER_MODEL`          | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Reranker model name                                              |
| `LLM_PROVIDER`            | `ollama`                                     | LLM backend:`vllm`, `openai`, `gemini`, `ollama`, `hf` |
| `VLLM_MODEL`              | `google/gemma-3-4b-it`                       | Model for vLLM provider                                          |
| `VLLM_BASE_URL`           | `http://localhost:8000/v1`                   | vLLM server endpoint                                             |
| `OPENAI_API_KEY`          | *(empty)*                                    | OpenAI API key                                                   |
| `OPENAI_MODEL`            | `gpt-4o-mini`                                | Model for OpenAI provider                                        |
| `GOOGLE_API_KEY`          | *(empty)*                                    | Google API key for Gemini                                        |
| `GEMINI_MODEL`            | `gemini-2.5-flash`                           | Model for Gemini provider                                        |
| `OLLAMA_BASE_URL`         | `http://localhost:11434`                     | Ollama server endpoint                                           |
| `OLLAMA_MODEL`            | `gemma3:4b`                                  | Model for Ollama provider                                        |
| `HF_MODEL_NAME`           | `Qwen/Qwen3.6-35B-A3B`                       | Model for hugging-face-inference-api                             |
| `HF_BASE_URL`             | `https://router.huggingface.co/v1`           | hugging-face-inference endpoint                                  |
| `HF_TOKEN`                | *(empty)*                                    | Your hugging-face api token                                      |
| `TESSERACT_LANGUAGES`     | `nep`                                        | OCR language configuration                                       |
| `CHUNK_SIZE`              | `256`                                        | Maximum chunk size in tokens                                     |
| `CHUNK_OVERLAP_SENTENCES` | `2`                                          | Sentence overlap between chunks                                  |
| `TOP_K_RETRIEVAL`         | `20`                                         | Hybrid search candidate count                                    |
| `TOP_K_RERANK`            | `5`                                          | Final reranked result count                                      |
| `DENSE_WEIGHT`            | `0.9`                                        | Dense score weight in hybrid search                              |
| `SPARSE_WEIGHT`           | `0.1`                                        | Sparse score weight in hybrid search                             |
| `LLM_TEMPERATURE`         | `0.3`                                        | LLM generation temperature                                       |
| `USE_TOOLS`               | `true`                                       | Enable tool-calling mode                                         |

## Supported Document Formats

| Format                    | Extensions          | Method                     |
| ------------------------- | ------------------- | -------------------------- |
| PDF (digital and scanned) | `.pdf`            | Tesseract OCR on all pages |
| Plain Text                | `.txt`            | Direct read                |
| Markdown                  | `.md`             | Direct read                |
| Word                      | `.docx`           | python-docx                |
| PowerPoint                | `.pptx`           | python-pptx                |
| Excel                     | `.xlsx`           | openpyxl                   |
| CSV                       | `.csv`            | Built-in csv               |
| RTF                       | `.rtf`            | striprtf                   |
| HTML                      | `.html`, `.htm` | BeautifulSoup              |
| EPUB                      | `.epub`           | EbookLib                   |
| OpenDocument              | `.odt`            | odfpy                      |

## LLM Providers

The system supports five LLM backends through a unified `get_llm()` factory. Switch providers by changing the `LLM_PROVIDER` environment variable or passing `--provider` to the CLI.

| Provider             | Setup                                                   | Model                                                                                  |
| -------------------- | ------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **`ollama`** | `ollama pull gemma3:4b`                               | Set `model = "gemma3:4b"` OR any other model but in the ollama format.               |
| **`vllm`**   | Start vLLM server with `--model google/gemma-3-4b-it` | Set `model = "google/gemma-3-4b-it"` OR any other model but in the vllm format.    |
| **`openai`** | Set `OPENAI_API_KEY` or pass `--api-key`            | Set `model = "gpt-4o-mini"` OR any other model based on the OpenAI API docs.        |
| **`gemini`** | Set `GOOGLE_API_KEY` or pass `--api-key`            | Set `model = "gemini-2.5-flash"` OR any other model based on the Google API docs. |
| **`hf`**     | Set `HF_TOKEN` or pass `--api-key`                  | Set `model = "Qwen/Qwen3.6-35B-A3B"` OR any other model repository.                  |

When no LLM is available, the system operates in **test mode**: the full retrieval pipeline runs but returns the formatted prompt instead of a generated answer.

## API Endpoints

Start the server with `nepali-rag serve`.

| Method   | Path                           | Purpose                                      |
| -------- | ------------------------------ | -------------------------------------------- |
| `POST` | `/api/v1/upload`             | Upload documents for ingestion               |
| `POST` | `/api/v1/query`              | Query with question and conversation history |
| `GET`  | `/api/v1/health`             | System health and LLM availability           |
| `GET`  | `/api/v1/collections`        | List all Qdrant collections                  |
| `GET`  | `/api/v1/collections/{name}` | Collection details                           |

### Upload example

```bash
curl -X POST http://localhost:8080/api/v1/upload \
  -F "files=@document.pdf" \
  -F "collection_name=my_docs"
```

### Query example

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "नेपालको राजधानी कहाँ हो?",
    "collection_name": "my_docs"
  }'
```

## CLI Reference

```
nepali-rag ingest <file> [<file> ...]
    --qdrant-url URL          (required) Qdrant instance URL
    --qdrant-api-key KEY      (required) Qdrant API key
    -c, --collection NAME     collection name (default: nepali_docs)
    --chunk-size N            token chunk size (default: 256)
    --chunk-overlap N         sentence overlap (default: 2)
    -v                        enable debug logging

nepali-rag query "<question>"
    --qdrant-url URL          (required) Qdrant instance URL
    --qdrant-api-key KEY      (required) Qdrant API key
    -c, --collection NAME     collection name (default: nepali_docs)
    --provider PROVIDER       LLM provider (default: ollama)
    --model MODEL             LLM model name (default: gemma3:4b)
    --api-key KEY             API key for LLM provider (required for openai, gemini, hf)
    --llm-base-url URL        base URL for LLM server (for vllm, ollama, hf)
    --temperature FLOAT       LLM temperature (default: 0.3)
    --top-k N                 retrieval candidates (default: 20)
    --top-r N                 rerank candidates (default: 5)
    --history FILE_OR_JSON    conversation history (JSON file path or inline JSON string)
    --test                    test mode (no LLM, returns raw prompt)
    --disable-tools           disable tool-calling mode (forces retrieval on every query)
    -v                        enable debug logging

nepali-rag serve
    --env-file PATH           path to .env file (default: .env in CWD)
    --host HOST               bind address (default: 0.0.0.0)
    --port PORT               port (default: 8080)
    --reload                  auto-reload for development
    -v                        enable debug logging
```

## Project Structure

```
nepali-rag/
├── nepali_rag/                  # Core package
│   ├── __init__.py              # Public API exports and __version__
│   ├── config/settings.py       # Centralized settings (pydantic-settings)
│   ├── ingestion/               # Document parsers (OCR, docx, pptx, etc.)
│   ├── preprocessing/           # Unicode NFC normalization, Nepali detection
│   ├── chunking/                # Two-level sentence-aware chunker
│   ├── embeddings/              # multilingual-e5-large with prefix handling
│   ├── vectorstore/             # Qdrant dense + sparse hybrid store
│   ├── retrieval/               # Hybrid retrieval + cross-encoder reranking
│   ├── rag/                     # RAG chain (always-retrieve + tool-calling)
│   ├── llm/                     # Multi-provider LLM factory
│   ├── pipeline/                # Full document ingestion orchestrator
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

## Component Details

### Architecture

![architecture](./nepali_rag/data/architecture/architecture.svg)

### Document Ingestion (`nepali_rag/ingestion/`)

Handles loading and text extraction from 11+ document formats. PDFs are rendered page-by-page to 300 DPI images using **PyMuPDF** (`fitz`) and processed through **Tesseract OCR** with Nepali (`nep`) + English (`eng`) language data. Other formats use dedicated parsers:

| Library                                   | Formats                      |
| ----------------------------------------- | ---------------------------- |
| **pytesseract** + **PyMuPDF** | `.pdf` (all pages via OCR) |
| **python-docx**                     | `.docx`                    |
| **python-pptx**                     | `.pptx`                    |
| **openpyxl**                        | `.xlsx`                    |
| **striprtf**                        | `.rtf`                     |
| **BeautifulSoup**                   | `.html`, `.htm`          |
| **EbookLib**                        | `.epub`                    |
| **odfpy**                           | `.odt`                     |
| Built-in                                  | `.txt`, `.md`, `.csv`  |

Every page is returned as a `PageResult` dataclass carrying the extracted text, source path, page number, and extraction method.

---

### Text Preprocessing (`nepali_rag/preprocessing/`)

Normalizes raw extracted text for consistent downstream processing:

1. **Unicode NFC normalization** — ensures composed Devanagari forms (e.g., combining sequences are collapsed into single codepoints).
2. **Zero-width character removal** — strips `U+200B` (zero-width space), `U+200C`/`U+200D` (joiners), and `U+FEFF` (BOM).
3. **Whitespace normalization** — collapses runs of spaces/tabs to a single space, and 3+ newlines to a double newline.
4. **Nepali detection** — `is_nepali_text()` classifies text as primarily Devanagari if >30% of non-whitespace characters fall in the `U+0900`–`U+097F` range.

---

### Chunking (`nepali_rag/chunking/`)

A two-level, structure-aware chunker that respects both paragraph boundaries and Nepali sentence delimiters (purna viram `।`, double danda `॥`, and standard `.`, `!`, `?`).

- **Level 1**: **LangChain's `RecursiveCharacterTextSplitter`** splits on structural separators (`\n\n`, `\n`, ) to keep paragraphs intact. A post-processing pass (`_ensure_sentence_boundaries`) trims every block to end on a sentence delimiter, carrying incomplete fragments forward.
- **Level 2**: Oversized blocks are split at sentence boundaries with configurable overlap (default: 2 sentences).
- **Tokenizer**: Uses the **HuggingFace `AutoTokenizer`** for the configured embedding model (`intfloat/multilingual-e5-large`) to count tokens accurately. Default chunk size is **256 tokens** (half the model's 512-token context limit).
- Small blocks below half the chunk size are merged with neighbors to avoid tiny fragments.

---

### Embeddings (`nepali_rag/embeddings/`)

- **Model**: [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) — a 1024-dimension multilingual embedding model with strong Nepali support.
- **Library**: **sentence-transformers** (`SentenceTransformer`).
- **Prefix handling**: Following the e5 specification, documents are prefixed with `"passage: "` and queries with `"query: "` before encoding.
- **Normalization**: All embeddings are L2-normalized at encode time (`normalize_embeddings=True`).

---

### Vector Store (`nepali_rag/vectorstore/`)

- **Database**: **Qdrant** (Qdrant Cloud).
- **Dual vector storage**: Each chunk is stored with both a **dense vector** (1024-dim cosine, from the embedding model) and a **sparse vector** (term-frequency based, computed via MD5 hashing of whitespace-split tokens into a 2²⁰-dimensional space).
- **Hybrid search**: Retrieval combines dense and sparse results using **Reciprocal Rank Fusion** (RRF, k=60) with configurable weights (default: 0.9 dense, 0.1 sparse).
- **Library**: `qdrant-client`.

> **P.S.** Current setup only supports *Qdrant Cloud*. For local/docker Qdrant setups use

```bash
from nepali_rag import QdrantStore
from qdrant_client import QdrantClient

local_client = QdrantClient("CONFIGURE DOCKER PORT/LOCAL TYPE HERE") # for custom local client
qdrant_store = QdrantStore(collection_name="SOME_COLLECTION_NAME", client=local_client)

```

---

### Retrieval (`nepali_rag/retrieval/`)

Two-stage retrieval pipeline:

1. **Stage 1 — Hybrid search**: Embeds the query, runs both dense and sparse search against Qdrant, and fuses results via RRF (default: top 20 candidates).
2. **Stage 2 — Cross-encoder reranking**: Scores each `(query, passage)` pair with a cross-encoder and returns the top results (default: top 5).

- **Reranker model**: [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) — a multilingual cross-encoder trained on MS MARCO, providing strong relevance scoring for Nepali text.
- **Library**: **sentence-transformers** (`CrossEncoder`).

---

### RAG Chain (`nepali_rag/rag/`)

Connects retrieval to generation with three operating modes:

| Mode                                        | Behavior                                                                                                                                                                |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Always-retrieve** (default)         | Every user message triggers the full retrieval pipeline. Retrieved context is injected into the prompt.                                                                 |
| **Tool-calling** (`use_tools=True`) | The LLM receives a `retrieve_documents` tool definition and autonomously decides whether to call it. Follow-ups, translations, and greetings skip retrieval entirely. |
| **Test mode** (`llm=None`)          | The full retrieval pipeline runs but returns the formatted prompt instead of an LLM-generated answer.                                                                   |

- **Framework**: **LangChain** (`langchain-core` messages: `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`).
- **Prompt design**: Bilingual system prompt instructs the LLM to cite retrieved chunks by number (e.g., `[1]`, `[2]`), answer in the user's language, and avoid fabrication when context is insufficient.
- **Conversation support**: Optional multi-turn history is prepended to the LLM messages for continuity but is not used for retrieval.

---

### Tool-Calling Mode

By default, the RAG chain operates in **tool-calling mode**, the LLM receives a `retrieve_documents` tool definition and autonomously decides whether retrieval is necessary for each message.

**How it works:**

1. The user sends a message (e.g., a question about Nepali law).
2. The LLM evaluates whether it needs to search documents to answer.
3. If yes → it calls `retrieve_documents`, the retrieval pipeline runs, and the LLM generates an answer grounded in the retrieved context.
4. If no → the LLM responds directly from conversation context (e.g., for greetings, follow-up clarifications, translations of a previous answer).

---

### LLM Integration (`nepali_rag/llm/`)

A factory-based architecture (`get_llm()`) that returns a **LangChain `BaseChatModel`** for any supported provider. Provider-specific logic is isolated in individual modules:

| Provider                    | LangChain Class                             | Default Model            |
| --------------------------- | ------------------------------------------- | ------------------------ |
| **Ollama**            | `ChatOllama`                              | `gemma3:4b`            |
| **vLLM**              | `ChatOpenAI` (OpenAI-compatible endpoint) | `google/gemma-3-4b-it` |
| **OpenAI**            | `ChatOpenAI`                              | `gpt-4o-mini`          |
| **Gemini**            | `ChatGoogleGenerativeAI`                  | `gemini-2.5-flash`     |
| **Hugging Face (HF)** | `ChatOpenAI` (OpenAI-compatible endpoint) | `Qwen/Qwen3.6-35B-A3B` |

All providers expose the same interface — callers use `get_llm()` and never import provider modules directly.

---

### Document Pipeline (`nepali_rag/pipeline/`)

The `DocumentPipeline` class orchestrates the full ingestion workflow in a single call:

```
load_document() → normalize_text() → chunk_pages() → embed_documents() → insert_chunks()
```

- Auto-generates a unique collection name (e.g., `doc_a1b2c3d4e5f6`) if none is provided.
- Supports batch ingestion of multiple files via `ingest_batch()`.
- Returns a summary dict with `collection_name`, `filename`, `pages_extracted`, `chunks_created`, and `chunks_indexed`.

---

### FastAPI Server (`nepali_rag/api/`)

A REST API layer built with **FastAPI** and **Uvicorn**:

- **Lifespan initialization**: The embedding model and reranker are loaded once at server startup and shared across all requests via `app.state`, avoiding per-request loading overhead.
- **Conversation persistence**: Each query session is tracked by a `conversation_id`. Conversation history is stored as JSON files in `nepali_rag/api/conversations/` and automatically loaded/appended on subsequent queries.
- **Graceful fallback**: If the LLM fails at connection or invocation time, the API falls back to test mode rather than returning a 500 error.
- **CORS**: Enabled for all origins to support frontend integrations.

---

### Configuration (`nepali_rag/config/`)

All settings are managed through **pydantic-settings** (`BaseSettings`), loaded from environment variables or a `.env` file. This provides type validation, defaults, and a single source of truth for the entire system.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
