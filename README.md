# Nepali RAG

Retrieval Augmented Generation framework purpose-built for the Nepali language documents, articles, books etc.

Plugable into any chatbot, search interface, or LLM wrapper to give your application grounded, citation-backed answers from Nepali documents.

**Why this exists:** Most RAG tooling assumes Latin-script languages. Nepali RAG handles the specific challenges of Devanagari text end-to-end.

## Architecture

```
Document Upload
      |
      v
+------------------+     +------------------+     +------------------------+
| Document         | --> | Text             | --> | PageStructure-Aware    |
| Ingestion        |     | Preprocessing    |     | Chunking               |
| (OCR + Parsers)  |     | (NFC + Cleanup)  |     | (256-token, Devanagari |
|                  |     |                  |     |  delimiter boundaries) |
+------------------+     +------------------+     +------------------------+
                                                            |
                                                            v
                                                  +------------------------+
                                                  | Embeddings             |
                                                  | (multilingual-e5-large)|
                                                  +------------------------+
                                                            |
                                                            v
                                                  +------------------------+
                                                  | Qdrant Vector Store    |
                                                  | (dense + sparse)       |
                                                  +------------------------+
                                                            |
      User Query                                            |
           |                                                |
           v                                                v
+------------------+     +------------------+     +------------------------+
| Query Embedding  | --> | Hybrid Search    | --> | Cross-Encoder          |
| (e5-large)       |     | (0.9 dense +     |     | Reranking              |
|                  |     |  0.1 sparse)     |     | (mmarco-mMiniLMv2)     |
+------------------+     +------------------+     +------------------------+
                                                            |
                                                            v
                                                  +------------------------+
                                                  | RAG Chain              |
                                                  | (LangChain + LLM)      |
                                                  | Always-retrieve or     |
                                                  | Tool-calling mode      |
                                                  +------------------------+
                                                            |
                                                            v
                                                       Answer + Sources
```

## Prerequisites

- **Python 3.10+**
- **Tesseract OCR** with Nepali language data (`nep` trained data)
  - Linux: `sudo apt install tesseract-ocr tesseract-ocr-nep`
  - macOS: `brew install tesseract` then download `nep.traineddata`
  - Windows: install from [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki) and add Nepali language data from [here](https://github.com/tesseract-ocr/tessdata/blob/main/nep.traineddata)
- **Qdrant** vector database instance (local or cloud)
  - Cloud: create a cluster at [cloud.qdrant.io](https://cloud.qdrant.io)
  - Docker: `docker run -p 6333:6333 qdrant/qdrant`
  - Local: If you wish, pass a custom `client` to `QdrantStore(client=...)` (not recommended for production)
- **LLM backend** (at least one):
  - Ollama (easiest): `ollama pull gemma3:4b` or any other model of choice (would recommend a model that supports tool-calling)
  - vLLM server for OpenAI-compatible API
  - OpenAI API key
  - Google Gemini API key
  - Hugging-face API key

## Installation

### From source (development)

```bash
git clone https://github.com/Swastik-Aryal/Nepali-RAG.git
cd Nepali-RAG
pip install -e ".[all]"
```

## Quick Start

### 1. Configure environment

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

At minimum, set your Qdrant connection and LLM provider:

```env
QDRANT_URL=http://localhost:6333
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma3:4b #or any other model of choice
```

### 2. Ingest a document

```bash
nepali-rag ingest path/to/document.pdf

# All available options:
nepali-rag ingest path/to/document.pdf \
  -c my_collection \          # target Qdrant collection name
  --chunk-size 256 \          # token chunk size override
  -v                          # enable debug logging
```

Or from Python:

```python
from nepali_rag import DocumentPipeline

pipeline = DocumentPipeline()
result = pipeline.ingest("path/to/document.pdf")
print(result)
# {'collection_name': 'doc_a1b2c3d4e5f6', 'filename': '...', 'pages_extracted': 12, 'chunks_created': 45, 'chunks_indexed': 45}


## OR FOR MORE CONTROL


from nepali_rag import DocumentPipeline, NepaliEmbeddingModel, QdrantStore
from qdrant_client import QdrantClient # Only for custom local clients

embedding_model = NepaliEmbeddingModel()
local_client = QdrantClient(":memory:") # for custom local client
qdrant_store = QdrantStore(collection_name="SOME_COLLECTION_NAME", client=local_client)
pipeline = DocumentPipeline(embedding_model=embedding_model, vector_store=qdrant_store)
result = pipeline.ingest("path/to/document.pdf")
print(result)
# {'collection_name': 'SOME_COLLECTION_NAME', 'filename': '...', 'pages_extracted': 12, 'chunks_created': 45, 'chunks_indexed': 45}
```

### 3. Query your documents

```bash
nepali-rag query "नेपालको संविधानमा के लेखिएको छ?"

# All available options:
nepali-rag query "नेपालको संविधानमा के लेखिएको छ?" \
  -c my_collection \          # Qdrant collection to search
  --provider ollama \         # LLM provider (vllm, openai, gemini, ollama)
  --model gemma3:4b \         # LLM model name override
  --top-k 20 \                # number of retrieval candidates
  --test \                    # run in test mode (no LLM, returns raw prompt)
  --disable-tools \           # disable tool-calling mode
  -v                          # enable debug logging
```

Or from Python:

```python
from nepali_rag import Retriever, QdrantStore, RAGChain, get_llm

store = QdrantStore(collection_name="EXISTING_COLLECTION_NAME")
retriever = Retriever(store=store)
llm = get_llm() # defaults to provider="ollama", model="gemma3:4b"
chain = RAGChain(retriever=retriever, llm=llm)

result = chain.run(question="नेपालको संविधानमा के लेखिएको छ?")
print(result["answer"])


## OR FOR MORE CONTROL


from nepali_rag import (
    Retriever, Reranker, QdrantStore,
    NepaliEmbeddingModel, RAGChain, get_llm,
)
from qdrant_client import QdrantClient # Only for custom local clients

local_client = QdrantClient(":memory:") # for custom local client
store = QdrantStore(collection_name="EXISTING_COLLECTION_NAME", client=local_client)

embedding_model = NepaliEmbeddingModel()
reranker = Reranker()

retriever = Retriever(embedding_model=embedding_model, reranker=reranker, store=store)
llm = get_llm(provider="hf", model="Qwen/Qwen3.6-35B-A3B") # api_key set in .env
chain = RAGChain(retriever=retriever, llm=llm)

result = chain.run(question="नेपालको संविधानमा के लेखिएको छ?")
print(result["answer"])
```

### 4. Start the API server

```bash
nepali-rag serve

# All available options:
nepali-rag serve \
  --host 0.0.0.0 \            # bind address (default: 0.0.0.0)
  --port 8080 \                # port (default: 8080)
  --reload \                   # enable auto-reload for development
  -v                           # enable debug logging
```

The API will be available at `http://localhost:8080`. Interactive docs at `http://localhost:8080/docs`.

## Configuration

All settings are managed through environment variables or a `.env` file. See [`.env.example`](.env.example) for the full list.

| Setting                     | Default                                        | Purpose                                                  |
| --------------------------- | ---------------------------------------------- | -------------------------------------------------------- |
| `QDRANT_URL`              | *(empty)*                                    | Vector database connection URL                           |
| `QDRANT_API_KEY`          | *(empty)*                                    | Qdrant Cloud authentication key                          |
| `COLLECTION_NAME`         | `nepali_docs`                                | Default Qdrant collection                                |
| `EMBEDDING_MODEL`         | `intfloat/multilingual-e5-large`             | Embedding model name                                     |
| `RERANKER_MODEL`          | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Reranker model name                                      |
| `LLM_PROVIDER`            | `ollama`                                     | LLM backend:`vllm`, `openai`, `gemini`, `ollama` |
| `VLLM_MODEL`              | `google/gemma-3-4b-it`                       | Model for vLLM provider                                  |
| `VLLM_BASE_URL`           | `http://localhost:8000/v1`                   | vLLM server endpoint                                     |
| `OPENAI_API_KEY`          | *(empty)*                                    | OpenAI API key                                           |
| `OPENAI_MODEL`            | `gpt-4o-mini`                                | Model for OpenAI provider                                |
| `GOOGLE_API_KEY`          | *(empty)*                                    | Google API key for Gemini                                |
| `GEMINI_MODEL`            | `gemini-2.5-flash`                           | Model for Gemini provider                                |
| `OLLAMA_BASE_URL`         | `http://localhost:11434`                     | Ollama server endpoint                                   |
| `OLLAMA_MODEL`            | `gemma3:4b`                                  | Model for Ollama provider                                |
| `HF_MODEL_NAME`           | `Qwen/Qwen3.6-35B-A3B`                       | Model for hugging-face-inference-api                     |
| `HF_BASE_URL`             | `https://router.huggingface.co/v1`           | hugging-face-inference endpoint                          |
| `HF_TOKEN`                | *(empty)*                                    | Your hugging-face api token                              |
| `TESSERACT_LANGUAGES`     | `nep`                                        | OCR language configuration                               |
| `CHUNK_SIZE`              | `256`                                        | Maximum chunk size in tokens                       |
| `CHUNK_OVERLAP_SENTENCES` | `2`                                          | Sentence overlap between chunks                          |
| `TOP_K_RETRIEVAL`         | `20`                                         | Hybrid search candidate count                            |
| `TOP_K_RERANK`            | `5`                                          | Final reranked result count                              |
| `DENSE_WEIGHT`            | `0.9`                                        | Dense score weight in hybrid search                      |
| `SPARSE_WEIGHT`           | `0.1`                                        | Sparse score weight in hybrid search                     |
| `LLM_TEMPERATURE`         | `0.3`                                        | LLM generation temperature                               |
| `USE_TOOLS`               | `true`                                       | Enable tool-calling mode                                 |

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

The system supports five LLM backends through a unified `get_llm()` factory. Switch providers by changing the `LLM_PROVIDER` environment variable.

| Provider         | Setup                                                   | Notes                                                                          |
| ---------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Ollama** | `ollama pull gemma3:4b`                               | Easiest local option, no API key needed                                        |
| **vLLM**   | Start vLLM server with `--model google/gemma-3-4b-it` | OpenAI-compatible API, no API key needed                                       |
| **OpenAI** | Set `OPENAI_API_KEY`                                  | Direct OpenAI API (GPT-4o, etc.)                                               |
| **Gemini** | Set `GOOGLE_API_KEY`                                  | Google Gemini (gemini-2.5-flash, etc.)                                         |
| **HF**     | Set `HF_TOKEN`                                        | Open source models from Hugging Face (choose any repo from the HF models page) |

When no LLM is available, the system operates in **test mode**: the full retrieval pipeline runs but returns the formatted prompt instead of a generated answer.

## API Endpoints

Start the server with `nepali-rag serve` (requires `pip install nepali-rag[api]`).

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
nepali-rag ingest <file> [-c COLLECTION] [--chunk-size N]
nepali-rag query "<question>" [-c COLLECTION] [--provider PROVIDER] [--model MODEL] [--top-k N] [--test] [--disable-tools]
nepali-rag serve [--host HOST] [--port PORT] [--reload]
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

- **Database**: **Qdrant** (local Docker or Qdrant Cloud).
- **Dual vector storage**: Each chunk is stored with both a **dense vector** (1024-dim cosine, from the embedding model) and a **sparse vector** (term-frequency based, computed via MD5 hashing of whitespace-split tokens into a 2²⁰-dimensional space).
- **Hybrid search**: Retrieval combines dense and sparse results using **Reciprocal Rank Fusion** (RRF, k=60) with configurable weights (default: 0.9 dense, 0.1 sparse).
- **Library**: `qdrant-client`.

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

### LLM Integration (`nepali_rag/llm/`)

A factory-based architecture (`get_llm()`) that returns a **LangChain `BaseChatModel`** for any supported provider. Provider-specific logic is isolated in individual modules:

| Provider               | LangChain Class                             | Default Model            |
| ---------------------- | ------------------------------------------- | ------------------------ |
| **Ollama**       | `ChatOllama`                              | `gemma3:4b`            |
| **vLLM**         | `ChatOpenAI` (OpenAI-compatible endpoint) | `google/gemma-3-4b-it` |
| **OpenAI**       | `ChatOpenAI`                              | `gpt-4o-mini`          |
| **Gemini**       | `ChatGoogleGenerativeAI`                  | `gemini-2.5-flash`     |
| **Hugging Face** | `ChatOpenAI` (OpenAI-compatible endpoint) | `Qwen/Qwen3.6-35B-A3B` |

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
