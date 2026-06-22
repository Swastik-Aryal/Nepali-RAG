# FastAPI Server

Full documentation for the Nepali-RAG HTTP API (FastAPI) layer.

## Starting the Server

```bash
# Uses .env from CWD by default
nepali-rag serve --host 0.0.0.0 --port 8080 --reload -v

# Point to a .env file anywhere on disk
nepali-rag serve --env-file /path/to/.env --port 8080 -v
```

The API will be available at `http://localhost:8080`. Interactive docs at `http://localhost:8080/docs`.

> **Note:** The API server reads configuration from a `.env` file (Qdrant credentials, LLM provider, etc.). Use `--env-file` to specify the path if it's not in your current directory.

## Endpoints

Start the server with `nepali-rag serve`. Interactive docs available at `http://localhost:8080/docs`.

| Method   | Path                                        | Purpose                                      |
| -------- | ------------------------------------------- | -------------------------------------------- |
| `POST` | `/api/v1/upload`                          | Upload documents for ingestion               |
| `POST` | `/api/v1/query`                           | Query with question and conversation history |
| `GET`  | `/api/v1/health`                          | System health and LLM availability           |
| `GET`  | `/api/v1/collections`                     | List all Qdrant collections                  |
| `GET`  | `/api/v1/collections/{name}`              | Collection details                           |
| `GET`  | `/api/v1/conversations`                   | List all persisted conversation IDs          |
| `GET`  | `/api/v1/conversations/{conversation_id}` | Full turn history for a conversation         |

---

### `POST /api/v1/upload`

Upload and index one or more documents via `DocumentPipeline`. Each file is validated against supported formats, saved to a temporary directory, passed through the full ingest pipeline (load -> normalize -> chunk -> embed -> Qdrant), and then cleaned up.

**Form parameters (multipart/form-data):**

- **`files`** (required) - One or more files to upload. Must be a supported format (`.pdf`, `.docx`, `.txt`, `.md`, `.html`, etc.).
- **`collection_name`** (optional) - Qdrant collection to index into. If omitted, `DocumentPipeline` auto-generates a `doc_<hex>` name.
- **`chunk_size`** (optional) - Maximum number of tokens per chunk. Falls back to `settings.chunk_size` (256), overridden by `CHUNK_SIZE` in ENV.

**Response fields:**

- **`results`** - List of per-file dicts, each with `collection_name`, `filename`, `pages_extracted`, `chunks_created`, and `chunks_indexed`.
- **`total_files`** - Number of files processed.
- **`total_chunks_indexed`** - Sum of all chunks stored in Qdrant.

**Errors:** `400` unsupported file format · `500` ingestion/indexing failure.

**Example:**

```bash
curl -X POST http://localhost:8080/api/v1/upload \
  -F "files=@document.pdf" \
  -F "collection_name=my_docs" \
  -F "chunk_size=512"
```

```json
{
    "results": [
        {
            "collection_name": "my_docs",
            "filename": "document.pdf",
            "pages_extracted": 4,
            "chunks_created": 12,
            "chunks_indexed": 12
        }
    ],
    "total_files": 1,
    "total_chunks_indexed": 12
}
```

---

### `POST /api/v1/query`

Run a RAG query: retrieve relevant chunks, generate an answer, and persist the conversation. The pipeline retrieves candidate passages from Qdrant, optionally reranks them, and feeds the context to the configured LLM for answer generation. Conversation turns are persisted to `api/conversations/<conversation_id>.json` so subsequent queries can leverage multi-turn context.

**JSON body parameters:**

- **`question`** (str, required) - The user's query. Must not be empty.
- **`conversation_id`** (str, optional) - Unique conversation identifier. If omitted, a random 16-character hex ID is generated.
- **`conversation_history`** (list, optional) - Explicit turn history (list of `{"role": "...", "content": "..."}` dicts) to replace any file-persisted history.
- **`collection_name`** (str, optional) - Qdrant collection to search. Falls back to `settings.collection_name` (`nepali_docs`), overridden by `COLLECTION_NAME` in ENV.
- **`top_k_retrieval`** (int, optional) - Hybrid-search candidate count. Falls back to `settings.top_k_retrieval` (20), overridden by `TOP_K_RETRIEVAL` in ENV.
- **`top_k_rerank`** (int, optional) - Final result count after reranking. Falls back to `settings.top_k_rerank` (5), overridden by `TOP_K_RERANK` in ENV.
- **`llm_provider`** (str, optional) - LLM provider override (e.g. `"ollama"`, `"openai"`, `"gemini"`, `"huggingface"`, `"vllm"`). Falls back to `settings.llm_provider`, overridden by `LLM_PROVIDER` in ENV.
- **`llm_model`** (str, optional) - Model name override. Falls back to the provider-specific model in settings (e.g. `OLLAMA_MODEL`, `OPENAI_MODEL`, `GEMINI_MODEL`, etc. in ENV).
- **`temperature`** (float, optional) - Sampling temperature. Falls back to `settings.llm_temperature` (0.3), overridden by `LLM_TEMPERATURE` in ENV.
- **`use_tools`** (bool, optional) - Whether the LLM should decide if retrieval is needed. Falls back to `settings.use_tools` (`True`), overridden by `USE_TOOLS` in ENV.

> **Note:** LLM credentials and connection details (API keys, base URLs) are not passed per-request - they are configured via environment variables / `settings.py`. See `.env.example` for the full list.

**Response fields:**

- **`answer`** - The generated response from the LLM (or formatted prompt in test mode).
- **`sources`** - Retrieved chunks, each with `text`, `metadata`, `score`, and `rerank_score`.
- **`context`** - The formatted context string fed to the LLM.
- **`collection_name`** - The Qdrant collection that was queried.
- **`conversation_id`** - The conversation ID (generated or provided).
- **`test_mode`** - `true` if the LLM was unavailable and the response is a raw prompt.
- **`used_tool_calling`** - `true` if the LLM invoked the `retrieve_documents` tool.

**Errors:** `400` empty question · `503` LLM initialization failure · `500` query processing failure.

**Example:**

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "त्यहाँको जनसंख्या कति हो?",
    "collection_name": "nepali_docs",
    "conversation_history": [
        {"role": "user", "content": "नेपालको राजधानी के हो?"},
        {"role": "assistant", "content": "नेपालको राजधानी काठमाडौं हो।"}
    ],
    "top_k_retrieval": 10,
    "top_k_rerank": 3
  }'
```

```json
{
    "answer": "नेपालको राजधानी काठमाडौं हो।",
    "sources": [
        {
            "text": "काठमाडौं नेपालको राजधानी हो...",
            "metadata": {"source": "nepal_info.pdf", "page_number": 1, "chunk_index": 0},
            "score": 0.87,
            "rerank_score": 0.95
        }
    ],
    "context": "[1](Source: nepal_info.pdf, Page: 1) काठमाडौं नेपालको राजधानी हो...",
    "collection_name": "nepali_docs",
    "conversation_id": "a1b2c3d4e5f67890",
    "test_mode": false,
    "used_tool_calling": true
}
```

---

### `GET /api/v1/health`

Return API health status, configured providers, and LLM reachability. Attempts to initialize the default LLM to determine reachability - no inference call is made, only the connection handshake is tested.

**Response fields:**

- **`status`** - Always `"ok"` if the API itself is running.
- **`llm_provider`** - The configured LLM provider name (e.g. `"ollama"`, `"openai"`).
- **`llm_available`** - `true` if LLM initialization succeeds without error.
- **`embedding_model`** - The configured embedding model identifier.
- **`default_collection`** - The default Qdrant collection name.

**Example:**

```bash
curl http://localhost:8080/api/v1/health
```

```json
{
    "status": "ok",
    "llm_provider": "ollama",
    "llm_available": true,
    "embedding_model": "intfloat/multilingual-e5-large",
    "default_collection": "nepali_docs"
}
```

---

### `GET /api/v1/collections`

List all Qdrant collections with their point counts and statuses.

**Response fields (list):** Each item contains:

- **`name`** - The collection name.
- **`points_count`** - Number of indexed points (chunks).
- **`status`** - Collection status (e.g. `"green"`).

**Errors:** `500` Qdrant connection or metadata fetch failure.

**Example:**

```bash
curl http://localhost:8080/api/v1/collections
```

```json
[
    {
        "name": "nepali_docs",
        "points_count": 142,
        "status": "green"
    },
    {
        "name": "doc_a1b2c3d4e5f6",
        "points_count": 38,
        "status": "green"
    }
]
```

---

### `GET /api/v1/collections/{name}`

Return point count and status for a single Qdrant collection.

**Path parameters:**

- **`name`** (str, required) - The exact Qdrant collection name to look up.

**Response fields:**

- **`name`** - The collection name (echoed back).
- **`points_count`** - Number of indexed points (chunks).
- **`status`** - Collection status (e.g. `"green"`).

**Errors:** `500` collection does not exist or Qdrant connection failure.

**Example:**

```bash
curl http://localhost:8080/api/v1/collections/nepali_docs
```

```json
{
    "name": "nepali_docs",
    "points_count": 142,
    "status": "green"
}
```

---

### `GET /api/v1/conversations`

List all persisted conversation IDs. Scans `api/conversations/` for `*.json` files and returns each file's stem (basename without the `.json` extension). Returns an empty list when no conversations have been persisted yet.

**Response:** Sorted list of conversation ID strings.

**Example:**

```bash
curl http://localhost:8080/api/v1/conversations
```

```json
["061004efc64840f7", "a3f8b2e1c9d74501", "qwen_fun"]
```

---

### `GET /api/v1/conversations/{conversation_id}`

Return the full turn history for a single persisted conversation.

**Path parameters:**

- **`conversation_id`** (str, required) - The bare conversation ID (filename stem), e.g. `061004efc64840f7` - **without** the `.json` suffix.

**Response fields:**

- **`conversation_id`** - The requested conversation ID (echoed back).
- **`turns`** - Ordered list of `{"role": "...", "content": "..."}` dicts representing the full conversation history.

**Errors:** `404` conversation not found · `500` file cannot be parsed as valid JSON.

**Example:**

```bash
curl http://localhost:8080/api/v1/conversations/061004efc64840f7
```

```json
{
    "conversation_id": "061004efc64840f7",
    "turns": [
        {"role": "user", "content": "नेपालको राजधानी के हो?"},
        {"role": "assistant", "content": "नेपालको राजधानी काठमाडौं हो।"}
    ]
}
```
