# MCP Server : Nepali-RAG

This directory contains the **Model Context Protocol (MCP)** server for Nepali-RAG. It exposes document ingestion and semantic retrieval as MCP tools that any MCP-compatible AI client can call directly.

**Transport:** Streamable HTTP, listens at `http://localhost:8000/mcp` by default.

> **Note:** The MCP server **DOESNOT** require a `.env` file. Only add one in the CWD if you wish to alter the `EMBEDDING_MODEL` OR/AND `RERANKER_MODEL`

---

## Contents

1. [Quick Start](#quick-start)
2. [Tools](#tools)
   - [`ingest_document`](#1-ingest_document)
   - [`retrieve`](#2-retrieve)
3. [Storage: Local Qdrant](#storage-local-qdrant-by-default)
4. [Configuration Examples](#configuration-examples)
   - [Claude Desktop (ngx)](#claude-desktop-ngx)
   - [Claude Code](#claude-code)
   - [Cursor](#cursor)
   - [VS Code](#vs-code)
   - [Windsurf](#windsurf)
   - [Zed](#zed)
5. [CLI Reference](#cli-reference)

---

## Quick Start

```bash
nepali-rag mcp
```

Server starts at `http://localhost:8000/mcp`. Configure it in your AI client (see [Configuration Examples](#configuration-examples)).

### Modes

| Flag          | Behavior                                                                          |
| ------------- | --------------------------------------------------------------------------------- |
| *(none)*    | Eager mode: embedding model and reranker load at startup (faster first tool call) |
| `--no-init` | Lazy mode: models load on first tool call (faster startup) (Not recommended)      |
| `-v`        | Enable debug logging                                                              |

---

## Tools

### 1. `ingest_document`

Takes one or more document file paths (PDF, image, Office, text, etc.), extracts and chunks the text, embeds each chunk using the configured embedding model, and indexes the vectors into a Qdrant collection for later retrieval. Supports both local embedded Qdrant (zero-infrastructure, default) and remote Qdrant servers.

**Inputs:**

- `file_paths` (list of strings, required): absolute paths to the document files to ingest.
- `qdrant_path` or `qdrant_url` (one required): path to a local Qdrant storage directory, or URL of a remote Qdrant server (`qdrant_api_key` optional for remote). Defaults to local embedded Qdrant with no external infra.
- `collection_name` (optional, auto-generated if omitted): Qdrant collection name (e.g. `session_a3f8b2c1`). Auto-generated names include a random hash to avoid collisions.
- `chunk_size` (optional, default `256`): max tokens per chunk before embedding.
- `chunk_overlap_sentences` (optional, default `2`): overlapping sentences between consecutive chunks to preserve boundary context.

**Output:** collection name, qdrant storage path, and per-file stats (pages extracted, chunks created, chunks indexed).

```json
{
  "collection_name": "session_a3f8b2c1",
  "qdrant_path": "C:/Users/me/my-project/qdrant_data",
  "results": [
    {
      "collection_name": "session_a3f8b2c1",
      "filename": "document.pdf",
      "pages_extracted": 12,
      "chunks_created": 45,
      "chunks_indexed": 45
    }
  ],
  "total_files": 1,
  "total_chunks_indexed": 45
}
```

---

### 2. `retrieve`

Takes a natural language query, runs hybrid search (dense embedding similarity + sparse keyword matching) against a Qdrant collection to fetch candidate chunks, then reranks them with a cross-encoder model and returns the most relevant results.

**Inputs:**

- `query` (string, required): the search query in natural language.
- `collection_name` (string, required): Qdrant collection to search (use the name from `ingest_document` output).
- `qdrant_path` or `qdrant_url` (one required): same storage config as `ingest_document`. `qdrant_api_key` for remote.
- `top_k_retrieval` (optional, default `20`): number of candidates retrieved via hybrid search before reranking.
- `top_k_rerank` (optional, default `5`): number of results returned after cross-encoder reranking.

**Output:** ranked results with text, source metadata, hybrid search score, and reranker score.

```json
{
  "results": [
    {
      "text": "काठमाडौं नेपालको राजधानी हो...",
      "metadata": {
        "source": "nepal_info.pdf",
        "page_number": 1,
        "chunk_index": 0
      },
      "score": 0.87,
      "rerank_score": 0.95
    }
  ]
}
```

---

## Storage: Local Qdrant by Default

Local file-based Qdrant is the default and requires zero external infrastructure.

### Local mode (default)

Pass `qdrant_path` with an absolute path to a directory. Data persists on disk at that path.

**Path guidelines for the AI agent:**

- If the agent has workspace access (Cursor, VS Code, Windsurf): use `<project_root>/qdrant_data` (e.g. `C:/Users/me/my-project/qdrant_data`).
- If no project directory (Claude Desktop, Claude Code): ask the user. Recommend `~/.nepali_rag/qdrant_data` (resolved to an absolute path).

### Remote mode (opt-in)

Pass `qdrant_url` (and optionally `qdrant_api_key`) to connect to a remote Qdrant server. Always ask the user before switching to remote mode.

> **Note:** The Qdrant client is cached per storage location. Ingest and retrieve share the same connection, avoiding locking conflicts with local embedded Qdrant.

---

## Configuration Examples

### Claude Desktop (ngx)

Edit `claude_desktop_config.json` (Claude → Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "nepali-rag": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:8000/mcp"
      ]
    }
}
```

### Claude Code

Add to `.claude/settings.json` in your project (or `~/.claude/settings.json` globally):

```json
{
  "mcpServers": {
    "nepali-rag": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "nepali-rag": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### VS Code

Add to `.vscode/mcp.json` in your project root (or configure via Settings → MCP):

```json
{
  "servers": {
    "nepali-rag": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### Windsurf

Add to `.windsurf/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "nepali-rag": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

### Zed

Add to `~/.config/zed/settings.json`:

```json
{
  "mcp_servers": {
    "nepali-rag": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

---

## CLI Reference

```
nepali-rag mcp
    --no-init                 skip eager model loading (load on first tool call)
    -v                        enable debug logging

    Server URL: http://localhost:8000/mcp (not configurable yet)
```

---

## Notes

- The server uses **streamable-http** transport from the `mcp` Python package (`FastMCP`).
- The embedding model and reranker are cached in memory once loaded.
- For questions or bugs, open an issue at [github.com/Swastik-Aryal/Nepali-RAG](https://github.com/Swastik-Aryal/Nepali-RAG).
