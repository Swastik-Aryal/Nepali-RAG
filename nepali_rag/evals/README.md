# Evaluation Module

This module provides a two-step pipeline for evaluating the Nepali RAG system across different embedding and reranker combinations:

1. **Generate** RAG outputs (`eval-generate`) - runs every embedding + reranker combo against your test dataset, producing result CSVs inside each datapoint's `rag_results/` folder.
2. **Score** those outputs (`eval-score`) - reads the result CSVs produced by step 1, runs DeepEval's LLM-as-judge metrics, and writes structured JSON reports.

> **`eval-generate` MUST be run before `eval-score`.** The scoring step reads from the `rag_results/` CSVs that the generation step creates. If a datapoint has no `rag_results/` directory, `eval-score` will skip it entirely.

Both steps are incremental: they skip combinations that already have outputs, so you can safely re-run after adding new models or datapoints.

## Installation

Install the base package with the `evals` extra to pull in DeepEval:

```bash
pip install nepali-rag[evals]
```
Or add `deepeval` manually:

```bash
pip install deepeval
```

## Dataset Format

Organize your dataset as a directory of "datapoint" sub-folders. Each sub-folder needs a `data.csv` and a `documents/` directory:

```
my_eval_dataset/
    test1/
        data.csv
        documents/
            doc1.pdf
            doc2.docx
    test2/
        data.csv
        documents/
            article.html
    ...
```

**`data.csv` columns:**

| Column | Description |
|---|---|
| `question` | The question to ask the RAG pipeline |
| `actual_answer` | The ground-truth / expected answer |

Example `data.csv`:

```csv
question,actual_answer
लक्ष्मीप्रसाद देवकोटाको जन्म कहाँ भएको थियो?,लक्ष्मीप्रसाद देवकोटाको जन्म काठमाडौँमा भएको थियो।
```

**`documents/`** can contain any file type the ingestion pipeline supports (PDF, DOCX, PPTX, HTML, ODT, etc.).

## Environment Setup

### Step 1: RAG infrastructure variables

These are needed for `eval-generate` to ingest documents and run queries:

```env
QDRANT_URL=https://your-qdrant-instance.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# LLM for RAG answer generation (uses your existing nepali_rag LLM config)
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key
```

All standard RAG settings (`TOP_K_RETRIEVAL`, `TOP_K_RERANK`, `DENSE_WEIGHT`, `CHUNK_SIZE`, etc.) are picked up from your `.env` or `nepali_rag/config/settings.py`.

### Step 2: Judge model variables (for eval-score)

The scoring step uses a separate LLM as a "judge" to evaluate answer quality. Configure it with these variables:

```env
EVAL_PROVIDER=gemini
EVAL_MODEL=gemini-2.5-flash
EVAL_API_KEY=your_google_api_key
```

Or for an OpenAI-compatible provider (Groq, Together, etc.):

```env
EVAL_PROVIDER=openai_compatible
EVAL_MODEL=qwen/qwen3.6-27b
EVAL_API_KEY=gsk_...
EVAL_BASE_URL=https://api.groq.com/openai/v1
```

## Usage

### CLI

> Run step 1 first. Step 2 will fail (or skip everything) if no `rag_results/` CSVs exist yet.

```bash
# Step 1: Generate RAG outputs for all model combinations
nepali-rag eval-generate path/to/my_eval_dataset

# Step 2: Score the generated outputs (requires step 1 outputs)
nepali-rag eval-score path/to/my_eval_dataset
```

With model overrides:

```bash
nepali-rag eval-generate path/to/dataset \
    --embeddings BAAI/bge-m3 intfloat/multilingual-e5-small \
    --rerankers BAAI/bge-reranker-v2-m3

nepali-rag eval-score path/to/dataset \
    --output-dir ./my_results \
    --embeddings BAAI/bge-m3 \
    --rerankers BAAI/bge-reranker-v2-m3
```

With async evaluation (for `eval-score` only):

```bash
nepali-rag eval-score path/to/dataset --async

nepali-rag eval-score path/to/dataset --async --max-concurrent 4
```

| Flag | Description |
|---|---|
| `--async` | Run DeepEval evaluations asynchronously (default: off) |
| `--max-concurrent N` | Max parallel evaluations when `--async` is enabled (default: 2) |

### Python

```python
from nepali_rag.evals import run_rag, run_evaluation

# Generate RAG outputs
run_rag(
    dataset_dir="path/to/dataset",
    embedding_models=["BAAI/bge-m3"],
    reranker_models=["BAAI/bge-reranker-v2-m3"],
)

# Score them
run_evaluation(
    dataset_dir="path/to/dataset",
    output_dir="./results",
)

# Score with async
run_evaluation(
    dataset_dir="path/to/dataset",
    output_dir="./results",
    run_async=True,
    max_concurrent=4,
)
```

## Output Structure

### 1. eval-generate

Creates a `rag_results/` folder inside each datapoint directory:

```
test1/
    rag_results/
        BAAI_bge_m3__BAAI_bge_reranker_v2_m3.csv
        intfloat_multilingual_e5_small__BAAI_bge_reranker_v2_m3.csv
```

Filenames follow the pattern `{embedding_slug}__{reranker_slug}.csv` where slugs are created by replacing `/` and `-` with `_`.

**Result CSV columns:**

| Column | Type | Description |
|---|---|---|
| `question` | string | Original question |
| `actual_answer` | string | Ground-truth answer |
| `output_answer` | string | RAG-generated answer |
| `retrieved_context` | JSON string (`list[str]`) | A JSON-serialized array of retrieved text chunks (e.g. `["chunk1", "chunk2"]`). The scorer parses this with `json.loads()` to build DeepEval test cases. |

### 2. eval-score

Writes structured JSON to an output directory (defaults to a sibling `eval_results/` next to the dataset):

```
eval_results/
    data_wise/
        test1/
            detailed.json       # per-question scores and reasons
            averages.json       # per-combo metric averages
        test2/
            detailed.json
            averages.json
        overall_averages.json   # cross-datapoint averages
```

**`detailed.json`** is keyed by `embedding_model > reranker_model > [array of per-question results]`. Each entry contains the question, answers, and per-metric score + reason.

**`averages.json`** aggregates scores per combo within that datapoint.

**`overall_averages.json`** averages across all datapoints for a global view.

## DeepEval Metrics

The scoring step runs these 5 standard RAG metrics from [DeepEval](https://docs.confident-ai.com/docs/metrics-introduction):

| Metric | What it measures |
|---|---|
| **Answer Relevancy** | Is the generated answer relevant to the question? |
| **Faithfulness** | Is the answer grounded in the retrieved context (no hallucination)? |
| **Contextual Recall** | Does the retrieved context cover the information in the expected answer? |
| **Contextual Precision** | Are the relevant chunks ranked higher in the retrieved context? |
| **Contextual Relevancy** | Is the retrieved context relevant to the question? |

All metrics use a default threshold of `0.5` (configurable via `build_metrics(threshold=0.7)`).

## Supported Judge Model Providers

The judge model (used by DeepEval to score outputs) supports these providers via `EVAL_PROVIDER`:

| Provider | `EVAL_PROVIDER` | `EVAL_BASE_URL` | Notes |
|---|---|---|---|
| OpenAI | `openai` | Not needed | Official OpenAI API |
| OpenAI-compatible | `openai_compatible` | **Required** | Groq, Together, any `/v1` endpoint |
| Anthropic | `anthropic` | Not needed | Claude models, needs `anthropic` pip package |
| Google Gemini | `gemini` | Not needed | Uses `EVAL_API_KEY` as Google API key |
| Ollama | `ollama` | Defaults to `http://localhost:11434` | Local models, no API key needed |
| vLLM | `vllm` | **Required** | Self-hosted vLLM `/v1` endpoint |
| HuggingFace | `huggingface` | Defaults to `https://router.huggingface.co/v1` | HF Inference API |

You can also construct the judge model programmatically:

```python
from nepali_rag.evals.eval_model import build_model, build_metrics

model = build_model(provider="anthropic", model_name="claude-sonnet-4-20250514")
metrics = build_metrics(model, threshold=0.7)
```

## Default Model Lists

When no `--embeddings` or `--rerankers` are specified, both steps sweep through these defaults:

**Embedding models:**
- `BAAI/bge-m3`
- `intfloat/multilingual-e5-small`
- `intfloat/multilingual-e5-base`
- `intfloat/multilingual-e5-large`
- `intfloat/multilingual-e5-large-instruct`
- `qwen/qwen3-embedding-0.6b`
- `qwen/qwen3-embedding-4b`
- `qwen/qwen3-embedding-8b`
- `jinaai/jina-embeddings-v3`
- `jinaai/jina-embeddings-v5-text-nano-retrieval`
- `jinaai/jina-embeddings-v5-text-small-retrieval`
- `alibaba-nlp/gte-multilingual-base`

**Reranker models:**
- `BAAI/bge-reranker-v2-m3`
- `Qwen/Qwen3-Reranker-0.6B`
- `BAAI/bge-reranker-v2-gemma`

That's 12 x 3 = 36 combinations by default.

## Module Files

| File | Purpose |
|---|---|
| `rag_output_generator.py` | Ingests docs, runs RAG queries, writes result CSVs |
| `evaluator.py` | Reads result CSVs, runs DeepEval metrics, writes JSON reports |
| `eval_model.py` | Judge model construction and metric configuration |

## Customization

This module is meant to be edited. Some things you might want to change:

- **Add/remove models** - edit `EMBEDDING_MODELS` and `RERANKER_MODELS` in `rag_output_generator.py` and `evaluator.py`, or just pass `--embeddings` / `--rerankers` at runtime.
- **Change metrics** - edit `build_metrics()` in `eval_model.py` to add, remove, or swap DeepEval metrics. See the [DeepEval metrics docs](https://docs.confident-ai.com/docs/metrics-introduction) for what's available.
- **Change the threshold** - pass a different `threshold` to `build_metrics()`.
- **Add a new judge provider** - add a new branch in `build_model()` in `eval_model.py`. Follow the `AnthropicModel` class as an example for wrapping non-OpenAI-compatible SDKs.
- **Change RAG settings per-eval** - the generator uses whatever `settings.py` / `.env` values are active. You can set env vars before running to override.

## Benchmark Results

### Setup

| Parameter | Value |
|---|---|
| **RAG LLM** | `gemma4:31b` (Google Gemini API) |
| **Judge LLM** | `Qwen3-235B-A22B-Instruct-2507` |
| **Dataset** | 8 document samples, 10 QnA pairs each (English + Nepali) |
| **Embedding models tested** | 10 of 12 supported |
| **Reranker models tested** | All 3 supported |
| **Total combinations** | 30 |
| **Skipped** | `Qwen3-Embedding-4B`, `Qwen3-Embedding-8B` (hardware limitations) |

### Results

> [!IMPORTANT]
> These evaluations were done on a randomly sampled general-purpose dataset that you can view **[here](https://github.com/Swastik-Aryal/Nepali-RAG/blob/main/nepali_rag/data/evaluation_tests)** and may not reflect the performance of the system for all usecases. For more accurate results, evaluations on domain-specific datasets are recommended.

Sorted by **Retrieval Avg** (average of Contextual Recall, Precision, and Relevancy). These 3 metrics directly measure retrieval quality, which is what the embedding + reranker combination controls. Answer Relevancy and Faithfulness are shown but are more LLM-dependent.

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

> **Best per metric**: Ans. Relevancy: `jina-v5-nano` + `bge-m3` (1.000) | Faithfulness: `jina-v5-small` + `bge-gemma` (0.983) | Ctx. Recall: `jina-v5-nano` + `bge-m3` (0.988) | Ctx. Precision: `gte-multilingual-base` + `bge-m3` (0.931) | Ctx. Relevancy: `jina-v5-small` + `bge-m3` (**0.433**)

## Replicating Results

The evaluation dataset and pre-computed result JSONs are bundled in [`nepali_rag/data/evaluation_tests/`](./../../nepali_rag/data/evaluation_tests/). To replicate the benchmark results below, point the eval command at that directory:

```bash
nepali-rag eval-score nepali_rag/data/evaluation_tests/eval_dataset
```

Pre-computed scores are in [`nepali_rag/data/evaluation_tests/eval_results`](./../../nepali_rag/data/evaluation_tests/eval_results)


