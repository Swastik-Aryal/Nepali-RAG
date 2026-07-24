"""RAG evaluation runner.

For every embedding + reranker combination, ingests documents into a fresh
Qdrant collection, runs each question through the RAG chain, and writes a
results CSV.

Usage (CLI):
    python -m nepali_rag.evals.rag_output_generator path/to/dataset
    OR
    python -m nepali_rag.evals.rag_output_generator path/to/dataset \
        --embeddings BAAI/bge-m3 intfloat/multilingual-e5-small \
        --rerankers BAAI/bge-reranker-v2-m3

Usage (Python):
    from nepali_rag.evals.rag_output_generator import run_rag
    run_rag("path/to/dataset")
     OR 
    run_rag("path/to/dataset", embedding_models=["BAAI/bge-m3"])

Required .env / environment variables:
    QDRANT_URL          Qdrant server URL.
    QDRANT_API_KEY      Qdrant API key.
    GOOGLE_API_KEY      (or whichever LLM provider vars you use)

    Other RAG settings (top_k_retrieval, top_k_rerank, dense_weight, etc.)
    are read from settings / .env. See nepali_rag/config/settings.py.

Dataset layout:
    dataset_dir/
        test1/
            data.csv                 columns: question, actual_answer
            documents/
                doc1.pdf
                doc2.docx
                ...
        test2/
            data.csv
            documents/
                ...

Output (created inside each datapoint folder):
    test1/
        rag_results/
            <embedding_slug>__<reranker_slug>.csv
                columns: question, actual_answer, output_answer, retrieved_context
"""

import os
os.environ["USE_TF"] = "0"

import csv
import gc
import json
import logging
import uuid
from itertools import product
from pathlib import Path
import time

import torch

from nepali_rag.embeddings.embedding_model import NepaliEmbeddingModel
from nepali_rag.pipeline.document_pipeline import DocumentPipeline
from nepali_rag.retrieval.reranker import Reranker
from nepali_rag.retrieval.retriever import Retriever
from nepali_rag.vectorstore.qdrant_store import QdrantStore
from nepali_rag.rag.chain import RAGChain
from nepali_rag.llm.base import get_llm

logger = logging.getLogger(__name__)

EMBEDDING_MODELS = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "intfloat/multilingual-e5-large-instruct",
    "qwen/qwen3-embedding-0.6b",
    "qwen/qwen3-embedding-4b", # (too large so skipped in testing phase)
    "qwen/qwen3-embedding-8b", # (too large so skipped in testing phase)
    "jinaai/jina-embeddings-v3",
    "jinaai/jina-embeddings-v5-text-nano-retrieval",
    "jinaai/jina-embeddings-v5-text-small-retrieval",
    "alibaba-nlp/gte-multilingual-base",
]

RERANKER_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "Qwen/Qwen3-Reranker-0.6B",
    "BAAI/bge-reranker-v2-gemma",
]

def _slug(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def _find_csv(folder: Path) -> Path | None:
    csvs = list(folder.glob("*.csv"))
    return csvs[0] if csvs else None


def _find_documents(folder: Path) -> list[str]:
    docs_dir = folder / "documents"
    if not docs_dir.is_dir():
        return []
    return [str(p) for p in docs_dir.iterdir() if p.is_file()]


def _read_dataset_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def run_rag(
    dataset_dir: str,
    embedding_models: list[str] | None = None,
    reranker_models: list[str] | None = None,
):
    """Run the full RAG evaluation sweep.

    Args:
        dataset_dir: Root directory containing datapoint sub-folders.
        embedding_models: List of embedding model names to evaluate.
        reranker_models: List of reranker model names to evaluate.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    embedding_models = embedding_models or EMBEDDING_MODELS
    reranker_models = reranker_models or RERANKER_MODELS
    combos = list(product(embedding_models, reranker_models))

    llm = get_llm()

    datapoint_dirs = sorted(
        [d for d in dataset_path.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    if not datapoint_dirs:
        logger.warning("No datapoint folders found in %s", dataset_dir)
        return

    logger.info("Found %d datapoint(s), %d combo(s) to evaluate", len(datapoint_dirs), len(combos))

    for emb_name, rrk_name in combos:
        combo_slug = f"{_slug(emb_name)}__{_slug(rrk_name)}"
        logger.info("Loading models: embedding=%s  reranker=%s", emb_name, rrk_name)

        embedding_model = NepaliEmbeddingModel(model_name=emb_name)
        reranker = Reranker(model_name=rrk_name)

        for dp_dir in datapoint_dirs:
            out_csv = dp_dir / "rag_results" / f"{combo_slug}.csv"
            if out_csv.exists():
                logger.info("Skipping %s for %s (already exists)", dp_dir.name, combo_slug)
                continue

            csv_path = _find_csv(dp_dir)
            if csv_path is None:
                logger.warning("No CSV found in %s, skipping", dp_dir)
                continue

            doc_paths = _find_documents(dp_dir)
            if not doc_paths:
                logger.warning("No documents in %s/documents/, skipping", dp_dir)
                continue

            rows = _read_dataset_csv(csv_path)
            if not rows:
                logger.warning("CSV is empty in %s, skipping", dp_dir)
                continue

            collection_name = f"eval_{uuid.uuid4().hex[:10]}"
            logger.info(
                "Datapoint: %s | collection: %s | docs: %d | questions: %d",
                dp_dir.name, collection_name, len(doc_paths), len(rows),
            )

            store = QdrantStore(
                vector_size=embedding_model.dimension,
                collection_name=collection_name,
            )

            pipeline = DocumentPipeline(
                embedding_model=embedding_model,
                vector_store=store,
            )

            for doc_path in doc_paths:
                pipeline.ingest(doc_path)

            retriever = Retriever(
                embedding_model=embedding_model,
                store=store,
                reranker=reranker,
            )
            chain = RAGChain(retriever=retriever, llm=llm, use_tools=False)

            results_out = []
            for row in rows:
                question = row.get("question", "")
                actual_answer = row.get("actual_answer", "")

                if not question:
                    continue

                result = chain.run(question=question)

                retrieved_texts = [
                    s.get("text", "") for s in result.get("sources", [])
                ]

                results_out.append({
                    "question": question,
                    "actual_answer": actual_answer,
                    "output_answer": result.get("answer", ""),
                    "retrieved_context": json.dumps(retrieved_texts, ensure_ascii=False),
                })

                logger.info("Sleeping for 5 seconds")
                time.sleep(5)

            results_dir = dp_dir / "rag_results"
            results_dir.mkdir(exist_ok=True)
            out_csv = results_dir / f"{combo_slug}.csv"

            with open(out_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["question", "actual_answer", "output_answer", "retrieved_context"],
                )
                writer.writeheader()
                writer.writerows(results_out)

            logger.info("Wrote %d rows to %s", len(results_out), out_csv)

            try:
                store.delete_collection()
            except Exception:
                logger.warning("Failed to delete collection %s", collection_name)

        del embedding_model, reranker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description="Run RAG evaluation sweep")
    parser.add_argument("dataset_dir", help="Path to the dataset root directory")
    parser.add_argument("--embeddings", nargs="+", default=None, help="Override embedding models")
    parser.add_argument("--rerankers", nargs="+", default=None, help="Override reranker models")

    args = parser.parse_args()

    run_rag(
        dataset_dir=args.dataset_dir,
        embedding_models=args.embeddings,
        reranker_models=args.rerankers,
    )
