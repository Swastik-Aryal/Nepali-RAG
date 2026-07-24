"""RAG evaluation using DeepEval metrics.

Reads result CSVs produced by rag_output_generator.py, runs 5 DeepEval
RAG metrics on each row, and produces structured JSON output.

Output structure (inside output_dir):
  data_wise/
    {datapoint_name}/
      detailed.json     - per-question scores and reasons
      averages.json     - per-combo metric averages
    overall_averages.json - cross-datapoint averages

Usage:
  python -m nepali_rag.evals.evaluator <dataset_dir> \
      [--output-dir OUTPUT_DIR] \
      [--embeddings EMB1 EMB2 ...] \
      [--rerankers RRK1 RRK2 ...] \
      [--async] \
      [--max-concurrent MAX_CONCURRENT]

Flags:
  --async             Run DeepEval evaluations asynchronously (default: off).
  --max-concurrent N  Max parallel evaluations when --async is enabled (default: 2).
"""

import csv
import json
import logging
import os
import time
import re
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "180")
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "5")
os.environ.setdefault("DEEPEVAL_RETRY_INITIAL_SECONDS", "3.0")
os.environ.setdefault("DEEPEVAL_RETRY_JITTER", "3.0")
os.environ.setdefault("DEEPEVAL_RETRY_CAP_SECONDS", "30.0")

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.test_case import LLMTestCase

from nepali_rag.evals.eval_model import build_metrics, build_model

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


def _metric_key(metric_name: str) -> str:
    """Convert 'Answer Relevancy' -> 'answer_relevancy'."""
    s = metric_name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _parse_csv(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _build_slug_to_model_map(model_names: list[str]) -> dict[str, str]:
    return {_slug(name): name for name in model_names}


def _parse_csv_filename(filename: str) -> tuple[str, str] | None:
    """Extract emb_slug and rrk_slug from '{emb_slug}__{rrk_slug}.csv'."""
    stem = Path(filename).stem
    parts = stem.split("__", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _evaluate_combo(
    csv_path: Path,
    metrics: list,
    run_async: bool = False,
    max_concurrent: int = 2,
) -> list[dict]:
    rows = _parse_csv(csv_path)
    if not rows:
        return []

    test_cases = []
    for row in rows:
        question = row.get("question", "")
        if not question:
            continue
        retrieval_context = json.loads(row.get("retrieved_context", "[]"))
        tc = LLMTestCase(
            input=question,
            actual_output=row.get("output_answer", ""),
            expected_output=row.get("actual_answer", ""),
            retrieval_context=retrieval_context,
        )
        test_cases.append(tc)

    if not test_cases:
        return []

    eval_result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        display_config=DisplayConfig(print_results=False),
        async_config=AsyncConfig(run_async=run_async, max_concurrent=max_concurrent, throttle_value=5)
    )

    tc_lookup = {tc.input: tc for tc in test_cases}

    results = []
    for tr in eval_result.test_results:
        tc = tc_lookup.get(tr.input)
        if tc is None:
            continue
        entry = {
            "question": tc.input,
            "actual_answer": tc.expected_output,
            "output_answer": tc.actual_output,
        }
        for md in tr.metrics_data:
            key = _metric_key(md.name)
            entry[f"{key}_score"] = md.score
            entry[f"{key}_reason"] = md.reason
        results.append(entry)

    return results


def _compute_averages(detailed_entries: list[dict]) -> dict:
    if not detailed_entries:
        return {}
    score_keys = [k for k in detailed_entries[0] if k.endswith("_score")]
    avgs = {}
    for sk in score_keys:
        values = [e[sk] for e in detailed_entries if e.get(sk) is not None]
        avgs[sk] = sum(values) / len(values) if values else 0.0
    avgs["num_questions"] = len(detailed_entries)
    return avgs


def run_evaluation(
    dataset_dir: str,
    output_dir: str | None = None,
    embedding_models: list[str] | None = None,
    reranker_models: list[str] | None = None,
    run_async: bool = False,
    max_concurrent: int = 2,
):
    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if output_dir:
        out_root = Path(output_dir)
    else:
        out_root = dataset_path.parent / "eval_results"

    data_wise_dir = out_root / "data_wise"
    data_wise_dir.mkdir(parents=True, exist_ok=True)

    embedding_models = embedding_models or EMBEDDING_MODELS
    reranker_models = reranker_models or RERANKER_MODELS

    emb_slug_map = _build_slug_to_model_map(embedding_models)
    rrk_slug_map = _build_slug_to_model_map(reranker_models)

    datapoint_dirs = sorted(
        [d for d in dataset_path.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )

    if not datapoint_dirs:
        logger.warning("No datapoint folders found in %s", dataset_dir)
        return

    model = build_model()
    metrics = build_metrics(model)

    all_testdata_averages: dict[str, dict[str, dict[str, dict]]] = {}

    for dp_dir in datapoint_dirs:
        rag_results_dir = dp_dir / "rag_results"
        if not rag_results_dir.is_dir():
            logger.warning("No rag_results/ in %s, skipping", dp_dir.name)
            continue

        dp_out_dir = data_wise_dir / dp_dir.name
        dp_out_dir.mkdir(parents=True, exist_ok=True)

        detailed_path = dp_out_dir / "detailed.json"
        existing_detailed: dict = {}
        if detailed_path.exists():
            with open(detailed_path, encoding="utf-8") as f:
                existing_detailed = json.load(f)

        for csv_file in sorted(rag_results_dir.glob("*.csv")):
            parsed = _parse_csv_filename(csv_file.name)
            if parsed is None:
                continue
            emb_slug, rrk_slug = parsed

            if emb_slug not in emb_slug_map or rrk_slug not in rrk_slug_map:
                continue

            emb_name = emb_slug_map[emb_slug]
            rrk_name = rrk_slug_map[rrk_slug]

            if emb_name in existing_detailed and rrk_name in existing_detailed.get(emb_name, {}):
                logger.info("Skipping %s / %s in %s (already evaluated)", emb_name, rrk_name, dp_dir.name)
                continue

            logger.info("Evaluating %s / %s in %s", emb_name, rrk_name, dp_dir.name)

            max_retries = 5
            results = None
            for attempt in range(1, max_retries + 1):
                try:
                    results = _evaluate_combo(csv_file, metrics, run_async=run_async, max_concurrent=max_concurrent)
                    break
                except Exception:
                    if attempt < max_retries:
                        logger.warning(
                            "Attempt %d/%d failed for %s / %s in %s, retrying...",
                            attempt, max_retries, emb_name, rrk_name, dp_dir.name,
                            exc_info=True,
                        )
                        time.sleep(10)
                    else:
                        logger.error(
                            "All %d attempts failed for %s / %s in %s.",
                            max_retries, emb_name, rrk_name, dp_dir.name,
                            exc_info=True,
                        )
                        raise

            if not results:
                logger.warning("No results for %s / %s in %s", emb_name, rrk_name, dp_dir.name)
                continue

            existing_detailed.setdefault(emb_name, {})[rrk_name] = results

            with open(detailed_path, "w", encoding="utf-8") as f:
                json.dump(existing_detailed, f, ensure_ascii=False, indent=2)
            logger.info("Wrote %s", detailed_path)

        testdata_avgs: dict[str, dict[str, dict]] = {}
        for emb_name, rerankers in existing_detailed.items():
            for rrk_name, entries in rerankers.items():
                avgs = _compute_averages(entries)
                testdata_avgs.setdefault(emb_name, {})[rrk_name] = avgs

        if testdata_avgs:
            averages_path = dp_out_dir / "averages.json"
            with open(averages_path, "w", encoding="utf-8") as f:
                json.dump(testdata_avgs, f, ensure_ascii=False, indent=2)
            logger.info("Wrote %s", averages_path)

        all_testdata_averages[dp_dir.name] = testdata_avgs

    overall: dict[str, dict[str, dict]] = {}
    for testdata_name, testdata_avgs in all_testdata_averages.items():
        for emb_name, rerankers in testdata_avgs.items():
            for rrk_name, avgs in rerankers.items():
                combo = overall.setdefault(emb_name, {}).setdefault(rrk_name, {
                    "sums": defaultdict(float),
                    "testdata_used": [],
                    "count": 0,
                })
                score_keys = [k for k in avgs if k.endswith("_score")]
                for sk in score_keys:
                    combo["sums"][sk] += avgs[sk]
                combo["testdata_used"].append(testdata_name)
                combo["count"] += 1

    overall_out = {"results": {}}
    for emb_name, rerankers in overall.items():
        for rrk_name, data in rerankers.items():
            count = data["count"]
            entry = {}
            for sk, total in data["sums"].items():
                entry[sk] = total / count if count else 0.0
            entry["testdata_used"] = sorted(data["testdata_used"])
            entry["num_testdata"] = count
            overall_out["results"].setdefault(emb_name, {})[rrk_name] = entry

    overall_path = data_wise_dir / "overall_averages.json"
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall_out, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", overall_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    logging.getLogger("instructor").setLevel(logging.DEBUG)


    parser = argparse.ArgumentParser(description="Run DeepEval RAG evaluation on result CSVs")
    parser.add_argument("dataset_dir", help="Path to the dataset root directory")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: sibling eval_results/)")
    parser.add_argument(
        "--embeddings",
        nargs="+",
        default=None,
        help="Override embedding models to evaluate",
    )
    parser.add_argument(
        "--rerankers",
        nargs="+",
        default=None,
        help="Override reranker models to evaluate",
    )
    parser.add_argument("--async", action="store_true", dest="run_async", default=False, help="Enable async evaluation")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent async evaluations (requires --async)")

    args = parser.parse_args()

    run_evaluation(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        embedding_models=args.embeddings,
        reranker_models=args.rerankers,
        run_async=args.run_async,
        max_concurrent=args.max_concurrent,
    )
