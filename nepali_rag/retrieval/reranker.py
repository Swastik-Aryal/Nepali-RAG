import torch
from sentence_transformers import CrossEncoder
from nepali_rag.config.settings import settings


class Reranker:
    """Scores (query, passage) pairs with a CrossEncoder model.

    Args:
        - model_name (str, optional): Name or path of the cross-encoder model to load.
            Defaults to ``settings.reranker_model`` (BAAI/bge-reranker-v2-m3).
        - use_fp16 (bool, optional): If ``True`` and a CUDA GPU is available, loads the
            model in half-precision (float16) for faster inference and lower memory usage.
            Silently ignored on CPU-only machines. Defaults to ``False``.
    """

    def __init__(self, model_name: str = None, use_fp16: bool = False):
        """Load reranker model.

        - model_name (str, optional): Name or path of the cross-encoder model to load.
            Defaults to ``settings.reranker_model`` (BAAI/bge-reranker-v2-m3).
        - use_fp16 (bool, optional): If ``True`` and a CUDA GPU is available, loads the
            model in half-precision (float16) for faster inference and lower memory usage.
            Silently ignored on CPU-only machines. Defaults to ``False``.
        
        """
        if model_name is None:
            model_name = settings.reranker_model
        self.model = CrossEncoder(model_name)
        if use_fp16 and torch.cuda.is_available():
            self.model.model.half()
            print("Reranker model loaded in half-precision (float16)")

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = None,
    ) -> list[dict]:
        """Rerank hybrid-search candidates by cross-encoder relevance score.

        Args:
            query: User search query.
            results: Candidate dicts from hybrid search; each must have a ``text`` key.
            top_k: Number of results to return (default ``settings.top_k_rerank``, 5).

        Returns:
            Reranked list of result dicts, each with an added 'rerank_score' key.
        """
        if top_k is None:
            top_k = settings.top_k_rerank

        if not results:
            return []

        pairs = [[query, r["text"]] for r in results]
        scores = self.model.predict(pairs)

        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
