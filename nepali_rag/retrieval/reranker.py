from sentence_transformers import CrossEncoder
from nepali_rag.config.settings import settings


class Reranker:
    """Cross-encoder reranker for second-stage retrieval precision."""

    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = settings.reranker_model
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = None,
    ) -> list[dict]:
        """Score each (query, passage) pair and return top_k sorted by rerank score.

        Args:
            query: the user query string.
            results: list of dicts, each must have a 'text' key.
            top_k: how many results to return after reranking.

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
