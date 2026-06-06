"""
Reranker — cross-encoder that rescores retrieved chunks against a query.

Model: BAAI/bge-reranker-base (free, HuggingFace, ~280MB)
  - Cross-encoder: scores (query, passage) pairs jointly
  - Far more accurate than bi-encoder cosine similarity alone
  - Only runs on top-k results (not the whole corpus) so speed is fine

Usage:
    reranker = Reranker()
    reranked = reranker.rerank(query, retrieved_chunks, top_k=5)
"""

from __future__ import annotations

from utils.logging import logger
from utils.models import RetrievedChunk
from utils.settings import settings


class Reranker:
    """
    Cross-encoder reranker using bge-reranker-base.

    Args:
        model_name : HuggingFace model id
        device     : "cpu" | "cuda"
    """

    def __init__(
        self,
        model_name : str | None = None,
        device     : str | None = None,
    ):
        self.model_name = model_name or settings.reranker_model
        self.device     = device     or settings.embed_device
        self._model     = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading reranker model '{}' ...", self.model_name)
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Reranker ready")
        return self._model

    def rerank(
        self,
        query   : str,
        chunks  : list[RetrievedChunk],
        top_k   : int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rescore chunks using the cross-encoder.
        Returns chunks sorted by rerank_score descending.

        Args:
            query  : the user's search query
            chunks : candidates from hybrid search
            top_k  : return only the top N (None = return all)
        """
        if not chunks:
            return []

        pairs  = [(query, c.chunk.text) for c in chunks]
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.warning("Reranking failed ({}); returning original order", e)
            return chunks[:top_k] if top_k else chunks

        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        ranked = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
        result = ranked[:top_k] if top_k else ranked

        logger.debug(
            "Reranked {} → {} chunks  top_score={:.4f}",
            len(chunks), len(result),
            result[0].rerank_score if result else 0.0,
        )
        return result