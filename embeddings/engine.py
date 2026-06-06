"""
Embedding engine — converts DocumentChunks into dense vectors.

Model: intfloat/multilingual-e5-large (free, HuggingFace)
  - Supports 100+ languages including EN, FR, DE, ES, IT
  - 1024-dimensional embeddings
  - Runs on CPU (slow but works); set EMBED_DEVICE=cuda on Colab T4

e5 models require a prefix:
  - "query: ..."   for search queries
  - "passage: ..." for documents being indexed

Usage:
    engine = EmbeddingEngine()
    query_vec = engine.embed_query("What is Acme's revenue?")
    embedded  = engine.embed_chunks(chunks)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from utils.logging import logger
from utils.models import DocumentChunk, EmbeddedChunk
from utils.settings import settings


def _batches(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class EmbeddingEngine:
    """
    Wraps sentence-transformers for multilingual embedding.

    Args:
        model_name : HuggingFace model id (default: multilingual-e5-large)
        device     : "cpu" | "cuda" | "mps"
        batch_size : chunks per forward pass (reduce if OOM on GPU)
    """

    def __init__(
        self,
        model_name : str | None = None,
        device     : str | None = None,
        batch_size : int = 16,
    ):
        self.model_name = model_name or settings.embed_model
        self.device     = device     or settings.embed_device
        self.batch_size = batch_size
        self._model     = None   # lazy load on first use

    @property
    def model(self):
        if self._model is None:
            logger.info(
                "Loading embedding model '{}' on {} — first load downloads ~1.1GB ...",
                self.model_name, self.device
            )
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            logger.info(
                "Embedding model ready. dim={}",
                self._model.get_sentence_embedding_dimension()
            )
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    # ── Public API ────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query.
        Uses the "query: " prefix required by e5 models.
        """
        vec = self.model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].tolist()

    def embed_chunks(self, chunks: list[DocumentChunk]) -> list[EmbeddedChunk]:
        """
        Embed a list of DocumentChunks in batches.
        Uses the "passage: " prefix required by e5 models.
        Returns EmbeddedChunk objects (DocumentChunk + embedding vector).
        """
        if not chunks:
            return []

        logger.info(
            "Embedding {} chunks in batches of {} on {} ...",
            len(chunks), self.batch_size, self.device
        )
        embedded: list[EmbeddedChunk] = []

        for batch in _batches(chunks, self.batch_size):
            texts = [f"passage: {c.text}" for c in batch]
            vecs  = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
            for chunk, vec in zip(batch, vecs):
                embedded.append(EmbeddedChunk(
                    **chunk.model_dump(),
                    embedding  = vec.tolist(),
                    embed_model= self.model_name,
                ))

        logger.info("Embedded {} chunks", len(embedded))
        return embedded

    def embed_texts(
        self,
        texts    : list[str],
        is_query : bool = False,
    ) -> list[list[float]]:
        """Embed arbitrary strings. Useful for reranking and eval."""
        prefix  = "query: " if is_query else "passage: "
        prefixed = [f"{prefix}{t}" for t in texts]
        vecs    = self.model.encode(prefixed, normalize_embeddings=True)
        return [v.tolist() for v in vecs]
    