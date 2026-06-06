"""
Vector store — ChromaDB (dense) + BM25 (sparse) with RRF hybrid fusion.

Architecture:
  VectorStore  — wraps ChromaDB for upsert + cosine similarity search
  BM25Index    — in-memory lexical index rebuilt from ChromaDB on startup
  HybridStore  — unified interface: add_chunks() / search() / delete()

Retrieval pipeline:
  query → embed → ChromaDB top-K  ─┐
  query → tokenise → BM25 top-K   ─┤→ RRF fusion → reranker → top-N results
                                    

Usage:
    store = HybridStore()
    store.add_chunks(embedded_chunks)
    results = store.search("revenue growth 2023", query_vec, top_k=10)
"""

from __future__ import annotations

from pathlib import Path

from utils.logging import logger
from utils.models import ChunkType, DocumentChunk, EmbeddedChunk, Language, RetrievedChunk
from utils.settings import settings


# ── BM25 lexical index ─────────────────────────────────────────────────────

class BM25Index:
    """
    Lightweight BM25 index over chunk texts.
    Rebuilt in memory at startup from ChromaDB — no separate persistence needed.
    """

    def __init__(self):
        self._bm25   = None
        self._chunks : list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            logger.debug("BM25: no chunks to index")
            return
        from rank_bm25 import BM25Okapi
        tokenised  = [c.text.lower().split() for c in chunks]
        self._bm25  = BM25Okapi(tokenised)
        self._chunks = chunks
        logger.debug("BM25 index built with {} documents", len(chunks))

    def search(
        self,
        query : str,
        top_k : int = 20,
    ) -> list[tuple[DocumentChunk, float]]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            (self._chunks[i], float(s))
            for i, s in ranked
            if s > 0
        ]

    @property
    def size(self) -> int:
        return len(self._chunks)


# ── ChromaDB vector store ──────────────────────────────────────────────────

class VectorStore:
    """Thin wrapper around ChromaDB for upsert + dense cosine search."""

    def __init__(
        self,
        persist_dir     : Path | None = None,
        collection_name : str  | None = None,
    ):
        import chromadb

        persist_dir     = persist_dir     or settings.chroma_persist_dir
        collection_name = collection_name or settings.chroma_collection
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client     = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name     = collection_name,
            metadata = {"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB ready — collection='{}', {} chunks stored",
            collection_name, self._collection.count()
        )

    def add(self, chunks: list[EmbeddedChunk]) -> None:
        """Upsert embedded chunks. Existing IDs are overwritten."""
        if not chunks:
            return
        self._collection.upsert(
            ids        = [c.id for c in chunks],
            embeddings = [c.embedding for c in chunks],
            documents  = [c.text for c in chunks],
            metadatas  = [self._meta(c) for c in chunks],
        )
        logger.info("Upserted {} chunks into ChromaDB", len(chunks))

    def search(
        self,
        query_vector : list[float],
        top_k        : int = 20,
        where        : dict | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """Dense cosine similarity search. Returns (chunk, score) pairs."""
        count = self._collection.count()
        if count == 0:
            return []

        kwargs: dict = {
            "query_embeddings" : [query_vector],
            "n_results"        : min(top_k, count),
            "include"          : ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        res       = self._collection.query(**kwargs)
        ids       = res["ids"][0]
        docs      = res["documents"][0]
        metas     = res["metadatas"][0]
        distances = res["distances"][0]

        results: list[tuple[DocumentChunk, float]] = []
        for cid, text, meta, dist in zip(ids, docs, metas, distances):
            chunk = self._hydrate(cid, text, meta)
            # Chroma cosine distance ∈ [0,2]; convert to similarity ∈ [0,1]
            score = 1.0 - dist / 2.0
            results.append((chunk, score))
        return results

    def get_all(self) -> list[DocumentChunk]:
        """Fetch every stored chunk (used to rebuild BM25 index)."""
        count = self._collection.count()
        if count == 0:
            return []
        res = self._collection.get(include=["documents", "metadatas"])
        return [
            self._hydrate(cid, text, meta)
            for cid, text, meta in zip(
                res["ids"], res["documents"], res["metadatas"]
            )
        ]

    def delete_by_doc(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document."""
        self._collection.delete(where={"doc_id": doc_id})
        logger.info("Deleted chunks for doc_id={}", doc_id)

    @property
    def count(self) -> int:
        return self._collection.count()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _meta(c: EmbeddedChunk) -> dict:
        return {
            "doc_id"     : c.doc_id,
            "chunk_index": c.chunk_index,
            "chunk_type" : c.chunk_type.value,
            "language"   : c.language.value,
            "page_number": c.page_number if c.page_number is not None else -1,
            "token_count": c.token_count,
            "source"     : c.metadata.get("source", ""),
        }

    @staticmethod
    def _hydrate(cid: str, text: str, meta: dict) -> DocumentChunk:
        return DocumentChunk(
            id          = cid,
            doc_id      = meta.get("doc_id", ""),
            chunk_index = int(meta.get("chunk_index", 0)),
            chunk_type  = ChunkType(meta.get("chunk_type", "text")),
            text        = text,
            page_number = (
                int(meta["page_number"])
                if meta.get("page_number", -1) != -1 else None
            ),
            language    = Language(meta.get("language", "unknown")),
            token_count = int(meta.get("token_count", 0)),
            metadata    = {"source": meta.get("source", "")},
        )


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────

def _rrf(
    vector_results : list[tuple[DocumentChunk, float]],
    bm25_results   : list[tuple[DocumentChunk, float]],
    k              : int = 60,
) -> list[RetrievedChunk]:
    """
    Combine dense + sparse rankings with Reciprocal Rank Fusion.
    RRF score = Σ 1/(k + rank).  Higher is better.
    """
    scores: dict[str, dict] = {}

    for rank, (chunk, vscore) in enumerate(vector_results):
        scores.setdefault(chunk.id, {
            "chunk": chunk, "vector_score": 0.0,
            "bm25_score": 0.0, "rrf": 0.0,
        })
        scores[chunk.id]["vector_score"] = vscore
        scores[chunk.id]["rrf"]         += 1.0 / (k + rank + 1)

    for rank, (chunk, bscore) in enumerate(bm25_results):
        scores.setdefault(chunk.id, {
            "chunk": chunk, "vector_score": 0.0,
            "bm25_score": 0.0, "rrf": 0.0,
        })
        scores[chunk.id]["bm25_score"] = bscore
        scores[chunk.id]["rrf"]       += 1.0 / (k + rank + 1)

    ranked = sorted(scores.values(), key=lambda x: x["rrf"], reverse=True)
    return [
        RetrievedChunk(
            chunk        = item["chunk"],
            vector_score = item["vector_score"],
            bm25_score   = item["bm25_score"],
        )
        for item in ranked
    ]


# ── Unified hybrid store ───────────────────────────────────────────────────

class HybridStore:
    """
    Single interface for all retrieval operations.

    - add_chunks()  → embed + store in ChromaDB; invalidates BM25
    - search()      → hybrid dense+sparse search with RRF fusion
    - delete_doc()  → remove all chunks for a document

    BM25 index is rebuilt lazily on first search after any add/delete.
    """

    def __init__(self):
        self._vector = VectorStore()
        self._bm25   = BM25Index()
        self._stale  = True   # BM25 needs rebuild

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """Store embedded chunks. Marks BM25 index as stale."""
        self._vector.add(chunks)
        self._stale = True

    def search(
        self,
        query          : str,
        query_vector   : list[float],
        top_k          : int = 10,
        language_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Hybrid search: vector + BM25 → RRF fusion → top_k results.

        Args:
            query           : raw query text (for BM25)
            query_vector    : embedded query vector (for ChromaDB)
            top_k           : number of results to return
            language_filter : ISO 2-letter code to filter by language
        """
        self._maybe_rebuild_bm25()

        where = {"language": language_filter} if language_filter else None

        vector_results = self._vector.search(
            query_vector, top_k=top_k * 3, where=where
        )
        bm25_results   = self._bm25.search(query, top_k=top_k * 3)

        fused = _rrf(vector_results, bm25_results)
        return fused[:top_k]

    def delete_doc(self, doc_id: str) -> None:
        self._vector.delete_by_doc(doc_id)
        self._stale = True

    def _maybe_rebuild_bm25(self) -> None:
        if self._stale:
            logger.debug("Rebuilding BM25 index from ChromaDB ...")
            all_chunks  = self._vector.get_all()
            self._bm25.build(all_chunks)
            self._stale = False

    @property
    def count(self) -> int:
        return self._vector.count