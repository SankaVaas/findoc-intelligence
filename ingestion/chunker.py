"""
Chunking layer — splits raw text into DocumentChunk objects for embedding.

Three strategies:
  recursive  (default) — respects paragraph → sentence → word boundaries.
                          Best for most documents.
  fixed                — simple sliding window. Fastest, no dependencies.
  semantic             — groups sentences by meaning similarity.
                          Best for dense financial prose; needs embedding model.

Tables are always converted to markdown and appended as TABLE-type chunks.

Usage:
    chunker = DocumentChunker()
    chunks  = chunker.chunk(doc, text, tables)
"""

from __future__ import annotations

from typing import Literal

from utils.logging import logger
from utils.models import ChunkType, DocumentChunk, RawDocument

ChunkStrategy = Literal["recursive", "fixed", "semantic"]


class DocumentChunker:
    """
    Converts (RawDocument, text, tables) → list[DocumentChunk].

    Args:
        strategy      : chunking algorithm to use
        chunk_size    : target size in characters (recursive/fixed)
        chunk_overlap : overlap between consecutive chunks
        min_chunk_size: discard chunks smaller than this (noise filter)
    """

    def __init__(
        self,
        strategy      : ChunkStrategy = "recursive",
        chunk_size    : int = 1000,
        chunk_overlap : int = 200,
        min_chunk_size: int = 80,
    ):
        self.strategy       = strategy
        self.chunk_size     = chunk_size
        self.chunk_overlap  = chunk_overlap
        self.min_chunk_size = min_chunk_size

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def chunk(
        self,
        doc   : RawDocument,
        text  : str,
        tables: list[dict] | None = None,
    ) -> list[DocumentChunk]:
        """
        Split text into chunks and append table chunks.
        Returns an ordered list of DocumentChunk objects.
        """
        if not text.strip():
            logger.warning("Empty text for doc_id={} — no chunks produced", doc.id)
            return []

        text_chunks  = self._chunk_text(doc, text)
        table_chunks = self._chunk_tables(doc, tables or [],
                                          start_index=len(text_chunks))
        all_chunks   = text_chunks + table_chunks

        logger.info(
            "doc='{}' → {} text + {} table chunks  (strategy={})",
            doc.source_path.name, len(text_chunks),
            len(table_chunks), self.strategy,
        )
        return all_chunks

    # ─────────────────────────────────────────────────────────
    # Text chunking strategies
    # ─────────────────────────────────────────────────────────

    def _chunk_text(self, doc: RawDocument, text: str) -> list[DocumentChunk]:
        if self.strategy == "recursive":
            return self._recursive(doc, text)
        if self.strategy == "semantic":
            return self._semantic(doc, text)
        return self._fixed(doc, text)

    def _recursive(self, doc: RawDocument, text: str) -> list[DocumentChunk]:
        """
        LangChain RecursiveCharacterTextSplitter.
        Tries to split on paragraph breaks first, then sentences, then words.
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            logger.warning("langchain-text-splitters not installed; "
                           "falling back to fixed chunking. "
                           "Install: pip install langchain-text-splitters")
            return self._fixed(doc, text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size     = self.chunk_size,
            chunk_overlap  = self.chunk_overlap,
            separators     = ["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function= len,
        )
        return self._build(doc, splitter.split_text(text), ChunkType.TEXT)

    def _fixed(self, doc: RawDocument, text: str) -> list[DocumentChunk]:
        """Simple sliding-window split. No dependencies."""
        parts: list[str] = []
        start = 0
        while start < len(text):
            parts.append(text[start : start + self.chunk_size])
            start += self.chunk_size - self.chunk_overlap
        return self._build(doc, parts, ChunkType.TEXT)

    def _semantic(self, doc: RawDocument, text: str) -> list[DocumentChunk]:
        """
        SemanticChunker — groups sentences by embedding cosine similarity.
        Requires sentence-transformers + langchain-experimental.
        Falls back to recursive if dependencies are missing.
        """
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_experimental.text_splitter import SemanticChunker
            from utils.settings import settings

            embeddings = HuggingFaceEmbeddings(
                model_name   = settings.embed_model,
                model_kwargs = {"device": settings.embed_device},
            )
            splitter = SemanticChunker(
                embeddings                  = embeddings,
                breakpoint_threshold_type   = "percentile",
                breakpoint_threshold_amount = 85,
            )
            return self._build(doc, splitter.split_text(text), ChunkType.TEXT)

        except ImportError:
            logger.warning("langchain-experimental not installed; "
                           "falling back to recursive. "
                           "Install: pip install langchain-experimental")
            return self._recursive(doc, text)

    # ─────────────────────────────────────────────────────────
    # Table chunks
    # ─────────────────────────────────────────────────────────

    def _chunk_tables(
        self,
        doc        : RawDocument,
        tables     : list[dict],
        start_index: int = 0,
    ) -> list[DocumentChunk]:
        """
        Convert each extracted table into a single TABLE-type chunk.
        The table is formatted as markdown so the LLM can read it naturally.
        """
        chunks: list[DocumentChunk] = []
        for i, tbl in enumerate(tables):
            md = _table_to_markdown(tbl.get("data", []))
            if not md.strip():
                continue
            chunk = DocumentChunk(
                doc_id      = doc.id,
                chunk_index = start_index + i,
                chunk_type  = ChunkType.TABLE,
                text        = md,
                page_number = tbl.get("page"),
                language    = doc.language,
                token_count = _estimate_tokens(md),
                metadata    = {
                    "source"     : str(doc.source_path),
                    "table_index": tbl.get("table_index", i),
                },
            )
            chunks.append(chunk)
        return chunks

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _build(
        self,
        doc       : RawDocument,
        raw       : list[str],
        chunk_type: ChunkType,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for i, text in enumerate(raw):
            text = text.strip()
            if len(text) < self.min_chunk_size:
                continue
            chunks.append(DocumentChunk(
                doc_id      = doc.id,
                chunk_index = i,
                chunk_type  = chunk_type,
                text        = text,
                language    = doc.language,
                token_count = _estimate_tokens(text),
                metadata    = {"source": str(doc.source_path)},
            ))
        return chunks


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def _table_to_markdown(rows: list[list[str | None]]) -> str:
    """Convert a list-of-rows into a GitHub-flavoured markdown table."""
    if not rows:
        return ""
    clean = [
        [str(c).strip() if c is not None else "" for c in row]
        for row in rows
    ]
    # Normalise column count
    width  = max(len(r) for r in clean)
    clean  = [r + [""] * (width - len(r)) for r in clean]
    header = clean[0]
    sep    = ["---"] * width
    body   = clean[1:]
    lines  = (
        ["| " + " | ".join(header) + " |"]
        + ["| " + " | ".join(sep)    + " |"]
        + ["| " + " | ".join(row)    + " |" for row in body]
    )
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """~1 token per 4 characters — fast approximation for chunking."""
    return max(1, len(text) // 4)