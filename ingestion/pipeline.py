"""
Ingestion pipeline — orchestrates loader → language detection → chunker.

This is the single entry-point for Layer 1.
Later layers (embeddings, vector store) call ingest_file() / ingest_directory()
and receive ready-to-embed DocumentChunk objects.

Usage:
    from ingestion.pipeline import IngestionPipeline

    pipe   = IngestionPipeline()
    chunks = pipe.ingest_file("data/raw/loan_app.pdf")
    print(f"Got {len(chunks)} chunks")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.logging import logger
from utils.models import DocumentChunk, Language, RawDocument
from ingestion.loader   import DocumentLoader
from ingestion.chunker  import DocumentChunker
from ingestion.language import detect_language


class IngestionPipeline:
    """
    Full ingestion pipeline: file → RawDocument + chunks.

    Args:
        strategy      : chunking strategy ("recursive" | "fixed" | "semantic")
        chunk_size    : target chunk size in characters
        chunk_overlap : overlap between chunks
        whisper_model : Whisper model size for audio transcription
    """

    def __init__(
        self,
        strategy     : str = "recursive",
        chunk_size   : int = 1000,
        chunk_overlap: int = 200,
        whisper_model: str = "base",
    ):
        self.loader  = DocumentLoader(whisper_model=whisper_model)
        self.chunker = DocumentChunker(
            strategy      = strategy,
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
        )

    def ingest_file(
        self,
        path            : str | Path,
        language_hint   : Language | None = None,
        metadata        : dict[str, Any]  | None = None,
    ) -> list[DocumentChunk]:
        """
        Process a single file end-to-end.

        Returns list of DocumentChunk objects ready for embedding.
        Raises FileNotFoundError if path doesn't exist.
        """
        path = Path(path)

        # 1. Load
        doc, text, tables = self.loader.load(
            path,
            language_hint = language_hint,
            metadata      = metadata,
        )

        # 2. Detect language (if not provided)
        if doc.language == Language.UNKNOWN:
            doc.language = detect_language(text)

        # 3. Chunk
        chunks = self.chunker.chunk(doc, text, tables)

        self._log_summary(doc, chunks)
        return chunks

    def ingest_directory(
        self,
        directory  : str | Path,
        recursive  : bool       = True,
        extensions : list[str]  | None = None,
    ) -> dict[str, list[DocumentChunk]]:
        """
        Process all supported files in a directory.

        Returns:
            dict mapping filename → list[DocumentChunk]
            Files that fail are logged and skipped (not raised).
        """
        directory = Path(directory)
        results: dict[str, list[DocumentChunk]] = {}

        loaded = self.loader.load_directory(
            directory, recursive=recursive, extensions=extensions
        )

        for doc, text, tables in loaded:
            try:
                if doc.language == Language.UNKNOWN:
                    doc.language = detect_language(text)
                chunks = self.chunker.chunk(doc, text, tables)
                results[doc.source_path.name] = chunks
                self._log_summary(doc, chunks)
            except Exception as e:
                logger.error("Chunking failed for '{}': {}",
                             doc.source_path.name, e)

        total = sum(len(v) for v in results.values())
        logger.info("Directory ingestion complete: {} files, {} total chunks",
                    len(results), total)
        return results

    @staticmethod
    def _log_summary(doc: RawDocument, chunks: list[DocumentChunk]) -> None:
        avg_tokens = (
            sum(c.token_count for c in chunks) / len(chunks)
            if chunks else 0
        )
        logger.info(
            "  ✓ '{}' | type={} | lang={} | chunks={} | avg_tokens={:.0f}",
            doc.source_path.name,
            doc.doc_type.value,
            doc.language.value,
            len(chunks),
            avg_tokens,
        )