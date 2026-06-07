"""
Agent pipeline — LangGraph StateGraph wiring all 5 nodes together.

Graph flow:
  retrieval → extraction → synthesis → validation → rpa → END
                                            ↑
                                     retry once if failed

Usage:
    pipeline = AgentPipeline()
    pipeline.ingest("data/raw/report.pdf")
    state = pipeline.run("What is the debt-to-equity ratio?")
    print(state.synthesis)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.logging import logger
from utils.models  import AgentState, Language
from utils.llm_client import LLMClient
from embeddings.engine  import EmbeddingEngine
from embeddings.reranker import Reranker
from vectorstore.store   import HybridStore
from agents.nodes import (
    retrieval_node,
    extraction_node,
    synthesis_node,
    validation_node,
    rpa_node,
)


def build_graph(store, embedder, reranker, llm):
    """
    Compile a LangGraph StateGraph.
    Each node receives/returns a plain dict (LangGraph requirement).
    """
    from langgraph.graph import StateGraph, END

    def _retrieval(state: dict) -> dict:
        return retrieval_node(AgentState(**state), store, embedder, reranker).model_dump()

    def _extraction(state: dict) -> dict:
        return extraction_node(AgentState(**state), llm).model_dump()

    def _synthesis(state: dict) -> dict:
        return synthesis_node(AgentState(**state), llm).model_dump()

    def _validation(state: dict) -> dict:
        return validation_node(AgentState(**state), llm).model_dump()

    def _rpa(state: dict) -> dict:
        return rpa_node(AgentState(**state)).model_dump()

    def _route(state: dict) -> str:
        """Retry synthesis once if validation failed and we haven't retried yet."""
        if not state.get("validation_passed") and not state.get("_retried"):
            state["_retried"] = True
            logger.info("[router] validation failed — retrying synthesis")
            return "synthesis"
        return "rpa"

    g = StateGraph(dict)
    g.add_node("retrieval",  _retrieval)
    g.add_node("extraction", _extraction)
    g.add_node("synthesis",  _synthesis)
    g.add_node("validation", _validation)
    g.add_node("rpa",        _rpa)

    g.set_entry_point("retrieval")
    g.add_edge("retrieval",  "extraction")
    g.add_edge("extraction", "synthesis")
    g.add_edge("synthesis",  "validation")
    g.add_conditional_edges(
        "validation", _route,
        {"synthesis": "synthesis", "rpa": "rpa"}
    )
    g.add_edge("rpa", END)

    return g.compile()


class AgentPipeline:
    """
    High-level interface: initialise once, call .run() or .ingest() anywhere.

    The pipeline is stateless between calls — each .run() is independent.
    All state lives in AgentState which is created fresh per query.
    """

    def __init__(self):
        logger.info("Initialising AgentPipeline ...")
        self.embedder = EmbeddingEngine()
        self.reranker = Reranker()
        self.store    = HybridStore()
        self.llm      = LLMClient()
        self._graph   = build_graph(
            self.store, self.embedder, self.reranker, self.llm
        )
        logger.info(
            "AgentPipeline ready — {} chunks in store, LLM={}",
            self.store.count, self.llm.model
        )

    # ── Query ─────────────────────────────────────────────────────────────

    def run(self, query: str, language: str = "unknown") -> AgentState:
        """
        Run the full agent pipeline for a query.
        Returns the final AgentState with synthesis, metrics, validation.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        initial = AgentState(
            query    = query,
            language = Language(language),
        )
        logger.info("Pipeline run | query='{}' | trace={}", query[:60], initial.trace_id[:8])
        result = self._graph.invoke(initial.model_dump())
        state  = AgentState(**result)

        logger.info(
            "Pipeline complete | validation={} | synthesis_len={} | actions={}",
            state.validation_passed,
            len(state.synthesis),
            len(state.rpa_actions),
        )
        return state

    # ── Ingest ────────────────────────────────────────────────────────────

    def ingest(self, file_path: str | Path) -> int:
        """
        Ingest a single document: load → chunk → embed → store.
        Returns number of chunks ingested.
        """
        from ingestion.pipeline import IngestionPipeline

        pipe     = IngestionPipeline()
        chunks   = pipe.ingest_file(str(file_path))
        embedded = self.embedder.embed_chunks(chunks)
        self.store.add_chunks(embedded)

        logger.info("Ingested '{}' → {} chunks", Path(file_path).name, len(embedded))
        return len(embedded)

    def ingest_directory(self, directory: str | Path) -> int:
        """Ingest all supported documents from a directory."""
        from ingestion.pipeline import IngestionPipeline

        pipe    = IngestionPipeline()
        results = pipe.ingest_directory(str(directory))
        total   = 0
        for fname, chunks in results.items():
            embedded = self.embedder.embed_chunks(chunks)
            self.store.add_chunks(embedded)
            total += len(embedded)

        logger.info("Ingested directory → {} total chunks", total)
        return total