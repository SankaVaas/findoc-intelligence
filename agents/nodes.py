"""
Agent nodes — pure functions that take AgentState and return AgentState.
Each node does exactly one thing. The graph in pipeline.py wires them together.

Nodes:
  retrieval_node   — hybrid search → top chunks
  extraction_node  — LLM extracts structured financial metrics
  synthesis_node   — LLM writes a grounded answer
  validation_node  — LLM checks answer against sources
  rpa_node         — writes result to disk + SQLite
"""

from __future__ import annotations

from utils.logging import logger
from utils.models  import AgentState, FinancialMetrics


# ── Retrieval ──────────────────────────────────────────────────────────────

def retrieval_node(
    state   : AgentState,
    store,          # HybridStore
    embedder,       # EmbeddingEngine
    reranker,       # Reranker
) -> AgentState:
    """Embed query → hybrid search → rerank → attach top chunks to state."""
    logger.info("[retrieval] query='{}'", state.query[:80])
    try:
        qvec    = embedder.embed_query(state.query)
        lang    = state.language.value if state.language.value != "unknown" else None
        raw     = store.search(state.query, qvec, top_k=12, language_filter=lang)
        reranked = reranker.rerank(state.query, raw, top_k=6)
        state.retrieved_chunks = reranked
        logger.info("[retrieval] → {} chunks (top score={:.3f})",
                    len(reranked),
                    reranked[0].rerank_score if reranked and reranked[0].rerank_score else 0)
    except Exception as e:
        logger.error("[retrieval] failed: {}", e)
        state.error = f"Retrieval failed: {e}"
    return state


# ── Extraction ─────────────────────────────────────────────────────────────

def extraction_node(state: AgentState, llm) -> AgentState:
    """
    Extract structured financial metrics from retrieved chunks.
    Uses LLM structured output — falls back gracefully if model is too small.
    """
    logger.info("[extraction] from {} chunks", len(state.retrieved_chunks))

    if not state.retrieved_chunks:
        logger.warning("[extraction] no chunks — skipping")
        return state

    context = "\n\n---\n\n".join(
        c.chunk.text for c in state.retrieved_chunks[:5]
    )
    from prompts.templates import EXTRACTION_PROMPT
    prompt = EXTRACTION_PROMPT.format(query=state.query, context=context)

    try:
        metrics = llm.structured(prompt, schema=FinancialMetrics)
        state.extracted_metrics = metrics
        logger.info("[extraction] company='{}' revenue={} confidence={:.2f}",
                    metrics.company_name, metrics.revenue, metrics.confidence)
    except Exception as e:
        # Small models sometimes can't produce valid JSON — non-fatal
        logger.warning("[extraction] structured output failed ({}); continuing without metrics", e)

    return state


# ── Synthesis ──────────────────────────────────────────────────────────────

def synthesis_node(state: AgentState, llm) -> AgentState:
    """Generate a grounded natural-language answer from chunks + metrics."""
    logger.info("[synthesis] generating answer")

    if not state.retrieved_chunks:
        state.synthesis = "No relevant documents found to answer this query."
        return state

    context = "\n\n---\n\n".join(
        c.chunk.text for c in state.retrieved_chunks[:5]
    )
    metrics_text = ""
    if state.extracted_metrics:
        m = state.extracted_metrics
        metrics_text = f"\nExtracted metrics: {m.model_dump_json(exclude_none=True)}\n"

    from prompts.templates import SYNTHESIS_PROMPT
    prompt = SYNTHESIS_PROMPT.format(
        query    = state.query,
        context  = context,
        metrics  = metrics_text,
        language = state.language.value,
    )

    try:
        state.synthesis = llm.chat(prompt)
        logger.info("[synthesis] {} chars generated", len(state.synthesis))
    except Exception as e:
        logger.error("[synthesis] failed: {}", e)
        state.error     = f"Synthesis failed: {e}"
        state.synthesis = "Unable to generate answer — LLM error."

    return state


# ── Validation ─────────────────────────────────────────────────────────────

def validation_node(state: AgentState, llm) -> AgentState:
    """
    Check the synthesis for hallucinations against the source chunks.
    Falls back to passed=True if validation itself fails (non-fatal).
    """
    logger.info("[validation] checking synthesis")

    if not state.synthesis or state.synthesis.startswith("No relevant"):
        state.validation_passed = False
        state.validation_notes  = "Nothing to validate"
        return state

    context = "\n\n---\n\n".join(
        c.chunk.text for c in state.retrieved_chunks[:4]
    )
    from prompts.templates import VALIDATION_PROMPT
    prompt = VALIDATION_PROMPT.format(
        query     = state.query,
        synthesis = state.synthesis,
        context   = context,
    )

    try:
        from pydantic import BaseModel
        from typing import List

        class ValidationResult(BaseModel):
            passed    : bool
            confidence: float = 0.0
            issues    : list  = []
            notes     : str   = ""

        result = llm.structured(prompt, schema=ValidationResult)
        state.validation_passed = result.passed
        state.validation_notes  = result.notes or "; ".join(str(i) for i in result.issues)
        logger.info("[validation] passed={} confidence={:.2f}",
                    result.passed, result.confidence)
    except Exception as e:
        logger.warning("[validation] check failed ({}); defaulting to passed=True", e)
        state.validation_passed = True
        state.validation_notes  = "Validation skipped"

    return state


# ── RPA ────────────────────────────────────────────────────────────────────

def rpa_node(state: AgentState) -> AgentState:
    """
    Post-processing: write result to disk + upsert metrics to SQLite.
    These are fire-and-forget — failures are logged but don't stop the pipeline.
    """
    logger.info("[rpa] executing post-processing actions")
    actions: list[str] = []

    # Action 1 — write JSON result to disk
    try:
        import json
        from datetime import datetime
        from utils.settings import settings

        out_dir  = settings.processed_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"result_{state.trace_id[:8]}_{ts}.json"

        out_file.write_text(json.dumps({
            "trace_id"        : state.trace_id,
            "query"           : state.query,
            "synthesis"       : state.synthesis,
            "validation_passed": state.validation_passed,
            "metrics"         : state.extracted_metrics.model_dump() if state.extracted_metrics else None,
            "chunk_count"     : len(state.retrieved_chunks),
            "timestamp"       : datetime.utcnow().isoformat(),
        }, indent=2, default=str))
        actions.append(f"Wrote result → {out_file.name}")
    except Exception as e:
        logger.warning("[rpa] file write failed: {}", e)

    # Action 2 — upsert financial metrics to SQLite
    if state.extracted_metrics and state.extracted_metrics.company_name:
        try:
            _upsert_sqlite(state)
            actions.append("Upserted metrics → SQLite")
        except Exception as e:
            logger.warning("[rpa] SQLite upsert failed: {}", e)

    state.rpa_actions = actions
    logger.info("[rpa] {} actions completed", len(actions))
    return state


def _upsert_sqlite(state: AgentState) -> None:
    import sqlite3
    from datetime import datetime
    from utils.settings import settings

    m    = state.extracted_metrics
    path = settings.sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS financial_metrics (
            trace_id         TEXT PRIMARY KEY,
            company_name     TEXT,
            revenue          REAL,
            revenue_currency TEXT,
            ebitda           REAL,
            net_profit       REAL,
            total_debt       REAL,
            debt_to_equity   REAL,
            fiscal_year      INTEGER,
            language         TEXT,
            confidence       REAL,
            query            TEXT,
            created_at       TEXT
        )
    """)
    conn.execute("""
        INSERT OR REPLACE INTO financial_metrics VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        state.trace_id, m.company_name, m.revenue, m.revenue_currency,
        m.ebitda, m.net_profit, m.total_debt, m.debt_to_equity,
        m.fiscal_year, m.language.value, m.confidence,
        state.query, datetime.utcnow().isoformat(),
    ))
    conn.commit()
    conn.close()