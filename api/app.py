"""
FastAPI backend — REST API for findoc-intelligence.

Endpoints:
  POST /api/ingest          — upload and process a document
  POST /api/query           — full agent pipeline query
  POST /api/query/stream    — streaming query response
  GET  /api/health          — system health
  GET  /api/stats           — chunk count + recent results stats
  GET  /api/results         — list recent query results from SQLite
  GET  /metrics             — Prometheus metrics

Run:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse

from utils.logging import logger, setup_logging
from utils.models  import (
    HealthResponse, QueryRequest, QueryResponse, Language
)
from utils.settings import settings

# ── Global pipeline instance ───────────────────────────────────────────────
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    setup_logging()
    settings.ensure_dirs()
    logger.info("Starting findoc-intelligence API ...")
    from agents.pipeline import AgentPipeline
    pipeline = AgentPipeline()
    logger.info("Pipeline ready — {} chunks in store", pipeline.store.count)
    yield
    logger.info("Shutting down ...")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "findoc-intelligence",
    description = "Multilingual Financial Document Intelligence API",
    version     = "0.1.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
async def health():
    components: dict[str, str] = {}
    try:
        components["chromadb"] = f"ok ({pipeline.store.count} chunks)"
    except Exception as e:
        components["chromadb"] = f"error: {e}"
    try:
        ok = pipeline.llm.health_check()
        components["llm"] = f"ok ({pipeline.llm.model})" if ok else "model not loaded"
    except Exception as e:
        components["llm"] = f"error: {e}"
    components["pipeline"] = "ready" if pipeline else "not initialised"
    status = "ok" if all("error" not in v for v in components.values()) else "degraded"
    return HealthResponse(status=status, version="0.1.0", components=components)


# ── Stats ──────────────────────────────────────────────────────────────────
@app.get("/api/stats")
async def stats():
    """Return high-level system stats for the dashboard."""
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready")

    # Count results from SQLite
    result_count  = 0
    recent_metrics: list[dict] = []
    try:
        import sqlite3, json
        conn = sqlite3.connect(str(settings.sqlite_path))
        conn.row_factory = sqlite3.Row
        result_count = conn.execute(
            "SELECT COUNT(*) FROM financial_metrics"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT * FROM financial_metrics ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        recent_metrics = [dict(r) for r in rows]
        conn.close()
    except Exception:
        pass

    # Count result JSON files
    result_files = 0
    try:
        result_files = len(list((settings.processed_dir / "results").glob("*.json")))
    except Exception:
        pass

    return {
        "chunks_in_store"  : pipeline.store.count,
        "queries_processed": result_count,
        "result_files"     : result_files,
        "llm_model"        : pipeline.llm.model,
        "embed_model"      : pipeline.embedder.model_name,
        "recent_metrics"   : recent_metrics,
    }


# ── Results ────────────────────────────────────────────────────────────────
@app.get("/api/results")
async def get_results(limit: int = 20):
    """Return recent query results from SQLite for the dashboard."""
    try:
        import sqlite3
        conn = sqlite3.connect(str(settings.sqlite_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM financial_metrics ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return {"results": [dict(r) for r in rows]}
    except Exception:
        return {"results": []}


# ── Ingest ─────────────────────────────────────────────────────────────────
@app.post("/api/ingest")
async def ingest(
    file    : UploadFile = File(...),
    language: str        = Form(default="unknown"),
):
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready")

    dest = settings.raw_docs_dir / file.filename
    dest.write_bytes(await file.read())
    logger.info("Saved upload: {}", dest.name)

    start = time.perf_counter()
    try:
        chunk_count = pipeline.ingest(str(dest))
        elapsed     = round((time.perf_counter() - start) * 1000, 1)
        return {
            "status"         : "ok",
            "filename"       : file.filename,
            "chunks_ingested": chunk_count,
            "latency_ms"     : elapsed,
        }
    except Exception as e:
        logger.error("Ingest failed: {}", e)
        raise HTTPException(500, str(e))


# ── Query ──────────────────────────────────────────────────────────────────
@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready")

    start = time.perf_counter()
    try:
        state   = pipeline.run(
            req.query,
            language=req.language.value if req.language else "unknown",
        )
        elapsed = round((time.perf_counter() - start) * 1000, 1)
        return QueryResponse(
            query      = req.query,
            answer     = state.synthesis or "No answer generated.",
            chunks     = state.retrieved_chunks[:req.top_k],
            metrics    = state.extracted_metrics,
            trace_id   = state.trace_id,
            latency_ms = elapsed,
        )
    except Exception as e:
        logger.error("Query failed: {}", e)
        raise HTTPException(500, str(e))


# ── Streaming query ────────────────────────────────────────────────────────
@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """Stream synthesis tokens as they're generated."""
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready")

    async def generate():
        from utils.models import AgentState, Language as Lang
        from agents.nodes import retrieval_node, extraction_node
        from prompts.templates import SYNTHESIS_PROMPT

        state = AgentState(
            query    = req.query,
            language = Lang(req.language.value if req.language else "unknown"),
        )
        state = retrieval_node(state, pipeline.store, pipeline.embedder, pipeline.reranker)
        state = extraction_node(state, pipeline.llm)

        context = "\n\n---\n\n".join(c.chunk.text for c in state.retrieved_chunks[:5])
        metrics_text = ""
        if state.extracted_metrics:
            metrics_text = f"\nExtracted: {state.extracted_metrics.model_dump_json(exclude_none=True)}\n"

        prompt = SYNTHESIS_PROMPT.format(
            query    = req.query,
            context  = context,
            metrics  = metrics_text,
            language = state.language.value,
        )
        async for token in pipeline.llm.astream(prompt):
            yield token

    return StreamingResponse(generate(), media_type="text/plain")


# ── Prometheus metrics ─────────────────────────────────────────────────────
@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host   = settings.api_host,
        port   = settings.api_port,
        reload = settings.api_reload,
    )