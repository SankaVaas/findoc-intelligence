"""
Shared Pydantic models used across ingestion, RAG, agents, and API layers.
Keeping all schemas here prevents circular imports and keeps the contract explicit.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────

class DocType(str, Enum):
    PDF         = "pdf"
    DOCX        = "docx"
    HTML        = "html"
    TXT         = "txt"
    AUDIO       = "audio"
    UNKNOWN     = "unknown"


class Language(str, Enum):
    EN = "en"
    FR = "fr"
    DE = "de"
    ES = "es"
    IT = "it"
    NL = "nl"
    PL = "pl"
    UNKNOWN = "unknown"


class ChunkType(str, Enum):
    TEXT    = "text"
    TABLE   = "table"
    HEADING = "heading"
    CAPTION = "caption"


class AgentName(str, Enum):
    ORCHESTRATOR = "orchestrator"
    RETRIEVAL    = "retrieval"
    EXTRACTION   = "extraction"
    SYNTHESIS    = "synthesis"
    VALIDATION   = "validation"
    RPA          = "rpa"


# ── Document models ────────────────────────────────────────────────────────

class RawDocument(BaseModel):
    """Represents a file immediately after ingestion, before any processing."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_path: Path
    doc_type: DocType
    language: Language = Language.UNKNOWN
    file_size_bytes: int = 0
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """A single chunk ready for embedding and vector storage."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str                    # parent RawDocument.id
    chunk_index: int
    chunk_type: ChunkType = ChunkType.TEXT
    text: str
    page_number: int | None = None
    language: Language = Language.UNKNOWN
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddedChunk(DocumentChunk):
    """A chunk with its embedding vector attached."""
    embedding: list[float] = Field(default_factory=list)
    embed_model: str = ""


# ── Retrieval models ───────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """A chunk returned from hybrid search, with scores."""
    chunk: DocumentChunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        if self.rerank_score is not None:
            return self.rerank_score
        return (self.vector_score + self.bm25_score) / 2


# ── Extraction models (financial domain) ──────────────────────────────────

class FinancialEntity(BaseModel):
    """Named entity extracted from a financial document."""
    text: str
    label: str           # ORG, MONEY, DATE, PERCENT, RATIO, etc.
    confidence: float = 0.0
    source_chunk_id: str = ""


class FinancialMetrics(BaseModel):
    """Structured financial data extracted from loan / underwriting docs."""
    company_name: str | None = None
    revenue: float | None = None
    revenue_currency: str | None = None
    ebitda: float | None = None
    net_profit: float | None = None
    total_debt: float | None = None
    equity: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    fiscal_year: int | None = None
    language: Language = Language.UNKNOWN
    raw_entities: list[FinancialEntity] = Field(default_factory=list)
    confidence: float = 0.0
    extraction_notes: str = ""


# ── Agent models ───────────────────────────────────────────────────────────

class AgentMessage(BaseModel):
    """Message passed between agents in the LangGraph graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: AgentName
    to_agent: AgentName | None = None   # None = broadcast / orchestrator decides
    content: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentState(BaseModel):
    """
    Shared mutable state passed through the LangGraph StateGraph.
    Each agent reads from and writes to this object.
    """
    query: str
    language: Language = Language.UNKNOWN
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    extracted_metrics: FinancialMetrics | None = None
    synthesis: str = ""
    validation_passed: bool | None = None
    validation_notes: str = ""
    rpa_actions: list[str] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)
    error: str | None = None
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# ── API request/response models ───────────────────────────────────────────

class IngestRequest(BaseModel):
    source_path: str
    language_hint: Language | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    language: Language | None = None
    rerank: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    chunks: list[RetrievedChunk]
    metrics: FinancialMetrics | None = None
    trace_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict[str, str]
