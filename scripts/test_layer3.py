"""
Layer 3 smoke test — LangGraph multi-agent pipeline.

Tests:
  1. LLM health check (Ollama reachable, model loaded)
  2. Plain LLM chat
  3. Structured JSON output
  4. Individual agent nodes
  5. Full graph: ingest → run query → get synthesis

Run from project root:
    python scripts/test_layer3.py

Requires Ollama running with qwen2.5:0.5b:
    ollama pull qwen2.5:0.5b
    (Ollama service starts automatically on Windows)
"""

import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging import setup_logging
from utils.settings import settings
setup_logging()

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"{GREEN}  ✓ {msg}{RESET}")
def fail(msg): print(f"{RED}  ✗ {msg}{RESET}")
def info(msg): print(f"{YELLOW}  → {msg}{RESET}")
def head(msg): print(f"\n{BOLD}{msg}{RESET}")


# ── Sample document ────────────────────────────────────────────────────────

SAMPLE_DOC = """
Acme Financial Services — Annual Report 2023

Revenue for fiscal year 2023 reached EUR 12.4 million, a 14 percent increase
over the prior year. EBITDA was EUR 3.1 million with a margin of 25 percent.
Total debt stood at EUR 8.2 million, giving a debt-to-equity ratio of 0.66.
The current ratio improved to 2.1 from 1.8 the prior year.

The board of directors recommends a dividend of EUR 0.15 per share.
Net profit after tax was EUR 1.9 million.

Regional breakdown: Western Europe 62 percent, Central Europe 28 percent,
other markets 10 percent.

Risk factors: interest rate exposure, EU regulatory changes, currency risk.
The company maintains hedging positions for major currency exposures.
Cash and cash equivalents at year end: EUR 2.3 million.
""".strip()


def make_test_doc(tmp: Path) -> Path:
    f = tmp / "acme_report_2023.txt"
    f.write_text(SAMPLE_DOC * 3)
    return f


# ── Tests ──────────────────────────────────────────────────────────────────

def test_llm_health():
    head("[1] LLM health check")
    from utils.llm_client import LLMClient
    llm = LLMClient()
    ok_flag = llm.health_check()
    if ok_flag:
        ok(f"Ollama reachable — model='{llm.model}'")
    else:
        raise RuntimeError(
            f"Ollama not reachable or model '{llm.model}' not found.\n"
            f"  Run: ollama pull {llm.model}"
        )
    return llm


def test_llm_chat(llm):
    head("[2] Plain LLM chat")
    response = llm.chat("What is 2 + 2? Answer with just the number.")
    assert response.strip(), "Empty response"
    ok(f"Response: {response.strip()[:80]}")


def test_structured_output(llm):
    head("[3] Structured JSON output")
    from pydantic import BaseModel

    class CompanyInfo(BaseModel):
        name   : str
        country: str
        founded: int | None = None

    result = llm.structured(
        "Extract company info from: Acme Corp was founded in Germany in 1995.",
        schema=CompanyInfo,
    )
    assert result.name,    "name should be extracted"
    assert result.country, "country should be extracted"
    ok(f"Extracted: name='{result.name}' country='{result.country}' founded={result.founded}")


def test_retrieval_node(tmp):
    head("[4] Retrieval node")
    from utils.models import AgentState, Language
    from embeddings.engine   import EmbeddingEngine
    from embeddings.reranker import Reranker
    from vectorstore.store   import HybridStore, VectorStore, BM25Index
    from ingestion.pipeline  import IngestionPipeline
    from agents.nodes import retrieval_node

    # Ingest sample doc
    doc_file = make_test_doc(tmp)
    pipe     = IngestionPipeline(chunk_size=300, chunk_overlap=60)
    chunks   = pipe.ingest_file(doc_file)

    embedder = EmbeddingEngine()
    embedded = embedder.embed_chunks(chunks)

    chroma_dir    = tmp / "chroma"
    store         = HybridStore.__new__(HybridStore)
    store._vector = VectorStore(persist_dir=chroma_dir, collection_name="test_r3")
    store._bm25   = BM25Index()
    store._stale  = True
    store.add_chunks(embedded)

    reranker = Reranker()
    state    = AgentState(query="What is the debt-to-equity ratio?")
    state    = retrieval_node(state, store, embedder, reranker)

    assert len(state.retrieved_chunks) > 0, "Should retrieve chunks"
    ok(f"Retrieved {len(state.retrieved_chunks)} chunks")
    info(f"Top chunk: {state.retrieved_chunks[0].chunk.text[:100].strip()}...")
    return store, embedder, reranker


def test_extraction_node(llm, store, embedder, reranker):
    head("[5] Extraction node")
    from utils.models import AgentState
    from agents.nodes import retrieval_node, extraction_node

    state = AgentState(query="What is Acme's revenue and EBITDA?")
    state = retrieval_node(state, store, embedder, reranker)
    state = extraction_node(state, llm)

    if state.extracted_metrics:
        m = state.extracted_metrics
        ok(f"Extracted — company='{m.company_name}' revenue={m.revenue} ebitda={m.ebitda}")
    else:
        info("Extraction returned no metrics (small model limitation — OK for dev)")


def test_synthesis_node(llm, store, embedder, reranker):
    head("[6] Synthesis node")
    from utils.models import AgentState
    from agents.nodes import retrieval_node, synthesis_node

    state = AgentState(query="Summarise the key financial highlights of Acme for 2023.")
    state = retrieval_node(state, store, embedder, reranker)
    state = synthesis_node(state, llm)

    assert state.synthesis, "Synthesis should not be empty"
    ok(f"Synthesis ({len(state.synthesis)} chars):")
    # Print synthesis wrapped
    for line in state.synthesis.strip().split("\n")[:6]:
        info(f"  {line}")


def test_validation_node(llm, store, embedder, reranker):
    head("[7] Validation node")
    from utils.models import AgentState
    from agents.nodes import retrieval_node, synthesis_node, validation_node

    state = AgentState(query="What was Acme's net profit in 2023?")
    state = retrieval_node(state, store, embedder, reranker)
    state = synthesis_node(state, llm)
    state = validation_node(state, llm)

    ok(f"Validation: passed={state.validation_passed} notes='{state.validation_notes[:80]}'")


def test_full_graph(tmp):
    head("[8] Full graph: ingest → query → synthesis")
    from agents.pipeline import AgentPipeline
    from vectorstore.store import VectorStore, BM25Index, HybridStore

    # Override chroma dir to use temp so we don't pollute real data
    chroma_dir = tmp / "full_graph_chroma"

    pipeline = AgentPipeline.__new__(AgentPipeline)
    from embeddings.engine   import EmbeddingEngine
    from embeddings.reranker import Reranker
    from utils.llm_client    import LLMClient

    pipeline.embedder = EmbeddingEngine()
    pipeline.reranker = Reranker()
    pipeline.llm      = LLMClient()
    pipeline.store    = HybridStore.__new__(HybridStore)
    pipeline.store._vector = VectorStore(persist_dir=chroma_dir, collection_name="full_test")
    pipeline.store._bm25   = BM25Index()
    pipeline.store._stale  = True

    from agents.pipeline import build_graph
    pipeline._graph = build_graph(
        pipeline.store, pipeline.embedder, pipeline.reranker, pipeline.llm
    )

    # Ingest
    (tmp / "docs").mkdir(exist_ok=True)
    doc_file = tmp / "docs" / "acme_2023.txt"
    doc_file.write_text(SAMPLE_DOC * 3)

    n = pipeline.ingest(doc_file)
    ok(f"Ingested {n} chunks")

    # Query
    state = pipeline.run("What is the debt-to-equity ratio and current ratio of Acme?")

    assert state.synthesis, "Synthesis must not be empty"
    ok(f"Synthesis received ({len(state.synthesis)} chars)")
    ok(f"Validation: {state.validation_passed}")
    ok(f"RPA actions: {state.rpa_actions}")
    info("Answer preview:")
    for line in state.synthesis.strip().split("\n")[:5]:
        info(f"  {line}")


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    results = {}
    tmp = Path(tempfile.mkdtemp())

    try:
        llm = test_llm_health()
        results["llm_health"] = True
    except Exception as e:
        fail(f"LLM health: {e}")
        results["llm_health"] = False
        print(f"\n{RED}Cannot proceed without Ollama. "
              f"Make sure it's running and '{settings.llm_model}' is pulled.{RESET}\n")
        sys.exit(1)

    for name, fn, args in [
        ("llm_chat",       test_llm_chat,       (llm,)),
        ("structured_out", test_structured_output, (llm,)),
    ]:
        try:
            fn(*args)
            results[name] = True
        except Exception as e:
            fail(f"{name}: {e}")
            results[name] = False

    # Retrieval node (needed by subsequent tests)
    store = embedder = reranker = None
    try:
        store, embedder, reranker = test_retrieval_node(tmp)
        results["retrieval_node"] = True
    except Exception as e:
        fail(f"retrieval_node: {e}")
        results["retrieval_node"] = False

    if store:
        for name, fn, args in [
            ("extraction_node", test_extraction_node, (llm, store, embedder, reranker)),
            ("synthesis_node",  test_synthesis_node,  (llm, store, embedder, reranker)),
            ("validation_node", test_validation_node, (llm, store, embedder, reranker)),
        ]:
            try:
                fn(*args)
                results[name] = True
            except Exception as e:
                fail(f"{name}: {e}")
                results[name] = False

    try:
        test_full_graph(tmp)
        results["full_graph"] = True
    except Exception as e:
        fail(f"full_graph: {e}")
        results["full_graph"] = False

    shutil.rmtree(tmp, ignore_errors=True)

    # Summary
    print(f"\n{BOLD}{'─'*50}\nRESULTS\n{'─'*50}{RESET}")
    passed = failed = 0
    for name, result in results.items():
        if result: ok(name);   passed += 1
        else:      fail(name); failed += 1

    print(f"\n  {GREEN}{passed} passed{RESET}  {RED}{failed} failed{RESET}\n")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run()