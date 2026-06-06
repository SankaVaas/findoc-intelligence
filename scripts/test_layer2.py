"""
Layer 2 smoke test — embeddings + ChromaDB + hybrid search.

What it tests:
  1. Embedding engine loads and produces correct-dimension vectors
  2. Chunks are embedded correctly (passage prefix)
  3. Query is embedded correctly (query prefix)
  4. ChromaDB stores and retrieves chunks
  5. BM25 index builds and searches
  6. Hybrid search (RRF fusion) works end-to-end
  7. Reranker rescores results
  8. Full pipeline: ingest file → embed → store → search

Run from project root:
    python scripts/test_layer2.py

NOTE: First run downloads multilingual-e5-large (~1.1GB) and
      bge-reranker-base (~280MB) from HuggingFace.
      Subsequent runs use the local cache.
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


# ── Sample chunks ──────────────────────────────────────────────────────────

def make_sample_chunks():
    """Create realistic financial document chunks for testing."""
    from utils.models import DocumentChunk, ChunkType, Language, RawDocument, DocType
    from ingestion.chunker import DocumentChunker

    doc_text = """
    Acme Financial Services Annual Report 2023.
    Revenue for fiscal year 2023 reached EUR 12.4 million, a 14 percent increase
    over the prior year. EBITDA was EUR 3.1 million with a margin of 25 percent.
    Total debt stood at EUR 8.2 million, yielding a debt-to-equity ratio of 0.66.
    The current ratio improved to 2.1 from 1.8 in the prior year.

    Regional performance: Western Europe contributed 62 percent of total revenue,
    Central and Eastern Europe 28 percent, and other markets 10 percent.
    The company expanded into three new markets during 2023: Poland, Czech Republic,
    and Romania.

    Risk factors include exposure to interest rate fluctuations, evolving EU regulatory
    requirements, and currency risk from non-EUR denominated contracts.
    The company maintains adequate hedging positions for its major currency exposures.

    Meridian Logistics GmbH reported operating profit of EUR 1.8 million.
    Net profit after tax was EUR 1.2 million. The board recommends a dividend
    of EUR 0.15 per share for fiscal year 2023.
    """.strip()

    from utils.models import RawDocument, DocType
    doc     = RawDocument(source_path=Path("test_report.txt"), doc_type=DocType.TXT)
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks  = chunker.chunk(doc, doc_text)
    return chunks


# ── Tests ──────────────────────────────────────────────────────────────────

def test_embedding_engine():
    head("[1] Embedding engine")
    from embeddings.engine import EmbeddingEngine

    engine = EmbeddingEngine()
    dim    = engine.dimension
    ok(f"Model loaded — dim={dim}")
    assert dim > 0, "Dimension must be positive"

    # Query embedding
    qvec = engine.embed_query("What is the revenue of Acme?")
    assert len(qvec) == dim, f"Query vec dim mismatch: {len(qvec)} != {dim}"
    ok(f"Query vector — dim={len(qvec)}, norm≈{sum(x**2 for x in qvec)**0.5:.4f}")

    # Passage embedding
    chunks   = make_sample_chunks()
    embedded = engine.embed_chunks(chunks)
    assert len(embedded) == len(chunks)
    assert all(len(e.embedding) == dim for e in embedded)
    assert all(e.embed_model for e in embedded)
    ok(f"Passage embedding — {len(embedded)} chunks, each dim={dim}")

    return engine, embedded


def test_vector_store(embedded):
    head("[2] ChromaDB vector store")

    # Use a temp dir so tests don't pollute real data
    tmp = Path(tempfile.mkdtemp()) / "test_chroma"

    from vectorstore.store import VectorStore
    from embeddings.engine import EmbeddingEngine

    store  = VectorStore(persist_dir=tmp, collection_name="test")
    engine = EmbeddingEngine()

    # Add
    store.add(embedded)
    assert store.count == len(embedded)
    ok(f"Stored {store.count} chunks in ChromaDB")

    # Search
    qvec    = engine.embed_query("revenue EBITDA margin")
    results = store.search(qvec, top_k=3)
    assert len(results) > 0
    assert all(0 <= score <= 1 for _, score in results)
    ok(f"Vector search returned {len(results)} results")
    for chunk, score in results:
        info(f"  score={score:.4f} | {chunk.text[:80].strip()}...")

    # get_all
    all_chunks = store.get_all()
    assert len(all_chunks) == len(embedded)
    ok(f"get_all() returned {len(all_chunks)} chunks")

    shutil.rmtree(tmp, ignore_errors=True)
    return results


def test_bm25(chunks):
    head("[3] BM25 lexical index")
    from vectorstore.store import BM25Index

    idx = BM25Index()
    idx.build(chunks)
    assert idx.size == len(chunks)
    ok(f"BM25 index built with {idx.size} documents")

    results = idx.search("debt equity ratio", top_k=3)
    assert len(results) > 0
    ok(f"BM25 search returned {len(results)} results")
    for chunk, score in results:
        info(f"  score={score:.4f} | {chunk.text[:80].strip()}...")


def test_hybrid_store(embedded):
    head("[4] Hybrid store (RRF fusion)")

    tmp = Path(tempfile.mkdtemp()) / "test_hybrid"

    from vectorstore.store import HybridStore, VectorStore, BM25Index
    from embeddings.engine import EmbeddingEngine

    # Manually wire up with temp dir
    store         = HybridStore.__new__(HybridStore)
    store._vector = VectorStore(persist_dir=tmp, collection_name="hybrid_test")
    store._bm25   = BM25Index()
    store._stale  = True

    engine = EmbeddingEngine()
    store.add_chunks(embedded)
    ok(f"Added {store.count} chunks to HybridStore")

    query    = "What is the debt-to-equity ratio?"
    qvec     = engine.embed_query(query)
    results  = store.search(query, qvec, top_k=5)

    assert len(results) > 0
    ok(f"Hybrid search returned {len(results)} results")
    for r in results:
        info(
            f"  vector={r.vector_score:.3f}  bm25={r.bm25_score:.3f}"
            f"  final={r.final_score:.3f} | {r.chunk.text[:70].strip()}..."
        )

    shutil.rmtree(tmp, ignore_errors=True)
    return results


def test_reranker(query, results):
    head("[5] Reranker (bge-reranker-base)")
    from embeddings.reranker import Reranker

    reranker = Reranker()
    reranked = reranker.rerank(query, results, top_k=3)

    assert len(reranked) <= 3
    assert all(r.rerank_score is not None for r in reranked)
    ok(f"Reranked to top {len(reranked)} results")
    for r in reranked:
        info(
            f"  rerank={r.rerank_score:.4f} | {r.chunk.text[:70].strip()}..."
        )


def test_full_pipeline():
    head("[6] Full pipeline: file → embed → store → search")
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())
    doc_file = tmp_dir / "loan_report.txt"
    doc_file.write_text(
        "Borrower: Nova Technologies SRL\n"
        "Loan application for EUR 2 million working capital facility.\n"
        "Revenue FY2023: EUR 5.4 million (growth 22% YoY).\n"
        "EBITDA: EUR 0.9 million. Net debt: EUR 1.1 million.\n"
        "Debt service coverage ratio: 1.6x. Current ratio: 1.9.\n"
        "Primary business: IT services and software development.\n"
        "Operating in Romania, Bulgaria, and Serbia.\n"
        "No significant litigation or regulatory issues reported.\n"
        * 5
    )

    chroma_dir = tmp_dir / "chroma"

    from ingestion.pipeline import IngestionPipeline
    from embeddings.engine  import EmbeddingEngine
    from vectorstore.store  import VectorStore, BM25Index, HybridStore, _rrf

    # Ingest
    pipe   = IngestionPipeline(chunk_size=300, chunk_overlap=60)
    chunks = pipe.ingest_file(doc_file)
    ok(f"Ingested {len(chunks)} chunks from file")

    # Embed
    engine   = EmbeddingEngine()
    embedded = engine.embed_chunks(chunks)
    ok(f"Embedded {len(embedded)} chunks")

    # Store
    store         = HybridStore.__new__(HybridStore)
    store._vector = VectorStore(persist_dir=chroma_dir, collection_name="pipeline_test")
    store._bm25   = BM25Index()
    store._stale  = True
    store.add_chunks(embedded)
    ok(f"Stored {store.count} chunks in ChromaDB")

    # Search
    query   = "What is the debt service coverage ratio?"
    qvec    = engine.embed_query(query)
    results = store.search(query, qvec, top_k=5)
    ok(f"Search returned {len(results)} results")

    # Verify relevant result in top-3
    top_texts = " ".join(r.chunk.text for r in results[:3]).lower()
    assert "debt" in top_texts or "coverage" in top_texts, \
        "Expected debt-related content in top results"
    ok("Relevant content found in top-3 results ✓")

    info(f"Top result: {results[0].chunk.text[:120].strip()}...")

    shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Main ───────────────────────────────────────────────────────────────────

def run():
    results_map = {}

    try:
        engine, embedded = test_embedding_engine()
        results_map["embedding_engine"] = True
    except Exception as e:
        fail(f"Embedding engine: {e}")
        results_map["embedding_engine"] = False
        embedded = None

    if embedded:
        try:
            test_vector_store(embedded)
            results_map["chromadb"] = True
        except Exception as e:
            fail(f"ChromaDB: {e}")
            results_map["chromadb"] = False

        chunks = make_sample_chunks()
        try:
            test_bm25(chunks)
            results_map["bm25"] = True
        except Exception as e:
            fail(f"BM25: {e}")
            results_map["bm25"] = False

        try:
            hybrid_results = test_hybrid_store(embedded)
            results_map["hybrid_search"] = True
        except Exception as e:
            fail(f"Hybrid store: {e}")
            results_map["hybrid_search"] = False
            hybrid_results = []

        if hybrid_results:
            try:
                test_reranker("debt equity ratio", hybrid_results)
                results_map["reranker"] = True
            except Exception as e:
                fail(f"Reranker: {e}")
                results_map["reranker"] = False

    try:
        test_full_pipeline()
        results_map["full_pipeline"] = True
    except Exception as e:
        fail(f"Full pipeline: {e}")
        results_map["full_pipeline"] = False

    # Summary
    print(f"\n{BOLD}{'─'*50}\nRESULTS\n{'─'*50}{RESET}")
    passed = failed = 0
    for name, result in results_map.items():
        if result:
            ok(name); passed += 1
        else:
            fail(name); failed += 1

    print(f"\n  {GREEN}{passed} passed{RESET}  {RED}{failed} failed{RESET}\n")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run()