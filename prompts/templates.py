"""
Prompt templates for all agents.
Kept here so prompts are easy to iterate without touching agent logic.
Designed to work with small models (qwen2.5:0.5b) as well as large ones.
"""

# ── Extraction agent ───────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are a financial data extraction specialist.
Extract structured financial information from the document excerpts below.

USER QUERY: {query}

DOCUMENT EXCERPTS:
{context}

Extract all financial metrics present. Omit fields not mentioned.
Be precise with numbers. Include currency when stated.
Set confidence 0.0-1.0 based on how clearly data appears in text.
"""

# ── Synthesis agent ────────────────────────────────────────────────────────

SYNTHESIS_PROMPT = """You are a senior financial analyst assistant.
Answer the user query using ONLY the document excerpts provided.
Do not invent or assume information not present in the excerpts.
If the answer cannot be found, say so clearly.
Be concise: 3-5 sentences maximum.

USER QUERY: {query}

{metrics}

DOCUMENT EXCERPTS:
{context}
"""

# ── Validation agent ───────────────────────────────────────────────────────

VALIDATION_PROMPT = """You are a fact-checker for financial documents.
Check if the generated answer is supported by the source excerpts.

USER QUERY: {query}

GENERATED ANSWER:
{synthesis}

SOURCE EXCERPTS:
{context}

Check for:
1. Claims not in the source (hallucinations)
2. Wrong numbers or names
3. Misleading omissions

Respond with JSON only:
{{"passed": true/false, "confidence": 0.0-1.0, "issues": [], "notes": ""}}
"""

# ── Orchestrator ───────────────────────────────────────────────────────────

ORCHESTRATOR_PROMPT = """Classify this financial query.

Query: {query}

Respond with JSON only:
{{"query_type": "financial_extraction|general_qa|summarisation", "language": "en|fr|de|es|it|unknown", "requires_extraction": true/false}}
"""