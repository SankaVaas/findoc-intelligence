"""
Language detection — identifies the language of a document's text.
Uses langdetect (no model download, rule-based).

Returns Language enum; falls back to Language.UNKNOWN on short/ambiguous text.
"""

from __future__ import annotations

from utils.logging import logger
from utils.models import Language

# Maps langdetect ISO codes → our Language enum
_MAP: dict[str, Language] = {
    "en": Language.EN,
    "fr": Language.FR,
    "de": Language.DE,
    "es": Language.ES,
    "it": Language.IT,
    "nl": Language.NL,
    "pl": Language.PL,
}


def detect_language(text: str, min_chars: int = 60) -> Language:
    """
    Detect the dominant language of `text`.

    Args:
        text      : document text (uses first 3000 chars for speed)
        min_chars : minimum text length to attempt detection

    Returns:
        Language enum value, or Language.UNKNOWN if uncertain.
    """
    snippet = text.strip()
    if len(snippet) < min_chars:
        logger.debug("Text too short for language detection ({} chars)", len(snippet))
        return Language.UNKNOWN

    try:
        from langdetect import detect, LangDetectException
        code = detect(snippet[:3000])
        lang = _MAP.get(code, Language.UNKNOWN)
        logger.debug("Detected language: {} (code='{}')", lang.value, code)
        return lang

    except ImportError:
        logger.warning("langdetect not installed — language will be UNKNOWN. "
                       "Install: pip install langdetect")
        return Language.UNKNOWN

    except Exception as e:
        logger.debug("Language detection failed: {}", e)
        return Language.UNKNOWN