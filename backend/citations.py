"""Programmatic, source-grounded citations for DOCMIND-AI.

Citations are built here in Python from the ACTUAL retrieved ``Document``
objects that were placed in the generation context (the final Top-5 from the
hybrid_rerank pipeline). The LLM never produces citation metadata itself, so it
cannot invent a source, page, or row: a citation exists only if the retrieved
chunk's metadata contains it.

Format (document-level, deliberately simple -- no claim-level tracking):

    PDF   ->  [Source: filename.pdf, Page: 5]
    CSV   ->  [Source: filename.csv, Row: 10]
    other ->  [Source: filename.txt]         (no page/row available)

Rules enforced here:
* Only the Documents passed in (the final generation context) can be cited.
* ``page`` is emitted only when it is a real, positive page number. PDFs carry
  1-indexed ints; DOCX/TXT/CSV carry the sentinel ``"N/A"`` -> page omitted.
* ``row`` is emitted only for CSV chunks (``CSVLoader`` sets an int ``row``);
  it is ``None`` for every other format -> row omitted.
* Missing/blank ``source`` -> that Document contributes no citation (never a
  fabricated placeholder).
* Duplicate (source, page, row) citations are collapsed, preserving first-seen
  order so the list mirrors retrieval ranking.
"""

from __future__ import annotations

from typing import Any

from backend.logging_config import get_logger

logger = get_logger(__name__)

# Shown when retrieval produced nothing to ground an answer in. The prompt also
# instructs the model to say the information is unavailable; this constant is the
# app-side fallback and the string the tests assert against.
INSUFFICIENT_CONTEXT_MESSAGE = (
    "I could not find this in the uploaded documents."
)


def _clean_source(value: Any) -> str | None:
    """Return a non-empty source filename, or None if there isn't one."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _valid_page(value: Any) -> int | None:
    """Return a real 1-indexed page number, or None.

    PDFs store an int page (already 1-indexed upstream). DOCX/TXT/CSV store the
    sentinel ``"N/A"``. Anything that is not a positive int is treated as 'no
    page' and omitted rather than guessed.
    """
    if isinstance(value, bool):  # bool is an int subclass; never a page
        return None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _valid_row(value: Any) -> int | None:
    """Return a CSV row index (>= 0), or None.

    ``CSVLoader`` sets a 0-based int ``row``; every other loader leaves it unset
    (None). We accept 0 as a legitimate first-row citation.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def format_citation(metadata: dict) -> str | None:
    """Build one citation string from a single Document's metadata.

    Returns None when there is no usable source, so callers can simply skip it.
    Never fabricates missing fields.
    """
    if not isinstance(metadata, dict):
        return None
    source = _clean_source(metadata.get("source"))
    if source is None:
        return None

    page = _valid_page(metadata.get("page"))
    row = _valid_row(metadata.get("row"))

    if page is not None:
        return f"[Source: {source}, Page: {page}]"
    if row is not None:
        return f"[Source: {source}, Row: {row}]"
    # Source known but no page/row available -> cite the source alone.
    return f"[Source: {source}]"


def build_citations(documents: list) -> list[str]:
    """Return de-duplicated, order-preserving citation strings for the docs.

    ``documents`` must be the ACTUAL final generation context (the Top-5
    ``Document`` objects). Only these can be cited. Each item may be a
    LangChain ``Document`` (has ``.metadata``) or a plain mapping with a
    ``metadata`` key -- both are handled so the function is easy to test.
    """
    citations: list[str] = []
    seen: set[str] = set()
    for doc in documents or []:
        metadata = getattr(doc, "metadata", None)
        if metadata is None and isinstance(doc, dict):
            metadata = doc.get("metadata")
        citation = format_citation(metadata or {})
        if citation and citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations


def build_citation_block(documents: list) -> str:
    """Render the citations as a trailing block to append after the answer.

    Empty string when there is nothing to cite, so appending is always safe.
    Logs only the COUNT of citations -- never source names or document content.
    """
    citations = build_citations(documents)
    if not citations:
        logger.info("No citable metadata in the generation context")
        return ""
    logger.info("Attached %d source citation(s)", len(citations))
    lines = "\n".join(f"- {citation}" for citation in citations)
    return f"\n\n**Sources**\n{lines}"


def is_insufficient_answer(answer: str) -> bool:
    """Heuristic: did the model report the answer isn't in the documents?

    Used to SKIP appending citations to a not-found answer (citing sources under
    an 'I couldn't find it' reply would be contradictory). Deliberately
    conservative -- matches only clear not-found phrasings.
    """
    if not answer:
        return True
    normalized = answer.lower()
    markers = (
        "could not find",
        "couldn't find",
        "not in the uploaded",
        "not available in the provided",
        "not available in the uploaded",
        "no information in the provided",
        "does not contain",
        "isn't in the",
        "is not in the",
    )
    return any(marker in normalized for marker in markers)
