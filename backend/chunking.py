"""Retrieval-oriented chunking for DocMind AI.

Chunking lives in its own module so the strategy can be tuned without touching
ingestion (the loaders in ``backend.retriever``), the embedding model, or the
FAISS vector store. ``backend.retriever.build_vectorstore`` is the only caller.

Contract with the rest of the pipeline
--------------------------------------
in : ``list[Document]`` exactly as produced by ``retriever.load_document()``
     (PDF -> one Document per page, DOCX/TXT -> one Document per file,
     CSV -> one Document per row).
out: ``list[Document]`` ready for ``FAISS.from_documents``. Chunk metadata is a
     deep copy of the parent Document's metadata (``source`` / ``page`` /
     ``row``). No metadata key is added, removed, or rewritten here.

Why the strategy changed
------------------------
1. The previous configuration used the splitter's *default* separator ladder
   ``["\\n\\n", "\\n", " ", ""]``, which has no sentence boundary. Pages
   extracted from PDFs are hard-wrapped at layout width, so a lone "\\n" is
   usually a mid-sentence line wrap rather than a real boundary. A sentence
   separator is now tried before the line separator.
2. ``chunk_size`` was 1000 characters. The embedding model configured in
   ``backend.retriever`` (``all-MiniLM-L6-v2``) encodes at most 256 word-piece
   tokens and silently truncates beyond that; ~1000 characters of technical
   English lands at or past that limit, so the tail of a chunk could be stored
   and shown to the LLM while never influencing the vector it was retrieved by.
   700 characters keeps whole chunks inside the encoder window.
3. ``CSVLoader`` emits one Document per row (~60 characters for a numeric CSV),
   which produced hundreds of near-identical micro-chunks. Consecutive rows of
   the same table are now packed up to ``chunk_size`` before splitting.
"""

from __future__ import annotations

import copy

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Parameters --------------------------------------------------------------
# Characters, not tokens: the splitter measures length with len().
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

# Boundary ladder, widest to narrowest. The splitter walks this list and uses
# the first separator present in the text, recursing with the remainder of the
# ladder whenever a piece is still larger than CHUNK_SIZE.
#
# Every entry is treated as a regular expression (is_separator_regex=True).
# The sentence pattern uses a look-behind so the terminating ".", "!" or "?"
# stays attached to the sentence it ends, and only the whitespace after it is
# consumed as the separator. ``[^\W\d]`` requires a letter before the period,
# which keeps numbered list markers ("1. ", "12. ") and decimals ("3.5") from
# being mistaken for sentence ends -- for a numbered Q&A document that moves the
# boundary to just *before* the next number, which is where it belongs.
SEPARATORS = [
    r"\n\n",                   # blank line -> paragraph / block break
    r"(?<=[^\W\d][.!?])\s+",   # sentence end (\s+ also covers a trailing "\n")
    r"\n",                     # single newline -> line / layout wrap
    r" ",                      # word
    r"",                       # hard character fallback
]

# Rows packed into one chunk are separated by a blank line so that the ladder
# above can still split a packed group at row boundaries if it ever needs to.
ROW_JOINER = "\n\n"


def build_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Return the configured splitter used by the production pipeline."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        is_separator_regex=True,
        keep_separator=True,
    )


def _is_row_document(document: Document) -> bool:
    """True for row-per-Document loader output (CSVLoader sets ``row``)."""
    return "row" in document.metadata


def _same_table(first: Document, second: Document) -> bool:
    """True when two row Documents came from the same file and page."""
    return (
        first.metadata.get("source") == second.metadata.get("source")
        and first.metadata.get("page") == second.metadata.get("page")
    )


def pack_row_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
) -> list[Document]:
    """Merge consecutive row Documents of the same table up to ``chunk_size``.

    Non-row Documents (PDF pages, DOCX/TXT files) pass through untouched and in
    order. A packed group inherits a deep copy of its *first* row's metadata, so
    ``source`` and ``page`` stay correct and ``row`` marks where the group
    starts. A single row larger than ``chunk_size`` is left alone for the text
    splitter to handle normally.
    """
    packed: list[Document] = []
    group: list[Document] = []
    group_length = 0

    def flush() -> None:
        nonlocal group, group_length
        if not group:
            return
        packed.append(
            Document(
                page_content=ROW_JOINER.join(document.page_content for document in group),
                metadata=copy.deepcopy(group[0].metadata),
            )
        )
        group = []
        group_length = 0

    for document in documents:
        if not _is_row_document(document):
            flush()
            packed.append(document)
            continue

        if group and not _same_table(group[-1], document):
            flush()

        addition = len(document.page_content) + (len(ROW_JOINER) if group else 0)
        if group and group_length + addition > chunk_size:
            flush()
            addition = len(document.page_content)

        group.append(document)
        group_length += addition

    flush()
    return packed


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Turn loaded Documents into retrieval chunks, preserving their metadata."""
    packed = pack_row_documents(documents, chunk_size=chunk_size)
    splitter = build_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(packed)
