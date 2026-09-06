"""Chunking test for DocMind AI (backend/chunking.py).

Run from the PROJECT ROOT:

    python tests/test_chunking.py                  # chunking only
    python tests/test_chunking.py --with-embeddings # + real embeddings / FAISS

Covers all four ingested formats. PDF and CSV inputs are taken from data/pdfs;
the DOCX and TXT fixtures are generated into a temporary directory and deleted
afterwards, so nothing is added to the app's document folder or its index.

Ingestion is the real thing: load_document() is imported from backend.retriever,
which means the embedding model configured there is constructed at import time.
That is expected -- this script never modifies the embedding model, the vector
store, or the retrieval strategy. The --with-embeddings flag only *reads* from
them to confirm the chunks the new strategy produces are still acceptable input.

Exit code is 0 when every invariant holds, 1 otherwise.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.chunking import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    pack_row_documents,
    split_documents,
)
from backend.retriever import PDF_DIR, load_document  # noqa: E402

PREVIEW_CHARS = 300

TXT_FIXTURE = """Retrieval Augmented Generation Field Notes

Retrieval quality sets the ceiling for a RAG system. A generator can only reason
over the passages it is handed, so a chunk that is cut in the middle of a
definition costs an answer even when the underlying document contains it. This is
why boundary selection matters more than raw chunk count.

Chunk size interacts with the embedding model. A bi-encoder has a fixed input
window, and any text beyond that window is truncated before pooling. Truncated
text still reaches the language model as context, which hides the problem: the
passage looks complete in the answer, yet the tail of it never contributed to the
vector that made the passage retrievable in the first place.

Overlap exists to protect facts that straddle a boundary. One sentence of
carryover is usually enough. Much larger overlaps inflate the index and return
near duplicate neighbours for the same query.
"""

DOCX_PARAGRAPHS = [
    "DocMind AI Ingestion Notes",
    "The ingestion layer accepts four formats. PDF files are read page by page, so "
    "each page arrives as its own document and keeps a one indexed page number for "
    "citation. Word documents arrive as a single document because the extractor "
    "concatenates every paragraph into one string.",
    "Because a DOCX file arrives as one long string, chunking is the only thing that "
    "gives it structure. The extractor separates paragraphs with a blank line, which "
    "means the paragraph separator in the boundary ladder lines up exactly with the "
    "author's own paragraph breaks.",
    "Plain text files behave the same way. Comma separated files are the exception: "
    "the loader emits one document per row, so the chunking layer packs consecutive "
    "rows back together before splitting.",
    "Metadata must survive the split. Every chunk carries a deep copy of its parent "
    "document's metadata, so the source filename and the page number used by the "
    "citation prompt are never lost.",
]

_DOCX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""

_DOCX_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def write_minimal_docx(path: Path, paragraphs: list[str]) -> None:
    """Write a small but valid .docx (readable by Word and by docx2txt)."""
    body = "".join(
        '<w:p><w:r><w:t xml:space="preserve">'
        + paragraph.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</w:t></w:r></w:p>"
        for paragraph in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _DOCX_CONTENT_TYPES)
        archive.writestr("_rels/.rels", _DOCX_RELS)
        archive.writestr("word/document.xml", document_xml)


def check_invariants(label: str, documents: list, chunks: list) -> list[str]:
    """Return a list of invariant violations (empty list means everything held)."""
    failures = []
    packed = pack_row_documents(documents, chunk_size=CHUNK_SIZE)
    parent_key_sets = {frozenset(document.metadata) for document in documents}

    if any(not chunk.page_content.strip() for chunk in chunks):
        failures.append("produced an empty / whitespace-only chunk")

    oversized = [len(chunk.page_content) for chunk in chunks if len(chunk.page_content) > CHUNK_SIZE]
    if oversized:
        failures.append(f"{len(oversized)} chunk(s) exceed CHUNK_SIZE (max {max(oversized)})")

    if any(frozenset(chunk.metadata) not in parent_key_sets for chunk in chunks):
        failures.append("chunk metadata key set differs from the loader's")

    for chunk in chunks:
        parents = [
            document
            for document in packed
            if document.metadata.get("source") == chunk.metadata.get("source")
            and document.metadata.get("page") == chunk.metadata.get("page")
        ]
        if not any(chunk.page_content in parent.page_content for parent in parents):
            failures.append(f"chunk text is not contiguous in its parent ({chunk.metadata})")
            break

    if label == "PDF":
        loaded_pages = {document.metadata["page"] for document in documents}
        chunk_pages = {chunk.metadata["page"] for chunk in chunks}
        if not chunk_pages <= loaded_pages:
            failures.append(f"invented page numbers: {sorted(chunk_pages - loaded_pages)}")
        if chunk_pages and min(chunk_pages) < 1:
            failures.append("page metadata is not 1-indexed")
        text_by_page = {document.metadata["page"]: document.page_content for document in documents}
        for chunk in chunks:
            page = chunk.metadata["page"]
            if page in text_by_page and chunk.page_content not in text_by_page[page]:
                failures.append(f"chunk text does not appear on the page it cites ({page})")
                break

    if label == "CSV":
        loaded_rows = {document.metadata["row"] for document in documents}
        chunk_rows = {chunk.metadata["row"] for chunk in chunks}
        if not chunk_rows <= loaded_rows:
            failures.append(f"invented row numbers: {sorted(chunk_rows - loaded_rows)}")

    return failures


def report(label: str, path: Path, previews: int = 2) -> tuple[list, list[str]]:
    print("=" * 78)
    print(f"{label}  |  {path.name}")
    print("=" * 78)

    documents = load_document(path)
    with_text = [document for document in documents if document.page_content.strip()]
    chunks = split_documents(documents)

    print(f"loader output   : {len(documents)} document(s), {len(with_text)} with extractable text")
    print(f"loader metadata : {documents[0].metadata if documents else '-'}")
    if label == "CSV":
        packed = pack_row_documents(documents, chunk_size=CHUNK_SIZE)
        print(f"after packing   : {len(packed)} document(s)")
    print(f"chunks          : {len(chunks)}")
    if chunks:
        lengths = [len(chunk.page_content) for chunk in chunks]
        print(f"chunk chars     : min {min(lengths)} / mean {sum(lengths) // len(lengths)} / max {max(lengths)}")

    failures = check_invariants(label, documents, chunks)
    print(f"invariants      : {'PASS' if not failures else 'FAIL'}")
    for failure in failures:
        print(f"  !! {failure}")

    for index, chunk in enumerate(chunks[:previews], start=1):
        text = chunk.page_content
        preview = text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + " ..."
        print(f"\n  chunk {index}/{len(chunks)} | {len(text)} chars | metadata={chunk.metadata}")
        for line in preview.splitlines():
            print(f"    | {line}")
    print()
    return chunks, failures


def check_embedding_compatibility(documents: list, chunks: list) -> list[str]:
    """Read-only check that the chunks still suit the configured encoder + FAISS."""
    from backend.retriever import build_vectorstore, embeddings, get_retrieved_chunks

    print("=" * 78)
    print("EMBEDDING / VECTOR STORE COMPATIBILITY")
    print("=" * 78)

    failures = []
    encoder = embeddings.client
    window = encoder.max_seq_length
    tokenizer = encoder.tokenizer
    token_counts = [
        len(tokenizer(chunk.page_content, add_special_tokens=True)["input_ids"])
        for chunk in chunks
    ]
    truncated = [count for count in token_counts if count > window]
    print(f"encoder input window : {window} tokens")
    print(
        f"chunk tokens         : min {min(token_counts)} / "
        f"mean {sum(token_counts) // len(token_counts)} / max {max(token_counts)}"
    )
    print(
        f"chunks truncated     : {len(truncated)} / {len(chunks)} "
        f"({100 * len(truncated) / len(chunks):.1f}%)"
    )
    if truncated:
        failures.append(f"{len(truncated)} chunk(s) still exceed the {window}-token window")

    vectorstore = build_vectorstore(documents)
    print(f"FAISS index built    : {vectorstore.index.ntotal} vectors")
    results = get_retrieved_chunks(vectorstore, "Who is the author of this book?", k=3)
    for result in results:
        print(
            f"  rank {result['rank']}: {result['source']} | page {result['page']} | "
            f"score {result['score']:.4f}"
        )
    if not results:
        failures.append("retrieval returned no results from the rebuilt index")
    print()
    return failures


def main() -> int:
    with_embeddings = "--with-embeddings" in sys.argv
    print(f"backend/chunking.py: CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}\n")

    temp_dir = Path(tempfile.mkdtemp(prefix="docmind_chunk_fixtures_"))
    all_failures: list[str] = []
    sample_documents: list = []
    sample_chunks: list = []
    try:
        txt_path = temp_dir / "rag_field_notes.txt"
        txt_path.write_text(TXT_FIXTURE, encoding="utf-8")
        docx_path = temp_dir / "ingestion_notes.docx"
        write_minimal_docx(docx_path, DOCX_PARAGRAPHS)

        targets: list[tuple[str, Path]] = []
        pdfs = sorted(PDF_DIR.glob("*.pdf"))
        if pdfs:
            targets.append(("PDF", pdfs[0]))
        else:
            print("!! no PDF in data/pdfs - PDF case skipped\n")
        targets.append(("DOCX", docx_path))
        targets.append(("TXT", txt_path))
        csvs = sorted(PDF_DIR.glob("*.csv"))
        if csvs:
            targets.append(("CSV", csvs[0]))
        else:
            print("!! no CSV in data/pdfs - CSV case skipped\n")

        for label, path in targets:
            chunks, failures = report(label, path)
            all_failures.extend(f"[{label}] {failure}" for failure in failures)
            if label in {"DOCX", "TXT"}:
                sample_documents.extend(load_document(path))
                sample_chunks.extend(chunks)

        if with_embeddings and sample_chunks:
            all_failures.extend(
                f"[embeddings] {failure}"
                for failure in check_embedding_compatibility(sample_documents, sample_chunks)
            )
        elif not with_embeddings:
            print("(re-run with --with-embeddings to also check the encoder window and FAISS)\n")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if all_failures:
        print(f"RESULT: {len(all_failures)} failure(s)")
        for failure in all_failures:
            print(f"  - {failure}")
        return 1
    print("RESULT: all chunking invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
