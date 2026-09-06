"""Local, on-disk persistence for the DocMind AI FAISS index.

This module is **additive**. It does not touch ingestion, chunking, the BGE
embedding configuration, BM25, hybrid retrieval, or the cross-encoder. It wraps
the existing dense pipeline so the FAISS index survives across processes instead
of living only in Streamlit ``session_state``:

    * the FAISS index itself is written with ``FAISS.save_local`` (produces
      ``index.faiss`` + ``index.pkl``). The ``.pkl`` already carries every chunk
      Document -- ``page_content`` plus the ``source`` / ``page`` / ``row``
      metadata -- so the retrieval contract is preserved byte-for-byte with no
      metadata rewriting on our side.
    * a small JSON *manifest* records what the index was built from: the
      embedding model name, the chunking parameters, and a SHA-256 of every
      source file in ``data/pdfs``. The manifest is what lets us decide, on a
      fresh process, whether the persisted index is still valid or must be
      rebuilt.

Rebuild happens when, and only when, the corpus or the build configuration has
changed -- a file added, removed, or modified (its hash changes), a different
embedding model, or different chunk parameters. When nothing has changed the
index is loaded from disk and **no embeddings are recomputed**.

Everything is local-only: plain files under ``data/faiss_index``. No database,
no Chroma/Pinecone/Qdrant/Weaviate, no cloud storage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS

from backend.chunking import CHUNK_OVERLAP, CHUNK_SIZE
from backend.logging_config import get_logger
from backend.retriever import (
    DATA_DIR,
    EMBEDDING_MODEL,
    PDF_DIR,
    SUPPORTED_EXTENSIONS,
    build_vectorstore,
    embeddings as default_embeddings,
    get_saved_pdf_names,
    load_documents_from_saved_pdfs,
)

logger = get_logger(__name__)

# --- Where persisted state lives (local only) --------------------------------
INDEX_DIR = DATA_DIR / "faiss_index"
# FAISS.save_local writes "<index_name>.faiss" and "<index_name>.pkl".
FAISS_INDEX_NAME = "index"
MANIFEST_PATH = INDEX_DIR / "manifest.json"

# Manifest schema version -- bump if the manifest layout ever changes so an old
# manifest is treated as invalid (forcing a safe rebuild) rather than misread.
MANIFEST_VERSION = 1

_HASH_CHUNK_BYTES = 1 << 20  # read files in 1 MiB blocks when hashing


def _hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes (content-based)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_corpus_signature() -> dict[str, str]:
    """Map each saved source file name to the SHA-256 of its current content.

    Hashing content (not mtime) is what makes change detection reliable: a file
    that is added, removed, or edited changes this mapping, and an untouched
    file keeps the same hash across processes and machines.
    """
    signature: dict[str, str] = {}
    for name in get_saved_pdf_names():
        path = PDF_DIR / name
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            signature[name] = _hash_file(path)
    return signature


def build_manifest(embedding_model: str = EMBEDDING_MODEL) -> dict:
    """Describe the corpus + build configuration the index should match."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "embedding_model": embedding_model,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "files": compute_corpus_signature(),
    }


def _read_manifest() -> dict | None:
    """Load the persisted manifest, or None if absent/unreadable/corrupt."""
    if not MANIFEST_PATH.exists():
        return None
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        # unreadable file or invalid JSON -> treat as no manifest (rebuild)
        logger.warning("FAISS manifest unreadable or invalid; will rebuild index")
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_manifest(manifest: dict) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def manifest_matches(saved: dict | None, current: dict) -> bool:
    """True when the persisted index can be reused for the current corpus.

    Every field that affects the vectors must agree: manifest version, embedding
    model, both chunk parameters, and the exact set of files with identical
    hashes. Any difference (add / remove / modify / model change / chunk change)
    returns False, which forces a rebuild.
    """
    if not isinstance(saved, dict):
        return False
    return (
        saved.get("manifest_version") == current["manifest_version"]
        and saved.get("embedding_model") == current["embedding_model"]
        and saved.get("chunk_size") == current["chunk_size"]
        and saved.get("chunk_overlap") == current["chunk_overlap"]
        and saved.get("files") == current["files"]
    )


def _index_files_present() -> bool:
    return (
        (INDEX_DIR / f"{FAISS_INDEX_NAME}.faiss").exists()
        and (INDEX_DIR / f"{FAISS_INDEX_NAME}.pkl").exists()
    )


def save_vectorstore(vectorstore: FAISS, manifest: dict) -> None:
    """Persist the FAISS index and its manifest to the local index directory."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR), index_name=FAISS_INDEX_NAME)
    _write_manifest(manifest)
    logger.info("Saved FAISS index to %s", INDEX_DIR)


def load_vectorstore(embeddings=default_embeddings) -> FAISS | None:
    """Load the persisted FAISS index, or None if it is missing or corrupt.

    Metadata is restored intact because ``index.pkl`` holds the original chunk
    Documents. Any failure (missing files, truncated pickle, load error) returns
    None rather than raising, so the caller can safely fall back to a rebuild.
    """
    if not _index_files_present():
        return None
    try:
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            index_name=FAISS_INDEX_NAME,
            # save_local/load_local uses pickle; the file is one we wrote to a
            # local app-owned directory, so deserializing it is safe here.
            allow_dangerous_deserialization=True,
        )
    except Exception:
        # corrupted / partially written / incompatible pickle -> rebuild
        logger.warning("Persisted FAISS index could not be loaded; will rebuild")
        return None


def get_or_build_vectorstore(
    embeddings=default_embeddings,
    embedding_model: str = EMBEDDING_MODEL,
) -> tuple[FAISS, bool]:
    """Return a FAISS index for the current corpus, loading it when still valid.

    Returns ``(vectorstore, rebuilt)``. ``rebuilt`` is False when the persisted
    index was reused (no embeddings recomputed) and True when it was rebuilt.

    Decision:
        * persisted manifest matches the current corpus + model + chunk params
          AND the index loads cleanly -> reuse it (no recompute).
        * otherwise (changed corpus, model/chunk change, or missing/corrupt
          persistence) -> rebuild from the loaded documents and persist.

    The caller is responsible for only invoking this when there is at least one
    document to index; ``build_vectorstore`` requires a non-empty corpus.
    """
    logger.info("Checking persistent FAISS index")
    current = build_manifest(embedding_model)
    saved = _read_manifest()

    if manifest_matches(saved, current):
        logger.info("Corpus unchanged; loading existing FAISS index")
        vectorstore = load_vectorstore(embeddings)
        if vectorstore is not None:
            logger.info("Loaded existing FAISS index (no embeddings recomputed)")
            return vectorstore, False
        # manifest was valid but the index files are missing/corrupt -> rebuild
        logger.warning("Manifest matched but index missing/corrupt; rebuilding")
    else:
        logger.info("Corpus or config changed; rebuilding FAISS index")

    documents = load_documents_from_saved_pdfs()
    vectorstore = build_vectorstore(documents)
    save_vectorstore(vectorstore, current)
    return vectorstore, True
