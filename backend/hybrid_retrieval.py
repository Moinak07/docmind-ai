"""Hybrid retrieval for DocMind AI: dense (BGE) + sparse (BM25) + reranking.

This module is additive. It does not touch ingestion, chunking, the BGE
embedding configuration, or the existing dense pipeline in
``backend.retriever`` -- it *composes* them. ``backend.retriever`` stays the
single source of truth for loading, chunking (via ``backend.chunking``) and the
FAISS/BGE index; everything here builds on the exact same chunk objects so the
sparse and dense stages rank an identical candidate set.

Four retrieval modes are exposed through one ``HybridRetriever`` (and its thin
``retrieve(query, mode=...)`` entry point):

    dense         FAISS + BGE only                 (unchanged baseline behaviour)
    bm25          BM25 (rank_bm25.BM25Okapi) only
    hybrid        dense + bm25 fused with RRF       (no raw score mixing)
    hybrid_rerank hybrid candidate pool reordered by a cross-encoder

Flow for hybrid_rerank:

    query -> [BGE dense list] + [BM25 list]
          -> Reciprocal Rank Fusion  -> candidate pool (POOL_SIZE)
          -> cross-encoder/ms-marco-MiniLM-L-6-v2 rescoring of ONLY that pool
          -> final top-k

Metadata contract: every result is emitted through ``_as_result`` which copies
``source`` / ``page`` / ``row`` straight off the chunk's metadata, so the shape
matches ``backend.retriever.get_retrieved_chunks`` exactly (rank, score, source,
page, row, content) and downstream/eval code needs no special-casing.

Design notes
------------
* RRF, not score mixing. FAISS returns an L2 *distance* (smaller = closer) while
  BM25 returns an unbounded relevance score (larger = better); the two are not
  on a comparable scale, so they are fused by *rank* using Reciprocal Rank
  Fusion, ``score = sum 1/(RRF_K + rank_i)``. This is exactly why the task says
  "Do NOT combine raw FAISS and BM25 scores directly."
* The cross-encoder reranks only the fused candidate pool (POOL_SIZE items), not
  the corpus -- that is the whole point of a two-stage retrieve-then-rerank
  design and keeps the expensive model off hundreds of chunks.
* Dense and BM25 remain independently callable (modes ``dense`` / ``bm25``), so
  nothing here removes the ability to run either leg alone.
"""

from __future__ import annotations

import re
import time
from typing import Callable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.chunking import split_documents
from backend.logging_config import get_logger

logger = get_logger(__name__)

# --- Parameters --------------------------------------------------------------
# Chunks pulled from EACH leg (dense, bm25) before fusion. Kept at the existing
# RETRIEVER_K so the dense leg behaves like the current pipeline.
LEG_K = 20

# Size of the fused candidate pool handed to the cross-encoder. The reranker
# only ever sees this many chunks, never the whole corpus.
POOL_SIZE = 20

# RRF dampening constant. 60 is the value from the original RRF paper (Cormack
# et al., 2009) and the common default; it keeps any single top rank from
# dominating the fused score.
RRF_K = 60

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def default_tokenizer(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer used for BM25.

    Deliberately simple and dependency-free. It lowercases and keeps only runs
    of ``[a-z0-9]``, discarding every other character (whitespace, punctuation
    and underscores). This matters for this corpus: CSV chunks render as
    ``sepal_length: 5.1`` and PDF text carries punctuation, so a value like
    ``5.1`` becomes the tokens ``5`` and ``1`` and ``sepal_length`` becomes
    ``sepal`` and ``length`` -- the colon, period and underscore never glue
    onto a term. Because the identical tokenizer is applied to both the stored
    chunks and the query, this split is consistent on both sides, so BM25 term
    matching is unaffected.
    """
    return _TOKEN_RE.findall(text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    rrf_k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Fuse several ranked lists of chunk indices into one RRF-scored order.

    Each input list is chunk indices ordered best-first. Returns ``(index,
    score)`` pairs sorted by descending fused score, where
    ``score = sum over lists of 1 / (rrf_k + rank)`` and ``rank`` is 1-based.
    Fusing by rank (not raw score) is what keeps incomparable FAISS distances
    and BM25 scores from being mixed directly.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, index in enumerate(ranked, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class BM25Index:
    """Modular BM25 sparse index over a fixed chunk list (rank_bm25.BM25Okapi).

    Built from the SAME ``Document`` chunks that back the FAISS index, so a
    chunk's position here is a stable identity usable for rank fusion. Metadata
    is never touched -- the chunks are held by reference and their ``source`` /
    ``page`` / ``row`` are read straight off later.
    """

    def __init__(
        self,
        chunks: list[Document],
        tokenizer: Callable[[str], list[str]] = default_tokenizer,
    ) -> None:
        # Imported lazily so importing this module never hard-requires rank_bm25
        # (e.g. when only the dense mode is used, or in a sandbox without it).
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.tokenizer = tokenizer
        self._bm25 = BM25Okapi([tokenizer(chunk.page_content) for chunk in chunks])

    def search(self, query: str, k: int) -> list[int]:
        """Return indices of the top-k chunks for the query, best-first."""
        scores = self._bm25.get_scores(self.tokenizer(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return ranked[:k]


def _as_result(chunk: Document, rank: int, score: float) -> dict:
    """Shape one chunk like backend.retriever.get_retrieved_chunks output."""
    return {
        "rank": rank,
        "score": float(score),
        "source": chunk.metadata.get("source", "unknown"),
        "page": chunk.metadata.get("page", "unknown"),
        "row": chunk.metadata.get("row"),
        "content": chunk.page_content,
    }


class HybridRetriever:
    """Compose dense (BGE/FAISS) and sparse (BM25) retrieval over one chunk set.

    The chunk list is produced once by ``backend.chunking.split_documents`` --
    the identical function the FAISS index is built from -- and shared by both
    legs, guaranteeing BM25 and FAISS rank the same candidates.
    """

    def __init__(
        self,
        documents: list[Document],
        embeddings,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
        vectorstore: FAISS | None = None,
    ) -> None:
        self.chunks = split_documents(documents)
        self.embeddings = embeddings
        self.cross_encoder_model = cross_encoder_model
        self._cross_encoder = None  # lazily loaded on first rerank

        # Dense index. Two ways in, same chunk set either way:
        #   * vectorstore=None (default, e.g. eval harness): build a fresh FAISS
        #     index from these chunks.
        #   * vectorstore=<existing FAISS> (production): REUSE the already-built,
        #     persistent index as the dense source so no embeddings are recomputed.
        # In both cases the dense leg maps a FAISS hit back to a chunk index by
        # content+metadata identity, so the injected index must have been built
        # from the SAME documents (it is -- backend.retriever.build_vectorstore
        # splits with the identical split_documents(), giving identical chunks).
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            # FAISS.from_documents stores the chunks in insertion order; we keep
            # our own copy in the SAME order so a FAISS hit can be mapped back to
            # a chunk index for fusion via content+metadata identity.
            self.vectorstore = FAISS.from_documents(documents=self.chunks, embedding=embeddings)
        self._index_by_identity = {
            self._identity(chunk): position for position, chunk in enumerate(self.chunks)
        }

        self._bm25: BM25Index | None = None  # built on first BM25/hybrid use

    @staticmethod
    def _identity(chunk: Document) -> tuple:
        """Stable identity of a chunk for mapping a FAISS hit to its index."""
        meta = chunk.metadata
        return (
            meta.get("source"),
            meta.get("page"),
            meta.get("row"),
            chunk.page_content,
        )

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            logger.debug(
                "Building BM25 index over %d chunk(s)",
                len(self.chunks),
                extra={"component": "BM25"},
            )
            self._bm25 = BM25Index(self.chunks)
        return self._bm25

    # --- individual legs (independently available) --------------------------
    def _dense_ranked_indices(self, query: str, k: int) -> list[int]:
        """FAISS/BGE hits as chunk indices, best-first (lowest L2 distance)."""
        hits = self.vectorstore.similarity_search_with_score(query, k=k)
        indices: list[int] = []
        for doc, _score in hits:  # hits already sorted by ascending distance
            index = self._index_by_identity.get(self._identity(doc))
            if index is not None and index not in indices:
                indices.append(index)
        return indices

    def _bm25_ranked_indices(self, query: str, k: int) -> list[int]:
        return self.bm25.search(query, k=k)

    # --- public retrieval ----------------------------------------------------
    def retrieve(self, query: str, mode: str = "dense", k: int = 5) -> list[dict]:
        """Return the top-k chunks for the query under the chosen mode.

        mode:
            "dense"          FAISS + BGE only
            "bm25"           BM25 only
            "hybrid"         dense + bm25 fused with RRF
            "hybrid_rerank"  fused pool reordered by the cross-encoder
        """
        started = time.perf_counter()
        logger.debug(
            "Hybrid retrieval started (mode=%s, k=%d)",
            mode,
            k,
            extra={"component": "Retriever"},
        )

        if mode == "dense":
            indices = self._dense_ranked_indices(query, k)
            logger.debug(
                "Dense retrieval completed - Candidates: %d",
                len(indices),
                extra={"component": "Dense"},
            )
            return [
                _as_result(self.chunks[i], rank, 1.0 / rank)
                for rank, i in enumerate(indices[:k], start=1)
            ]

        if mode == "bm25":
            indices = self._bm25_ranked_indices(query, k)
            logger.debug(
                "BM25 retrieval completed - Candidates: %d",
                len(indices),
                extra={"component": "BM25"},
            )
            return [
                _as_result(self.chunks[i], rank, 1.0 / rank)
                for rank, i in enumerate(indices[:k], start=1)
            ]

        if mode in ("hybrid", "hybrid_rerank"):
            dense_indices = self._dense_ranked_indices(query, LEG_K)
            logger.debug(
                "Dense retrieval completed - Candidates: %d",
                len(dense_indices),
                extra={"component": "Dense"},
            )
            bm25_indices = self._bm25_ranked_indices(query, LEG_K)
            logger.debug(
                "BM25 retrieval completed - Candidates: %d",
                len(bm25_indices),
                extra={"component": "BM25"},
            )
            fused = reciprocal_rank_fusion([dense_indices, bm25_indices])
            logger.debug(
                "RRF fusion completed - Candidates: %d",
                len(fused),
                extra={"component": "RRF"},
            )

            if mode == "hybrid":
                results = [
                    _as_result(self.chunks[i], rank, score)
                    for rank, (i, score) in enumerate(fused[:k], start=1)
                ]
                logger.debug(
                    "Hybrid retrieval returning %d chunk(s)",
                    len(results),
                    extra={"component": "Retriever"},
                )
                return results

            # hybrid_rerank: cross-encoder rescoring of ONLY the fused pool.
            pool_indices = [index for index, _ in fused[:POOL_SIZE]]
            logger.debug(
                "Cross-encoder reranking - Candidates: %d",
                len(pool_indices),
                extra={"component": "Reranker"},
            )
            reranked = self._rerank(query, pool_indices)
            logger.debug(
                "Cross-encoder reranking completed - Candidates: %d",
                len(reranked),
                extra={"component": "Reranker"},
            )
            results = [
                _as_result(self.chunks[i], rank, score)
                for rank, (i, score) in enumerate(reranked[:k], start=1)
            ]
            logger.info(
                "Retrieval completed successfully - Chunks: %d, Latency: %dms",
                len(results),
                round((time.perf_counter() - started) * 1000),
                extra={"component": "Retriever"},
            )
            return results

        raise ValueError(
            f"Unknown retrieval mode {mode!r}; expected one of "
            "'dense', 'bm25', 'hybrid', 'hybrid_rerank'."
        )

    def _rerank(self, query: str, pool_indices: list[int]) -> list[tuple[int, float]]:
        """Rescore a candidate pool with the cross-encoder, best-first.

        Only the pooled chunks are scored -- never the whole corpus. Returns
        ``(chunk_index, cross_encoder_score)`` sorted by descending score.
        """
        if not pool_indices:
            return []
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            logger.debug(
                "Loading cross-encoder model: %s",
                self.cross_encoder_model,
                extra={"component": "Reranker"},
            )
            self._cross_encoder = CrossEncoder(self.cross_encoder_model)

        pairs = [[query, self.chunks[i].page_content] for i in pool_indices]
        scores = self._cross_encoder.predict(pairs)  # higher = more relevant
        ranked = sorted(
            zip(pool_indices, (float(s) for s in scores)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked
