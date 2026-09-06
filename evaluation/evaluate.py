import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.hybrid_retrieval import HybridRetriever
from backend.retriever import (
    embeddings,
    get_saved_pdf_names,
    load_documents_from_saved_pdfs,
)
from mrr import mean_reciprocal_rank, reciprocal_rank
from test_questions import TEST_QUESTIONS

TOP_K = 5

# Task 5: the four retrieval modes to compare, in reporting order. Labels are
# what appears in the summary table; each maps to a HybridRetriever.retrieve
# mode. "dense" is the existing BGE + FAISS baseline.
STRATEGIES = [
    ("BGE", "dense"),
    ("BM25", "bm25"),
    ("Hybrid", "hybrid"),
    ("Hybrid + Reranker", "hybrid_rerank"),
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def matches_expected(result: dict, expected: dict) -> bool:
    """True when a retrieved chunk is the one the ground truth points at.

    A chunk is located by source plus whichever locator its format provides:
    PDFs use ``page``, CSV rows use ``row``. Only the locators actually present
    in the ground truth are required, so a CSV entry does not need ``page``
    (it is always "N/A" for non-PDF formats) and a PDF entry does not need
    ``row`` (only CSVLoader sets it).
    """
    if result["source"] != expected["source"]:
        return False
    if "page" in expected and result.get("page") != expected["page"]:
        return False
    if "row" in expected and result.get("row") != expected["row"]:
        return False
    return normalize_text(expected["text_contains"]) in normalize_text(result["content"])


def describe_expected(expected: dict) -> str:
    """Render the ground truth for logging, omitting locators it does not use."""
    parts = [expected["source"]]
    if "page" in expected:
        parts.append(f"page {expected['page']}")
    if "row" in expected:
        parts.append(f"row {expected['row']}")
    parts.append(f"contains: {expected['text_contains']}")
    return " | ".join(parts)


def describe_result(result: dict) -> str:
    """Render one retrieved chunk, showing row only when the format provides it."""
    locator = f"page {result['page']}"
    if result.get("row") is not None:
        locator += f" | row {result['row']}"
    return f"{result['source']} | {locator} | score {result['score']:.4f}"


def find_expected_rank(results: list[dict], expected: dict) -> int | None:
    for result in results:
        if matches_expected(result, expected):
            return result["rank"]
    return None


def validate_ground_truth() -> None:
    saved_pdfs = set(get_saved_pdf_names())
    missing_sources = sorted(
        {
            item["expected"]["source"]
            for item in TEST_QUESTIONS
            if item["expected"]["source"] not in saved_pdfs
        }
    )
    if missing_sources:
        raise RuntimeError(
            "Evaluation ground-truth source files are not available in data/pdfs: "
            + ", ".join(missing_sources)
        )


def evaluate_strategy(retriever: HybridRetriever, mode: str, verbose: bool) -> tuple[float, float]:
    """Run all questions under one retrieval mode; return (hit_rate, mrr)."""
    ranks: list[int | None] = []
    for item in TEST_QUESTIONS:
        expected = item["expected"]
        results = retriever.retrieve(item["question"], mode=mode, k=TOP_K)
        rank = find_expected_rank(results, expected)
        ranks.append(rank)

        if verbose:
            print(f"\n[{item['id']}]")
            print(f"Question: {item['question']}")
            print(f"Expected: {describe_expected(expected)}")
            print(f"Hit@{TOP_K}: {'yes' if rank is not None else 'no'}")
            print(f"First matching rank: {rank if rank is not None else 'not retrieved'}")
            print(f"Reciprocal rank: {reciprocal_rank(rank):.4f}")
            print("Retrieved results:")
            for result in results:
                print(f"  {result['rank']}. {describe_result(result)}")

    hits = sum(rank is not None for rank in ranks)
    hit_rate = hits / len(TEST_QUESTIONS) if TEST_QUESTIONS else 0.0
    return hit_rate, mean_reciprocal_rank(ranks)


def run_evaluation(verbose: bool = True) -> None:
    validate_ground_truth()
    documents = load_documents_from_saved_pdfs()
    # One retriever, one chunk set, one FAISS index -- reused across every mode
    # so all four strategies rank the identical candidates.
    retriever = HybridRetriever(documents, embeddings)

    print(f"Retrieval evaluation (top {TOP_K}) over {len(TEST_QUESTIONS)} questions")
    print(f"Indexed files: {', '.join(get_saved_pdf_names())}")

    summary: list[tuple[str, float, float]] = []
    for label, mode in STRATEGIES:
        if verbose:
            print(f"\n{'=' * 70}\nStrategy: {label}  (mode={mode})\n{'=' * 70}")
        hit_rate, mrr = evaluate_strategy(retriever, mode, verbose)
        summary.append((label, hit_rate, mrr))

    print("\n\nOverall results")
    print(f"{'Strategy':<22}{'Hit Rate@' + str(TOP_K):<14}{'MRR@' + str(TOP_K)}")
    for label, hit_rate, mrr in summary:
        print(f"{label:<22}{hit_rate:<14.4f}{mrr:.4f}")


if __name__ == "__main__":
    run_evaluation()
