import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.retriever import (
    build_vectorstore,
    get_retrieved_chunks,
    get_saved_pdf_names,
    load_documents_from_saved_pdfs,
)
from mrr import mean_reciprocal_rank, reciprocal_rank
from test_questions import TEST_QUESTIONS

TOP_K = 5


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def find_expected_rank(results: list[dict], expected: dict) -> int | None:
    expected_text = normalize_text(expected["text_contains"])
    for result in results:
        if (
            result["source"] == expected["source"]
            and result["page"] == expected["page"]
            and expected_text in normalize_text(result["content"])
        ):
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


def run_evaluation() -> None:
    validate_ground_truth()
    documents = load_documents_from_saved_pdfs()
    vectorstore = build_vectorstore(documents)

    ranks: list[int | None] = []
    print(f"Baseline retrieval evaluation (top {TOP_K})")
    print(f"Indexed PDFs: {', '.join(get_saved_pdf_names())}")

    for item in TEST_QUESTIONS:
        expected = item["expected"]
        results = get_retrieved_chunks(vectorstore, item["question"], k=TOP_K)
        rank = find_expected_rank(results, expected)
        ranks.append(rank)

        print(f"\n[{item['id']}]")
        print(f"Question: {item['question']}")
        print(
            "Expected: "
            f"{expected['source']} | page {expected['page']} | "
            f"contains: {expected['text_contains']}"
        )
        print(f"Hit@{TOP_K}: {'yes' if rank is not None else 'no'}")
        print(f"First matching rank: {rank if rank is not None else 'not retrieved'}")
        print(f"Reciprocal rank: {reciprocal_rank(rank):.4f}")
        print("Retrieved results:")
        for result in results:
            print(
                f"  {result['rank']}. {result['source']} | "
                f"page {result['page']} | score {result['score']:.4f}"
            )

    hits = sum(rank is not None for rank in ranks)
    hit_rate = hits / len(TEST_QUESTIONS) if TEST_QUESTIONS else 0.0
    mrr = mean_reciprocal_rank(ranks)

    print("\nOverall results")
    print(f"Hit Rate@{TOP_K}: {hits}/{len(TEST_QUESTIONS)} = {hit_rate:.4f}")
    print(f"MRR@{TOP_K}: {mrr:.4f}")


if __name__ == "__main__":
    run_evaluation()
