import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# FIX 1: increased k so topics spread across many pages are fully covered
RETRIEVER_K = 20
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def save_uploaded_pdfs(uploaded_files) -> list[str]:
    saved = []
    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name
        dest = PDF_DIR / safe_name
        with open(dest, "wb") as file:
            file.write(uploaded_file.getvalue())
        saved.append(safe_name)
    return saved


def delete_saved_pdf(filename: str) -> bool:
    target = PDF_DIR / Path(filename).name
    if target.exists():
        target.unlink()
        return True
    return False


def get_saved_pdf_names() -> list[str]:
    return sorted(p.name for p in PDF_DIR.glob("*.pdf"))


def get_saved_pdf_count() -> int:
    return len(get_saved_pdf_names())


def load_documents_from_saved_pdfs() -> list:
    """Load all PDFs. Store clean filename + FIX 3: 1-indexed page numbers."""
    documents = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
            # FIX 3: PyPDFLoader uses 0-based page index; add 1 so it matches
            # the page number the user sees in their PDF viewer.
            raw_page = doc.metadata.get("page", 0)
            doc.metadata["page"] = raw_page + 1
        documents.extend(docs)
    return documents


def build_vectorstore(documents: list) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = text_splitter.split_documents(documents)
    return FAISS.from_documents(documents=splits, embedding=embeddings)


def get_retrieved_chunks(vectorstore: FAISS, question: str, k: int = RETRIEVER_K) -> list[dict]:
    """Return retrieved chunks with source/page metadata for the debug expander."""
    results = vectorstore.similarity_search_with_score(question, k=k)
    chunks = []
    for index, (doc, score) in enumerate(results, start=1):
        chunks.append(
            {
                "rank": index,
                "score": float(score),
                "source": doc.metadata.get("source", "unknown"),
                # page is already 1-indexed after load_documents_from_saved_pdfs
                "page": doc.metadata.get("page", "unknown"),
                "content": doc.page_content,
            }
        )
    return chunks
