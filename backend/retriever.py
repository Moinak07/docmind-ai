import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

from backend.chunking import CHUNK_OVERLAP, CHUNK_SIZE, split_documents  # noqa: F401
from backend.logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}

# --- Constants ---
# CHUNK_SIZE / CHUNK_OVERLAP are defined in backend.chunking and re-exported
# here so existing imports from this module keep working.
# FIX 1: increased k so topics spread across many pages are fully covered
RETRIEVER_K = 20
# TASK 4: embedding model upgraded from all-MiniLM-L6-v2 to BAAI/bge-small-en-v1.5.
# bge-small-en-v1.5 is also a 384-dim sentence-transformer, so the FAISS index
# (which derives its dimension from the embeddings at build time) stays
# compatible with no other change.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# HuggingFaceBgeEmbeddings is the installed LangChain version's BGE-aware wrapper
# (langchain_community 0.2.16). It applies BGE's retrieval query instruction to
# queries only -- embed_instruction defaults to "" so documents are embedded
# verbatim -- keeping document and query vectors in the same 384-dim space.
# normalize_embeddings=True is the configuration BGE expects; combined with the
# FAISS L2 index already in use it yields cosine-equivalent ranking without any
# change to the FAISS/retrieval logic itself.
logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
embeddings = HuggingFaceBgeEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)


def save_uploaded_pdfs(uploaded_files) -> list[str]:
    saved = []
    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name
        dest = PDF_DIR / safe_name
        with open(dest, "wb") as file:
            file.write(uploaded_file.getvalue())
        saved.append(safe_name)
        # Log the filename only -- never the file contents.
        logger.info("Document uploaded: %s", safe_name)
    return saved


def delete_saved_pdf(filename: str) -> bool:
    target = PDF_DIR / Path(filename).name
    if target.exists():
        target.unlink()
        logger.info("Document removed: %s", target.name)
        return True
    logger.warning("Delete requested for missing document: %s", target.name)
    return False


def get_saved_pdf_names() -> list[str]:
    return sorted(
        p.name
        for p in PDF_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def get_saved_pdf_count() -> int:
    return len(get_saved_pdf_names())


def load_document(document_path: Path) -> list:
    """Load one supported document and normalize its source/page metadata."""
    extension = document_path.suffix.lower()
    if extension == ".pdf":
        loader = PyPDFLoader(str(document_path))
    elif extension == ".docx":
        loader = Docx2txtLoader(str(document_path))
    elif extension == ".txt":
        loader = TextLoader(str(document_path), autodetect_encoding=True)
    elif extension == ".csv":
        loader = CSVLoader(str(document_path), autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported document type: {document_path.suffix}")

    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = document_path.name
        if extension == ".pdf":
            # FIX 3: PyPDFLoader uses 0-based page index; add 1 so it matches
            # the page number the user sees in their PDF viewer.
            raw_page = doc.metadata.get("page", 0)
            doc.metadata["page"] = raw_page + 1
        else:
            doc.metadata["page"] = "N/A"
    return docs


def load_documents_from_saved_pdfs() -> list:
    """Load all supported documents. PDFs retain 1-indexed page numbers."""
    logger.info("Loading documents")
    documents = []
    source_paths = sorted(
        path
        for path in PDF_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    for document_path in source_paths:
        documents.extend(load_document(document_path))
    logger.info("Loaded %d document section(s) from %d file(s)", len(documents), len(source_paths))
    return documents


def build_vectorstore(documents: list) -> FAISS:
    """Chunk the loaded documents (see backend.chunking) and index them in FAISS."""
    logger.info("Creating chunks")
    splits = split_documents(documents)
    logger.info("Created %d chunk(s); building FAISS index", len(splits))
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    logger.info("FAISS index built (%d vectors)", len(splits))
    return vectorstore


def get_retrieved_chunks(vectorstore: FAISS, question: str, k: int = RETRIEVER_K) -> list[dict]:
    """Return retrieved chunks with source/page/row metadata for the debug expander."""
    logger.debug("Dense similarity search for debug chunks (k=%d)", k)
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
                # row is only set by CSVLoader (0-based over data rows); None for
                # every other format. Exposed so callers can locate a CSV chunk,
                # which has page "N/A" and is therefore not addressable by page.
                "row": doc.metadata.get("row"),
                "content": doc.page_content,
            }
        )
    logger.debug("Retrieved %d debug chunk(s)", len(chunks))
    return chunks
