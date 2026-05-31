import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import messages_from_dict, messages_to_dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"
CHAT_HISTORY_FILE = DATA_DIR / "chat_histories.json"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# --- Constants ---
DEFAULT_MODEL = "llama-3.1-8b-instant"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# FIX 1: increased k so topics spread across many pages are fully covered
RETRIEVER_K = 20
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_saved_groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "")


def load_saved_histories() -> dict:
    if not CHAT_HISTORY_FILE.exists():
        return {}
    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as file:
        saved_histories = json.load(file)
    return {
        session: ChatMessageHistory(messages=messages_from_dict(messages))
        for session, messages in saved_histories.items()
    }


def save_histories(histories: dict) -> None:
    CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                session: messages_to_dict(history.messages)
                for session, history in histories.items()
            },
            file,
            indent=2,
        )


def get_session_history(histories: dict, session: str) -> ChatMessageHistory:
    if session not in histories:
        histories[session] = ChatMessageHistory()
    return histories[session]


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


def answer_upload_status_question(question: str) -> str | None:
    normalized_question = question.lower()
    upload_terms = ("upload", "uploaded", "pdf", "document", "file")
    if not any(term in normalized_question for term in upload_terms):
        return None
    status_terms = ("did i", "have i", "already", "uploaded", "do you have", "can you see")
    if not any(term in normalized_question for term in status_terms):
        return None
    pdf_names = get_saved_pdf_names()
    if not pdf_names:
        return "No PDF is currently saved in the app. Please upload a PDF from the sidebar."
    if len(pdf_names) == 1:
        return f"Yes, one PDF is uploaded and ready: {pdf_names[0]}."
    return f"Yes, {len(pdf_names)} PDFs are uploaded and ready: {', '.join(pdf_names)}."


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


def build_conversational_rag_chain(
    api_key: str,
    vectorstore: FAISS,
    histories: dict,
    model_name: str = DEFAULT_MODEL,
):
    llm = ChatGroq(groq_api_key=api_key, model_name=model_name, streaming=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # FIX 1 & 2: system prompt now explicitly instructs the LLM to:
    #   - synthesize in its own words (not copy-paste)
    #   - list ALL items completely without truncation
    #   - use only context from the named PDF when a specific file is mentioned
    system_prompt = (
        "You are DocMind AI, a helpful question-answering assistant inside a PDF RAG app.\n\n"
        "The context below comes from PDF files uploaded by the user. "
        "Each chunk includes its source filename and page number.\n\n"
        "STRICT RULES — follow every rule for every response:\n"
        "1. SYNTHESIZE — never copy-paste raw text from the context. "
        "Always rewrite the answer in your own clear, readable words.\n"
        "2. BE COMPLETE — if asked for a list (subtopics, characteristics, steps, features, etc.), "
        "go through ALL retrieved chunks and include EVERY item you find. "
        "Do not stop early, do not say 'and more', do not truncate.\n"
        "3. USE ONLY THE RELEVANT SOURCE — if the user mentions a specific PDF by name "
        "(e.g. 'Business.pdf'), use ONLY chunks from that file. Ignore chunks from other PDFs.\n"
        "4. DO NOT MIX SOURCES — never combine content from different PDFs in one answer "
        "unless the user explicitly asks you to compare or combine them.\n"
        "5. CITE CORRECTLY — when citing a source, write: (Source: filename, Page X). "
        "Use the page number exactly as provided in the metadata — it already matches the "
        "page number visible in the PDF viewer.\n"
        "6. IF NOT FOUND — if the answer is genuinely not in the retrieved context, say: "
        "'I could not find this in the uploaded PDF content.'\n"
        "7. DO NOT use any external or pretrained knowledge. Only use the context provided.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # FIX 3: document_prompt now shows 1-indexed page (already corrected in metadata)
    document_prompt = PromptTemplate.from_template(
        "Source: {source} | Page: {page}\n{page_content}"
    )

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt,
        document_prompt=document_prompt,
    )
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def history_for_chain(session: str) -> ChatMessageHistory:
        return get_session_history(histories, session)

    return RunnableWithMessageHistory(
        rag_chain,
        history_for_chain,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


def stream_rag_answer(chain, question: str, session_id: str):
    """Yield only answer text chunks from a streaming RAG chain."""
    for chunk in chain.stream(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    ):
        if isinstance(chunk, dict):
            answer_chunk = chunk.get("answer")
            if isinstance(answer_chunk, str):
                yield answer_chunk
            elif hasattr(answer_chunk, "content"):
                yield answer_chunk.content
        elif hasattr(chunk, "content"):
            yield chunk.content