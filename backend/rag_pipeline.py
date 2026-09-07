import json
import os
from pathlib import Path
from typing import List

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.messages import messages_from_dict, messages_to_dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory

from backend.generator import DEFAULT_MODEL, get_llm, stream_rag_answer
from backend.hybrid_retrieval import HybridRetriever
from backend.logging_config import get_logger
from backend.retriever import (
    build_vectorstore,
    delete_saved_pdf,
    embeddings,
    get_retrieved_chunks,
    get_saved_pdf_count,
    get_saved_pdf_names,
    load_documents_from_saved_pdfs,
    save_uploaded_pdfs,
)

logger = get_logger(__name__)

# Number of final reranked chunks handed to the generator (BGE dense + BM25 ->
# RRF -> cross-encoder -> Top-K). Kept small on purpose: the cross-encoder has
# already ranked relevance, so the LLM only needs the best few chunks.
PRODUCTION_TOP_K = 5


class HybridRerankRetriever(BaseRetriever):
    """LangChain retriever that runs the existing hybrid_rerank pipeline.

    This is the production retrieval path. It is a thin adapter: it owns no
    retrieval logic of its own and simply delegates to the existing
    ``HybridRetriever.retrieve(query, mode="hybrid_rerank", k=...)`` (BGE dense +
    BM25 -> RRF -> cross-encoder), then converts the returned chunk dicts back
    into LangChain ``Document`` objects so the rest of the chain
    (history-aware retriever -> stuff-documents -> generator) is unchanged.

    Metadata (``source``/``page``/``row``) is copied straight through so the
    citation prompt and the debug expander keep working exactly as before.
    """

    hybrid: HybridRetriever
    k: int = PRODUCTION_TOP_K

    # HybridRetriever is a plain (non-pydantic) object; allow it as a field.
    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.hybrid.retrieve(query, mode="hybrid_rerank", k=self.k)
        documents: List[Document] = []
        for chunk in results:
            documents.append(
                Document(
                    page_content=chunk["content"],
                    metadata={
                        "source": chunk.get("source"),
                        "page": chunk.get("page"),
                        "row": chunk.get("row"),
                    },
                )
            )
        logger.debug(
            "Production hybrid_rerank retriever returned %d document(s)",
            len(documents),
            extra={"component": "Retriever"},
        )
        return documents

DATA_DIR = Path("data")
CHAT_HISTORY_FILE = DATA_DIR / "chat_histories.json"


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


def answer_upload_status_question(question: str) -> str | None:
    normalized_question = question.lower()
    upload_terms = ("upload", "uploaded", "pdf", "document", "file")
    if not any(term in normalized_question for term in upload_terms):
        return None
    status_terms = ("did i", "have i", "already", "uploaded", "do you have", "can you see")
    if not any(term in normalized_question for term in status_terms):
        return None
    document_names = get_saved_pdf_names()
    if not document_names:
        return "No document is currently saved in the app. Please upload a document from the sidebar."
    if len(document_names) == 1:
        return f"Yes, one document is uploaded and ready: {document_names[0]}."
    return f"Yes, {len(document_names)} documents are uploaded and ready: {', '.join(document_names)}."


def build_conversational_rag_chain(
    api_key: str,
    vectorstore: FAISS,
    histories: dict,
    model_name: str = DEFAULT_MODEL,
):
    logger.debug("Building conversational RAG chain", extra={"component": "RAG"})
    llm = get_llm(api_key, model_name)

    # PRODUCTION RETRIEVAL: BGE dense + BM25 -> RRF -> cross-encoder -> Top-K.
    # The dense leg reuses the SAME persistent FAISS index passed in as
    # ``vectorstore`` (no embeddings recomputed); the chunk set is rebuilt from
    # the saved documents with the identical split_documents() the index was
    # built from, so BM25 and the dense leg rank an identical candidate set.
    # This replaces the previous FAISS-only ``vectorstore.as_retriever()`` path.
    documents = load_documents_from_saved_pdfs()
    hybrid = HybridRetriever(documents, embeddings, vectorstore=vectorstore)
    retriever = HybridRerankRetriever(hybrid=hybrid, k=PRODUCTION_TOP_K)

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

    # Grounded-generation prompt. The answer must be grounded ONLY in the
    # retrieved context below. Citations are NOT written by the model — the app
    # appends verified, programmatically-built citations from the real Document
    # metadata after generation, so the model is told not to invent its own.
    system_prompt = (
        "You are DocMind AI, a question-answering assistant for documents the user uploaded.\n\n"
        "The context below is retrieved from those documents. Each chunk shows its source "
        "filename and page number.\n\n"
        "STRICT RULES — follow every rule for every response:\n"
        "1. GROUND EVERY FACT IN THE CONTEXT — base your answer ONLY on the retrieved context "
        "below. Do NOT use outside or pretrained knowledge as factual evidence. Do NOT invent "
        "facts, do NOT guess, and do NOT make unsupported assumptions.\n"
        "2. IF THE CONTEXT IS INSUFFICIENT — if the retrieved context does not contain enough "
        "information to answer, say exactly: 'I could not find this in the uploaded documents.' "
        "Do not try to fill the gap with anything from outside the context.\n"
        "3. CONVERSATION HISTORY IS FOR REFERENCE ONLY — you may use prior turns to understand "
        "what the user is referring to (pronouns, follow-ups), but history must NOT override or "
        "substitute for the current retrieved evidence. Facts still come only from the context.\n"
        "4. SYNTHESIZE — never copy-paste raw text from the context. Rewrite the answer in your "
        "own clear, readable words.\n"
        "5. BE COMPLETE — if asked for a list (subtopics, characteristics, steps, features, etc.), "
        "go through ALL retrieved chunks and include EVERY item you find. Do not stop early, do "
        "not say 'and more', do not truncate.\n"
        "6. USE ONLY THE RELEVANT SOURCE — if the user names a specific file (e.g. 'Business.pdf'), "
        "use ONLY chunks from that file and ignore chunks from other files. Never combine content "
        "from different files unless the user explicitly asks you to compare or combine them.\n"
        "7. DO NOT WRITE CITATIONS YOURSELF — do not add '(Source: ...)', page numbers, row "
        "numbers, or a sources list. The application appends verified source citations "
        "automatically from the document metadata. Never invent a source, page, or row.\n\n"
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
