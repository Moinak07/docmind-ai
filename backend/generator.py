from langchain_groq import ChatGroq

from backend.logging_config import get_logger

logger = get_logger(__name__)

# --- Constants ---
DEFAULT_MODEL = "qwen/qwen3.8-27b"


def get_llm(api_key: str, model_name: str = DEFAULT_MODEL) -> ChatGroq:
    logger.info("Initializing LLM: %s", model_name)
    return ChatGroq(groq_api_key=api_key, model_name=model_name, streaming=True)


def stream_rag_answer(chain, question: str, session_id: str, context_sink=None):
    """Yield only answer text chunks from a streaming RAG chain.

    ``context_sink`` (optional): a mutable list. If provided, the actual
    retrieved Top-5 ``Document`` objects emitted by the chain under its
    ``context`` key are stored there so the caller can build source-grounded
    citations from real metadata AFTER the stream finishes. Passing it does not
    change what is yielded (still only answer text), so streaming and
    ``st.write_stream`` behave exactly as before. Callers that omit it are
    unaffected.
    """
    # Log the query boundary and the LLM hand-off. Never log the full question
    # text, the retrieved context, or the conversation history.
    logger.info("Processing query (length=%d chars)", len(question))
    logger.info("Sending context to LLM")
    produced_any = False
    for chunk in chain.stream(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    ):
        if isinstance(chunk, dict):
            # Capture the real generation context (the final Top-5 Documents).
            # create_retrieval_chain emits "context" alongside "answer"; it may
            # arrive in its own streamed step, so we record it whenever present.
            if context_sink is not None:
                context_docs = chunk.get("context")
                if context_docs:
                    context_sink.clear()
                    context_sink.extend(context_docs)
            answer_chunk = chunk.get("answer")
            if isinstance(answer_chunk, str):
                produced_any = produced_any or bool(answer_chunk)
                yield answer_chunk
            elif hasattr(answer_chunk, "content"):
                produced_any = True
                yield answer_chunk.content
        elif hasattr(chunk, "content"):
            produced_any = True
            yield chunk.content
    if produced_any:
        logger.info("Answer generated")
    else:
        logger.warning("LLM stream produced no answer content")
