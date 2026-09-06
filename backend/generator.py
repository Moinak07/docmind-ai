from langchain_groq import ChatGroq

from backend.logging_config import get_logger

logger = get_logger(__name__)

# --- Constants ---
DEFAULT_MODEL = "qwen/qwen3.8-27b"


def get_llm(api_key: str, model_name: str = DEFAULT_MODEL) -> ChatGroq:
    logger.info("Initializing LLM: %s", model_name)
    return ChatGroq(groq_api_key=api_key, model_name=model_name, streaming=True)


def stream_rag_answer(chain, question: str, session_id: str):
    """Yield only answer text chunks from a streaming RAG chain."""
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
