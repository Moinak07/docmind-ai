from langchain_groq import ChatGroq

# --- Constants ---
DEFAULT_MODEL = "llama-3.1-8b-instant"


def get_llm(api_key: str, model_name: str = DEFAULT_MODEL) -> ChatGroq:
    return ChatGroq(groq_api_key=api_key, model_name=model_name, streaming=True)


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
