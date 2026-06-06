SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided context. "
    "Follow these rules strictly:\n"
    "1. Prioritize the provided context as your primary knowledge source. "
    "Base your answer on the context whenever possible.\n"
    "2. Mirror the language of the user's question. "
    "If the user asks in Nepali, respond in Nepali. "
    "If the user asks in English, respond in English. "
    "If the user explicitly requests a specific language at any point, switch to that language.\n"
    "3. If the context does not contain enough information to answer the question, "
    "say so clearly. Do not fabricate or guess information.\n"
    "4. Cite relevant parts of the context by referencing the chunk numbers "
    "(e.g. [1], [2]) when supporting your answer.\n"
    "5. Keep answers concise and directly relevant to the question."
)


def format_context(retrieved_results: list[dict]) -> str:
    """Format retrieved chunks into a numbered context string for the LLM.

    Each chunk is labeled with its index, source filename, and page number
    so the LLM can reference them in citations.

    Args:
        retrieved_results: list of result dicts from the retriever,
            each containing 'text' and 'metadata' keys.

    Returns:
        A formatted string with all chunks numbered sequentially.
    """
    parts = []
    for i, result in enumerate(retrieved_results, start=1):
        metadata = result.get("metadata", {})
        source = metadata.get("source", "unknown")
        page = metadata.get("page_number", "?")

        header = f"[{i}] (Source: {source}, Page: {page})"
        parts.append(f"{header}\n{result['text']}")

    return "\n\n".join(parts)


def build_user_message(question: str, context: str) -> str:
    """Build the user-facing message that combines context and question.

    Args:
        question: the user's current question.
        context: the formatted context string from format_context().

    Returns:
        The combined message string.
    """
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
