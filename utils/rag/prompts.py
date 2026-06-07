SYSTEM_PROMPT = (
    "You are a helpful assistant that answers ``Question`` using the provided ``Context``. "
    "Follow these rules strictly:\n"
    "1. Prioritize the provided ``Context`` as a knowledge source. You donot need to rely fully on it.\n"
    "2. If the ``Context`` or the previous conversations does not contain enough information to answer the ``Question`` and you donot know the answer yourself confidently, "
    "then donot fabricate information.\n"
    "3. Cite relevant parts of the ``Context`` by referencing the chunk numbers if you have used any relevant parts."
    "(e.g. [1], [2]) when supporting your answer.\n"
    "4. Keep answers concise and directly relevant to the ``Question``,\n"
    "5. Answer in the same language this prompt end in unless the user explicitly requests a specific language at any point.\n"

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
