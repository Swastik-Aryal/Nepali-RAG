SYSTEM_PROMPT = (
    "You are a helpful assistant that answers ``Question`` using the provided ``Context``. "
    "Follow these rules strictly:\n"
    "1. Prioritize the provided ``Context`` as a knowledge source. You donot need to rely fully on it.\n"
    "2. If the ``Context`` or the previous conversations does not contain enough information to answer the ``Question`` and you donot know the answer yourself confidently, "
    "then donot fabricate information.\n"
    "3. Keep answers concise and directly relevant to the ``Question``,\n"
    "4. Answer in the same language the user's `Question` is in unless the user explicitly requests a specific language at any point.\n"

)

TOOL_CALLING_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a document knowledge base.\n\n"
    "You have a tool called ``retrieve_documents`` that searches the knowledge base "
    "for relevant information. Use it wisely:\n\n"
    "**CALL the tool** when the user asks a question that likely requires information "
    "from the knowledge base — factual questions, specific details, references, etc.\n\n"
    "**DO NOT call the tool** when:\n"
    "- The user is asking for a translation, rephrasing, or summary of something "
    "already discussed in the conversation.\n"
    "- The user is making small talk, greeting, or thanking you.\n"
    "- The user is asking a follow-up that can be answered from the conversation "
    "history alone (e.g. 'Can you tell me that in English?').\n"
    "- You are confident you can answer from general knowledge or prior conversation.\n\n"
    "Answer in the same language the user's `Question` is in unless they explicitly "
    "request a specific language."
)

# LangChain-compatible tool definition for retrieve_documents.
# This dict is passed to llm.bind_tools() and works uniformly across
# OpenAI, Gemini, Ollama, and vLLM providers.
RETRIEVAL_TOOL_DEFINITION = {
    "name": "retrieve_documents",
    "description": (
        "Search the document knowledge base for relevant information. "
        "Call this tool when you need factual information, specific details, "
        "or references from the uploaded documents to answer the user's question. "
        "Do NOT call this tool for conversational follow-ups, translations, "
        "or rephrasing of previously discussed content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The search query to look up in the knowledge base. "
                    "Rephrase the user's question into a clear, concise search "
                    "query optimized for retrieval."
                ),
            },
        },
        "required": ["query"],
    },
}


def format_context(retrieved_results: list[dict]) -> str:
    """Format retrieved chunks as numbered context for the LLM prompt.

    Each chunk is labeled ``[n] (Source: …, Page: …)`` for citation.

    Args:
        retrieved_results: list of result dicts from the retriever,
            each containing 'text' and 'metadata' keys.

    Returns:
        A formatted string with all chunks numbered sequentially from 1.
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
