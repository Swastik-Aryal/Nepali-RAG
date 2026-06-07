from utils.retrieval.retriever import Retriever
from utils.rag.prompts import SYSTEM_PROMPT, format_context, build_user_message
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


class RAGChain:
    """RAG chain that connects retrieval to generation.

    Supports a test mode (no LLM) that returns the fully formatted prompt
    instead of a generated answer. When an LLM is provided, it sends the
    prompt through LangChain's ChatOpenAI interface.

    Conversation history is accepted optionally so multi-turn context
    reaches the LLM, but retrieval always uses only the current query.
    """

    def __init__(self, retriever: Retriever = None, llm=None):
        self.retriever = retriever or Retriever()
        self.llm = llm

    def run(
        self,
        question: str,
        conversation_history: list[dict] = None,
        top_k_retrieval: int = None,
        top_k_rerank: int = None,
    ) -> dict:
        """Execute the full RAG pipeline for a single question.

        Args:
            question: the user's current question (used for retrieval).
            conversation_history: optional list of *previous* turns, each a dict
                with 'role' ('user' or 'assistant') and 'content' keys.
                Do NOT include the current question here; it goes in the
                ``question`` parameter. These are prepended to the LLM
                messages for multi-turn context but are NOT used for
                retrieval. Expected format::

                    [
                        {"role": "user", "content": "What is the capital of Nepal?"},
                        {"role": "assistant", "content": "The capital of Nepal is Kathmandu."},
                    ]
            top_k_retrieval: override for number of hybrid search candidates.
            top_k_rerank: override for number of final reranked results.

        Returns:
            A dict with keys:
                answer: the LLM response (or the formatted prompt in test mode).
                sources: the list of retrieved result dicts.
                context: the formatted context string.
        """
        results = self.retriever.retrieve(
            query=question,
            top_k_retrieval=top_k_retrieval,
            top_k_rerank=top_k_rerank,
        )

        context_str = format_context(results)
        user_message = build_user_message(question, context_str)

        if self.llm is None:
            prompt_lines = [
                f"SYSTEM: {SYSTEM_PROMPT}",
            ]
            if conversation_history:
                for turn in conversation_history:
                    role = turn["role"].upper()
                    prompt_lines.append(f"{role}: {turn['content']}")
            prompt_lines.append(f"USER: {user_message}")
            full_prompt = "\n\n".join(prompt_lines)

            return {
                "answer": full_prompt,
                "sources": results,
                "context": context_str,
            }

        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        if conversation_history:
            for turn in conversation_history:
                if turn["role"] == "user":
                    messages.append(HumanMessage(content=turn["content"]))
                else:
                    messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=user_message))

        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "answer": answer,
            "sources": results,
            "context": context_str,
        }
