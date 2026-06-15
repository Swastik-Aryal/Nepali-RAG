import logging

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from nepali_rag.rag.prompts import (
    RETRIEVAL_TOOL_DEFINITION,
    SYSTEM_PROMPT,
    TOOL_CALLING_SYSTEM_PROMPT,
    build_user_message,
    format_context,
)
from nepali_rag.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGChain:
    """RAG chain that connects retrieval to generation.

    Supports three operating modes:

    1. **Always-retrieve** (``use_tools=False``, default): Every user message
       triggers a full retrieval pipeline. The retrieved context is injected
       into the prompt before the LLM generates an answer.

    2. **Tool-calling** (``use_tools=True``): The LLM receives a
       ``retrieve_documents`` tool and decides autonomously whether retrieval
       is needed for the current message. Follow-ups, translations, and
       greetings skip retrieval entirely.

    3. **Test mode** (``llm=None``): Returns the fully formatted prompt
       without calling an LLM. Always uses the always-retrieve path.

    Conversation history is accepted optionally so multi-turn context
    reaches the LLM, but retrieval always uses only the current query.

    Args:
        - retriever (Retriever, optional): Hybrid-search retriever instance. If ``None``, a
            default ``Retriever`` is created. Defaults to ``None``.
        - llm (optional): LangChain chat model obtained from ``get_llm``. ``None`` enables
            test mode (returns formatted prompts without calling an LLM).
            Defaults to ``None``.
        - use_tools (bool, optional): If ``True``, the LLM decides whether to retrieve; if
            ``False``, every query triggers retrieval.
            Defaults to ``settings.use_tools`` (True).
    """

    def __init__(self, retriever: Retriever = None, llm=None, use_tools: bool = True):
        """Wire up retriever and LLM.

        Args:
            - retriever (Retriever, optional): Hybrid-search retriever instance. If ``None``, a
                default ``Retriever`` is created. Defaults to ``None``.
            - llm (optional): LangChain chat model obtained from ``get_llm``. ``None`` enables
                test mode (returns formatted prompts without calling an LLM).
                Defaults to ``None``.
            - use_tools (bool, optional): If ``True``, the LLM decides whether to retrieve; if
                ``False``, every query triggers retrieval.
                Defaults to ``settings.use_tools`` (True).
                
        """
        self.retriever = retriever or Retriever()
        self.llm = llm
        self.use_tools = use_tools

    def run(
        self,
        question: str,
        conversation_history: list[dict] = None,
        top_k_retrieval: int = None,
        top_k_rerank: int = None,
    ) -> dict:
        """Execute the RAG pipeline for a single question.

        Dispatches to the tool-calling path or the always-retrieve path
        based on ``self.use_tools``.

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
            top_k_retrieval: Hybrid-search candidate count (default ``settings.top_k_retrieval``, 20).
            top_k_rerank: Final result count (default ``settings.top_k_rerank``, 5).

        Returns:
            A dict with keys:
                answer: the LLM response (or the formatted prompt in test mode).
                sources: the list of retrieved result dicts (empty if tool
                    calling decided to skip retrieval).
                context: the formatted context string (empty if skipped).
                tool_called: bool indicating whether retrieval was invoked.
        """
        # Tool-calling requires an LLM; fall back to always-retrieve if absent.
        if self.use_tools and self.llm is not None:
            try:
                return self._run_with_tools(
                    question=question,
                    conversation_history=conversation_history,
                    top_k_retrieval=top_k_retrieval,
                    top_k_rerank=top_k_rerank,
                )
            except Exception as exc:
                logger.warning(
                    "Tool-calling failed (%s), falling back to always-retrieve",
                    exc,
                )

        return self._run_always_retrieve(
            question=question,
            conversation_history=conversation_history,
            top_k_retrieval=top_k_retrieval,
            top_k_rerank=top_k_rerank,
        )

    # Always-retrieve path (original behavior)
    def _run_always_retrieve(
        self,
        question: str,
        conversation_history: list[dict] = None,
        top_k_retrieval: int = None,
        top_k_rerank: int = None,
    ) -> dict:
        """Original RAG flow: retrieve on every message."""
        results = self.retriever.retrieve(
            query=question,
            top_k_retrieval=top_k_retrieval,
            top_k_rerank=top_k_rerank,
        )

        context_str = format_context(results)
        user_message = build_user_message(question, context_str)

        # Test mode no LLM available.
        if self.llm is None:
            prompt_lines = [f"SYSTEM: {SYSTEM_PROMPT}"]
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
                "tool_called": False,
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
            "tool_called": False,
        }

    # Tool-calling path
    def _run_with_tools(
        self,
        question: str,
        conversation_history: list[dict] = None,
        top_k_retrieval: int = None,
        top_k_rerank: int = None,
    ) -> dict:
        """Tool-calling flow: LLM decides whether to retrieve.

        The LLM receives the ``retrieve_documents`` tool and the full
        conversation history. It either:
        - Calls the tool → we run retrieval, feed results back, get answer.
        - Responds directly → no retrieval, answer from conversation context.
        """
        llm_with_tools = self.llm.bind_tools(
            [RETRIEVAL_TOOL_DEFINITION],
            tool_choice="auto",
        )

        # Build messages with the tool-calling system prompt.
        messages = [SystemMessage(content=TOOL_CALLING_SYSTEM_PROMPT)]

        if conversation_history:
            for turn in conversation_history:
                if turn["role"] == "user":
                    messages.append(HumanMessage(content=turn["content"]))
                else:
                    messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=question))

        # First LLM call: decide whether to use the tool.
        response = llm_with_tools.invoke(messages)

        # LLM chose NOT to call the tool 
        if not response.tool_calls:
            answer = response.content if hasattr(response, "content") else str(response)
            logger.info("Tool-calling mode: LLM skipped retrieval for '%s'", question[:80])
            return {
                "answer": answer,
                "sources": [],
                "context": "",
                "tool_called": False,
            }

        # LLM called retrieve_documents
        tool_call = response.tool_calls[0]
        search_query = tool_call.get("args", {}).get("query", question)
        logger.info(
            "Tool-calling mode: LLM requested retrieval with query='%s'",
            search_query[:80],
        )

        # Run the actual retrieval.
        results = self.retriever.retrieve(
            query=search_query,
            top_k_retrieval=top_k_retrieval,
            top_k_rerank=top_k_rerank,
        )

        context_str = format_context(results)

        # Build the tool response message to feed back to the LLM.
        # Append the AI message (with tool call) and the tool result,
        # then let the LLM generate the final answer.
        messages.append(response)  # AI message containing the tool call
        messages.append(
            ToolMessage(
                content=context_str if context_str else "No relevant documents found.",
                tool_call_id=tool_call.get("id", "retrieve_documents"),
            )
        )

        # Second LLM call: generate the answer using the retrieved context.
        # Use the base LLM (without tools bound) for the final response.
        final_response = self.llm.invoke(messages)
        answer = (
            final_response.content
            if hasattr(final_response, "content")
            else str(final_response)
        )

        return {
            "answer": answer,
            "sources": results,
            "context": context_str,
            "tool_called": True,
        }
