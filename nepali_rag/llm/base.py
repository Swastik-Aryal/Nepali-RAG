from langchain_core.language_models import BaseChatModel
from nepali_rag.config.settings import settings


PROVIDERS = {
    "anthropic": "nepali_rag.llm.anthropic_provider.get_anthropic_llm",
    "vllm": "nepali_rag.llm.vllm_provider.get_vllm_llm",
    "openai": "nepali_rag.llm.openai_provider.get_openai_llm",
    "gemini": "nepali_rag.llm.gemini_provider.get_gemini_llm",
    "ollama": "nepali_rag.llm.ollama_provider.get_ollama_llm",
    "hf": "nepali_rag.llm.hf_provider.get_hf_llm",
}


def get_llm(provider: str = None, **kwargs) -> BaseChatModel:
    """Factory that returns a LangChain BaseChatModel for the requested provider.

    This is the single entry point for all LLM access across the codebase.
    Provider-specific logic stays inside each provider file; callers only
    interact with the standard LangChain chat model interface.

    Args:
        - provider: One of ``anthropic``, ``vllm``, ``hf``, ``openai``, ``gemini``, ``ollama``.
            Defaults to ``settings.llm_provider`` (``ollama``).
        - **kwargs: Forwarded to the provider factory.
            - Common keys:
                - ``model``: Default model to use for the LLM.
                        - default ``settings.anthropic_model`` for anthropic ("claude-opus-4-6"),
                        - default ``settings.vllm_model`` for vllm ("google/gemma-4-E4B-it"),
                        - default ``settings.hf_model_name`` for hf ("Qwen/Qwen3.6-35B-A3B"),
                        - default ``settings.openai_model`` for openai ("gpt-4o-mini"),
                        - default ``settings.gemini_model`` for gemini ("gemini-2.5-flash"),
                        - default ``settings.ollama_model`` for ollama ("gemma4")
                - ``temperature``: Default temperature to use for the LLM.
                        - default ``settings.llm_temperature`` (0.3)
                - ``api_key``: API key for authenticating with the LLM.
                        - default ``settings.anthropic_api_key`` for anthropic (""),
                        - default ``settings.openai_api_key`` for openai ("")
                        - default ``settings.google_api_key`` for gemini ("")
                        - default ``settings.hf_token`` for hf ("")
                - ``base_url``: Base URL of the LLM.
                        - default ``settings.vllm_base_url`` for vllm (``http://localhost:8000/v1``)
                        - default ``settings.openai_base_url`` for openai (``""``, uses official API)
                        - default ``settings.ollama_base_url`` for ollama (``http://localhost:11434``)
                        - default ``settings.hf_base_url`` for hf (``https://router.huggingface.co/v1``)
    
    Returns:
        A configured BaseChatModel instance.

    Raises:
        ValueError: if the provider name is not recognized.
    """
    provider = (provider or settings.llm_provider).lower().strip()

    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported providers: {', '.join(sorted(PROVIDERS))}."
        )

    module_path, func_name = PROVIDERS[provider].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    factory = getattr(module, func_name)

    return factory(**kwargs)
