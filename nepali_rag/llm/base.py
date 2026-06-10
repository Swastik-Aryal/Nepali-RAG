from langchain_core.language_models import BaseChatModel
from nepali_rag.config.settings import settings


PROVIDERS = {
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
        provider: one of "vllm", "hf","openai", "gemini", "ollama". Falls back to the
            LLM_PROVIDER setting if not specified.
        **kwargs: forwarded to the provider constructor. Common keys:
            model, temperature, api_key, base_url.

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
