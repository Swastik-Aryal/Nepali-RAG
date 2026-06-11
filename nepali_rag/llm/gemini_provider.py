"""Google Gemini LLM provider."""

from langchain_core.language_models import BaseChatModel
from nepali_rag.config.settings import settings


def get_gemini_llm(**kwargs) -> BaseChatModel:
    """Return a ``ChatGoogleGenerativeAI`` instance for Google Gemini.

    Requires ``GOOGLE_API_KEY`` in ``.env`` or ``api_key`` kwarg.

    Kwargs default to settings: ``model`` (``settings.gemini_model``, ``gemini-2.5-flash``),
    ``temperature`` (``settings.llm_temperature``, 0.3).
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai is required for the Gemini provider. "
            "Install it with: pip install langchain-google-genai"
        )

    api_key = kwargs.get("api_key", settings.google_api_key)
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Set it in your .env file or pass api_key= to get_llm()."
        )

    return ChatGoogleGenerativeAI(
        model=kwargs.get("model", settings.gemini_model),
        google_api_key=api_key,
        temperature=kwargs.get("temperature", settings.llm_temperature),
    )
