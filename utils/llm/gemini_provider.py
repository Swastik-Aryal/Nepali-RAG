from langchain_core.language_models import BaseChatModel
from utils.config.settings import settings


def get_gemini_llm(**kwargs) -> BaseChatModel:
    """Connect to Google Gemini via langchain-google-genai.

    Requires the GOOGLE_API_KEY environment variable to be set.
    Also used by the RAGAS evaluation module as the judge LLM.
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
