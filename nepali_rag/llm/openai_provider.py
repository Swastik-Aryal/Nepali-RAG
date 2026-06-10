from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from nepali_rag.config.settings import settings


def get_openai_llm(**kwargs) -> BaseChatModel:
    """Connect to the OpenAI API directly (GPT-4o, etc.).

    Requires the OPENAI_API_KEY environment variable to be set.
    """
    api_key = kwargs.get("api_key", settings.openai_api_key)
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Set it in your .env file or pass api_key= to get_llm()."
        )

    return ChatOpenAI(
        model=kwargs.get("model", settings.openai_model),
        openai_api_key=api_key,
        temperature=kwargs.get("temperature", settings.llm_temperature),
    )
