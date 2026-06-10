from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from nepali_rag.config.settings import settings


def get_hf_llm(**kwargs) -> BaseChatModel:
    """Connect to the HF API directly (Llama, etc.).

    Requires the HF_TOKEN environment variable to be set.
    """
    api_key = kwargs.get("api_key", settings.hf_token)
    if not api_key:
        raise ValueError(
            "HF_TOKEN is not set. "
            "Set it in your .env file or pass api_key= to get_llm()."
        )

    return ChatOpenAI(
        model=kwargs.get("model", settings.hf_model_name),
        base_url=kwargs.get("base_url", settings.hf_base_url),
        api_key=api_key,
        temperature=kwargs.get("temperature", settings.llm_temperature),
    )
