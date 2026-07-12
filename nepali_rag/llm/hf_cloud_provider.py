"""Hugging Face Inference API LLM provider."""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from nepali_rag.config.settings import settings


def get_hf_llm(**kwargs) -> BaseChatModel:
    """Return a ``ChatOpenAI`` client for the Hugging Face router API.

    Requires ``HF_TOKEN`` in ``.env`` or ``api_key`` kwarg.

    Kwargs default to settings: ``model`` (``settings.hf_model_name``, ``Qwen/Qwen3.6-35B-A3B``),
    ``base_url`` (``settings.hf_base_url``, ``https://router.huggingface.co/v1``),
    ``temperature`` (``settings.llm_temperature``, 0.3).
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
