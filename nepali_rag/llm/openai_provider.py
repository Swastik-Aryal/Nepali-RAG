"""OpenAI API LLM provider.

Also supports OpenAI-compatible providers (DeepSeek, Moonshot, Qwen, etc.)
by setting ``OPENAI_BASE_URL`` in ``.env`` or passing ``base_url`` kwarg.
"""

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from nepali_rag.config.settings import settings


def get_openai_llm(**kwargs) -> BaseChatModel:
    """Return a ``ChatOpenAI`` instance.

    Works with the official OpenAI API **and** any provider that exposes an
    OpenAI-compatible ``/v1`` endpoint (DeepSeek, Moonshot, Qwen, Yi, etc.).

    To use a third-party provider, set ``OPENAI_BASE_URL`` in your ``.env``
    (e.g. ``https://api.deepseek.com/v1``) or pass ``base_url=`` directly.

    Requires ``OPENAI_API_KEY`` in ``.env`` or ``api_key`` kwarg.

    Kwargs default to settings: ``model`` (``settings.openai_model``, ``gpt-4o-mini``),
    ``temperature`` (``settings.llm_temperature``, 0.3).
    """
    api_key = kwargs.get("api_key", settings.openai_api_key)
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. "
            "Set it in your .env file or pass api_key= to get_llm()."
        )

    base_url = kwargs.get("base_url", settings.openai_base_url) or None

    # ChatOpenAI appends /chat/completions automatically, so strip it if the
    # user accidentally included it in the base URL.
    if base_url:
        base_url = base_url.rstrip("/")
        for suffix in ("/chat/completions", "/chat"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]

    llm_kwargs = dict(
        model=kwargs.get("model", settings.openai_model),
        openai_api_key=api_key,
        temperature=kwargs.get("temperature", settings.llm_temperature),
    )

    if base_url:
        llm_kwargs["base_url"] = base_url

    return ChatOpenAI(**llm_kwargs)

