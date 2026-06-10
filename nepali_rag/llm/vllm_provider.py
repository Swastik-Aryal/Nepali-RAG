from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from nepali_rag.config.settings import settings


def get_vllm_llm(**kwargs) -> BaseChatModel:
    """Connect to a local vLLM server via LangChain's ChatOpenAI.

    vLLM exposes an OpenAI-compatible /v1 endpoint, so ChatOpenAI works
    directly when pointed at the local server. No API key is required.
    """
    return ChatOpenAI(
        model=kwargs.get("model", settings.vllm_model),
        openai_api_base=kwargs.get("base_url", settings.vllm_base_url),
        openai_api_key=kwargs.get("api_key", "not-needed"),
        temperature=kwargs.get("temperature", settings.llm_temperature),
    )
