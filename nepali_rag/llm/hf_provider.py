"""Hugging Face local LLM provider."""

from langchain_core.language_models import BaseChatModel

from nepali_rag.config.settings import settings


def get_hf_llm(**kwargs) -> BaseChatModel:
    """Return a local Hugging Face chat model via ``transformers``.

    No API key required. The model is downloaded and run locally.

    Kwargs default to settings: ``model`` (``settings.hf_model_name``, ``Qwen/Qwen3.6-35B-A3B``),
    ``temperature`` (``settings.llm_temperature``, 0.3).
    """
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    except ImportError as exc:
        raise ImportError(
            "langchain-huggingface is required for the local Hugging Face provider. "
            "Install it with: pip install langchain-huggingface"
        ) from exc

    temperature = kwargs.get("temperature", settings.llm_temperature)
    pipeline = HuggingFacePipeline.from_model_id(
        model_id=kwargs.get("model", settings.hf_model_name),
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 512,
            "do_sample": temperature > 0,
            **({"temperature": temperature} if temperature > 0 else {}),
        },
    )
    return ChatHuggingFace(llm=pipeline)
