"""Eval LLM (judge model) configuration for DeepEval metrics.

Configuration is driven by environment variables:

    EVAL_PROVIDER   - One of: openai, openai_compatible, anthropic, gemini,
                      ollama, vllm, huggingface
    EVAL_MODEL      - Model name/identifier for the chosen provider.
    EVAL_API_KEY    - API key (not needed for ollama).
    EVAL_BASE_URL   - Custom endpoint URL (required for openai_compatible,
                      vllm, huggingface; optional for others).

Provider notes:
    openai            : Official OpenAI API. No EVAL_BASE_URL needed.
    openai_compatible : Any OpenAI-compatible endpoint (Groq, Together, etc).
                        Set EVAL_BASE_URL to the provider's /v1 endpoint.
    anthropic         : Anthropic Claude models. Uses the anthropic SDK.
    gemini            : Google Gemini. EVAL_API_KEY = Google API key.
    ollama            : Local Ollama instance. EVAL_BASE_URL defaults to
                        http://localhost:11434.
    vllm              : Self-hosted vLLM server. Equivalent to
                        openai_compatible with EVAL_BASE_URL pointing to
                        your vLLM /v1 endpoint.
    huggingface       : HuggingFace Inference API (OpenAI-compatible router).
                        EVAL_BASE_URL defaults to https://router.huggingface.co/v1

Example .env additions:
    EVAL_PROVIDER=openai_compatible
    EVAL_MODEL=qwen/qwen3.6-27b
    EVAL_API_KEY=gsk_...
    EVAL_BASE_URL=https://api.groq.com/openai/v1

Programmatic override (takes priority over env vars):
    from nepali_rag.evals.eval_model import build_model, build_metrics

    model = build_model(provider="anthropic", model_name="claude-sonnet-4-20250514")
    metrics = build_metrics(model)
"""

# import asyncio
# import os

# import instructor
# from pydantic import BaseModel

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import (
    AnthropicModel,
    GeminiModel,
    LocalModel,
    OllamaModel,
)
from deepeval.models.base_model import DeepEvalBaseLLM

from nepali_rag.config.settings import settings

DEFAULT_THRESHOLD = 0.5


# /* EXPERIMENTAL */

# class BedrockInstructorModel(DeepEvalBaseLLM):
#     def __init__(self, model_name: str, region: str, aws_access_key_id: str, aws_secret_access_key: str):
#         self._model_name = model_name
#         os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
#         os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
#         os.environ["AWS_DEFAULT_REGION"] = region
#         self.client = instructor.from_provider(f"bedrock/{self._model_name}")

#     def load_model(self):
#         return self.client

#     def generate(self, prompt: str, schema: BaseModel | None = None) -> BaseModel | str:
#         resp = self.client.create(
#             modelId=self._model_name,
#             messages=[{"role": "user", "content": prompt}],
#             response_model=schema,
#             max_retries= 5
#         )
#         return resp

#     async def a_generate(self, prompt: str, schema: BaseModel | None = None) -> BaseModel | str:
#         return await asyncio.to_thread(self.generate, prompt, schema)

#     def get_model_name(self):
#         return self._model_name


def build_model(
    provider: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> DeepEvalBaseLLM:
    """Construct the judge LLM.

    Priority: explicit args > env vars. Raises ValueError if provider is
    missing or unknown.
    """
    provider = provider or settings.eval_provider
    model_name = model_name or settings.eval_model
    api_key = api_key if api_key is not None else settings.eval_api_key
    base_url = base_url if base_url is not None else settings.eval_base_url

    if not provider:
        raise ValueError(
            "No eval provider configured. Set EVAL_PROVIDER env var."
        )
    if not model_name:
        raise ValueError(
            "No eval model configured. Set EVAL_MODEL env var."
        )

    if provider == "openai":
        return LocalModel(
            model=model_name,
            api_key=api_key,
            base_url="https://api.openai.com/v1",
        )

    if provider == "openai_compatible":
        if not base_url:
            raise ValueError("openai_compatible requires EVAL_BASE_URL.")
        return LocalModel(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
        )

    if provider == "anthropic":
        return AnthropicModel(model=model_name, api_key=api_key)

    if provider == "gemini":
        return GeminiModel(model=model_name, api_key=api_key)

    if provider == "ollama":
        return OllamaModel(
            model=model_name,
            base_url=base_url or "http://localhost:11434",
        )

    if provider == "vllm":
        if not base_url:
            raise ValueError("vllm requires EVAL_BASE_URL pointing to your vLLM /v1 endpoint.")
        return LocalModel(
            model=model_name,
            api_key=api_key or "EMPTY",
            base_url=base_url,
        )

    if provider == "huggingface":
        return LocalModel(
            model=model_name,
            api_key=api_key,
            base_url=base_url or "https://router.huggingface.co/v1",
        )

    # /* EXPERIMENTAL */
    # if provider == "bedrock":
    #     return BedrockInstructorModel(
    #         model_name=model_name,
    #         region="us-east-1",
    #         aws_access_key_id="YOUR_AWS_ACCESS_ID",
    #         aws_secret_access_key="YOUR_AWS_SECRET_ACCESS_KEY",
    #     )

    raise ValueError(
        f"Unknown EVAL_PROVIDER: {provider!r}. "
        "Supported: openai, openai_compatible, anthropic, gemini, ollama, vllm, huggingface"
    )


def build_metrics(
    model: DeepEvalBaseLLM | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> list:
    """Build the standard 5 DeepEval RAG metrics."""
    if model is None:
        model = build_model()
    return [
        AnswerRelevancyMetric(model=model, threshold=threshold),
        FaithfulnessMetric(model=model, threshold=threshold),
        ContextualRecallMetric(model=model, threshold=threshold),
        ContextualPrecisionMetric(model=model, threshold=threshold),
        ContextualRelevancyMetric(model=model, threshold=threshold),
    ]
