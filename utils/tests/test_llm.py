import sys
sys.stdout.reconfigure(encoding="utf-8")

from utils.llm.base import get_llm, PROVIDERS
from utils.config.settings import settings
from langchain_core.language_models import BaseChatModel


def test_factory_unknown_provider():
    print("[Test 1] Unknown provider raises ValueError")
    try:
        get_llm(provider="nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)
        assert "Supported providers" in str(e)
        print(f"  Got expected error: {e}")
    print("Test 1 passed\n")


def test_factory_vllm_returns_model():
    print("[Test 2] vLLM provider returns a BaseChatModel")
    llm = get_llm(provider="vllm")
    assert isinstance(llm, BaseChatModel), f"Expected BaseChatModel, got {type(llm)}"
    print(f"  Type: {type(llm).__name__}")
    print("Test 2 passed\n")


def test_factory_vllm_custom_params():
    print("[Test 3] vLLM provider accepts custom parameters")
    llm = get_llm(
        provider="vllm",
        model="custom/model",
        temperature=0.5,
        max_tokens=512,
        base_url="http://localhost:9999/v1",
    )
    assert isinstance(llm, BaseChatModel)
    print(f"  Type: {type(llm).__name__}")
    print("Test 3 passed\n")


def test_factory_openai_missing_key():
    print("[Test 4] OpenAI provider raises error when API key is empty")
    original = settings.openai_api_key
    settings.openai_api_key = ""
    try:
        get_llm(provider="openai")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "OPENAI_API_KEY" in str(e)
        print(f"  Got expected error: {e}")
    finally:
        settings.openai_api_key = original
    print("Test 4 passed\n")


def test_factory_openai_with_key():
    print("[Test 5] OpenAI provider returns model when API key is provided")
    llm = get_llm(provider="openai", api_key="sk-test-key-for-validation")
    assert isinstance(llm, BaseChatModel)
    print(f"  Type: {type(llm).__name__}")
    print("Test 5 passed\n")


def test_factory_gemini_missing_key():
    print("[Test 6] Gemini provider raises error when API key is empty")
    original = settings.google_api_key
    settings.google_api_key = ""
    try:
        get_llm(provider="gemini")
        assert False, "Should have raised ValueError or ImportError"
    except (ValueError, ImportError) as e:
        print(f"  Got expected error: {type(e).__name__}: {e}")
    finally:
        settings.google_api_key = original
    print("Test 6 passed\n")


def test_factory_default_provider():
    print("[Test 7] Default provider uses settings.llm_provider")
    original = settings.llm_provider
    settings.llm_provider = "vllm"
    llm = get_llm()
    assert isinstance(llm, BaseChatModel)
    print(f"  Default provider '{settings.llm_provider}' returned {type(llm).__name__}")
    settings.llm_provider = original
    print("Test 7 passed\n")


def test_provider_registry():
    print("[Test 8] All expected providers are registered")
    expected = {"vllm", "openai", "gemini"}
    assert expected == set(PROVIDERS.keys()), (
        f"Expected providers {expected}, got {set(PROVIDERS.keys())}"
    )
    print(f"  Registered providers: {sorted(PROVIDERS.keys())}")
    print("Test 8 passed\n")


def test_case_insensitive_provider():
    print("[Test 9] Provider name is case-insensitive")
    for name in ["VLLM", "Vllm", " vllm "]:
        llm = get_llm(provider=name)
        assert isinstance(llm, BaseChatModel)
    print("Test 9 passed\n")


def test_rag_chain_integration():
    print("[Test 10] Integration: get_llm() instance works with RAGChain constructor")
    from utils.rag.chain import RAGChain

    llm = get_llm(provider="vllm")
    chain = RAGChain(llm=llm)
    assert chain.llm is llm
    print(f"  RAGChain accepted LLM of type {type(llm).__name__}")
    print("Test 10 passed\n")


def run_test():
    test_factory_unknown_provider()
    test_factory_vllm_returns_model()
    test_factory_vllm_custom_params()
    test_factory_openai_missing_key()
    test_factory_openai_with_key()
    test_factory_gemini_missing_key()
    test_factory_default_provider()
    test_provider_registry()
    test_case_insensitive_provider()
    test_rag_chain_integration()
    print("All LLM integration tests passed")


if __name__ == "__main__":
    run_test()
