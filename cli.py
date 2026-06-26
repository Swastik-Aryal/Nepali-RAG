import argparse
import logging
import sys


def _status(msg: str):
    """Print a minimal status message to stderr."""
    print(f"  ⟶  {msg}", file=sys.stderr, flush=True)


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_ingest(args):
    """Ingest one or more documents through the full pipeline."""
    _setup_logging(args.verbose)
    from nepali_rag.embeddings.embedding_model import NepaliEmbeddingModel
    from nepali_rag.vectorstore.qdrant_store import QdrantStore
    from nepali_rag.pipeline.document_pipeline import DocumentPipeline

    _status("Loading embedding model...")
    embedding_model = NepaliEmbeddingModel(model_name=args.embedding)

    _status("Connecting to Qdrant...")
    store = QdrantStore(
        vector_size=embedding_model.dimension,
        collection_name=args.collection,
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
    )
    pipeline = DocumentPipeline(
        embedding_model=embedding_model,
        vector_store=store,
        collection_name=args.collection,
        chunk_size=args.chunk_size
    )

    _status("Ingesting files...")
    results = pipeline.ingest_batch(args.files)

    for result in results:
        print(f"\nCollection : {result['collection_name']}")
        print(f"File       : {result['filename']}")
        print(f"Pages      : {result['pages_extracted']}")
        print(f"Chunks     : {result['chunks_created']}")
        print(f"Indexed    : {result['chunks_indexed']}")

    if len(results) > 1:
        total_chunks = sum(r['chunks_indexed'] for r in results)
        print(f"\n Total: {len(results)} files, {total_chunks} chunks indexed")


def _load_history(history_arg: str) -> list[dict] | None:
    """Load conversation history from a JSON file path or inline JSON string."""
    if not history_arg:
        return None
    import json
    from pathlib import Path

    path = Path(history_arg)
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(history_arg)


def cmd_query(args):
    """Run the full RAG chain and print the answer."""
    _setup_logging(args.verbose)
    from nepali_rag.embeddings.embedding_model import NepaliEmbeddingModel
    from nepali_rag.vectorstore.qdrant_store import QdrantStore
    from nepali_rag.retrieval.retriever import Retriever
    from nepali_rag.retrieval.reranker import Reranker
    from nepali_rag.rag.chain import RAGChain

    _status("Loading models…")
    embedding_model = NepaliEmbeddingModel(model_name=args.embedding)
    reranker = Reranker(model_name=args.reranker)

    _status("Connecting to Qdrant...")
    store = QdrantStore(
        vector_size=embedding_model.dimension,
        collection_name=args.collection,
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
    )
    
    _status("Retrieving document context...")
    retriever = Retriever(embedding_model=embedding_model, reranker=reranker, store=store)

    llm = None
    if not args.test:
        try:
            from nepali_rag.llm.base import get_llm
            llm_kwargs = {"provider": args.provider, "model": args.model, "temperature": args.temperature}
            if args.api_key:
                llm_kwargs["api_key"] = args.api_key
            if args.llm_base_url:
                llm_kwargs["base_url"] = args.llm_base_url
            llm = get_llm(**llm_kwargs)
        except Exception as exc:
            logging.getLogger(__name__).warning("LLM unavailable (%s), running in test mode", exc)

    chain = RAGChain(retriever=retriever, llm=llm, use_tools=args.tools)

    conversation_history = _load_history(args.history)

    _status("Thinking…")
    result = chain.run(
        question=args.question,
        conversation_history=conversation_history,
        top_k_retrieval=args.top_k,
        top_k_rerank=args.top_r,
    )

    print(result["answer"])

def cmd_serve(args):
    """Start the FastAPI server."""
    _setup_logging(args.verbose)
    from pathlib import Path
    from dotenv import load_dotenv
    import uvicorn

    env_path = Path(args.env_file)
    if not env_path.is_file():
        print(f"Warning: env file not found at '{env_path.resolve()}', server will use defaults.", file=sys.stderr)
    else:
        load_dotenv(env_path, override=True)

    uvicorn.run(
        "nepali_rag.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_mcp(args):
    """Start the MCP HTTP server."""
    _setup_logging(args.verbose)

    _status("Initializing models…")
    from nepali_rag.mcp.server import run_server
    run_server(no_init=args.no_init)


def main():
    parser = argparse.ArgumentParser(
        prog="nepali-rag",
        description="Nepali RAG System: ingest documents, query them, or start the API server.",
    )

    # Shared parent parser so -v/--verbose works in any position
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ingest 
    p_ingest = subparsers.add_parser("ingest", parents=[common], help="ingest documents into the vector store")
    p_ingest.add_argument("files", nargs="+", help="path(s) to the document file(s)")
    p_ingest.add_argument("--qdrant-url", required=True, help="Qdrant instance URL (e.g. https://xyz.cloud.qdrant.io)")
    p_ingest.add_argument("--qdrant-api-key", required=True, help="Qdrant API key")
    p_ingest.add_argument("-c", "--collection", default="nepali_docs", help="target Qdrant collection name (default: nepali_docs)")
    p_ingest.add_argument("--chunk-size", type=int, default=256, help="token chunk size (default: 256)")
    p_ingest.add_argument("--embedding", default="intfloat/multilingual-e5-large", help="embedding model name (default: intfloat/multilingual-e5-large)")
    p_ingest.set_defaults(func=cmd_ingest)

    # query 
    p_query = subparsers.add_parser("query", parents=[common], help="query the RAG pipeline")
    p_query.add_argument("question", help="the question to ask")
    p_query.add_argument("--qdrant-url", required=True, help="Qdrant instance URL (e.g. https://xyz.cloud.qdrant.io)")
    p_query.add_argument("--qdrant-api-key", required=True, help="Qdrant API key")
    p_query.add_argument("-c", "--collection", default="nepali_docs", help="Qdrant collection to search (default: nepali_docs)")
    p_query.add_argument("--provider", default="ollama", help="LLM provider: ollama, vllm, openai, gemini, hf, anthropic (default: ollama)")
    p_query.add_argument("--model", default="gemma4", help="LLM model name (default: gemma4)")
    p_query.add_argument("--api-key", default=None, help="API key for the LLM provider (required for openai, gemini, hf)")
    p_query.add_argument("--llm-base-url", default=None, help="base URL for the LLM server (for vllm, ollama, hf)")
    p_query.add_argument("--temperature", type=float, default=0.3, help="LLM temperature (default: 0.3)")
    p_query.add_argument("--top-k", type=int, default=20, help="number of retrieval candidates (default: 20)")
    p_query.add_argument("--top-r", type=int, default=5, help="number of rerank candidates (default: 5)")
    p_query.add_argument("--history", default=None, help="conversation history as a JSON file path or inline JSON string")
    p_query.add_argument("--test", action="store_true", help="test mode (no LLM, returns raw prompt)")
    p_query.add_argument("--disable-tools", action="store_false", dest="tools", default=True, help="disable tool-calling mode")
    p_query.add_argument("--embedding", default="intfloat/multilingual-e5-large", help="embedding model name (default: intfloat/multilingual-e5-large)")
    p_query.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3", help="reranker model name (default: BAAI/bge-reranker-v2-m3)")
    p_query.set_defaults(func=cmd_query)

    # serve 
    p_serve = subparsers.add_parser("serve", parents=[common], help="start the FastAPI server")
    p_serve.add_argument("--env-file", default=".env", help="path to .env file (default: .env in CWD)")
    p_serve.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8080, help="port (default: 8080)")
    p_serve.add_argument("--reload", action="store_true", help="enable auto-reload for development")
    p_serve.set_defaults(func=cmd_serve)

    # mcp
    p_mcp = subparsers.add_parser("mcp", parents=[common], help="start the MCP server (HTTP transport at http://localhost:8000)")
    p_mcp.add_argument("--no-init", action="store_true", default=False, help="skip eager model loading; load models on first tool call instead")
    p_mcp.set_defaults(func=cmd_mcp)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
