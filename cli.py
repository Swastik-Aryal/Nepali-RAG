import argparse
import logging
import sys


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_ingest(args):
    """Ingest a document through the full pipeline (load, normalize, chunk, embed, index)."""
    _setup_logging(args.verbose)
    from nepali_rag.pipeline.document_pipeline import DocumentPipeline

    kwargs = {}
    if args.collection:
        kwargs["collection_name"] = args.collection
    if args.chunk_size:
        kwargs["chunk_size"] = args.chunk_size

    pipeline = DocumentPipeline(**kwargs)
    result = pipeline.ingest(args.file)

    print(f"Collection : {result['collection_name']}")
    print(f"File       : {result['filename']}")
    print(f"Pages      : {result['pages_extracted']}")
    print(f"Chunks     : {result['chunks_created']}")
    print(f"Indexed    : {result['chunks_indexed']}")


def cmd_query(args):
    """Run the full RAG chain and print the answer."""
    _setup_logging(args.verbose)
    from nepali_rag.config.settings import settings
    from nepali_rag.retrieval.retriever import Retriever
    from nepali_rag.vectorstore.qdrant_store import QdrantStore
    from nepali_rag.rag.chain import RAGChain

    collection = args.collection or settings.collection_name

    llm = None
    if not args.test:
        try:
            from nepali_rag.llm.base import get_llm
            llm_kwargs = {}
            if args.provider:
                llm_kwargs["provider"] = args.provider
            if args.model:
                llm_kwargs["model"] = args.model
            llm = get_llm(**llm_kwargs)
        except Exception as exc:
            logging.getLogger(__name__).warning("LLM unavailable (%s), running in test mode", exc)

    store = QdrantStore(collection_name=collection)
    retriever = Retriever(store=store)
    chain = RAGChain(retriever=retriever, llm=llm, use_tools=args.tools)

    result = chain.run(
        question=args.question,
        top_k_retrieval=args.top_k,
    )

    print(result["answer"])

    if result.get("sources"):
        print(f"\nSources ({len(result['sources'])} chunks):")
        for i, src in enumerate(result["sources"], 1):
            meta = src.get("metadata", {})
            score = src.get("rerank_score") or src.get("score", 0)
            print(f"  [{i}] {meta.get('source', '?')} p.{meta.get('page_number', '?')}  (score {score:.3f})")


def cmd_serve(args):
    """Start the FastAPI server."""
    _setup_logging(args.verbose)
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed. Install the api extras:\n"
            "  pip install nepali-rag[api]",
            file=sys.stderr,
        )
        sys.exit(1)

    uvicorn.run(
        "nepali_rag.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="nepali-rag",
        description="Nepali RAG System: ingest documents, query them, or start the API server.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = subparsers.add_parser(
        "ingest",
        help="ingest a document into the vector store",
    )
    p_ingest.add_argument("file", help="path to the document file")
    p_ingest.add_argument("-c", "--collection", help="target Qdrant collection name")
    p_ingest.add_argument("--chunk-size", type=int, help="token chunk size override")
    p_ingest.set_defaults(func=cmd_ingest)

    # query
    p_query = subparsers.add_parser(
        "query",
        help="query the RAG pipeline",
    )
    p_query.add_argument("question", help="the question to ask")
    p_query.add_argument("-c", "--collection", help="Qdrant collection to search")
    p_query.add_argument("--provider", help="LLM provider override (vllm, openai, gemini, ollama)")
    p_query.add_argument("--model", help="LLM model name override")
    p_query.add_argument("--top-k", type=int, help="number of retrieval candidates")
    p_query.add_argument("--test", action="store_true", help="run in test mode (no LLM)")
    p_query.add_argument("--disable-tools", action="store_false", dest="tools", default=True, help="disable tool-calling mode")
    p_query.set_defaults(func=cmd_query)

    # serve
    p_serve = subparsers.add_parser(
        "serve",
        help="start the FastAPI server",
    )
    p_serve.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8080, help="port (default: 8080)")
    p_serve.add_argument("--reload", action="store_true", help="enable auto-reload for development")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
