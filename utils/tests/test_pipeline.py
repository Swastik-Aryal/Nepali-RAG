import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
from pathlib import Path
from utils.pipeline.document_pipeline import DocumentPipeline


SAMPLE_PDF = str(Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "test.pdf")
SAMPLE_PDF2 = str(Path(__file__).resolve().parent.parent / "data" / "sample_docs" / "test1.pdf")
TEST_COLLECTION = "test_pipeline_ingest"


def run_test():
    print(f"Test file: {SAMPLE_PDF}")
    assert Path(SAMPLE_PDF).exists(), f"Sample PDF not found: {SAMPLE_PDF}"

    pipeline = DocumentPipeline(collection_name=TEST_COLLECTION)

    print("Running full pipeline ingest_batch")
    results = pipeline.ingest_batch([SAMPLE_PDF, SAMPLE_PDF2])

    total_pages = 0
    total_chunks_created = 0
    total_chunks_indexed = 0

    for r in results:
        print(f"  file: {r['filename']}")
        print(f"    collection: {r['collection_name']}, pages: {r['pages_extracted']}, chunks: {r['chunks_created']}, indexed: {r['chunks_indexed']}")
        assert r["collection_name"] == TEST_COLLECTION
        assert r["pages_extracted"] > 0, f"No pages from {r['filename']}"
        assert r["chunks_created"] > 0, f"No chunks from {r['filename']}"
        assert r["chunks_indexed"] == r["chunks_created"]
        total_pages += r["pages_extracted"]
        total_chunks_created += r["chunks_created"]
        total_chunks_indexed += r["chunks_indexed"]

    print(f"Totals: {total_pages} pages, {total_chunks_created} chunks, {total_chunks_indexed} indexed")
    assert total_chunks_indexed == total_chunks_created
    print("Batch ingest passed")

    time.sleep(1)

    print("\nVerifying Qdrant collection")
    info = pipeline.vector_store.get_collection_info()
    print(f"Collection: {info['name']}, points: {info['points_count']}, status: {info['status']}")
    assert info["points_count"] == total_chunks_indexed, (
        f"Expected {total_chunks_indexed} points, got {info['points_count']}"
    )
    print("Collection verification passed")

    # print(f"\nCleaning up: deleting collection {TEST_COLLECTION}")
    # pipeline.vector_store.delete_collection()
    print("All pipeline tests passed")


if __name__ == "__main__":
    run_test()
