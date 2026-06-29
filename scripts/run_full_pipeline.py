"""
run_full_pipeline.py
════════════════════
End-to-end ingestion pipeline:
  1. Load PDFs      (PDFLoader)
  2. Chunk text     (Chunker)
  3. Embed chunks   (EmbeddingModel)
  4. Store in DB    (VectorStore / ChromaDB)

Usage:
    python scripts/run_full_pipeline.py
"""

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Must happen before any other import that touches settings
import os; os.environ["LOG_FORMAT"] = "console"

from app_logging.logger import setup_logging, get_logger
from config.settings import get_settings
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.processors.chunker import Chunker
from rag.embeddings import EmbeddingModel, load_embedding_model
from rag.vector_store import VectorStore, get_vector_store


setup_logging(level="INFO", log_format="console")
log = get_logger(__name__)


def main() -> None:
    t0 = time.perf_counter()
    settings = get_settings()

    # ── 1. Find PDFs ───────────────────────────────────────────────────────────
    raw_dir = settings.ingestion.data_dir
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        log.error("pipeline.no_pdfs_found", data_dir=str(raw_dir))
        return

    log.info("pipeline.start", pdf_files=[p.name for p in pdf_paths], count=len(pdf_paths))

    # ── 2. Load PDFs ───────────────────────────────────────────────────────────
    loader = PDFLoader()
    extractions = loader.load_batch(pdf_paths)
    successful = [r for r in extractions if r.status in ("success", "partial")]
    log.info("pipeline.loaded", total=len(extractions), successful=len(successful))

    # ── 3. Chunk ───────────────────────────────────────────────────────────────
    chunker = Chunker()
    chunking_results = []
    for extraction in successful:
        result = chunker.chunk_document(extraction)
        chunking_results.append(result)
        log.info("pipeline.chunked", source=result.source, chunks=result.total_chunks)

    # ── 4. Embed ───────────────────────────────────────────────────────────────
    model = load_embedding_model()
    log.info("pipeline.embedding_model_loaded", model=model.model_name)

    all_texts: list[str] = []
    all_sources: list[str] = []
    all_pages: list[int] = []
    chunk_batches = []

    for cr in chunking_results:
        for c in cr.chunks:
            all_texts.append(c.text)
            all_sources.append(c.source)
            all_pages.append(c.page)
        chunk_batches.append(cr.to_chromadb_batch())

    if not all_texts:
        log.warning("pipeline.no_chunks_to_embed")
        return

    embedding_result = model.embed_documents(
        all_texts,
        sources=all_sources,
        pages=all_pages,
    )
    log.info("pipeline.embedded", count=embedding_result.total_embedded)

    # ── 5. Store in ChromaDB ───────────────────────────────────────────────────
    store = get_vector_store()
    info = store.create_collection(exist_ok=True)
    log.info("pipeline.collection_ready", name=info.name, docs_before=info.document_count)

    # Flatten all chunk batches into one upsert
    merged_ids: list[str] = []
    merged_docs: list[str] = []
    merged_metas: list[dict] = []

    for batch in chunk_batches:
        merged_ids.extend(batch["ids"])
        merged_docs.extend(batch["documents"])
        merged_metas.extend(batch["metadatas"])

    insert_result = store.add_documents(
        ids=merged_ids,
        documents=merged_docs,
        embeddings=embedding_result.embeddings_only(),
        metadatas=merged_metas,
    )

    # ── 6. Verify ──────────────────────────────────────────────────────────────
    info_after = store.get_collection_info()

    elapsed = (time.perf_counter() - t0) * 1000

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  PDFs processed:    {len(successful)}")
    print(f"  Chunks created:    {len(merged_ids)}")
    print(f"  Chunks embedded:   {embedding_result.total_embedded}")
    print(f"  Chunks stored:     {insert_result.total_inserted}")
    print(f"  Collection total:  {info_after.document_count}")
    print(f"  Elapsed:           {elapsed:.0f} ms")
    print("=" * 60)

    # Peek at stored docs
    print("\n  Sample stored documents:")
    for doc in store.peek(3):
        print(f"    [{doc['id']}] {doc['text'][:80]}...")

    print("\n  [OK] PDFs -> Chunks -> Embeddings -> ChromaDB pipeline works!")
    print("=" * 60)


if __name__ == "__main__":
    main()
