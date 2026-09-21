"""
api/routers/chat.py
────────────────────
POST /chat — Clinical question answering via RAG + Ollama.

Request \u2192 Retriever \u2192 LLM \u2192 ChatResponse

The router is intentionally thin: it validates the request,
delegates retrieval to the Retriever dependency, delegates
generation to the LLM client, assembles the response, and returns.
No business logic lives here.

LLM integration:
    Uses langchain_ollama.OllamaLLM (or langchain_community fallback).
    The LLM is called synchronously in a thread-pool executor so the
    async event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, status

from api.dependencies import (
    RequestIdDep,
    RetrieverDep,
    SettingsDep,
)
from api.schemas.models import ChatRequest, ChatResponse, SourceReference
from app_logging.logger import get_logger

if TYPE_CHECKING:
    from clinical.retrieval.act_retriever import (
        ACTEvidenceResult,
        ACTRetriever,
    )
    from clinical.retrieval.cbt_retriever import CBTRetriever
    from clinical.retrieval.humanistic_retriever import HumanisticRetriever
    from clinical.retrieval.models import CBTEvidenceResult, HumanisticEvidenceResult
    from config.settings import Settings
    from rag.retriever import RetrievalResult, RetrievedChunk

_log = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── LLM prompt template ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a clinical psychology assistant supporting licensed psychologists \
and clinicians. You answer questions accurately, citing the provided context. \
You do not diagnose patients. If the context does not contain sufficient \
information to answer, say so clearly rather than speculating.

Always maintain clinical precision. Use DSM-5 / ICD-11 terminology where \
appropriate. Do not provide dosage or prescription advice.
"""

_RAG_PROMPT_TEMPLATE = """\
Use ONLY the following clinical reference excerpts to answer the question. \
If the answer is not in the excerpts, say you don't have enough information.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {query}

Answer:"""


def _build_prompt(query: str, context: str) -> str:
    return _RAG_PROMPT_TEMPLATE.format(
        context=context,
        query=query,
    )


def _call_llm(prompt: str, settings) -> str:
    """
    Call Ollama synchronously via LangChain.

    This function is designed to be run inside asyncio.run_in_executor()
    so it never blocks the event loop.

    Tries langchain_ollama first (preferred), falls back to
    langchain_community for older installs.
    """
    try:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(
            model=settings.llm.ollama_model,
            base_url=settings.llm.ollama_base_url,
            temperature=settings.llm.temperature,
            top_k=settings.llm.top_k,
            top_p=settings.llm.top_p,
            num_predict=settings.llm.max_tokens,
        )
    except ImportError:
        from langchain_community.llms import Ollama  # type: ignore[import]
        llm = Ollama(
            model=settings.llm.ollama_model,
            base_url=settings.llm.ollama_base_url,
            temperature=settings.llm.temperature,
        )

    full_prompt = f"{_SYSTEM_PROMPT}\n\n{prompt}"
    return llm.invoke(full_prompt)


# ── Framework-scoped retrieval + user-documents merge ────────────────────────

_USER_DOCUMENTS_COLLECTION = "user_documents"


def _get_framework_retriever(
    framework: str,
) -> ACTRetriever | CBTRetriever | HumanisticRetriever:
    """
    Return the process-wide retriever for a curated clinical framework.

    Imported lazily so the chat router stays light and the framework
    retrieval modules (which import the heavy embedding stack on demand)
    are only loaded when a framework is actually requested.
    """
    from clinical.retrieval import (
        get_act_retriever,
        get_cbt_retriever,
        get_humanistic_retriever,
    )

    factories = {
        "cbt": get_cbt_retriever,
        "act": get_act_retriever,
        "humanistic": get_humanistic_retriever,
    }
    return factories[framework]()


def _framework_result_to_retrieval_result(
    result: ACTEvidenceResult | CBTEvidenceResult | HumanisticEvidenceResult,
    query: str,
    n_results: int,
) -> RetrievalResult:
    """
    Adapt a framework evidence result (CBT/ACT/Humanistic) to the generic
    ``RetrievalResult`` shape consumed by the chat response assembly.
    """
    from rag.retriever import RetrievalResult, RetrievedChunk

    chunks = [
        RetrievedChunk(
            text=item.content,
            source=item.source,
            page=item.page,
            score=item.quality_rating or 0.0,
            chunk_id=item.chunk_id,
            rank=rank,
            metadata={
                "framework": getattr(item, "framework", None),
                "chapter": getattr(item, "chapter", None),
                "topic": getattr(item, "topic", None),
            },
        )
        for rank, item in enumerate(result.retrieved, start=1)
    ]
    return RetrievalResult(
        query=query,
        chunks=chunks,
        total_candidates=result.total_candidates,
        threshold_used=0.0,
        n_results_requested=n_results,
        elapsed_ms=result.elapsed_ms,
        embedding_ms=0.0,
        search_ms=result.elapsed_ms,
    )


def _retrieve_user_documents(
    query: str,
    settings: Settings,
    n_results: int,
    collection: str = _USER_DOCUMENTS_COLLECTION,
) -> list[RetrievedChunk]:
    """
    Dense-only retrieval against the user's uploaded-documents collection.

    Runs in a thread executor from the request handler so the async event
    loop is never blocked by embedding or vector search. Returns an empty
    list when the collection does not exist (no documents uploaded yet).
    """
    from rag.embeddings import embed_text
    from rag.retriever import RetrievedChunk
    from rag.vector_store import VectorStore

    store = VectorStore(
        collection_name=collection,
        persist_dir=settings.chroma.persist_dir,
    )
    if not store.collection_exists():
        _log.warning("chat.user_collection_missing", collection=collection)
        return []

    query_vector = embed_text(query, is_query=True)
    results = store.query_documents(query_vector=query_vector, n_results=n_results)
    return [
        RetrievedChunk(
            text=r.text,
            source=r.source,
            page=r.page,
            score=r.score,
            chunk_id=r.chunk_id,
            rank=r.rank,
            metadata=r.metadata,
        )
        for r in results
    ]


def _merge_chunks(
    primary: RetrievalResult,
    extra_chunks: list[RetrievedChunk],
    n_results: int,
) -> RetrievalResult:
    """
    Merge curated + user-document chunks, dedupe by chunk_id, re-rank by
    descending score, and truncate to ``n_results``.
    """
    from rag.retriever import RetrievalResult, RetrievedChunk

    seen: dict[str, RetrievedChunk] = {}
    for chunk in list(primary.chunks) + list(extra_chunks):
        key = chunk.chunk_id or f"{chunk.source}:{chunk.page}:{chunk.rank}"
        existing = seen.get(key)
        if existing is None or chunk.score > existing.score:
            seen[key] = chunk

    sorted_chunks = sorted(seen.values(), key=lambda c: c.score, reverse=True)[:n_results]
    merged = [
        RetrievedChunk(
            text=c.text,
            source=c.source,
            page=c.page,
            score=c.score,
            chunk_id=c.chunk_id,
            rank=rank,
            metadata=c.metadata,
        )
        for rank, c in enumerate(sorted_chunks, start=1)
    ]
    return RetrievalResult(
        query=primary.query,
        chunks=merged,
        total_candidates=primary.total_candidates + len(extra_chunks),
        threshold_used=primary.threshold_used,
        n_results_requested=n_results,
        elapsed_ms=primary.elapsed_ms,
        embedding_ms=primary.embedding_ms,
        search_ms=primary.search_ms,
    )


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Answer a clinical question via RAG",
    description=(
        "Embeds the query, retrieves the most relevant document chunks from "
        "ChromaDB, and generates a grounded answer via Ollama (Llama 3.1 8B). "
        "Returns the answer and source citations."
    ),
    responses={
        200: {"description": "Answer generated successfully"},
        400: {"description": "Invalid query"},
        503: {"description": "Retriever or LLM unavailable"},
    },
)
async def chat(
    body: ChatRequest,
    retriever: RetrieverDep,
    settings: SettingsDep,
    request_id: RequestIdDep,
) -> ChatResponse:
    t_total = time.perf_counter()

    _log.info(
        "chat.request",
        request_id=request_id,
        query_length=len(body.query),
        session_id=body.session_id,
        n_sources=body.n_sources,
        source_filter=body.source_filter,
        framework=body.framework,
        include_user_documents=body.include_user_documents,
    )

    # ── 1. Retrieve relevant chunks ───────────────────────────────────────────
    t_retrieval = time.perf_counter()
    try:
        if body.framework is not None:
            framework_retriever = _get_framework_retriever(body.framework)
            framework_result = await framework_retriever.retrieve_async(body.query)
            retrieval_result = _framework_result_to_retrieval_result(
                framework_result,
                body.query,
                body.n_sources,
            )
        else:
            page_range_tuple = tuple(body.page_range) if body.page_range else None
            retrieval_result = await retriever.aretrieve(
                body.query,
                n_results=body.n_sources,
                source_filter=body.source_filter,
                page_range=page_range_tuple,
            )

        # Merge the user's uploaded documents when requested.
        if body.include_user_documents:
            try:
                user_chunks = await asyncio.to_thread(
                    _retrieve_user_documents,
                    body.query,
                    settings,
                    body.n_sources,
                )
            except Exception as exc:
                _log.warning(
                    "chat.user_documents_failed",
                    request_id=request_id,
                    error=str(exc),
                )
                user_chunks = []
            if user_chunks:
                retrieval_result = _merge_chunks(
                    retrieval_result,
                    user_chunks,
                    body.n_sources,
                )
                _log.info(
                    "chat.user_documents_merged",
                    request_id=request_id,
                    merged_chunks=len(retrieval_result.chunks),
                )
    except Exception as exc:
        _log.error("chat.retrieval_failed", request_id=request_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    retrieval_ms = (time.perf_counter() - t_retrieval) * 1000

    _log.info(
        "chat.retrieved",
        request_id=request_id,
        chunks_found=len(retrieval_result.chunks),
        top_score=retrieval_result.top.score if retrieval_result.top else None,
        retrieval_ms=round(retrieval_ms, 2),
    )

    # ── 2. Build context string ───────────────────────────────────────────────
    if retrieval_result.found:
        context = retrieval_result.to_cited_context()
    else:
        context = "No relevant clinical references were found in the knowledge base."
        _log.warning(
            "chat.no_context",
            request_id=request_id,
            query=body.query[:80],
        )

    # ── 3. Generate answer via Ollama ─────────────────────────────────────────
    t_generation = time.perf_counter()
    prompt = _build_prompt(body.query, context)

    try:
        loop = asyncio.get_event_loop()
        answer: str = await loop.run_in_executor(
            None,
            _call_llm,
            prompt,
            settings,
        )
        answer = answer.strip()
    except Exception as exc:
        _log.error("chat.generation_failed", request_id=request_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"LLM generation failed: {exc}. "
                "Ensure Ollama is running: ollama serve"
            ),
        ) from exc

    generation_ms = (time.perf_counter() - t_generation) * 1000
    total_ms = (time.perf_counter() - t_total) * 1000

    # ── 4. Build source references ────────────────────────────────────────────
    sources = [
        SourceReference(
            text=chunk.text,
            source=chunk.source,
            page=chunk.page,
            score=chunk.score,
        )
        for chunk in retrieval_result.chunks
    ]

    _log.info(
        "chat.complete",
        request_id=request_id,
        answer_length=len(answer),
        sources=len(sources),
        retrieval_ms=round(retrieval_ms, 2),
        generation_ms=round(generation_ms, 2),
        total_ms=round(total_ms, 2),
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        session_id=body.session_id,
        model=settings.llm.ollama_model,
        retrieval_ms=round(retrieval_ms, 2),
        generation_ms=round(generation_ms, 2),
        total_ms=round(total_ms, 2),
    )
