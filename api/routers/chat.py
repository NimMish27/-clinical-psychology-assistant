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
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import (
    RequestIdDep,
    RetrieverDep,
    SettingsDep,
)
from api.schemas.models import ChatRequest, ChatResponse, SourceReference
from app_logging.logger import get_logger

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
    )

    # ── 1. Retrieve relevant chunks ───────────────────────────────────────────
    t_retrieval = time.perf_counter()
    try:
        page_range_tuple = tuple(body.page_range) if body.page_range else None
        retrieval_result = await retriever.aretrieve(
            body.query,
            n_results=body.n_sources,
            source_filter=body.source_filter,
            page_range=page_range_tuple,
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
