from __future__ import annotations

from typing import Any, AsyncGenerator

from langgraph.graph import END, START, StateGraph

from clinical.graph.nodes import (
    node_case_understanding,
    node_clinical_formulation,
    node_evidence_synthesis,
    node_missing_info,
    node_query_generation,
    node_response_generation,
    node_retrieval,
    node_safety_validation,
    node_therapeutic_planning,
)
from clinical.graph.state import GraphState
from app_logging.logger import get_logger

_log = get_logger(__name__)

_NODE_NAMES: dict[str, str] = {
    node_case_understanding.__name__: "case_understanding",
    node_query_generation.__name__: "query_generation",
    node_retrieval.__name__: "retrieval",
    node_evidence_synthesis.__name__: "evidence_synthesis",
    node_clinical_formulation.__name__: "clinical_formulation",
    node_missing_info.__name__: "missing_info",
    node_therapeutic_planning.__name__: "therapeutic_planning",
    node_response_generation.__name__: "response_generation",
    node_safety_validation.__name__: "safety_validation",
}

SEQUENCE = [
    "case_understanding",
    "query_generation",
    "retrieval",
    "evidence_synthesis",
    "clinical_formulation",
    "missing_info",
    "therapeutic_planning",
    "response_generation",
]


SEQUENCE = [
    "case_understanding",
    "query_generation",
    "retrieval",
    "evidence_synthesis",
    "clinical_formulation",
    "missing_info",
    "therapeutic_planning",
    "response_generation",
    "safety_validation",
]


def _tag(name: str) -> str:
    """Short label for logs / traces."""
    return f"gr_{name}"


# ── Builder ──────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Construct the LangGraph clinical pipeline.

    The graph evaluates **sequentially**: each node is invoked only after
    the previous node completes.  This avoids the complexity of fan-out /
    fan-in with conditional joins while keeping the pipeline predictable.

    Every node catches its own errors and records them in ``state["errors"]``
    so that downstream nodes can still operate on partial data.
    """
    builder = StateGraph(GraphState)

    builder.add_node(_tag("case_understanding"), node_case_understanding)
    builder.add_node(_tag("query_generation"), node_query_generation)
    builder.add_node(_tag("retrieval"), node_retrieval)
    builder.add_node(_tag("evidence_synthesis"), node_evidence_synthesis)
    builder.add_node(_tag("clinical_formulation"), node_clinical_formulation)
    builder.add_node(_tag("missing_info"), node_missing_info)
    builder.add_node(_tag("therapeutic_planning"), node_therapeutic_planning)
    builder.add_node(_tag("response_generation"), node_response_generation)
    builder.add_node(_tag("safety_validation"), node_safety_validation)

    # Sequential chain
    for i in range(len(SEQUENCE) - 1):
        builder.add_edge(_tag(SEQUENCE[i]), _tag(SEQUENCE[i + 1]))

    builder.add_edge(_tag("safety_validation"), END)
    builder.set_entry_point(_tag("case_understanding"))

    return builder


# ── Compiled graph (cached singleton) ────────────────────────

_graph: StateGraph | None = None


def get_graph() -> StateGraph:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── High-level runner ────────────────────────────────────────

async def run_pipeline(
    text: str,
    session_id: str | None = None,
) -> GraphState:
    """Convenience wrapper: build or fetch the compiled graph and run it.

    Returns the final state with all module outputs populated.
    """
    import time

    graph = get_graph()
    compiled = graph.compile()

    initial: dict[str, Any] = {
        "text": text,
        "session_id": session_id,
        "errors": {},
    }

    t0 = time.perf_counter()
    final = await compiled.ainvoke(initial)
    elapsed = time.perf_counter() - t0

    _log.info(
        "graph.pipeline_complete",
        elapsed_ms=round(elapsed * 1000, 2),
        error_count=len(final.get("errors", {})),
    )
    return final


# ── Streaming runner ─────────────────────────────────────────

async def stream_pipeline(
    text: str,
    session_id: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run the pipeline with per-node streaming.

    Yields a dict for every completed node:
    ``{"node": "<name>", "status": "ok" | "error", "error": ...}``
    followed by a final ``{"node": "__done__", "state": ...}``.
    """
    graph = get_graph()
    compiled = graph.compile()

    initial: dict[str, Any] = {
        "text": text,
        "session_id": session_id,
        "errors": {},
    }

    async for event in compiled.astream(initial):
        for node_name, output in event.items():
            raw = node_name.replace("gr_", "")
            if isinstance(output, dict):
                errors = output.get("errors", {})
                node_errs = {k: v for k, v in errors.items() if k == raw}
                if node_errs:
                    yield {"node": raw, "status": "error", "error": list(node_errs.values())[0]}
                    continue
            yield {"node": raw, "status": "ok"}

    yield {"node": "__done__", "state": compiled.ainvoke(initial)}
