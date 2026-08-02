from __future__ import annotations

import time
from typing import Any, AsyncGenerator

from langgraph.graph import END, START, StateGraph

from clinical.graph.nodes import (
    node_case_understanding,
    node_clinical_formulation,
    node_evidence_synthesis,
    node_framework_analysis,
    node_framework_comparison,
    node_framework_selection,
    node_missing_info,
    node_query_generation,
    node_response_generation,
    node_retrieval,
    node_safety_validation,
    node_session_planning,
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
    node_framework_analysis.__name__: "framework_analysis",
    node_framework_comparison.__name__: "framework_comparison",
    node_framework_selection.__name__: "framework_selection",
    node_clinical_formulation.__name__: "clinical_formulation",
    node_missing_info.__name__: "missing_info",
    node_therapeutic_planning.__name__: "therapeutic_planning",
    node_session_planning.__name__: "session_planning",
    node_response_generation.__name__: "response_generation",
    node_safety_validation.__name__: "safety_validation",
}


SEQUENCE = [
    "case_understanding",
    "query_generation",
    "retrieval",
    "evidence_synthesis",
    "framework_analysis",
    "framework_comparison",
    "framework_selection",
    "clinical_formulation",
    "missing_info",
    "therapeutic_planning",
    "session_planning",
    "response_generation",
    "safety_validation",
]


def _tag(name: str) -> str:
    """Short label for logs / traces."""
    return f"gr_{name}"


def _after_framework_selection(state: GraphState) -> str:
    """Route past the human-in-the-loop selection gate.

    Returns ``"pause"`` when the run is waiting for the psychologist to
    choose a framework, otherwise ``"continue"`` into the downstream chain.
    """
    if state.get("awaiting_selection"):
        return "pause"
    return "continue"


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
    builder.add_node(_tag("framework_analysis"), node_framework_analysis)
    builder.add_node(_tag("framework_comparison"), node_framework_comparison)
    builder.add_node(_tag("framework_selection"), node_framework_selection)
    builder.add_node(_tag("clinical_formulation"), node_clinical_formulation)
    builder.add_node(_tag("missing_info"), node_missing_info)
    builder.add_node(_tag("therapeutic_planning"), node_therapeutic_planning)
    builder.add_node(_tag("session_planning"), node_session_planning)
    builder.add_node(_tag("response_generation"), node_response_generation)
    builder.add_node(_tag("safety_validation"), node_safety_validation)

    # Sequential chain — the edge out of framework_selection is
    # conditional (pause for selection vs. continue downstream).
    for i in range(len(SEQUENCE) - 1):
        src, dst = SEQUENCE[i], SEQUENCE[i + 1]
        if src == "framework_selection":
            continue
        builder.add_edge(_tag(src), _tag(dst))

    builder.add_conditional_edges(
        _tag("framework_selection"),
        _after_framework_selection,
        {
            "continue": _tag("clinical_formulation"),
            "pause": END,
        },
    )

    builder.add_edge(_tag("safety_validation"), END)
    builder.set_entry_point(_tag("case_understanding"))

    return builder


RESUME_SEQUENCE = [
    "framework_selection",
    "clinical_formulation",
    "missing_info",
    "therapeutic_planning",
    "session_planning",
    "response_generation",
    "safety_validation",
]


def build_resume_graph() -> StateGraph:
    """Construct the downstream subgraph used to resume a paused workflow.

    Re-enters the pipeline at the framework-selection gate. All upstream
    outputs are supplied via the persisted state, so the subgraph only
    contains the nodes after the human-in-the-loop gate.
    """
    builder = StateGraph(GraphState)

    builder.add_node(_tag("framework_selection"), node_framework_selection)
    builder.add_node(_tag("clinical_formulation"), node_clinical_formulation)
    builder.add_node(_tag("missing_info"), node_missing_info)
    builder.add_node(_tag("therapeutic_planning"), node_therapeutic_planning)
    builder.add_node(_tag("session_planning"), node_session_planning)
    builder.add_node(_tag("response_generation"), node_response_generation)
    builder.add_node(_tag("safety_validation"), node_safety_validation)

    builder.add_conditional_edges(
        _tag("framework_selection"),
        _after_framework_selection,
        {
            "continue": _tag("clinical_formulation"),
            "pause": END,
        },
    )

    for i in range(len(RESUME_SEQUENCE) - 1):
        src, dst = RESUME_SEQUENCE[i], RESUME_SEQUENCE[i + 1]
        if src == "framework_selection":
            continue
        builder.add_edge(_tag(src), _tag(dst))

    builder.add_edge(_tag("safety_validation"), END)
    builder.set_entry_point(_tag("framework_selection"))

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
    text: str | None = None,
    session_id: str | None = None,
    *,
    workflow_id: str | None = None,
    initial_state: dict[str, Any] | None = None,
    reported_diagnosis: str | None = None,
    patient_metadata: dict[str, Any] | None = None,
) -> GraphState:
    """Run the clinical pipeline and return the final state.

    Two modes:
      - **Fresh run**: provide ``text`` (optionally ``session_id`` and
        ``workflow_id`` to enable the human-in-the-loop pause).
      - **Resume**: provide ``initial_state`` (a previously persisted
        paused state); execution re-enters at ``framework_selection``.

    ``reported_diagnosis`` and ``patient_metadata`` are optional inputs
    provided by the psychologist; they are passed to the framework
    analyses (clearly labelled as reported, never validated) and used by
    the safety validator to ensure the final report does not present a
    reported diagnosis as confirmed.

    In fresh resumable runs (``workflow_id`` set) the graph halts at the
    framework-selection gate with ``awaiting_selection=True``; the caller
    persists the returned state and later resumes it via ``initial_state``.
    """
    if initial_state is not None:
        graph = build_resume_graph()
        compiled = graph.compile()
        final = await compiled.ainvoke(dict(initial_state))
        _log.info(
            "graph.pipeline_resumed",
            error_count=len(final.get("errors", {})),
        )
        return final

    if not text:
        raise ValueError("run_pipeline requires text or initial_state")

    graph = get_graph()
    compiled = graph.compile()

    initial: dict[str, Any] = {
        "text": text,
        "session_id": session_id,
        "workflow_id": workflow_id,
        "reported_diagnosis": reported_diagnosis,
        "patient_metadata": patient_metadata,
        "errors": {},
    }

    t0 = time.perf_counter()
    final = await compiled.ainvoke(initial)
    elapsed = time.perf_counter() - t0

    _log.info(
        "graph.pipeline_complete",
        elapsed_ms=round(elapsed * 1000, 2),
        error_count=len(final.get("errors", {})),
        awaiting_selection=final.get("awaiting_selection", False),
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
