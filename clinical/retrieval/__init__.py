"""
clinical/retrieval/ — Focused clinical literature retrieval.

Retrievers for framework-specific literature:
- ``CBTRetriever`` — CBT literature with hybrid retrieval, metadata filtering,
  and cross-encoder reranking.
- ``ACTRetriever`` — ACT literature with the same infrastructure.
- ``HumanisticRetriever`` — Humanistic therapy literature.

Usage::

    from clinical.retrieval import get_cbt_retriever, get_act_retriever

    cbt = get_cbt_retriever()
    cbt_result = cbt.retrieve("cognitive restructuring", ...)

    act = get_act_retriever()
    act_result = act.retrieve("acceptance and values", ...)
"""

from clinical.retrieval.act_retriever import (
    ACTEvidenceFilter,
    ACTEvidenceItem,
    ACTEvidenceResult,
    ACTRetriever,
    get_act_retriever,
)
from clinical.retrieval.cbt_retriever import CBTRetriever, get_cbt_retriever
from clinical.retrieval.humanistic_retriever import (
    HumanisticRetriever,
    get_humanistic_retriever,
)
from clinical.retrieval.models import (
    CBTEvidenceFilter,
    CBTEvidenceItem,
    CBTEvidenceResult,
    HumanisticEvidenceFilter,
    HumanisticEvidenceItem,
    HumanisticEvidenceResult,
)

__all__ = [
    "ACTRetriever",
    "get_act_retriever",
    "ACTEvidenceFilter",
    "ACTEvidenceItem",
    "ACTEvidenceResult",
    "CBTRetriever",
    "get_cbt_retriever",
    "CBTEvidenceFilter",
    "CBTEvidenceItem",
    "CBTEvidenceResult",
    "HumanisticRetriever",
    "get_humanistic_retriever",
    "HumanisticEvidenceFilter",
    "HumanisticEvidenceItem",
    "HumanisticEvidenceResult",
]
