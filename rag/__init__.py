"""
rag — Retrieval-Augmented Generation core
"""
from rag.embeddings import (
    EmbeddedDocument, EmbeddingError, EmbeddingInferenceError,
    EmbeddingModel, EmbeddingResult, EmptyInputError, ModelLoadError,
    embed_documents, embed_text, load_embedding_model,
)
from rag.vector_store import (
    CollectionInfo, InsertResult, QueryError, QueryResult,
    VectorStore, VectorStoreError, get_vector_store,
)
from rag.retriever import (
    RetrievedChunk, RetrievalResult, Retriever, RetrieverError,
    EmptyQueryError, EmbeddingFailedError, SearchFailedError, NoResultsError,
    get_retriever,
)

__all__ = [
    "load_embedding_model", "embed_text", "embed_documents",
    "EmbeddingModel", "EmbeddingResult", "EmbeddedDocument",
    "EmbeddingError", "ModelLoadError", "EmbeddingInferenceError", "EmptyInputError",
    "get_vector_store", "VectorStore", "QueryResult", "InsertResult",
    "CollectionInfo", "VectorStoreError", "QueryError",
    "get_retriever", "Retriever", "RetrievedChunk", "RetrievalResult",
    "RetrieverError", "EmptyQueryError", "EmbeddingFailedError",
    "SearchFailedError", "NoResultsError",
]
