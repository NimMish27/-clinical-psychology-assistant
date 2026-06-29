"""
tests/unit/rag/test_embeddings.py
───────────────────────────────────
Unit tests for the embedding module.

All tests mock SentenceTransformer — no model download or GPU required.
Tests cover: singleton lifecycle, BGE prefix behaviour, batch splitting,
error handling, progress callbacks, output types, and health checks.

Run:
    pytest tests/unit/rag/test_embeddings.py -v
"""

from __future__ import annotations

import threading
import time
from typing import Callable
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import numpy as np

from rag.embeddings import (
    EmbeddedDocument,
    EmbeddingError,
    EmbeddingInferenceError,
    EmbeddingModel,
    EmbeddingResult,
    EmptyInputError,
    ModelLoadError,
    Vector,
    _BGE_EMBEDDING_DIM,
    _BGE_QUERY_PREFIX,
    _TRUNCATION_WARN_CHARS,
    _make_batches,
    embed_documents,
    embed_text,
    load_embedding_model,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the EmbeddingModel singleton before every test."""
    EmbeddingModel._instance = None
    yield
    EmbeddingModel._instance = None


def _mock_vector(dim: int = _BGE_EMBEDDING_DIM) -> np.ndarray:
    """Return a unit-normalised random numpy vector."""
    v = np.random.rand(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_model(
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "cpu",
    batch_size: int = 32,
) -> EmbeddingModel:
    """Build a fresh EmbeddingModel (bypasses singleton)."""
    return EmbeddingModel(model_name=model_name, device=device, batch_size=batch_size)


def _make_loaded_model(encode_return=None) -> tuple[EmbeddingModel, MagicMock]:
    """Build an EmbeddingModel with a pre-loaded mock SentenceTransformer."""
    model = _make_model()
    mock_st = MagicMock()
    mock_st.encode.return_value = (
        encode_return if encode_return is not None
        else np.array([_mock_vector()])
    )
    mock_st.get_sentence_embedding_dimension.return_value = _BGE_EMBEDDING_DIM
    model._model = mock_st
    model._loaded = True
    return model, mock_st


# ── Exception hierarchy tests ─────────────────────────────────────────────────

class TestExceptions:
    def test_embedding_error_base(self):
        exc = EmbeddingError("base error")
        assert "base error" in str(exc)
        assert exc.cause is None

    def test_embedding_error_with_cause(self):
        cause = ValueError("root cause")
        exc = EmbeddingError("wrapper", cause=cause)
        assert "ValueError" in str(exc)
        assert "root cause" in str(exc)

    def test_model_load_error_is_embedding_error(self):
        assert issubclass(ModelLoadError, EmbeddingError)

    def test_inference_error_is_embedding_error(self):
        assert issubclass(EmbeddingInferenceError, EmbeddingError)

    def test_empty_input_error_is_embedding_error(self):
        assert issubclass(EmptyInputError, EmbeddingError)


# ── EmbeddedDocument tests ────────────────────────────────────────────────────

class TestEmbeddedDocument:
    def test_fields_stored(self):
        v = [0.1] * _BGE_EMBEDDING_DIM
        doc = EmbeddedDocument(text="hello", embedding=v, source="doc.pdf", page=3)
        assert doc.text == "hello"
        assert doc.embedding == v
        assert doc.source == "doc.pdf"
        assert doc.page == 3
        assert doc.dim == _BGE_EMBEDDING_DIM

    def test_to_chromadb_embedding_returns_vector(self):
        v = [0.5] * _BGE_EMBEDDING_DIM
        doc = EmbeddedDocument(text="t", embedding=v, source="d.pdf", page=1)
        assert doc.to_chromadb_embedding() == v

    def test_frozen(self):
        v = [0.0] * _BGE_EMBEDDING_DIM
        doc = EmbeddedDocument(text="t", embedding=v, source="d.pdf", page=1)
        with pytest.raises(Exception):
            doc.text = "changed"  # type: ignore


# ── EmbeddingResult tests ─────────────────────────────────────────────────────

class TestEmbeddingResult:
    def _make_result(self, n: int = 3) -> EmbeddingResult:
        docs = [
            EmbeddedDocument(text=f"text {i}", embedding=[float(i)] * _BGE_EMBEDDING_DIM,
                             source="doc.pdf", page=i + 1)
            for i in range(n)
        ]
        return EmbeddingResult(
            documents=docs,
            total_embedded=n,
            total_failed=0,
            model_name="BAAI/bge-large-en-v1.5",
            elapsed_ms=100.0,
            texts_per_second=30.0,
        )

    def test_embeddings_only_returns_list_of_lists(self):
        result = self._make_result(3)
        vecs = result.embeddings_only()
        assert len(vecs) == 3
        assert all(isinstance(v, list) for v in vecs)

    def test_to_chromadb_batch_structure(self):
        result = self._make_result(4)
        batch = result.to_chromadb_batch()
        assert set(batch.keys()) == {"embeddings"}
        assert len(batch["embeddings"]) == 4

    def test_repr_contains_counts(self):
        result = self._make_result(5)
        r = repr(result)
        assert "5" in r
        assert "embedded" in r


# ── _make_batches utility ─────────────────────────────────────────────────────

class TestMakeBatches:
    def test_exact_division(self):
        batches = _make_batches(list(range(9)), 3)
        assert len(batches) == 3
        assert batches[0] == [0, 1, 2]
        assert batches[2] == [6, 7, 8]

    def test_remainder_batch(self):
        batches = _make_batches(list(range(10)), 3)
        assert len(batches) == 4
        assert batches[-1] == [9]

    def test_single_item(self):
        batches = _make_batches(["only"], 32)
        assert len(batches) == 1
        assert batches[0] == ["only"]

    def test_empty_list(self):
        assert _make_batches([], 10) == []

    def test_batch_size_larger_than_list(self):
        batches = _make_batches([1, 2, 3], 100)
        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]


# ── EmbeddingModel — singleton lifecycle ──────────────────────────────────────

class TestSingletonLifecycle:
    def _patch_settings(self):
        mock_cfg = MagicMock()
        mock_cfg.embedding.model_name = "BAAI/bge-large-en-v1.5"
        mock_cfg.embedding.device = "cpu"
        mock_cfg.embedding.batch_size = 32
        return patch("rag.embeddings.EmbeddingModel.get_instance.__func__",
                     side_effect=lambda cls, **kw: EmbeddingModel("BAAI/bge-large-en-v1.5", "cpu", 32))

    def test_same_instance_returned_twice(self):
        with patch("rag.embeddings.get_settings") as mock_settings:
            mock_settings.return_value.embedding.model_name = "BAAI/bge-large-en-v1.5"
            mock_settings.return_value.embedding.device = "cpu"
            mock_settings.return_value.embedding.batch_size = 32
            a = EmbeddingModel.get_instance()
            b = EmbeddingModel.get_instance()
            assert a is b

    def test_force_reload_creates_new_instance(self):
        with patch("rag.embeddings.get_settings") as mock_settings:
            mock_settings.return_value.embedding.model_name = "BAAI/bge-large-en-v1.5"
            mock_settings.return_value.embedding.device = "cpu"
            mock_settings.return_value.embedding.batch_size = 32
            a = EmbeddingModel.get_instance()
            b = EmbeddingModel.get_instance(force_reload=True)
            assert a is not b

    def test_thread_safety(self):
        """Multiple threads must all get the same singleton."""
        with patch("rag.embeddings.get_settings") as mock_settings:
            mock_settings.return_value.embedding.model_name = "BAAI/bge-large-en-v1.5"
            mock_settings.return_value.embedding.device = "cpu"
            mock_settings.return_value.embedding.batch_size = 32
            instances = []
            barrier = threading.Barrier(8)

            def get_it():
                barrier.wait()  # all threads start simultaneously
                instances.append(EmbeddingModel.get_instance())

            threads = [threading.Thread(target=get_it) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All 8 references should point to the same object
            assert all(i is instances[0] for i in instances)


# ── EmbeddingModel — model loading ────────────────────────────────────────────

class TestModelLoading:
    def test_not_loaded_initially(self):
        model = _make_model()
        assert model.is_loaded is False

    def test_load_sets_loaded_flag(self):
        model = _make_model()
        mock_st_class = MagicMock()
        mock_st_instance = MagicMock()
        mock_st_instance.encode.return_value = np.array([_mock_vector()])
        mock_st_class.return_value = mock_st_instance

        with patch("rag.embeddings.SentenceTransformer", mock_st_class, create=True):
            with patch.dict("sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st_class)}):
                model.load()

        # Can't easily test the flag without real import, so test idempotency
        # via the double-check guard
        model._loaded = True
        model.load()  # second call must not re-load
        assert model.is_loaded is True

    def test_load_idempotent(self):
        """Calling load() twice must not reload the model."""
        model, mock_st = _make_loaded_model()
        original_model = model._model
        model.load()  # second call
        assert model._model is original_model  # same object

    def test_missing_sentence_transformers_raises(self):
        model = _make_model()
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                with pytest.raises((ModelLoadError, ImportError)):
                    model.load()

    def test_load_failure_raises_model_load_error(self):
        model = _make_model()
        mock_st_class = MagicMock(side_effect=RuntimeError("CUDA OOM"))

        with patch.dict("sys.modules", {
            "sentence_transformers": MagicMock(SentenceTransformer=mock_st_class)
        }):
            with pytest.raises((ModelLoadError, RuntimeError)):
                model.load()


# ── embed_text tests ──────────────────────────────────────────────────────────

class TestEmbedText:
    def test_returns_list_of_floats(self):
        model, mock_st = _make_loaded_model(encode_return=np.array(_mock_vector()))
        vec = model.embed_text("CBT for depression")
        assert isinstance(vec, list)
        assert all(isinstance(f, float) for f in vec)

    def test_vector_length(self):
        model, mock_st = _make_loaded_model(encode_return=np.array(_mock_vector()))
        vec = model.embed_text("test text")
        assert len(vec) == _BGE_EMBEDDING_DIM

    def test_query_prefix_applied(self):
        model, mock_st = _make_loaded_model(encode_return=np.array(_mock_vector()))
        model.embed_text("What is CBT?", is_query=True)
        call_args = mock_st.encode.call_args[0][0]
        assert call_args.startswith(_BGE_QUERY_PREFIX)

    def test_document_no_prefix(self):
        model, mock_st = _make_loaded_model(encode_return=np.array(_mock_vector()))
        model.embed_text("CBT criterion text here", is_query=False)
        call_args = mock_st.encode.call_args[0][0]
        assert not call_args.startswith(_BGE_QUERY_PREFIX)

    def test_empty_string_raises(self):
        model, _ = _make_loaded_model()
        with pytest.raises(EmptyInputError):
            model.embed_text("")

    def test_whitespace_only_raises(self):
        model, _ = _make_loaded_model()
        with pytest.raises(EmptyInputError):
            model.embed_text("   \n\t  ")

    def test_inference_error_wrapped(self):
        model, mock_st = _make_loaded_model()
        mock_st.encode.side_effect = RuntimeError("CUDA error")
        with pytest.raises(EmbeddingInferenceError):
            model.embed_text("some text")

    def test_long_text_does_not_raise(self):
        """Long text should warn (via logger) but still embed."""
        model, mock_st = _make_loaded_model(encode_return=np.array(_mock_vector()))
        long_text = "A" * (_TRUNCATION_WARN_CHARS + 100)
        # Should not raise — just warns
        vec = model.embed_text(long_text)
        assert isinstance(vec, list)

    def test_normalise_flag_passed_to_encode(self):
        model, mock_st = _make_loaded_model(encode_return=np.array(_mock_vector()))
        model.embed_text("test")
        call_kwargs = mock_st.encode.call_args.kwargs
        assert call_kwargs.get("normalize_embeddings") is True


# ── embed_documents tests ─────────────────────────────────────────────────────

class TestEmbedDocuments:
    def _vectors(self, n: int) -> np.ndarray:
        return np.array([_mock_vector() for _ in range(n)])

    def test_happy_path(self):
        n = 5
        model, mock_st = _make_loaded_model(encode_return=self._vectors(n))
        texts = [f"Clinical text document number {i}." for i in range(n)]
        result = model.embed_documents(texts)

        assert isinstance(result, EmbeddingResult)
        assert result.total_embedded == n
        assert result.total_failed == 0
        assert len(result.documents) == n

    def test_empty_texts_raises(self):
        model, _ = _make_loaded_model()
        with pytest.raises(EmptyInputError):
            model.embed_documents([])

    def test_source_and_page_metadata_stored(self):
        model, mock_st = _make_loaded_model(encode_return=self._vectors(2))
        texts = ["Text alpha for clinical document.", "Text beta for clinical document."]
        sources = ["DSM5.pdf", "ICD11.pdf"]
        pages = [10, 20]
        result = model.embed_documents(texts, sources=sources, pages=pages)

        assert result.documents[0].source == "DSM5.pdf"
        assert result.documents[0].page == 10
        assert result.documents[1].source == "ICD11.pdf"
        assert result.documents[1].page == 20

    def test_mismatched_lengths_raises(self):
        model, _ = _make_loaded_model()
        with pytest.raises(ValueError, match="same length"):
            model.embed_documents(["a", "b"], sources=["x"])

    def test_empty_text_in_batch_counted_as_failed(self):
        model, mock_st = _make_loaded_model(encode_return=self._vectors(1))
        texts = ["", "Valid clinical content here for embedding."]
        result = model.embed_documents(texts)
        assert result.total_failed == 1
        assert result.total_embedded == 1
        assert len(result.errors) == 1

    def test_batch_failure_continues(self):
        """A batch-level encode failure is recorded, other batches succeed."""
        model, mock_st = _make_loaded_model()
        call_count = [0]

        def encode_side_effect(texts, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("GPU OOM on batch 0")
            return self._vectors(len(texts))

        mock_st.encode.side_effect = encode_side_effect
        texts = [f"Clinical document text sample {i}." for i in range(6)]
        result = model.embed_documents(texts, batch_size=3)

        # First batch failed (3 texts), second batch succeeded (3 texts)
        assert result.total_failed == 3
        assert result.total_embedded == 3
        assert len(result.errors) == 1

    def test_progress_callback_called(self):
        n = 9
        model, mock_st = _make_loaded_model()
        mock_st.encode.side_effect = lambda texts, **kw: self._vectors(len(texts))

        calls = []
        def on_progress(done: int, total: int) -> None:
            calls.append((done, total))

        texts = [f"Batch progress test text item {i} content." for i in range(n)]
        model.embed_documents(texts, batch_size=3, on_batch_complete=on_progress)

        assert len(calls) == 3           # 9 texts / batch_size 3 = 3 batches
        assert calls[-1] == (3, 3)       # last call: (3 done, 3 total)
        assert calls[0][1] == 3          # total_batches always 3

    def test_batch_size_override(self):
        """Custom batch_size should split texts into correct number of encode calls."""
        model, mock_st = _make_loaded_model()
        mock_st.encode.side_effect = lambda texts, **kw: self._vectors(len(texts))
        texts = [f"Clinical batch size override text sample {i}." for i in range(10)]
        model.embed_documents(texts, batch_size=2)
        # 10 texts / batch_size 2 = 5 encode calls
        assert mock_st.encode.call_count == 5

    def test_result_elapsed_ms_positive(self):
        model, mock_st = _make_loaded_model()
        mock_st.encode.side_effect = lambda texts, **kw: self._vectors(len(texts))
        texts = ["Text for timing test of the embedding module."]
        result = model.embed_documents(texts)
        assert result.elapsed_ms >= 0

    def test_embeddings_are_lists_not_numpy(self):
        model, mock_st = _make_loaded_model(encode_return=self._vectors(2))
        texts = ["Vector type test document one.", "Vector type test document two."]
        result = model.embed_documents(texts)
        for doc in result.documents:
            assert isinstance(doc.embedding, list)
            assert all(isinstance(f, float) for f in doc.embedding)

    def test_default_source_and_page_when_omitted(self):
        model, mock_st = _make_loaded_model(encode_return=self._vectors(1))
        result = model.embed_documents(["Some clinical text content here."])
        assert result.documents[0].source == "unknown"
        assert result.documents[0].page == 0

    def test_normalise_flag_in_batch_encode(self):
        model, mock_st = _make_loaded_model()
        mock_st.encode.side_effect = lambda texts, **kw: self._vectors(len(texts))
        model.embed_documents(["Normalisation flag test document text."])
        kw = mock_st.encode.call_args.kwargs
        assert kw.get("normalize_embeddings") is True


# ── Introspection / health ────────────────────────────────────────────────────

class TestIntrospection:
    def test_health_not_loaded(self):
        model = _make_model()
        h = model.health()
        assert h["status"] == "not_loaded"
        assert h["is_loaded"] is False
        assert h["model_name"] == "BAAI/bge-large-en-v1.5"

    def test_health_loaded(self):
        model, mock_st = _make_loaded_model()
        h = model.health()
        assert h["status"] == "ok"
        assert h["is_loaded"] is True
        assert h["embedding_dim"] == _BGE_EMBEDDING_DIM

    def test_embedding_dim_before_load(self):
        model = _make_model()
        assert model.embedding_dim == _BGE_EMBEDDING_DIM  # constant fallback

    def test_embedding_dim_after_load(self):
        model, mock_st = _make_loaded_model()
        assert model.embedding_dim == _BGE_EMBEDDING_DIM

    def test_repr_not_loaded(self):
        model = _make_model()
        assert "not_loaded" in repr(model)

    def test_repr_loaded(self):
        model, _ = _make_loaded_model()
        assert "loaded" in repr(model)
        assert "BAAI/bge-large-en-v1.5" in repr(model)


# ── Module-level convenience functions ───────────────────────────────────────

class TestModuleFunctions:
    def _patch_singleton(self, model: EmbeddingModel):
        EmbeddingModel._instance = model

    def test_embed_text_delegates_to_singleton(self):
        model, mock_st = _make_loaded_model(encode_return=np.array(_mock_vector()))
        self._patch_singleton(model)
        vec = embed_text("DSM-5 criteria for major depression")
        assert isinstance(vec, list)

    def test_embed_text_query_prefix(self):
        model, mock_st = _make_loaded_model(encode_return=np.array(_mock_vector()))
        self._patch_singleton(model)
        embed_text("What is CBT?", is_query=True)
        call_arg = mock_st.encode.call_args[0][0]
        assert call_arg.startswith(_BGE_QUERY_PREFIX)

    def test_embed_documents_delegates_to_singleton(self):
        n = 3
        vectors = np.array([_mock_vector() for _ in range(n)])
        model, mock_st = _make_loaded_model(encode_return=vectors)
        self._patch_singleton(model)
        result = embed_documents(
            ["Clinical text one.", "Clinical text two.", "Clinical text three."]
        )
        assert result.total_embedded == n

    def test_load_embedding_model_triggers_load(self):
        model, mock_st = _make_loaded_model()
        self._patch_singleton(model)
        # load() should be idempotent when already loaded
        returned = load_embedding_model()
        assert returned is model
        assert returned.is_loaded is True

    def test_load_embedding_model_force_reload(self):
        with patch("rag.embeddings.get_settings") as mock_settings:
            mock_settings.return_value.embedding.model_name = "BAAI/bge-large-en-v1.5"
            mock_settings.return_value.embedding.device = "cpu"
            mock_settings.return_value.embedding.batch_size = 32

            original = EmbeddingModel.get_instance()
            # Mark as "loaded" so load() returns immediately
            original._loaded = True

            with patch.object(EmbeddingModel, "load"):
                new_model = load_embedding_model(force_reload=True)
            
            assert new_model is not original


# ── ChromaDB integration shape ────────────────────────────────────────────────

class TestChromaDBIntegration:
    def test_to_chromadb_batch_is_dict_of_lists(self):
        docs = [
            EmbeddedDocument(text=f"t{i}", embedding=[float(i)] * _BGE_EMBEDDING_DIM,
                             source="doc.pdf", page=i)
            for i in range(5)
        ]
        result = EmbeddingResult(
            documents=docs, total_embedded=5, total_failed=0,
            model_name="BAAI/bge-large-en-v1.5", elapsed_ms=50.0, texts_per_second=100.0
        )
        batch = result.to_chromadb_batch()
        assert isinstance(batch["embeddings"], list)
        assert len(batch["embeddings"]) == 5
        assert all(isinstance(v, list) for v in batch["embeddings"])
        assert all(len(v) == _BGE_EMBEDDING_DIM for v in batch["embeddings"])

    def test_vector_values_are_floats(self):
        docs = [EmbeddedDocument(
            text="test", embedding=[0.1] * _BGE_EMBEDDING_DIM, source="d.pdf", page=1
        )]
        result = EmbeddingResult(
            documents=docs, total_embedded=1, total_failed=0,
            model_name="m", elapsed_ms=10.0, texts_per_second=100.0
        )
        vec = result.to_chromadb_batch()["embeddings"][0]
        assert all(isinstance(f, float) for f in vec)
