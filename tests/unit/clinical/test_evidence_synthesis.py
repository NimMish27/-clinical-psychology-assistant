from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from clinical.evidence_synthesis import (
    Agreement,
    EvidenceSynthesisResult,
    EvidenceSynthesizer,
    Finding,
    Implication,
    Theme,
    Uncertainty,
)


# ═══════════════════════════════════════════════════════════════
# Model unit tests
# ═══════════════════════════════════════════════════════════════

class TestFinding:
    def test_basic_creation(self):
        f = Finding(
            statement="CBT is effective for GAD",
            supporting_sources=["Roth&Fonagy.pdf, p.45"],
            confidence=0.85,
        )
        assert "CBT" in f.statement
        assert len(f.supporting_sources) == 1
        assert f.confidence == 0.85

    def test_frozen(self):
        f = Finding(statement="test finding", supporting_sources=["a"], confidence=0.5)
        with pytest.raises(Exception):
            f.statement = "changed"

    def test_min_length_enforced_statement(self):
        with pytest.raises(ValueError):
            Finding(statement="ab", supporting_sources=["a"], confidence=0.5)

    def test_min_length_enforced_sources(self):
        with pytest.raises(ValueError):
            Finding(statement="valid statement", supporting_sources=[], confidence=0.5)

    def test_confidence_clamped(self):
        with pytest.raises(ValueError):
            Finding(statement="test", supporting_sources=["a"], confidence=1.5)
        with pytest.raises(ValueError):
            Finding(statement="test", supporting_sources=["a"], confidence=-0.1)


class TestTheme:
    def test_basic_creation(self):
        t = Theme(
            name="CBT efficacy",
            description="CBT consistently outperforms waitlist",
            prevalence=0.9,
        )
        assert t.name == "CBT efficacy"
        assert t.prevalence == 0.9

    def test_frozen(self):
        t = Theme(name="test theme", description="valid description", prevalence=0.5)
        with pytest.raises(Exception):
            t.name = "changed"


class TestAgreement:
    def test_basic_creation(self):
        a = Agreement(
            topic="CBT first-line",
            consensus="All guidelines recommend CBT as first-line",
            supporting_sources=["NICE.pdf", "APA.pdf"],
        )
        assert a.topic == "CBT first-line"
        assert len(a.supporting_sources) == 2

    def test_frozen(self):
        a = Agreement(topic="test topic", consensus="valid consensus text", supporting_sources=["s"])
        with pytest.raises(Exception):
            a.topic = "changed"


class TestUncertainty:
    def test_basic_creation(self):
        u = Uncertainty(
            topic="Long-term outcomes",
            description="Few studies beyond 12-month follow-up",
            implications="Caution in predicting long-term prognosis",
        )
        assert u.topic == "Long-term outcomes"

    def test_frozen(self):
        u = Uncertainty(topic="test topic", description="valid description", implications="valid implication")
        with pytest.raises(Exception):
            u.topic = "changed"


class TestImplication:
    def test_basic_creation(self):
        imp = Implication(
            recommendation="Monitor for relapse monthly",
            strength=0.7,
            source="BPD Guideline, p.22",
        )
        assert imp.strength == 0.7
        assert imp.source == "BPD Guideline, p.22"

    def test_no_source(self):
        imp = Implication(
            recommendation="Monitor for relapse",
            strength=0.6,
        )
        assert imp.source is None

    def test_frozen(self):
        imp = Implication(recommendation="valid recommendation", strength=0.5)
        with pytest.raises(Exception):
            imp.recommendation = "changed"


class TestEvidenceSynthesisResult:
    def test_defaults(self):
        r = EvidenceSynthesisResult()
        assert r.key_findings == []
        assert r.common_themes == []
        assert r.areas_of_agreement == []
        assert r.areas_of_uncertainty == []
        assert r.practical_implications == []
        assert r.overall_summary == ""
        assert r.confidence == 0.0
        assert r.chunks_analysed == 0
        assert r.synthesis_ms >= 0.0
        assert r.synthesised_at is not None

    def test_with_data(self):
        r = EvidenceSynthesisResult(
            key_findings=[
                Finding(statement="Finding 1", supporting_sources=["s1"], confidence=0.8),
            ],
            common_themes=[
                Theme(name="Theme 1", description="Desc text here", prevalence=0.7),
            ],
            areas_of_agreement=[
                Agreement(topic="Topic 1", consensus="Consensus", supporting_sources=["s1"]),
            ],
            areas_of_uncertainty=[
                Uncertainty(topic="Uncertainty 1", description="Valid description", implications="Valid implications"),
            ],
            practical_implications=[
                Implication(recommendation="Rec 1", strength=0.6),
            ],
            overall_summary="Summary text",
            confidence=0.75,
            chunks_analysed=5,
        )
        assert len(r.key_findings) == 1
        assert len(r.common_themes) == 1
        assert len(r.areas_of_agreement) == 1
        assert len(r.areas_of_uncertainty) == 1
        assert len(r.practical_implications) == 1
        assert r.overall_summary == "Summary text"
        assert r.confidence == 0.75
        assert r.chunks_analysed == 5

    def test_frozen(self):
        r = EvidenceSynthesisResult(overall_summary="test", confidence=0.5)
        with pytest.raises(Exception):
            r.overall_summary = "changed"


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@dataclass
class FakeChunk:
    text: str
    source: str = "DSM5.pdf"
    page: int = 12
    score: float = 0.85


GOOD_JSON_RESPONSE = """{
  "key_findings": [
    {
      "statement": "CBT is the first-line treatment for GAD",
      "supporting_sources": ["NICE Guideline.pdf, p.15", "APA Practice.pdf, p.22"],
      "confidence": 0.9
    },
    {
      "statement": "SSRIs show moderate effect sizes for GAD",
      "supporting_sources": ["Bandelow2020.pdf, p.8"],
      "confidence": 0.75
    }
  ],
  "common_themes": [
    {
      "name": "CBT as gold standard",
      "description": "CBT consistently recommended across all major guidelines as first-line psychological treatment for GAD",
      "prevalence": 0.95
    }
  ],
  "areas_of_agreement": [
    {
      "topic": "CBT efficacy",
      "consensus": "All reviewed sources agree CBT is superior to no treatment and most active comparators",
      "supporting_sources": ["NICE Guideline.pdf", "APA Practice.pdf", "Cochrane2021.pdf"]
    }
  ],
  "areas_of_uncertainty": [
    {
      "topic": "Optimal number of CBT sessions",
      "description": "Studies report varying session counts from 6 to 20 with no clear dose-response consensus",
      "implications": "Clinicians should individualise treatment duration based on response"
    }
  ],
  "practical_implications": [
    {
      "recommendation": "Start with 12 weekly sessions of CBT before considering medication",
      "strength": 0.85,
      "source": "NICE Guideline.pdf"
    }
  ],
  "overall_summary": "CBT is strongly supported as first-line treatment for GAD. SSRIs offer a moderate second-line option. The optimal session count remains uncertain.",
  "confidence": 0.85
}"""

EMPTY_JSON_RESPONSE = """{
  "key_findings": [],
  "common_themes": [],
  "areas_of_agreement": [],
  "areas_of_uncertainty": [],
  "practical_implications": [],
  "overall_summary": "No relevant evidence found.",
  "confidence": 0.0
}"""

PARTIAL_JSON_RESPONSE = """{
  "key_findings": [
    {
      "statement": "Only finding",
      "supporting_sources": ["source.pdf"],
      "confidence": 0.5
    }
  ]
}"""

MARKDOWN_FENCED_RESPONSE = """```json
{
  "key_findings": [
    {
      "statement": "Finding from markdown fence",
      "supporting_sources": ["test.pdf"],
      "confidence": 0.7
    }
  ],
  "common_themes": [],
  "areas_of_agreement": [],
  "areas_of_uncertainty": [],
  "practical_implications": [],
  "overall_summary": "Markdown fenced response test.",
  "confidence": 0.7
}
```"""

INVALID_JSON_RESPONSE = "This is not JSON at all"


# ═══════════════════════════════════════════════════════════════
# EvidenceSynthesizer tests
# ═══════════════════════════════════════════════════════════════

class TestEvidenceSynthesizer:
    def make_synthesizer(self, response: str = GOOD_JSON_RESPONSE) -> EvidenceSynthesizer:
        llm = MagicMock()
        llm.generate = AsyncMock(return_value=response)
        return EvidenceSynthesizer(llm)

    def make_chunks(self, n: int = 3) -> list[FakeChunk]:
        return [
            FakeChunk(
                text=f"Cognitive behavioural therapy is effective for treating anxiety disorders chunk {i}",
                source=f"source_{i}.pdf",
                page=10 + i,
                score=0.9 - (i * 0.05),
            )
            for i in range(n)
        ]

    # ── Happy path ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_basic_synthesis(self):
        synth = self.make_synthesizer()
        chunks = self.make_chunks(3)
        result = await synth.synthesise(chunks, query="CBT for GAD")
        assert isinstance(result, EvidenceSynthesisResult)
        assert len(result.key_findings) == 2
        assert len(result.common_themes) == 1
        assert len(result.areas_of_agreement) == 1
        assert len(result.areas_of_uncertainty) == 1
        assert len(result.practical_implications) == 1
        assert result.overall_summary
        assert result.confidence == 0.85
        assert result.chunks_analysed == 3
        assert result.synthesis_ms > 0

    @pytest.mark.asyncio
    async def test_synthesis_with_case_context(self):
        synth = self.make_synthesizer()
        chunks = self.make_chunks(2)
        result = await synth.synthesise(
            chunks,
            query="CBT for GAD",
            case_context="42-year-old female with generalised anxiety disorder",
        )
        assert len(result.key_findings) == 2
        assert result.chunks_analysed == 2

    @pytest.mark.asyncio
    async def test_synthesis_without_query(self):
        synth = self.make_synthesizer()
        chunks = self.make_chunks(2)
        result = await synth.synthesise(chunks)
        assert len(result.key_findings) == 2

    # ── Empty / edge cases ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        synth = self.make_synthesizer()
        result = await synth.synthesise([])
        assert result.key_findings == []
        assert result.chunks_analysed == 0
        assert "No evidence chunks" in result.overall_summary

    @pytest.mark.asyncio
    async def test_single_chunk(self):
        synth = self.make_synthesizer()
        chunks = self.make_chunks(1)
        result = await synth.synthesise(chunks)
        assert result.chunks_analysed == 1
        assert len(result.key_findings) == 2

    @pytest.mark.asyncio
    async def test_empty_llm_response(self):
        synth = self.make_synthesizer(response=EMPTY_JSON_RESPONSE)
        chunks = self.make_chunks(3)
        result = await synth.synthesise(chunks)
        assert result.key_findings == []
        assert result.common_themes == []
        assert result.chunks_analysed == 3

    # ── Response parsing ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_markdown_fenced_response(self):
        synth = self.make_synthesizer(response=MARKDOWN_FENCED_RESPONSE)
        chunks = self.make_chunks(2)
        result = await synth.synthesise(chunks)
        assert len(result.key_findings) == 1
        assert "markdown fence" in result.key_findings[0].statement

    @pytest.mark.asyncio
    async def test_invalid_json_response_falls_back(self):
        synth = self.make_synthesizer(response=INVALID_JSON_RESPONSE)
        chunks = self.make_chunks(2)
        result = await synth.synthesise(chunks)
        assert result.chunks_analysed == 2
        assert "failed" in result.overall_summary.lower()

    @pytest.mark.asyncio
    async def test_partial_json_response(self):
        synth = self.make_synthesizer(response=PARTIAL_JSON_RESPONSE)
        chunks = self.make_chunks(2)
        result = await synth.synthesise(chunks)
        assert len(result.key_findings) == 1
        assert result.overall_summary == ""

    # ── Chunk type handling ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_dict_chunks(self):
        synth = self.make_synthesizer()
        chunks = [
            {"text": "CBT is effective", "source": "source.pdf", "page": 5, "score": 0.9},
            {"text": "SSRIs are effective", "source": "source2.pdf", "page": 10, "score": 0.8},
        ]
        result = await synth.synthesise(chunks, query="treatment options")
        assert result.chunks_analysed == 2
        assert len(result.key_findings) == 2

    @pytest.mark.asyncio
    async def test_mixed_chunk_types(self):
        synth = self.make_synthesizer()
        chunks: list = [
            FakeChunk(text="CBT is effective", source="fake.pdf", page=1, score=0.9),
            {"text": "SSRIs are effective", "source": "dict.pdf", "page": 2, "score": 0.8},
        ]
        result = await synth.synthesise(chunks, query="test")
        assert result.chunks_analysed == 2

    # ── Chunk with very long text ───────────────────────────

    @pytest.mark.asyncio
    async def test_long_text_truncated_in_prompt(self):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value=GOOD_JSON_RESPONSE)
        synth = EvidenceSynthesizer(llm)

        chunks = [FakeChunk(text="A" * 2000, source="long.pdf", page=1, score=0.9)]
        await synth.synthesise(chunks)

        # Check that the prompt contained truncated (800 char) text
        prompt = llm.generate.call_args[1]["prompt"]
        assert len(prompt) < 1200  # truncated not full 2000

    # ── LLM error handling ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_llm_raises_exception(self):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("Ollama not available"))
        synth = EvidenceSynthesizer(llm)
        chunks = self.make_chunks(2)
        result = await synth.synthesise(chunks)
        assert result.chunks_analysed == 2
        assert "failed" in result.overall_summary.lower()

    # ── Confidence clamping ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_confidence_clamping_in_builders(self):
        llm = MagicMock()
        llm.generate = AsyncMock(return_value=
            '{"key_findings": [{"statement": "test finding", "supporting_sources": ["s"], "confidence": 1.5}], '
            '"common_themes": [{"name": "valid", "description": "valid desc here", "prevalence": -0.5}], '
            '"practical_implications": [{"recommendation": "valid recommend", "strength": 2.0}], '
            '"areas_of_agreement": [], "areas_of_uncertainty": [], '
            '"overall_summary": "test", "confidence": 1.2}')
        synth = EvidenceSynthesizer(llm)
        result = await synth.synthesise(self.make_chunks(1))
        assert result.key_findings[0].confidence == 1.0
        assert result.common_themes[0].prevalence == 0.0
        assert result.practical_implications[0].strength == 1.0
        assert result.confidence == 1.0

    # ── Unpack chunk edge cases ──────────────────────────────

    def test_unpack_chunk_with_missing_attrs(self):
        synth = self.make_synthesizer()
        class Minimal:
            text = "some text"
        text, source, page, score = synth._unpack_chunk(Minimal())
        assert text == "some text"
        assert source == "unknown"
        assert page == 0
        assert score == 0.0

    def test_unpack_unknown_type(self):
        synth = self.make_synthesizer()
        text, source, page, score = synth._unpack_chunk(42)
        assert text == "42"
        assert source == "unknown"
        assert page == 0
        assert score == 0.0

    # ── Builder resilience ──────────────────────────────────

    def test_build_findings_skips_invalid(self):
        synth = self.make_synthesizer()
        raw = [
            {"statement": "This is a valid finding", "supporting_sources": ["s"], "confidence": 0.8},
            {"statement": "", "supporting_sources": ["s"], "confidence": 0.5},
            {"not_statement": "missing"},
            42,
            None,
        ]
        findings = synth._build_findings(raw)
        assert len(findings) == 1
        assert findings[0].statement == "This is a valid finding"

    def test_build_themes_skips_invalid(self):
        synth = self.make_synthesizer()
        raw = [
            {"name": "Valid Theme", "description": "valid description", "prevalence": 0.7},
            {"name": "", "description": "valid", "prevalence": 0.5},
            "invalid",
        ]
        themes = synth._build_themes(raw)
        assert len(themes) == 1

    def test_build_agreements_skips_invalid(self):
        synth = self.make_synthesizer()
        raw = [
            {"topic": "Valid Topic", "consensus": "valid consensus text", "supporting_sources": ["s"]},
            {},
            {"topic": "", "consensus": "valid", "supporting_sources": ["s"]},
        ]
        agreements = synth._build_agreements(raw)
        assert len(agreements) == 1

    def test_build_uncertainties_skips_invalid(self):
        synth = self.make_synthesizer()
        raw = [
            {"topic": "Valid Topic", "description": "valid description", "implications": "valid implications"},
            {"topic": "", "description": "valid", "implications": "valid"},
        ]
        uncertainties = synth._build_uncertainties(raw)
        assert len(uncertainties) == 1

    def test_build_implications_skips_invalid(self):
        synth = self.make_synthesizer()
        raw = [
            {"recommendation": "Valid", "strength": 0.7},
            {"recommendation": "", "strength": 0.5},
            "invalid",
        ]
        implications = synth._build_implications(raw)
        assert len(implications) == 1
