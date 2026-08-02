"""
tests/unit/clinical/test_session_planning.py
────────────────────────────────────────────
Unit tests for the session planning stage: the SessionPlanner module
and the node_session_planning graph node.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clinical.session_planning.models import SessionPlanResult
from clinical.session_planning.planner import SessionPlanner


class TestSessionPlanner:
    @pytest.fixture
    def llm(self):
        llm = MagicMock()
        llm.generate = AsyncMock()
        return llm

    async def test_plan_builds_result(self, llm):
        llm.generate.return_value = (
            '{"disclaimer": "All session proposals are suggestions for '
            'clinician consideration.", "session_count": 6, "cadence": "weekly", '
            '"objectives": [{"title": "Engagement and formulation", '
            '"focus": "Establish rapport and review the formulation.", '
            '"suggested_activities": ["Review formulation", "Set goals"], '
            '"homework": "Reflect on goals", "expected_duration_minutes": 50}], '
            '"review_points": ["After session 3"]}'
        )
        planner = SessionPlanner(llm)
        result = await planner.plan(
            formulation='{"case_summary": "Low mood"}',
            plan_text='{"therapeutic_focus": []}',
            selected_framework="CBT",
            framework_lens="Framework: CBT",
        )
        assert isinstance(result, SessionPlanResult)
        assert result.session_count == 6
        assert result.cadence == "weekly"
        assert len(result.objectives) == 1
        assert result.objectives[0].title == "Engagement and formulation"
        assert result.objectives[0].homework == "Reflect on goals"
        assert result.review_points == ["After session 3"]
        # The prompt must include the selected framework lens.
        prompt_arg = llm.generate.await_args.kwargs["prompt"]
        assert "Framework: CBT" in prompt_arg
        assert "## Selected Framework" in prompt_arg

    async def test_plan_empty_input_returns_graceful(self, llm):
        planner = SessionPlanner(llm)
        result = await planner.plan()
        assert result.session_count == 0
        assert llm.generate.await_count == 0

    async def test_plan_llm_failure_returns_graceful(self, llm):
        llm.generate.side_effect = RuntimeError("LLM down")
        planner = SessionPlanner(llm)
        result = await planner.plan(
            formulation='{"case_summary": "Low mood"}',
            plan_text="plan",
        )
        assert result.session_count == 0
        assert "could not be completed" in result.disclaimer

    async def test_plan_skips_malformed_objectives(self, llm):
        llm.generate.return_value = (
            '{"disclaimer": "All session proposals are suggestions for '
            'clinician consideration.", "session_count": 1, "objectives": ['
            '{"focus": "missing title"}, {"title": "Valid session", '
            '"focus": "A valid focus area for this session."}], '
            '"review_points": []}'
        )
        planner = SessionPlanner(llm)
        result = await planner.plan(
            formulation='{"case_summary": "Low mood"}',
            plan_text="plan",
        )
        assert len(result.objectives) == 1
        assert result.objectives[0].title == "Valid session"


class TestNodeSessionPlanning:
    @pytest.mark.asyncio
    async def test_generates_session_plan(self):
        from clinical.graph.nodes import node_session_planning

        mock_plan = MagicMock()
        mock_plan.model_dump_json.return_value = '{"plan": true}'
        mock_plan.therapeutic_focus = []
        mock_plan.treatment_goals = []
        mock_form = MagicMock()
        mock_form.model_dump_json.return_value = '{"case_summary": "Low mood"}'
        mock_form.case_summary = "Low mood"
        mock_form.caution = ""
        mock_sp = MagicMock()
        mock_sp.plan = AsyncMock(return_value=SessionPlanResult(
            disclaimer="All session proposals are suggestions for clinician consideration.",
            session_count=4,
        ))

        with patch("clinical.graph.nodes._get_session_planner", return_value=mock_sp):
            state = {
                "text": "case",
                "formulation": mock_form,
                "plan": mock_plan,
                "errors": {},
            }
            result = await node_session_planning(state)
            assert "session_plan" in result
            assert result["session_plan"].session_count == 4

    @pytest.mark.asyncio
    async def test_missing_formulation_or_plan_returns_error(self):
        from clinical.graph.nodes import node_session_planning

        result = await node_session_planning({"text": "case", "errors": {}})
        assert "errors" in result
        assert "session_planning" in result["errors"]

    @pytest.mark.asyncio
    async def test_planner_failure_returns_error(self):
        from clinical.graph.nodes import node_session_planning

        mock_sp = MagicMock()
        mock_sp.plan = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("clinical.graph.nodes._get_session_planner", return_value=mock_sp):
            result = await node_session_planning({
                "text": "case",
                "formulation": MagicMock(model_dump_json=lambda: "{}", case_summary="x", caution=""),
                "plan": MagicMock(model_dump_json=lambda: "{}", therapeutic_focus=[], treatment_goals=[]),
                "errors": {},
            })
            assert "errors" in result
            assert "session_planning" in result["errors"]
