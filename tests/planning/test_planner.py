"""Tests for planning module — Planner, Plan, PlanStep, SelfCorrector."""

import json
from unittest.mock import AsyncMock

import pytest

from corail.models.base import Model
from corail.planning.correction import SelfCorrector
from corail.planning.planner import Plan, Planner, PlanStep


# --- Mock model ---

class _MockModel(Model):
    def __init__(self) -> None:
        self._generate = AsyncMock()

    async def generate(self, messages, **kwargs):
        return await self._generate(messages, **kwargs)


@pytest.fixture
def mock_model() -> _MockModel:
    return _MockModel()


# --- PlanStep tests ---

class TestPlanStep:
    def test_initial_status_is_pending(self):
        step = PlanStep(id="s1", description="Do something")
        assert step.status == "pending"
        assert step.result == ""

    def test_mark_started(self):
        step = PlanStep(id="s1", description="Do something")
        step.mark_started()
        assert step.status == "in_progress"

    def test_mark_completed(self):
        step = PlanStep(id="s1", description="Do something")
        step.mark_completed("done successfully")
        assert step.status == "completed"
        assert step.result == "done successfully"

    def test_mark_failed(self):
        step = PlanStep(id="s1", description="Do something")
        step.mark_failed("connection timeout")
        assert step.status == "failed"
        assert step.result == "connection timeout"

    def test_mark_skipped(self):
        step = PlanStep(id="s1", description="Do something")
        step.mark_skipped("not needed")
        assert step.status == "skipped"
        assert step.result == "not needed"


# --- Plan tests ---

class TestPlan:
    def test_current_step_returns_first_pending(self):
        steps = [
            PlanStep(id="s1", description="Step 1", status="completed"),
            PlanStep(id="s2", description="Step 2", status="pending"),
            PlanStep(id="s3", description="Step 3", status="pending"),
        ]
        plan = Plan(goal="test", steps=steps)
        assert plan.current_step is not None
        assert plan.current_step.id == "s2"

    def test_current_step_returns_none_when_all_done(self):
        steps = [
            PlanStep(id="s1", description="Step 1", status="completed"),
            PlanStep(id="s2", description="Step 2", status="failed"),
        ]
        plan = Plan(goal="test", steps=steps)
        assert plan.current_step is None

    def test_is_complete_when_all_finished(self):
        steps = [
            PlanStep(id="s1", description="Step 1", status="completed"),
            PlanStep(id="s2", description="Step 2", status="skipped"),
            PlanStep(id="s3", description="Step 3", status="failed"),
        ]
        plan = Plan(goal="test", steps=steps)
        assert plan.is_complete is True

    def test_is_not_complete_with_pending(self):
        steps = [
            PlanStep(id="s1", description="Step 1", status="completed"),
            PlanStep(id="s2", description="Step 2", status="pending"),
        ]
        plan = Plan(goal="test", steps=steps)
        assert plan.is_complete is False

    def test_progress_string(self):
        steps = [
            PlanStep(id="s1", description="Step 1", status="completed"),
            PlanStep(id="s2", description="Step 2", status="completed"),
            PlanStep(id="s3", description="Step 3", status="pending"),
            PlanStep(id="s4", description="Step 4", status="failed"),
            PlanStep(id="s5", description="Step 5", status="pending"),
        ]
        plan = Plan(goal="test", steps=steps)
        assert plan.progress == "2/5 steps completed"

    def test_completed_and_failed_step_lists(self):
        steps = [
            PlanStep(id="s1", description="A", status="completed"),
            PlanStep(id="s2", description="B", status="failed"),
            PlanStep(id="s3", description="C", status="completed"),
        ]
        plan = Plan(goal="test", steps=steps)
        assert len(plan.completed_steps) == 2
        assert len(plan.failed_steps) == 1

    def test_empty_plan(self):
        plan = Plan(goal="empty")
        assert plan.current_step is None
        assert plan.is_complete is True
        assert plan.progress == "0/0 steps completed"


# --- Planner.needs_planning tests ---

class TestNeedsPlanning:
    def test_short_question_does_not_need_planning(self):
        assert Planner.needs_planning("What is Python?") is False

    def test_single_question_mark_does_not_need_planning(self):
        assert Planner.needs_planning("How do I install this?") is False

    def test_short_input_does_not_need_planning(self):
        assert Planner.needs_planning("Hello world") is False

    def test_complex_task_needs_planning(self):
        task = "Create a new REST API endpoint and then write integration tests and also update the documentation"
        assert Planner.needs_planning(task) is True

    def test_action_verbs_with_conjunction_needs_planning(self):
        task = "Build the authentication module and integrate it with the existing user service after migrating the database"
        assert Planner.needs_planning(task) is True

    def test_long_statement_without_action_does_not_need_planning(self):
        stmt = "The weather today is really nice and warm and I am happy about the sunshine and the birds singing"
        assert Planner.needs_planning(stmt) is False

    def test_empty_string_does_not_need_planning(self):
        assert Planner.needs_planning("") is False


# --- Planner.create_plan tests ---

class TestCreatePlan:
    async def test_creates_plan_from_json_response(self, mock_model):
        mock_model._generate.return_value = json.dumps([
            "Search for configuration files",
            "Update the database schema",
            "Run migration scripts",
        ])
        planner = Planner(model=mock_model)
        plan = await planner.create_plan("migrate database", ["search", "execute_sql"])

        assert plan.goal == "migrate database"
        assert len(plan.steps) == 3
        assert plan.steps[0].description == "Search for configuration files"
        assert plan.steps[0].status == "pending"
        mock_model._generate.assert_awaited_once()

    async def test_creates_plan_from_numbered_list(self, mock_model):
        mock_model._generate.return_value = (
            "1. Find the config file\n"
            "2. Update the settings\n"
            "3. Restart the service"
        )
        planner = Planner(model=mock_model)
        plan = await planner.create_plan("update config", [])

        assert len(plan.steps) == 3
        assert plan.steps[0].description == "Find the config file"

    async def test_each_step_has_unique_id(self, mock_model):
        mock_model._generate.return_value = json.dumps(["Step A", "Step B", "Step C"])
        planner = Planner(model=mock_model)
        plan = await planner.create_plan("test", [])

        ids = [s.id for s in plan.steps]
        assert len(set(ids)) == len(ids)  # All unique

    async def test_empty_response_yields_empty_plan(self, mock_model):
        mock_model._generate.return_value = "[]"
        planner = Planner(model=mock_model)
        plan = await planner.create_plan("nothing", [])

        assert len(plan.steps) == 0
        assert plan.is_complete is True


# --- SelfCorrector tests ---

class TestSelfCorrector:
    async def test_suggest_alternative(self, mock_model):
        mock_model._generate.return_value = "Try using the backup API endpoint instead"
        corrector = SelfCorrector(model=mock_model)

        alternative = await corrector.suggest_alternative(
            failed_step="Call the primary API",
            error="Connection refused",
            available_tools=["http_call", "search"],
        )

        assert "backup" in alternative.lower() or len(alternative) > 0
        mock_model._generate.assert_awaited_once()

    async def test_includes_tools_in_prompt(self, mock_model):
        mock_model._generate.return_value = "Use search tool"
        corrector = SelfCorrector(model=mock_model)
        await corrector.suggest_alternative("failed", "error", ["search", "execute"])

        call_args = mock_model._generate.call_args[0][0]
        prompt_text = call_args[0]["content"]
        assert "search" in prompt_text
        assert "execute" in prompt_text
