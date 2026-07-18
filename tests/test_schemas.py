"""Schema / A·F 规则 / 种子 Tool 契约测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.schemas import (
    SEED_TOOL_NAMES,
    AgentRole,
    LocationHypothesis,
    ObservationField,
    ParamField,
    RevisionContext,
    RevisionSource,
    SubmitAnswerResult,
    ToolDefinition,
    ToolTier,
    Trajectory,
    VerificationResult,
    validate_tool_name,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "tool_registry.json"


def _status_error_fields() -> list[ObservationField]:
    return [
        ObservationField(
            name="status",
            type="string",
            nullable=False,
            description="执行状态，取值 success/empty/error",
        ),
        ObservationField(
            name="error_message",
            type="string",
            nullable=True,
            description="status=error 时填写，否则为 null",
        ),
    ]


def _minimal_draft_tool(**overrides: object) -> ToolDefinition:
    data: dict = {
        "name": "search_landmark",
        "description": "检索地标相关线索，供粗定位使用。",
        "tier": ToolTier.DRAFT,
        "params": [
            ParamField(
                name="query",
                type="string",
                required=True,
                description="检索查询语句，如[church spire]",
                example="church spire",
            )
        ],
        "observation_fields": [
            *_status_error_fields(),
            ObservationField(
                name="summary",
                type="string",
                nullable=True,
                description="摘要文本；无可用信息时为 null",
            ),
        ],
        "allowed_agents": [AgentRole.COARSE],
        "is_terminal": False,
        "executor_ref": None,
        "created_at": "2026-07-14T00:00:00Z",
    }
    data.update(overrides)
    return ToolDefinition.model_validate(data)


class TestToolNameRules:
    def test_seed_names_pass(self) -> None:
        for name in SEED_TOOL_NAMES:
            assert validate_tool_name(name) == name

    def test_a3_rejects_meaningless(self) -> None:
        with pytest.raises(ValueError, match="无意义"):
            validate_tool_name("helper")

    def test_a4_rejects_single_token_non_seed(self) -> None:
        with pytest.raises(ValueError, match="两个语义 token"):
            validate_tool_name("ocrify")

    def test_rejects_leading_underscore(self) -> None:
        with pytest.raises(ValueError):
            validate_tool_name("_bad_tool")


class TestToolDefinitionValidators:
    def test_f1_rejects_name_overlap(self) -> None:
        with pytest.raises(ValidationError, match="交集"):
            _minimal_draft_tool(
                params=[
                    ParamField(
                        name="summary",
                        type="string",
                        required=True,
                        description="与 observation 同名的非法参数名",
                        example="x",
                    )
                ],
                observation_fields=[
                    *_status_error_fields(),
                    ObservationField(
                        name="summary",
                        type="string",
                        nullable=True,
                        description="摘要文本；无可用信息时为 null",
                    ),
                ],
            )

    def test_draft_requires_null_executor_ref(self) -> None:
        with pytest.raises(ValidationError, match="executor_ref"):
            _minimal_draft_tool(executor_ref="pipeline.tools.web_search.execute")

    def test_terminal_requires_empty_observation(self) -> None:
        with pytest.raises(ValidationError, match="空列表"):
            _minimal_draft_tool(
                name="submit_answer",
                is_terminal=True,
                params=[
                    ParamField(
                        name="latitude",
                        type="float",
                        required=True,
                        description="最终纬度，如[48.8584]",
                        example=48.8584,
                    )
                ],
                observation_fields=_status_error_fields(),
                allowed_agents=[AgentRole.FINE],
            )

    def test_forbidden_param_image(self) -> None:
        with pytest.raises(ValidationError):
            ParamField(
                name="image_path",
                type="string",
                required=True,
                description="禁止的图像路径参数字段名示例",
                example="/tmp/a.jpg",
            )

    def test_required_param_cannot_have_default(self) -> None:
        with pytest.raises(ValidationError, match="default"):
            ParamField(
                name="query",
                type="string",
                required=True,
                description="检索查询语句，如[example]",
                example="example",
                default="x",
            )


class TestSeedRegistry:
    def test_all_seeds_validate(self) -> None:
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        assert isinstance(raw, list)
        tools = [ToolDefinition.model_validate(item) for item in raw]
        names = {t.name for t in tools}
        assert names == SEED_TOOL_NAMES
        by_name = {t.name: t for t in tools}
        expected_production = {
            "sun_position_calc": "pipeline.tools.sun_position.execute",
            "map_query": "pipeline.tools.map_query.execute",
            "web_search": "pipeline.tools.web_search.execute",
            "reverse_image_search": "pipeline.tools.reverse_image_search.execute",
            "ocr": "pipeline.tools.ocr.execute",
            "zoom_inspect": "pipeline.tools.zoom_inspect.execute",
        }
        for name, ref in expected_production.items():
            assert by_name[name].tier == ToolTier.PRODUCTION
            assert by_name[name].executor_ref == ref
        for t in tools:
            if t.name in expected_production:
                continue
            assert t.tier == ToolTier.DRAFT
            assert t.executor_ref is None

    def test_map_query_uses_resolved_latlng(self) -> None:
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        mq = next(t for t in raw if t["name"] == "map_query")
        tool = ToolDefinition.model_validate(mq)
        param_names = {p.name for p in tool.params}
        obs_names = {o.name for o in tool.observation_fields}
        assert "latlng" in param_names
        assert "resolved_latlng" in obs_names
        assert "latlng" not in obs_names
        assert param_names.isdisjoint(obs_names)

    def test_reject_map_query_with_obs_latlng(self) -> None:
        raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        mq = next(t for t in raw if t["name"] == "map_query")
        for field in mq["observation_fields"]:
            if field["name"] == "resolved_latlng":
                field["name"] = "latlng"
                field["description"] = "地点坐标 (lat,lng)；查无结果时为 null"
        with pytest.raises(ValidationError, match="交集"):
            ToolDefinition.model_validate(mq)


class TestHandoffAndRevision:
    def _coarse_hyp(self) -> LocationHypothesis:
        return LocationHypothesis(
            possible_countries=["France"],
            possible_regions=["Île-de-France"],
            reasoning_summary="Looks like Paris landmarks.",
            confidence=0.7,
            key_clues_remaining=["exact plaza"],
        )

    def _submit(self) -> SubmitAnswerResult:
        return SubmitAnswerResult(
            latitude=48.8584,
            longitude=2.2945,
            location_name="Eiffel Tower",
            confidence=0.9,
            reasoning="Matched landmark geometry.",
        )

    def test_fine_requires_coarse_handoff(self) -> None:
        with pytest.raises(ValidationError, match="coarse_handoff"):
            Trajectory(
                id="t1",
                agent_role=AgentRole.FINE,
                system_prompt="sys",
                user_query="locate",
                image_path="a.jpg",
                steps=[],
            )

    def test_verifier_requires_fine_handoff(self) -> None:
        with pytest.raises(ValidationError, match="fine_handoff"):
            Trajectory(
                id="t1",
                agent_role=AgentRole.VERIFIER,
                system_prompt="sys",
                user_query="verify",
                image_path="a.jpg",
                steps=[],
                coarse_handoff=self._coarse_hyp(),
            )

    def test_verification_pass_forbids_return_to(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(
                verdict="pass",
                failed_checks=[],
                suggested_recheck="none",
                return_to_agent=2,
            )

    def test_revision_system_feedback_requires_result(self) -> None:
        with pytest.raises(ValidationError, match="verification_result"):
            RevisionContext(
                source=RevisionSource.SYSTEM_FEEDBACK,
                parent_trajectory_id="p1",
                target_agent=AgentRole.FINE,
                revision_round=1,
            )
