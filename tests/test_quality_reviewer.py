from __future__ import annotations

from pipeline.quality import reviewer
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.quality import SemanticQualityReview
from pipeline.schemas.trajectory import Action, Trajectory, TrajectoryStep
from pipeline.schemas.transcript import TranscriptSegment


def test_reviewer_requires_evidence_references_and_never_rewrites(monkeypatch) -> None:
    captured = {}

    def fake_call(prompt, response_model, images=None, **kwargs):
        captured["prompt"] = prompt
        captured["images"] = images
        assert response_model is SemanticQualityReview
        return SemanticQualityReview(
            evidence_grounding=0.8,
            final_answer_support=0.9,
            reasoning_consistency=0.85,
            tool_semantics=0.75,
            input_alignment=0.95,
            summary="审核完成",
        )

    monkeypatch.setattr(reviewer, "call_structured", fake_call)
    freeform = FreeFormTrajectory.model_validate(
        {
            "source_video": "demo",
            "steps": [
                {
                    "event_type": "final",
                    "thought": "提交。",
                    "tool": "final_answer",
                    "params": {"location": "上海"},
                    "observation": None,
                }
            ],
        }
    )
    trajectory = Trajectory(
        id="demo__t01",
        system_prompt="system",
        user_query="query",
        image_paths=["selected.jpg"],
        steps=[
            TrajectoryStep(
                event_type="final",
                thought="提交。",
                action=Action(tool="final_answer", params={"location": "上海"}),
            )
        ],
    )
    result = reviewer.review_trajectory_semantics(
        freeform,
        trajectory,
        transcript=[TranscriptSegment(start=1, end=2, text="最终地点是上海")],
    )
    assert result.summary == "审核完成"
    assert captured["images"] == ["selected.jpg"]
    assert "只能审核，不得重写" in captured["prompt"]
    assert "transcript:start-end" in captured["prompt"]
    assert "没有来源的精确事实" in captured["prompt"]
