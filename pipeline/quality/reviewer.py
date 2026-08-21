"""独立语义审核 Agent：只评估，不改写轨迹。"""

from __future__ import annotations

import json

from pipeline.llm import call_structured
from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.freeform import FreeFormTrajectory
from pipeline.schemas.quality import SemanticQualityReview
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment

REVIEWER_RUBRIC = """
你是地理定位 SFT 轨迹的独立质量审核 Agent。你只能审核，不得重写、补全或美化轨迹。
必须区分：输入图片可见事实、字幕中讲解者报告的外部工具结果、Agent 自身合理推理、
以及没有来源的精确事实。Observation 只有在字幕明确报告了对应外部动作回执或提供了
真实工具结果时才算有依据；画面直观看到的事实应属于 reasoning，而不是伪造 Observation。

分别给出 0~1 分：
1. evidence_grounding：Thought/Observation 是否能在图片、字幕或工具回执中找到依据；
2. final_answer_support：最终 location 是否完整且由前述证据推出，是否擅自添加精细坐标；
3. reasoning_consistency：候选提出、排除、修正和收敛是否前后连贯；
4. tool_semantics：是否把纯思考伪装成 Tool，Tool/operation/inputs 是否符合调用目的；
5. input_alignment：轨迹依赖的图片和线索是否确实属于当前 task。

每个问题必须给出 step_index，并尽量引用 transcript:start-end、image 文件名或具体字段。
以下情况写入 hard_failures：明确虚假 Observation、最终答案缺失或数量不完整、答案泄露图、
无来源的精确坐标/距离/数据库命中、轨迹依赖的原图与实际输入明显不一致。
不要因为表达流畅而给高分；证据不足时应降低分数并明确写“未验证”。
""".strip()


def review_trajectory_semantics(
    freeform: FreeFormTrajectory,
    trajectory: Trajectory,
    *,
    transcript: list[TranscriptSegment] | None = None,
    task: GeoTaskSpec | None = None,
    image_paths: list[str] | None = None,
) -> SemanticQualityReview:
    """调用独立模型审核语义质量；返回结构化分数与证据引用。"""

    transcript_payload = [
        {"start": item.start, "end": item.end, "text": item.text}
        for item in transcript or []
    ]
    prompt = (
        f"{REVIEWER_RUBRIC}\n\n"
        f"Stage 1.5 task：{json.dumps(task.model_dump() if task else None, ensure_ascii=False)}\n\n"
        f"字幕：{json.dumps(transcript_payload, ensure_ascii=False)}\n\n"
        f"Stage 2：{freeform.model_dump_json()}\n\n"
        f"Stage 3：{trajectory.model_dump_json()}"
    )
    return call_structured(
        prompt,
        SemanticQualityReview,
        images=list(image_paths or trajectory.image_paths),
        lane="llm",
    )
