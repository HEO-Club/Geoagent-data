"""stage4：LLM 合成 NormalizedStep 中 Action 的 Observation。

依赖 stage3 产出的 NormalizedStep 与 tools.base.execute_action；
不访问 groundtruth，不生成 Thought（Thought 属于 stage5）。
"""

from __future__ import annotations

from typing import Optional

from pipeline.schemas import (
    AgentRole,
    NormalizedStep,
    ObservationExecutionResult,
    ObservationSource,
)
from pipeline.tools.base import execute_action


class ObservationSynthesisExhausted(RuntimeError):
    """非 terminal Action 的 Observation 合成在重试耗尽后仍失败。"""


def _is_synthesis_exhausted(result: ObservationExecutionResult) -> bool:
    """识别 LLM 合成耗尽（区别于权限/未知 tool 等预检错误）。"""
    return (
        result.status == "error"
        and result.source is ObservationSource.LLM_SYNTHESIZED
        and result.observation is None
    )


def generate_observations(
    normalized_steps: list[NormalizedStep],
    image_path: str,
    agent_role: AgentRole,
    *,
    registry_path: Optional[str] = None,
    use_cache: bool = True,
) -> list[ObservationExecutionResult]:
    """展开 normalized_steps 中的全部 Action，逐个 LLM 合成 Observation。

    thought_only 步（actions=[]）不产生 execution result。
    composed 步可产生多个 ObservationExecutionResult。
    将 step.move.narration 传入合成上下文（内部按角色消毒）。

    若任一非 terminal 合成耗尽，抛 :class:`ObservationSynthesisExhausted`，
    样本不得入库。

    Args:
        normalized_steps: stage3 规范化步骤列表。
        image_path: 当前关键帧图像路径。
        agent_role: 调用方 Agent 角色（权限与 purpose 约束）。
        registry_path: 可选 registry 路径覆盖（测试注入）。
        use_cache: 是否启用 diskcache。

    Returns:
        与展开后 Action 顺序一一对应的 ObservationExecutionResult 列表。
    """
    results: list[ObservationExecutionResult] = []
    for step in normalized_steps:
        narration = step.move.narration or ""
        for action in step.actions:
            results.append(
                execute_action(
                    action,
                    image_path,
                    agent_role,
                    narration=narration,
                    registry_path=registry_path,
                    use_cache=use_cache,
                )
            )

    exhausted = [r for r in results if _is_synthesis_exhausted(r)]
    if exhausted:
        tools = ", ".join(sorted({r.action.tool for r in exhausted}))
        raise ObservationSynthesisExhausted(
            f"Observation 合成重试耗尽（role={agent_role.value}, tools=[{tools}]）；"
            "样本不得入库"
        )
    return results
