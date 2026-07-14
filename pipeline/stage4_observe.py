"""stage4：执行 NormalizedStep 中的 Action → ObservationExecutionResult。

依赖 stage3 产出的 NormalizedStep 与 tools.base.execute_action；
不访问 groundtruth，不生成 Thought（Thought 属于 stage5）。
"""

from __future__ import annotations

from typing import Optional

from pipeline.schemas import AgentRole, NormalizedStep, ObservationExecutionResult
from pipeline.tools.base import execute_action


def generate_observations(
    normalized_steps: list[NormalizedStep],
    image_path: str,
    agent_role: AgentRole,
    *,
    registry_path: Optional[str] = None,
    use_cache: bool = True,
) -> list[ObservationExecutionResult]:
    """展开 normalized_steps 中的全部 Action，逐个 execute_action。

    thought_only 步（actions=[]）不产生 execution result。
    composed 步可产生多个 ObservationExecutionResult。

    Args:
        normalized_steps: stage3 规范化步骤列表。
        image_path: 当前帧图像路径（由 execute_action 统一传入各 Tool）。
        agent_role: 调用方 Agent 角色（权限与 purpose 约束）。
        registry_path: 可选 registry 路径覆盖（测试注入）。
        use_cache: 是否启用 diskcache。

    Returns:
        与展开后 Action 顺序一一对应的 ObservationExecutionResult 列表。
    """
    results: list[ObservationExecutionResult] = []
    for step in normalized_steps:
        for action in step.actions:
            results.append(
                execute_action(
                    action,
                    image_path,
                    agent_role,
                    registry_path=registry_path,
                    use_cache=use_cache,
                )
            )
    return results
