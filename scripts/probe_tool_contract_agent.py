"""用合成场景实测 LLM 是否理解 Canonical Tool operation/input_schema。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, Field

from pipeline.llm import call_structured
from pipeline.stage3_normalize_format.params import attach_operation_input_schemas
from pipeline.stage3_normalize_format.trees import load_forest


class ProbeDecision(BaseModel):
    case_id: str
    action: Literal["tool_call", "reasoning", "needs_input"]
    canonical_tool: str | None = None
    operation: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class ProbeResult(BaseModel):
    decisions: list[ProbeDecision]


CASES = [
    {
        "case_id": "osm_structured",
        "scenario": "查询郑州市范围内所有 bridge=yes 的 OSM 要素并返回几何，不要求模型手写 Overpass 代码。",
        "expected": ["osm_query", "query"],
        "expected_action": "tool_call",
    },
    {
        "case_id": "osm_export",
        "scenario": "把上一步 osm_result_02 导出成保留几何的 GeoJSON。",
        "expected": ["osm_query", "export"],
        "expected_action": "tool_call",
    },
    {
        "case_id": "satellite_time",
        "scenario": "比较兰州近水广场 2002 年和 2025 年卫星影像中的植被变化。",
        "expected": ["satellite_imagery_query", "compare_time"],
        "expected_action": "tool_call",
    },
    {
        "case_id": "satellite_candidates",
        "scenario": "在兰州近水广场、兰州水车园两个候选地的卫星影像中，比对‘台阶-植被-台阶’结构模板。",
        "expected": ["satellite_imagery_query", "compare_candidates"],
        "expected_action": "tool_call",
    },
    {
        "case_id": "site_search",
        "scenario": "只在 example.org 网站内检索关键词‘杨浦大桥 历史照片’，返回前 5 项。",
        "expected": ["web_search", "site_search"],
        "expected_action": "tool_call",
    },
    {
        "case_id": "flight_archive",
        "scenario": "查询 2025-09-21 中山市沙溪镇上空的历史航班记录。",
        "expected": ["flight_data_query", "search"],
        "expected_action": "tool_call",
    },
    {
        "case_id": "pure_reasoning",
        "scenario": "把已经获得的桥梁、河流和山体证据综合起来，排除不符合的候选；不访问任何新服务。",
        "expected": [None, None],
        "expected_action": "reasoning",
    },
    {
        "case_id": "crop_before_reverse_search",
        "scenario": "需要对当前完整图片右上角的桥梁标志做局部反向搜图，但目前只有 $current_image，没有裁剪图。下一步先做什么？",
        "expected": ["image_process", "crop"],
        "expected_action": "tool_call",
    },
    {
        "case_id": "query_before_osm_export",
        "scenario": "最终希望导出 GeoJSON，但还没有任何 OSM 查询结果；目前只知道区域是郑州市、标签 bridge=yes。下一步先做什么？",
        "expected": ["osm_query", "query"],
        "expected_action": "tool_call",
    },
    {
        "case_id": "reverse_search_without_any_image",
        "scenario": "希望进行反向搜图，但当前输入里没有图片、视频帧、图片 URL 或任何可引用的图片 ID。",
        "expected": [None, None],
        "expected_action": "needs_input",
    },
]


def main() -> None:
    forest = attach_operation_input_schemas(
        load_forest(Path("canonical_tool_catalog.json"))
    )
    selected = {
        "osm_query",
        "satellite_imagery_query",
        "web_search",
        "flight_data_query",
        "image_process",
        "reverse_image_search",
    }
    catalog = []
    for tree in forest.trees:
        if tree.canonical.name not in selected:
            continue
        catalog.append(
            {
                "name": tree.canonical.name,
                "description": tree.canonical.description,
                "usage": tree.canonical.usage,
                "operations": [item.model_dump() for item in tree.canonical.operations],
            }
        )
    prompt = (
        "你在测试地理定位 Agent 的 Tool 参数合同。对每个 case 独立判断：只有真实访问外部"
        "执行器时 action=tool_call；纯粹整合已有证据时 action=reasoning；如果必需的原始输入"
        "不存在且无法从上下文获得，action=needs_input。tool_call 必须从目录"
        "选择 canonical_tool 和 operation，并严格按该 operation 的 input_schema 生成 inputs。"
        "只能使用 scenario 已提供的值，不得补写场景中没有的坐标、日期、结果 ID 或代码。"
        "认真读取 acquisition_hint：缺局部图时先 crop/zoom，缺 source_result 时先执行前置 query，"
        "完全没有图片时不得伪造 image。"
        "OSM 默认使用结构化 area/tags，由执行器生成 Overpass QL，除非场景明确提供代码。"
        "每个 case_id 恰好返回一次。\n\n"
        f"目录：{json.dumps(catalog, ensure_ascii=False)}\n\n"
        f"测试场景：{json.dumps([{k: v for k, v in case.items() if k not in {'expected', 'expected_action'}} for case in CASES], ensure_ascii=False)}"
    )
    result = call_structured(prompt, ProbeResult, lane="llm")
    by_id = {item.case_id: item for item in result.decisions}
    rows = []
    for case in CASES:
        decision = by_id.get(case["case_id"])
        expected_tool, expected_op = case["expected"]
        expected_action = case["expected_action"]
        passed = bool(decision) and (
            (decision.action == expected_action and expected_tool is None)
            or (
                decision.action == expected_action == "tool_call"
                and decision.canonical_tool == expected_tool
                and decision.operation == expected_op
            )
        )
        rows.append(
            {
                **case,
                "passed": passed,
                "decision": decision.model_dump() if decision else None,
            }
        )
    report = {
        "passed": sum(item["passed"] for item in rows),
        "total": len(rows),
        "pass_rate": round(sum(item["passed"] for item in rows) / len(rows), 4),
        "cases": rows,
    }
    out = Path("data/runs/tool_contract_agent_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={out}")


if __name__ == "__main__":
    main()
