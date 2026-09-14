"""final_answer.submit 本地登记执行器测试；禁止真实付费 API 与网络。"""

from __future__ import annotations

from typing import Any

from tool import execute
from tool.contract import Observation, RuntimeContext


def _nested_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(value)
        for item in value.values():
            found.update(_nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_nested_keys(item))
    return found


def _submit(*, inputs: dict[str, Any], ctx: RuntimeContext | None = None) -> Observation:
    return execute(
        "final_answer",
        "submit",
        purpose="提交最终地点",
        inputs=inputs,
        ctx=ctx,
    )


def test_empty_inputs_are_missing_input() -> None:
    observation = _submit(inputs={})
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.result is None
    assert observation.error is not None
    assert "location" in observation.error


def test_single_string_is_registered() -> None:
    ctx = RuntimeContext()
    observation = _submit(inputs={"location": "上海市杨浦大桥"}, ctx=ctx)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["location"] == "上海市杨浦大桥"
    assert observation.result["kind"] == "single"
    assert observation.result["count"] == 1
    assert observation.result["applied"]["source_field"] == "location"
    assert ctx.extras["final_answer"]["location"] == "上海市杨浦大桥"
    assert ctx.previous_tool_result is None
    assert ctx.active_area is None
    assert any("未访问地理编码" in item for item in observation.result["assumptions"])
    assert any("不代表事实正确" in item for item in observation.result["assumptions"])
    assert "confirmed_location" not in _nested_keys(observation.result)


def test_multi_location_array_keeps_order() -> None:
    observation = _submit(
        inputs={"location": [" 云南丽江永胜县涛源镇 ", "上海市杨浦大桥"]},
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["location"] == ["云南丽江永胜县涛源镇", "上海市杨浦大桥"]
    assert observation.result["kind"] == "multi"
    assert observation.result["count"] == 2


def test_single_element_array_is_not_coerced_to_string() -> None:
    observation = _submit(inputs={"location": ["上海市杨浦大桥"]})
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["location"] == ["上海市杨浦大桥"]
    assert observation.result["kind"] == "multi"
    assert observation.result["count"] == 1


def test_aliases_地点_and_答案_are_accepted() -> None:
    place = _submit(inputs={"地点": "惠济区"})
    answer = _submit(inputs={"答案": "涛源镇"})
    assert place.ok is True and place.result is not None
    assert answer.ok is True and answer.result is not None
    assert place.result["location"] == "惠济区"
    assert place.result["applied"]["source_field"] == "地点"
    assert answer.result["location"] == "涛源镇"
    assert answer.result["applied"]["source_field"] == "答案"


def test_result_and_site_are_not_substitutes() -> None:
    by_result = _submit(inputs={"result": "上海市杨浦大桥"})
    by_site = _submit(inputs={"site": "上海市杨浦大桥"})
    assert by_result.ok is False
    assert by_site.ok is False
    assert by_result.error_code == "missing_input"
    assert by_site.error_code == "missing_input"
    assert by_result.error is not None
    assert "result/site" in by_result.error


def test_empty_and_blank_location_are_invalid() -> None:
    empty = _submit(inputs={"location": ""})
    blank = _submit(inputs={"location": "   "})
    empty_list = _submit(inputs={"location": []})
    blank_item = _submit(inputs={"location": ["甲地", "  "]})
    assert empty.error_code == "invalid_input"
    assert blank.error_code == "invalid_input"
    assert empty_list.error_code == "invalid_input"
    assert blank_item.error_code == "invalid_input"


def test_mixed_and_structured_location_are_invalid() -> None:
    as_dict = _submit(inputs={"location": {"name": "杨浦大桥", "lat": 31.25, "lng": 121.54}})
    as_number = _submit(inputs={"location": 31.25})
    mixed = _submit(inputs={"location": ["甲地", {"name": "乙地"}]})
    nested = _submit(inputs={"location": [["甲地"]]})
    json_text = _submit(inputs={"location": '["甲地", "乙地"]'})
    assert as_dict.error_code == "invalid_input"
    assert as_number.error_code == "invalid_input"
    assert mixed.error_code == "invalid_input"
    assert nested.error_code == "invalid_input"
    assert json_text.ok is True
    assert json_text.result is not None
    assert json_text.result["location"] == '["甲地", "乙地"]'
    assert json_text.result["kind"] == "single"


def test_extra_fields_are_not_merged_into_result() -> None:
    observation = _submit(
        inputs={
            "location": "某市某镇",
            "lat": 31.25,
            "lng": 121.54,
            "confidence": "高",
            "extensions": {"site": "应忽略"},
        },
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["location"] == "某市某镇"
    keys = _nested_keys(observation.result)
    assert "lat" not in keys
    assert "lng" not in keys
    assert "confidence" not in keys
    assert "extensions" not in keys
    assert "site" not in keys
    assert "confirmed_location" not in keys
    assert observation.extensions == {
        "lat": 31.25,
        "lng": 121.54,
        "confidence": "高",
        "site": "应忽略",
    }


def test_without_ctx_still_succeeds() -> None:
    observation = _submit(inputs={"location": "某地"}, ctx=None)
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["location"] == "某地"
