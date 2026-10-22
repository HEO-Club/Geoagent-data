"""final_answer 共享执行器：登记已提交地点，不查证、不补全。"""

from __future__ import annotations

from typing import Any

from tool.contract import Observation, RuntimeContext, declared_inputs

_EXTRAS_KEY = "final_answer"
_LOCATION_FIELD = "location"
_LOCATION_ALIASES = ("地点", "答案")
_FORBIDDEN_SUBSTITUTES = ("result", "site")
_KIND_SINGLE = "single"
_KIND_MULTI = "multi"
_ASSUMPTIONS = (
    "本步只登记已提交地点，未访问地理编码、地图或外部模型",
    "格式通过不代表事实正确；证据支持由后续评估负责",
    "未根据证据粒度自动补全精细经纬度或门牌",
)


class SubmitInputError(Exception):
    """location 无法按合同解析。"""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def execute_submit(
    *,
    purpose: str,
    inputs: dict[str, Any],
    ctx: RuntimeContext | None = None,
) -> Observation:
    """校验并登记 final_answer.submit 的 location，不访问外部服务。"""

    del purpose
    try:
        raw, source_field = _extract_location(inputs)
        location, kind = _normalize_location(raw)
    except SubmitInputError as exc:
        return _fail(str(exc), exc.error_code)

    payload = {
        "location": location,
        "count": 1 if kind == _KIND_SINGLE else len(location),
        "kind": kind,
        "applied": {"source_field": source_field},
        "assumptions": list(_ASSUMPTIONS),
    }
    _register(ctx, payload)
    return Observation(ok=True, result=payload)


def _extract_location(inputs: dict[str, Any]) -> tuple[Any, str]:
    declared = declared_inputs(inputs, _LOCATION_FIELD)
    if _LOCATION_FIELD in declared:
        return declared[_LOCATION_FIELD], _LOCATION_FIELD
    for alias in _LOCATION_ALIASES:
        if alias in inputs:
            return inputs[alias], alias
    hint = ""
    if any(name in inputs for name in _FORBIDDEN_SUBSTITUTES):
        hint = "；不得使用 result/site 等替代字段"
    raise SubmitInputError(f"缺少必填输入 location{hint}", "missing_input")


def _normalize_location(raw: Any) -> tuple[str | list[str], str]:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise SubmitInputError("location 必须是非空地点字符串", "invalid_input")
        return text, _KIND_SINGLE
    if isinstance(raw, list):
        if not raw:
            raise SubmitInputError("location 数组不能为空", "invalid_input")
        places: list[str] = []
        for index, item in enumerate(raw):
            if not isinstance(item, str):
                raise SubmitInputError(
                    f"location[{index}] 必须是地点字符串，不能使用对象、数字或嵌套列表",
                    "invalid_input",
                )
            text = item.strip()
            if not text:
                raise SubmitInputError(
                    f"location[{index}] 必须是非空地点字符串",
                    "invalid_input",
                )
            places.append(text)
        return places, _KIND_MULTI
    raise SubmitInputError(
        "location 必须是非空地点字符串或地点字符串数组，不能使用对象或数字",
        "invalid_input",
    )


def _register(ctx: RuntimeContext | None, payload: dict[str, Any]) -> None:
    if ctx is None:
        return
    ctx.extras[_EXTRAS_KEY] = dict(payload)


def _fail(message: str, error_code: str) -> Observation:
    return Observation(ok=False, result=None, error=message, error_code=error_code)
