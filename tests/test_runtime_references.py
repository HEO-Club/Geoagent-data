"""harness 受限引用解析和凭证脱敏测试。"""

from __future__ import annotations

import pytest

from tool.contract import RuntimeContext
from tool.runtime.references import (
    ReferenceResolutionError,
    contains_credential_fields,
    redact_credentials,
    resolve_inputs,
)


def test_resolves_nested_step_reference_and_tracks_dependency() -> None:
    resolved = resolve_inputs(
        {
            "bbox": "$step_1_tool_result.candidates[0].bbox",
            "nested": ["literal", "$step_1_tool_result.candidates[0].name"],
        },
        ctx=RuntimeContext(),
        step_results={
            1: {
                "candidates": [
                    {"name": "郑州市", "bbox": [113.4, 34.6, 114.0, 35.1]}
                ]
            }
        },
        current_step=2,
    )
    assert resolved.value["bbox"] == [113.4, 34.6, 114.0, 35.1]
    assert resolved.value["nested"][1] == "郑州市"
    assert resolved.dependency_steps == [1]
    assert len(resolved.bindings) == 2


def test_context_references_keep_producing_step_dependency() -> None:
    ctx = RuntimeContext(
        previous_tool_result={"result_id": "osm_demo"},
        active_area="郑州市",
        extras={"_previous_tool_step": 2, "_active_area_step": 1},
    )
    resolved = resolve_inputs(
        {"source": "$previous_tool_result.result_id", "area": "$active_area"},
        ctx=ctx,
        step_results={},
        current_step=3,
    )
    assert resolved.value == {"source": "osm_demo", "area": "郑州市"}
    assert resolved.dependency_steps == [1, 2]


@pytest.mark.parametrize(
    ("reference", "code"),
    [
        ("$step_2_tool_result", "forward_reference"),
        ("$step_1_tool_result.missing", "reference_path_missing"),
        ("$unknown_runtime_value", "unknown_reference"),
    ],
)
def test_reference_failures_are_structured(reference: str, code: str) -> None:
    with pytest.raises(ReferenceResolutionError) as caught:
        resolve_inputs(
            {"value": reference},
            ctx=RuntimeContext(),
            step_results={1: {"ok": True}},
            current_step=2,
        )
    assert caught.value.error_code == code


def test_credentials_are_detected_redacted_and_not_referenceable() -> None:
    payload = {"query": "x", "nested": {"api_key": "secret-value"}}
    assert contains_credential_fields(payload) is True
    assert redact_credentials(payload)["nested"]["api_key"] == "***REDACTED***"

    with pytest.raises(ReferenceResolutionError) as caught:
        resolve_inputs(
            {"value": "$step_1_tool_result.client_secret"},
            ctx=RuntimeContext(),
            step_results={1: {"client_secret": "secret-value"}},
            current_step=2,
        )
    assert caught.value.error_code == "credential_reference_forbidden"

    prefixed_secret = "sk-abcdefghijklmnopqrstuvwx"
    assert contains_credential_fields({"query": prefixed_secret}) is True
    assert redact_credentials(prefixed_secret) == "***REDACTED***"
