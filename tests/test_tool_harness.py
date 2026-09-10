"""真实 Tool harness：执行、失败隔离、断点恢复与安全合同。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from filelock import FileLock
from PIL import Image

from pipeline.schemas.trajectory import Action, Trajectory, TrajectoryStep
from tool import execute as real_execute
from tool.contract import Observation, RuntimeContext
from tool.runtime import HarnessError, ToolHarness, ToolHarnessConfig
from tool.runtime import harness as harness_module

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPO_ROOT / "canonical_tool_catalog_v2.json"


class FakeGeocodeClient:
    name = "fake_geocoder"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(
        self,
        query: str,
        *,
        limit: int,
        language: str | None,
        viewbox: tuple[float, float, float, float] | None,
    ) -> list[dict[str, Any]]:
        del limit, language, viewbox
        self.calls.append(query)
        return [
            {
                "osm_type": "relation",
                "osm_id": 1,
                "lat": "34.95",
                "lon": "113.50",
                "display_name": "郑州市",
                "boundingbox": ["34.9", "35.0", "113.4", "113.6"],
            }
        ]

    def reverse(self, lat: float, lon: float, *, language: str | None) -> dict[str, Any]:
        raise AssertionError((lat, lon, language))


class FakeOverpassClient:
    name = "fake_overpass"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def query(self, overpass_ql: str) -> dict[str, Any]:
        self.calls.append(overpass_ql)
        return {
            "elements": [
                {
                    "type": "way",
                    "id": 3,
                    "center": {"lat": 34.97, "lon": 113.52},
                    "tags": {"name": "候选桥梁", "bridge": "yes"},
                }
            ]
        }


def _config(tmp_path: Path, **updates: Any) -> ToolHarnessConfig:
    values: dict[str, Any] = {
        "workspace_dir": tmp_path / "runs",
        "catalog_path": CATALOG,
    }
    values.update(updates)
    return ToolHarnessConfig(**values)


def _geo_trajectory(*, trajectory_id: str = "geo-chain", location: str = "郑州市") -> Trajectory:
    return Trajectory(
        id=trajectory_id,
        system_prompt="system",
        user_query="Locate the place shown in the image.",
        steps=[
            TrajectoryStep(event_type="reasoning", thought="先解析范围"),
            TrajectoryStep(
                event_type="tool_call",
                thought="解析郑州市",
                action=Action(
                    tool="geocode",
                    params={
                        "operation": "geocode",
                        "purpose": "取得候选 bbox",
                        "inputs": {"query": "郑州市", "top_k": 1},
                    },
                ),
                observation={"distilled_only": "不得复用"},
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="查找候选桥梁",
                action=Action(
                    tool="osm_query",
                    params={
                        "operation": "query",
                        "purpose": "取得 bbox 内桥梁",
                        "inputs": {
                            "bbox": "$step_2_tool_result.candidates[0].bbox",
                            "feature_types": ["桥梁"],
                            "limit": 5,
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="统计真实桥梁结果",
                action=Action(
                    tool="osm_query",
                    params={
                        "operation": "count",
                        "purpose": "统计桥梁数量",
                        "inputs": {
                            "source_result": "$step_3_tool_result.result_id"
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="final",
                thought="提交答案",
                action=Action(tool="final_answer", params={"location": location}),
            ),
        ],
    )


def _context(
    geocoder: FakeGeocodeClient | None = None,
    overpass: FakeOverpassClient | None = None,
) -> RuntimeContext:
    return RuntimeContext(
        extras={
            "geocode_client": geocoder or FakeGeocodeClient(),
            "overpass_client": overpass or FakeOverpassClient(),
        }
    )


def test_harness_executes_chain_and_ignores_distilled_observations(tmp_path: Path) -> None:
    geocoder = FakeGeocodeClient()
    overpass = FakeOverpassClient()
    trajectory = _geo_trajectory()
    original = trajectory.model_dump(mode="json")
    harness = ToolHarness(
        _config(tmp_path),
        context=_context(geocoder, overpass),
    )
    report = harness.run(trajectory)

    assert report.status == "completed"
    assert report.counts == {
        "total": 5,
        "skipped_reasoning": 1,
        "succeeded": 3,
        "terminal": 1,
    }
    assert report.final_location == "郑州市"
    assert report.records[2].dependency_steps == [2]
    assert report.records[3].dependency_steps == [3]
    assert report.records[1].observation is not None
    assert report.records[1].observation.result["provider"] == "fake_geocoder"
    assert "distilled_only" not in report.records[1].observation.result
    assert geocoder.calls == ["郑州市"]
    assert len(overpass.calls) == 1
    assert trajectory.model_dump(mode="json") == original

    assert harness.last_report_path is not None
    persisted = json.loads(harness.last_report_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "completed"
    events = (harness.last_run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line)["event"] == "step_started" for line in events)
    assert json.loads(events[-1])["event"] == "run_finished"
    assert len(list((harness.last_run_dir / "results").glob("*.json"))) == 2


def test_harness_resume_reuses_successful_steps_without_provider_calls(tmp_path: Path) -> None:
    trajectory = _geo_trajectory()
    first = ToolHarness(_config(tmp_path), context=_context())
    assert first.run(trajectory).status == "completed"

    geocoder = FakeGeocodeClient()
    overpass = FakeOverpassClient()
    resumed = ToolHarness(
        _config(tmp_path),
        context=_context(geocoder, overpass),
    ).run(trajectory)
    assert resumed.resumed is True
    assert all(record.reused_from_checkpoint for record in resumed.records)
    assert geocoder.calls == []
    assert overpass.calls == []


def test_harness_refuses_changed_trajectory_checkpoint(tmp_path: Path) -> None:
    ToolHarness(_config(tmp_path), context=_context()).run(_geo_trajectory())
    with pytest.raises(HarnessError) as caught:
        ToolHarness(_config(tmp_path), context=_context()).run(
            _geo_trajectory(location="开封市")
        )
    assert caught.value.error_code == "trajectory_fingerprint_mismatch"


def test_harness_refuses_changed_runtime_or_execution_policy(tmp_path: Path) -> None:
    trajectory = _geo_trajectory(trajectory_id="runtime-fingerprint")
    original_context = _context()
    original_context.extras["harness_provider_revision"] = "v1"
    ToolHarness(_config(tmp_path), context=original_context).run(trajectory)

    changed_context = _context()
    changed_context.extras["harness_provider_revision"] = "v2"
    with pytest.raises(HarnessError) as runtime_error:
        ToolHarness(_config(tmp_path), context=changed_context).run(trajectory)
    assert runtime_error.value.error_code == "runtime_fingerprint_mismatch"

    policy_trajectory = _geo_trajectory(trajectory_id="policy-fingerprint")
    ToolHarness(_config(tmp_path), context=_context()).run(policy_trajectory)
    with pytest.raises(HarnessError) as policy_error:
        ToolHarness(
            _config(tmp_path, stop_on_error=True),
            context=_context(),
        ).run(policy_trajectory)
    assert policy_error.value.error_code == "execution_policy_mismatch"

    revision_trajectory = _geo_trajectory(trajectory_id="provider-revision")
    ToolHarness(
        _config(tmp_path, provider_revision="dataset-v1"),
        context=_context(),
    ).run(revision_trajectory)
    with pytest.raises(HarnessError) as revision_error:
        ToolHarness(
            _config(tmp_path, provider_revision="dataset-v2"),
            context=_context(),
        ).run(revision_trajectory)
    assert revision_error.value.error_code == "runtime_fingerprint_mismatch"


def test_harness_detects_checkpoint_terminal_tampering(tmp_path: Path) -> None:
    trajectory = _geo_trajectory(trajectory_id="checkpoint-tamper")
    harness = ToolHarness(_config(tmp_path), context=_context())
    harness.run(trajectory)
    report_path = harness.last_report_path
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["records"][-1]["final_location"] = "被篡改地点"
    report_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(HarnessError) as caught:
        ToolHarness(_config(tmp_path), context=_context()).run(trajectory)
    assert caught.value.error_code == "checkpoint_record_invalid"


def test_result_corruption_reruns_producer_and_dependents(tmp_path: Path) -> None:
    trajectory = _geo_trajectory(trajectory_id="result-corruption")
    first_harness = ToolHarness(_config(tmp_path), context=_context())
    first = first_harness.run(trajectory)
    result_id = first.records[1].observation.result["result_id"]
    result_path = first_harness.last_run_dir / "results" / f"{result_id}.json"
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    stored["payload"]["candidate_count"] = 999
    result_path.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")

    geocoder = FakeGeocodeClient()
    overpass = FakeOverpassClient()
    resumed = ToolHarness(
        _config(tmp_path),
        context=_context(geocoder, overpass),
    ).run(trajectory)
    assert resumed.status == "completed"
    assert geocoder.calls == ["郑州市"]
    assert len(overpass.calls) == 1
    assert resumed.records[1].reused_from_checkpoint is False
    assert resumed.records[2].reused_from_checkpoint is False
    assert resumed.records[3].reused_from_checkpoint is False


def test_missing_reference_blocks_only_dependent_step_by_default(tmp_path: Path) -> None:
    geocoder = FakeGeocodeClient()
    trajectory = Trajectory(
        id="missing-reference",
        system_prompt="system",
        user_query="query",
        steps=[
            TrajectoryStep(
                event_type="tool_call",
                thought="错误前向引用",
                action=Action(
                    tool="osm_query",
                    params={
                        "operation": "count",
                        "purpose": "统计",
                        "inputs": {"source_result": "$step_2_tool_result.result_id"},
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="独立查询仍应运行",
                action=Action(
                    tool="geocode",
                    params={
                        "operation": "geocode",
                        "purpose": "解析",
                        "inputs": {"query": "郑州市"},
                    },
                ),
            ),
            TrajectoryStep(
                event_type="final",
                thought="答案",
                action=Action(tool="final_answer", params={"location": "郑州市"}),
            ),
        ],
    )
    report = ToolHarness(
        _config(tmp_path),
        context=_context(geocoder=geocoder),
    ).run(trajectory)
    assert report.status == "completed_with_errors"
    assert report.records[0].status == "blocked"
    assert report.records[0].error_code == "forward_reference"
    assert report.records[1].status == "succeeded"
    assert geocoder.calls == ["郑州市"]


def test_parameter_compiler_exception_is_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("compiler exploded")

    monkeypatch.setattr(harness_module, "normalize_and_validate_tool_inputs", explode)
    report = ToolHarness(_config(tmp_path), context=_context()).run(
        _geo_trajectory(trajectory_id="compiler-isolation")
    )
    assert report.status == "completed_with_errors"
    assert report.records[1].status == "invalid"
    assert report.records[1].error_code == "parameter_validation_exception"
    assert report.records[2].status == "blocked"
    assert report.records[-1].status == "terminal"


def test_credentials_in_inputs_are_rejected_and_redacted(tmp_path: Path) -> None:
    secret = "secret-credential-value-123456"
    trajectory = Trajectory(
        id="credential-guard",
        system_prompt="system",
        user_query="query",
        steps=[
            TrajectoryStep(
                event_type="tool_call",
                thought="非法携带密钥",
                action=Action(
                    tool="geocode",
                    params={
                        "operation": "geocode",
                        "purpose": "解析",
                        "inputs": {"query": "郑州市", "api_key": secret},
                    },
                ),
            ),
            TrajectoryStep(
                event_type="final",
                thought="答案",
                action=Action(tool="final_answer", params={"location": "郑州市"}),
            ),
        ],
    )
    harness = ToolHarness(_config(tmp_path), context=_context())
    report = harness.run(trajectory)
    assert report.records[0].status == "invalid"
    assert report.records[0].error_code == "credential_in_inputs"
    assert report.records[0].raw_inputs["api_key"] == "***REDACTED***"
    assert secret not in harness.last_report_path.read_text(encoding="utf-8")


def test_credentials_in_executor_output_are_failed_and_redacted(tmp_path: Path) -> None:
    secret = "sk-abcdefghijklmnopqrstuvwx"

    def leaking_dispatcher(*_args: Any, **_kwargs: Any) -> Observation:
        return Observation(ok=True, result={"api_key": secret, "value": 1})

    trajectory = _geo_trajectory(trajectory_id="credential-output")
    harness = ToolHarness(
        _config(tmp_path, stop_on_error=True),
        context=_context(),
        dispatcher=leaking_dispatcher,
    )
    report = harness.run(trajectory)
    first_call = report.records[1]
    assert first_call.status == "failed"
    assert first_call.error_code == "credential_in_observation"
    assert first_call.observation.result["api_key"] == "***REDACTED***"
    assert secret not in harness.last_report_path.read_text(encoding="utf-8")


def test_executor_exception_is_isolated_and_stop_policy_is_explicit(tmp_path: Path) -> None:
    calls: list[str] = []

    def broken_dispatcher(
        tool_name: str,
        operation: str,
        **_kwargs: Any,
    ) -> Observation:
        calls.append(f"{tool_name}.{operation}")
        raise RuntimeError("provider exploded")

    trajectory = _geo_trajectory(trajectory_id="stop-on-error")
    report = ToolHarness(
        _config(tmp_path, stop_on_error=True),
        context=_context(),
        dispatcher=broken_dispatcher,
    ).run(trajectory)
    assert report.records[1].status == "failed"
    assert report.records[1].error_code == "executor_exception"
    assert report.records[2].status == "blocked"
    assert report.records[3].status == "blocked"
    assert report.records[4].status == "terminal"
    assert calls == ["geocode.geocode"]


@pytest.mark.parametrize("bad_value", [object(), float("nan"), float("inf")])
def test_invalid_provider_observation_is_isolated(
    tmp_path: Path,
    bad_value: Any,
) -> None:
    def invalid_dispatcher(*_args: Any, **_kwargs: Any) -> Observation:
        return Observation(ok=True, result={"value": bad_value})

    trajectory = Trajectory(
        id=f"invalid-observation-{type(bad_value).__name__}-{bad_value!r}",
        system_prompt="system",
        user_query="query",
        steps=[
            TrajectoryStep(
                event_type="tool_call",
                thought="测试异常回执",
                action=Action(
                    tool="geocode",
                    params={
                        "operation": "geocode",
                        "purpose": "验证边界",
                        "inputs": {"query": "郑州市"},
                    },
                ),
            ),
            TrajectoryStep(
                event_type="final",
                thought="答案",
                action=Action(tool="final_answer", params={"location": "郑州市"}),
            ),
        ],
    )
    report = ToolHarness(
        _config(tmp_path),
        context=_context(),
        dispatcher=invalid_dispatcher,
    ).run(trajectory)
    assert report.status == "completed_with_errors"
    assert report.records[0].status == "failed"
    assert report.records[0].error_code == "invalid_observation"
    assert report.records[1].status == "terminal"


def test_outer_contract_and_final_contract_are_strict(tmp_path: Path) -> None:
    bad_outer = _geo_trajectory(trajectory_id="bad-outer")
    bad_outer.steps[1].action.params["extra"] = "x"
    report = ToolHarness(_config(tmp_path), context=_context()).run(bad_outer)
    assert report.records[1].status == "invalid"
    assert report.records[1].error_code == "invalid_outer_contract"

    bad_final = _geo_trajectory(trajectory_id="bad-final")
    bad_final.steps[-1].action.params["confidence"] = 1
    with pytest.raises(HarnessError) as caught:
        ToolHarness(_config(tmp_path), context=_context()).run(bad_final)
    assert caught.value.error_code == "invalid_final_contract"


def test_single_writer_lock_rejects_concurrent_same_trajectory(tmp_path: Path) -> None:
    config = _config(tmp_path)
    harness = ToolHarness(config, context=_context())
    run_dir = harness._select_run_dir("geo-chain")
    run_dir.mkdir(parents=True)
    with (
        FileLock(str(run_dir / ".harness.lock")),
        pytest.raises(HarnessError) as caught,
    ):
        harness.run(_geo_trajectory())
    assert caught.value.error_code == "run_already_locked"


def test_missing_artifact_invalidates_transitive_dependent_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 20), (20, 40, 60)).save(source)
    trajectory = Trajectory(
        id="image-resume",
        system_prompt="system",
        user_query="query",
        image_paths=[str(source)],
        steps=[
            TrajectoryStep(
                event_type="tool_call",
                thought="裁剪",
                action=Action(
                    tool="image_edit",
                    params={
                        "operation": "crop",
                        "purpose": "裁剪区域",
                        "inputs": {
                            "image": "$current_image",
                            "region": [0, 0, 10, 10],
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="放大",
                action=Action(
                    tool="image_edit",
                    params={
                        "operation": "zoom",
                        "purpose": "放大裁剪图",
                        "inputs": {
                            "image": "$step_1_tool_result.image_id",
                            "region": [0, 0, 10, 10],
                            "scale": 2,
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="final",
                thought="答案",
                action=Action(tool="final_answer", params={"location": "测试地点"}),
            ),
        ],
    )
    calls: list[str] = []

    def counting_dispatcher(
        tool_name: str,
        operation: str,
        **kwargs: Any,
    ) -> Observation:
        calls.append(f"{tool_name}.{operation}")
        return real_execute(tool_name, operation, **kwargs)

    config = _config(tmp_path)
    first_harness = ToolHarness(config, dispatcher=counting_dispatcher)
    first = first_harness.run(trajectory)
    assert first.status == "completed"
    assert calls == ["image_edit.crop", "image_edit.zoom"]
    crop_path = Path(first.records[0].observation.artifacts["image_path"])
    crop_path.unlink()

    calls.clear()
    resumed = ToolHarness(config, dispatcher=counting_dispatcher).run(trajectory)
    assert resumed.status == "completed"
    assert calls == ["image_edit.crop", "image_edit.zoom"]
    assert resumed.records[0].reused_from_checkpoint is False
    assert resumed.records[1].reused_from_checkpoint is False


def test_modified_artifact_invalidates_transitive_dependent_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 20), (20, 40, 60)).save(source)
    trajectory = Trajectory(
        id="image-artifact-tamper",
        system_prompt="system",
        user_query="query",
        image_paths=[str(source)],
        steps=[
            TrajectoryStep(
                event_type="tool_call",
                thought="裁剪",
                action=Action(
                    tool="image_edit",
                    params={
                        "operation": "crop",
                        "purpose": "裁剪区域",
                        "inputs": {"image": "$current_image", "region": [0, 0, 10, 10]},
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="测量裁剪图",
                action=Action(
                    tool="image_measure",
                    params={
                        "operation": "measure",
                        "purpose": "量测宽度",
                        "inputs": {
                            "image": "$step_1_tool_result.image_id",
                            "measurement": "distance",
                            "region": [0, 0, 10, 10],
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="final",
                thought="答案",
                action=Action(tool="final_answer", params={"location": "测试地点"}),
            ),
        ],
    )
    calls: list[str] = []

    def counting_dispatcher(
        tool_name: str,
        operation: str,
        **kwargs: Any,
    ) -> Observation:
        calls.append(f"{tool_name}.{operation}")
        return real_execute(tool_name, operation, **kwargs)

    config = _config(tmp_path)
    first = ToolHarness(config, dispatcher=counting_dispatcher).run(trajectory)
    artifact = Path(first.records[0].observation.artifacts["image_path"])
    Image.new("RGB", (10, 10), (255, 0, 0)).save(artifact)

    calls.clear()
    resumed = ToolHarness(config, dispatcher=counting_dispatcher).run(trajectory)
    assert resumed.status == "completed"
    assert calls == ["image_edit.crop", "image_measure.measure"]
    assert resumed.records[0].reused_from_checkpoint is False
    assert resumed.records[1].reused_from_checkpoint is False


def test_process_interruption_leaves_running_checkpoint_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 20), (20, 40, 60)).save(source)
    trajectory = Trajectory(
        id="interrupted-resume",
        system_prompt="system",
        user_query="query",
        image_paths=[str(source)],
        steps=[
            TrajectoryStep(
                event_type="tool_call",
                thought="裁剪",
                action=Action(
                    tool="image_edit",
                    params={
                        "operation": "crop",
                        "purpose": "测试中断恢复",
                        "inputs": {"image": "$current_image", "region": [0, 0, 10, 10]},
                    },
                ),
            ),
            TrajectoryStep(
                event_type="final",
                thought="答案",
                action=Action(tool="final_answer", params={"location": "测试地点"}),
            ),
        ],
    )

    class CrashOnceDispatcher:
        def __init__(self) -> None:
            self.crashed = False

        def __call__(
            self,
            tool_name: str,
            operation: str,
            **kwargs: Any,
        ) -> Observation:
            if not self.crashed:
                self.crashed = True
                raise KeyboardInterrupt("simulated process interruption")
            return real_execute(tool_name, operation, **kwargs)

    dispatcher = CrashOnceDispatcher()
    first = ToolHarness(_config(tmp_path), dispatcher=dispatcher)
    with pytest.raises(KeyboardInterrupt, match="simulated"):
        first.run(trajectory)
    checkpoint = json.loads(first.last_report_path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "running"
    assert checkpoint["records"][0]["status"] == "running"

    resumed = ToolHarness(_config(tmp_path), dispatcher=dispatcher).run(trajectory)
    assert resumed.status == "completed"
    assert resumed.resumed is True
    assert resumed.records[0].status == "succeeded"
    assert any("上次中断于 running" in warning for warning in resumed.warnings)


def test_runtime_roots_are_additive_and_include_exact_input_image(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    Image.new("RGB", (4, 4), (1, 2, 3)).save(image)
    context_root = tmp_path / "context-root"
    config_root = tmp_path / "config-root"
    context_root.mkdir()
    config_root.mkdir()
    trajectory = Trajectory(
        id="root-union",
        system_prompt="system",
        user_query="query",
        image_paths=[str(image)],
        steps=[
            TrajectoryStep(
                event_type="final",
                thought="答案",
                action=Action(tool="final_answer", params={"location": "测试地点"}),
            )
        ],
    )
    harness = ToolHarness(
        _config(tmp_path, allowed_file_roots=[config_root]),
        context=RuntimeContext(allowed_file_roots=[context_root]),
    )
    run_dir = tmp_path / "manual-run"
    runtime = harness._runtime_context(trajectory, run_dir)
    assert runtime.allowed_file_roots is not None
    assert {path.resolve() for path in runtime.allowed_file_roots} == {
        run_dir.resolve(),
        config_root.resolve(),
        context_root.resolve(),
        image.resolve(),
    }
    assert runtime.current_image == "input_0001"


def test_changed_input_file_content_invalidates_checkpoint(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(image)
    trajectory = Trajectory(
        id="input-content-fingerprint",
        system_prompt="system",
        user_query="query",
        image_paths=[str(image)],
        steps=[
            TrajectoryStep(
                event_type="final",
                thought="答案",
                action=Action(tool="final_answer", params={"location": "测试地点"}),
            )
        ],
    )
    ToolHarness(_config(tmp_path)).run(trajectory)
    Image.new("RGB", (8, 8), (9, 8, 7)).save(image)
    with pytest.raises(HarnessError) as caught:
        ToolHarness(_config(tmp_path)).run(trajectory)
    assert caught.value.error_code == "runtime_fingerprint_mismatch"


def test_harness_controls_mixed_tool_chain_and_isolates_failed_branch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scene.png"
    Image.new("RGB", (64, 48), (12, 34, 56)).save(source)
    trajectory = Trajectory(
        id="mixed-acceptance",
        system_prompt="system",
        user_query="Locate the scene.",
        image_paths=[str(source)],
        steps=[
            TrajectoryStep(event_type="reasoning", thought="先读取本地图像证据"),
            TrajectoryStep(
                event_type="tool_call",
                thought="裁剪中心区域",
                action=Action(
                    tool="image_edit",
                    params={
                        "operation": "crop",
                        "purpose": "生成可测量局部图",
                        "inputs": {"image": "$current_image", "region": [8, 8, 40, 32]},
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="测量裁剪图宽度",
                action=Action(
                    tool="image_measure",
                    params={
                        "operation": "measure",
                        "purpose": "取得像素尺度",
                        "inputs": {
                            "image": "$step_2_tool_result.image_id",
                            "measurement": "distance",
                            "axis": "horizontal",
                            "region": [0, 0, 32, 24],
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="比较原图和裁剪图几何",
                action=Action(
                    tool="image_compare",
                    params={
                        "operation": "compare",
                        "purpose": "检查几何比例变化",
                        "inputs": {
                            "images": [
                                "$current_image",
                                "$step_2_tool_result.image_id",
                            ],
                            "method": "geometry",
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="读取原图文件元数据",
                action=Action(
                    tool="media_metadata_read",
                    params={
                        "operation": "file",
                        "purpose": "确认输入尺寸",
                        "inputs": {"file": "$current_image", "fields": ["width", "height"]},
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="解析候选城市",
                action=Action(
                    tool="geocode",
                    params={
                        "operation": "geocode",
                        "purpose": "取得城市 bbox",
                        "inputs": {"query": "郑州市", "top_k": 1},
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="查询候选桥梁",
                action=Action(
                    tool="osm_query",
                    params={
                        "operation": "query",
                        "purpose": "取得 bbox 内桥梁",
                        "inputs": {
                            "bbox": "$step_6_tool_result.candidates[0].bbox",
                            "feature_types": ["桥梁"],
                            "limit": 5,
                        },
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="统计桥梁结果",
                action=Action(
                    tool="osm_query",
                    params={
                        "operation": "count",
                        "purpose": "统计候选数量",
                        "inputs": {"source_result": "$step_7_tool_result.result_id"},
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="尝试尚未落地的地图图层",
                action=Action(
                    tool="map_layer_query",
                    params={
                        "operation": "load_layer",
                        "purpose": "验证失败隔离",
                        "inputs": {"area": "郑州市", "layers": ["roads"]},
                    },
                ),
            ),
            TrajectoryStep(
                event_type="tool_call",
                thought="引用失败分支应被阻止",
                action=Action(
                    tool="osm_query",
                    params={
                        "operation": "count",
                        "purpose": "验证依赖阻断",
                        "inputs": {"source_result": "$step_9_tool_result.result_id"},
                    },
                ),
            ),
            TrajectoryStep(
                event_type="final",
                thought="保留原轨迹答案",
                action=Action(tool="final_answer", params={"location": "郑州市"}),
            ),
        ],
    )
    report = ToolHarness(_config(tmp_path), context=_context()).run(trajectory)
    assert report.status == "completed_with_errors"
    assert report.counts == {
        "total": 11,
        "skipped_reasoning": 1,
        "succeeded": 7,
        "failed": 1,
        "blocked": 1,
        "terminal": 1,
    }
    assert report.records[2].observation.result["value"] == 32.0
    assert report.records[3].observation.result["summary"]["pair_count"] == 1
    assert report.records[4].observation.result["applied"]["fields"] == [
        "width",
        "height",
    ]
    assert report.records[7].observation.result["count"] == 1
    assert report.records[8].error_code == "not_implemented"
    assert report.records[9].error_code == "unresolved_reference"
    assert report.final_location == "郑州市"
