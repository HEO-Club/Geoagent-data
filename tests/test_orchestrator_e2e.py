"""Orchestrator + batch：Mock 端到端测试（禁止真实 API）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from pipeline.config import Settings, clear_settings_cache
from pipeline.schemas import (
    Action,
    AgentRole,
    AgentTimeSegment,
    LocationHypothesis,
    Move,
    NormalizationMode,
    NormalizedStep,
    ObservationExecutionResult,
    ObservationSource,
    PreprocessResult,
    StageStatus,
    SubmitAnswerResult,
    TimedScreenAction,
    Trajectory,
    TrajectoryStep,
    TrajectoryVerificationReport,
    TranscriptSegment,
    VerificationResult,
    VideoInput,
)
from run_one_video import (
    is_video_fully_completed,
    load_or_create_manifest,
    run_one_video,
    video_id_from_path,
)


GT = (48.8584, 2.2945)


@pytest.fixture()
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("ALLOW_REAL_API", "false")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("MAX_CONCURRENT_VIDEOS", "2")
    clear_settings_cache()
    settings = Settings(
        APP_ENV="test",
        ALLOW_REAL_API=False,
        INTERMEDIATE_DIR=str(tmp_path / "intermediate"),
        OUTPUT_DIR=str(tmp_path / "output"),
        CACHE_DIR=str(tmp_path / "cache"),
        MAX_CONCURRENT_VIDEOS=2,
        MAX_REVISION_ROUNDS=2,
    )
    yield settings
    clear_settings_cache()


def _transcript() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(start=0.0, end=5.0, text="看起来像欧洲的铁塔结构。"),
        TranscriptSegment(start=5.0, end=12.0, text="我在地图上确认一下具体位置。"),
        TranscriptSegment(
            start=12.0,
            end=15.0,
            text="答案是巴黎埃菲尔铁塔，坐标四十八点八五八四，二点二九四五。",
        ),
        TranscriptSegment(start=15.0, end=20.0, text="再核对一下街景是否吻合。"),
    ]


def _video_input(video_path: str) -> VideoInput:
    return VideoInput(
        video_path=video_path,
        transcript=_transcript(),
        groundtruth=GT,
        source_platform="test",
    )


def _hyp() -> LocationHypothesis:
    return LocationHypothesis(
        possible_countries=["France"],
        possible_regions=["Île-de-France"],
        reasoning_summary="Iron lattice.",
        confidence=0.8,
        key_clues_remaining=[],
    )


def _submit() -> SubmitAnswerResult:
    return SubmitAnswerResult(
        latitude=48.8584,
        longitude=2.2945,
        location_name="Eiffel Tower",
        confidence=0.95,
        reasoning="Map match.",
    )


def _move(role: AgentRole, start: float, end: float, narration: str) -> Move:
    return Move(
        start_time=start,
        end_time=end,
        narration=narration,
        screen_action="查看屏幕",
        visible_clues=["tower"],
        agent_role=role,
    )


def _make_hooks() -> dict[str, Any]:
    """注入确定性钩子，完全绕过真实视频 / LLM / 付费 API。"""

    def fake_preprocess(video_input: VideoInput) -> PreprocessResult:
        return PreprocessResult(
            answer_timestamp=12.0,
            agent_segments=[
                AgentTimeSegment(
                    agent_role=AgentRole.COARSE, start_time=0.0, end_time=5.0
                ),
                AgentTimeSegment(
                    agent_role=AgentRole.FINE, start_time=5.0, end_time=12.0
                ),
                AgentTimeSegment(
                    agent_role=AgentRole.VERIFIER, start_time=15.0, end_time=20.0
                ),
            ],
            revision_segments=[],
        )

    def fake_extract(video_path: str, time_range: tuple[float, float], fps: float = 1.0) -> list[str]:
        return [f"{video_path}::frame::{time_range[0]:.1f}"]

    def fake_detect(
        keyframes: list[str],
        narration_context: str,
        time_range: tuple[float, float],
    ) -> list[TimedScreenAction]:
        return [
            TimedScreenAction(
                start_time=time_range[0],
                end_time=min(time_range[0] + 1.0, time_range[1]),
                description="屏幕操作",
                visible_clues=["clue"],
            )
        ]

    def fake_moves(
        video_input: VideoInput,
        preprocess_result: PreprocessResult,
        screen_actions_by_role: dict[AgentRole, list[TimedScreenAction]],
    ) -> dict[AgentRole, list[Move]]:
        return {
            AgentRole.COARSE: [
                _move(AgentRole.COARSE, 0.0, 5.0, "宏观特征")
            ],
            AgentRole.FINE: [
                _move(AgentRole.FINE, 5.0, 10.0, "查地图"),
                _move(AgentRole.FINE, 10.0, 12.0, "提交"),
            ],
            AgentRole.VERIFIER: [
                _move(AgentRole.VERIFIER, 15.0, 20.0, "交叉验证")
            ],
        }

    def fake_normalize(moves: list[Move], agent_role: AgentRole) -> list[NormalizedStep]:
        steps: list[NormalizedStep] = []
        for i, move in enumerate(moves):
            if agent_role == AgentRole.FINE and i == len(moves) - 1:
                actions = [
                    Action(tool="submit_answer", params=_submit().model_dump())
                ]
            elif agent_role == AgentRole.FINE:
                actions = [Action(tool="map_query", params={"query": "tower"})]
            elif agent_role == AgentRole.VERIFIER:
                actions = [
                    Action(tool="map_query", params={"latlng": list(GT)})
                ]
            else:
                actions = [Action(tool="ocr", params={})]
            steps.append(
                NormalizedStep(
                    move=move,
                    thought_draft=move.narration,
                    actions=actions,
                    normalization_mode=NormalizationMode.MATCHED,
                    matched_tool_confidence=0.9,
                )
            )
        return steps

    def fake_observations(
        normalized_steps: list[NormalizedStep],
        image_path: str,
        agent_role: AgentRole,
        **kwargs: Any,
    ) -> list[ObservationExecutionResult]:
        results: list[ObservationExecutionResult] = []
        for step in normalized_steps:
            for action in step.actions:
                if action.tool == "submit_answer":
                    results.append(
                        ObservationExecutionResult(
                            action=action,
                            observation=None,
                            source=None,
                            status="skipped",
                        )
                    )
                elif action.tool == "map_query":
                    results.append(
                        ObservationExecutionResult(
                            action=action,
                            observation={
                                "status": "success",
                                "error_message": None,
                                "resolved_latlng": list(GT),
                                "formatted_address": "Paris",
                                "place_type": "tourist_attraction",
                                "viewport": None,
                                "place_id": None,
                            },
                            source=ObservationSource.REAL_EXECUTION,
                            status="success",
                        )
                    )
                else:
                    results.append(
                        ObservationExecutionResult(
                            action=action,
                            observation={
                                "status": "success",
                                "error_message": None,
                                "texts": ["Tour Eiffel"],
                            },
                            source=ObservationSource.REAL_EXECUTION,
                            status="success",
                        )
                    )
        return results

    def fake_reconstruct_all(
        all_steps: dict[AgentRole, list[NormalizedStep]],
        all_observations: dict[AgentRole, list[ObservationExecutionResult]],
        answer_timestamp: float,
        image_path: str,
    ) -> dict[AgentRole, Trajectory]:
        hyp = _hyp()
        submit = _submit()

        def _steps_from(
            role: AgentRole,
        ) -> list[TrajectoryStep]:
            units: list[TrajectoryStep] = []
            obs_list = all_observations[role]
            oi = 0
            for ns in all_steps[role]:
                for action in ns.actions:
                    obs_res = obs_list[oi]
                    oi += 1
                    units.append(
                        TrajectoryStep(
                            thought=ns.thought_draft,
                            action=action,
                            observation=obs_res.observation,
                            observation_source=obs_res.source,
                        )
                    )
            return units

        return {
            AgentRole.COARSE: Trajectory(
                id="main-coarse",
                agent_role=AgentRole.COARSE,
                system_prompt="coarse",
                user_query="粗定位",
                image_path=image_path,
                steps=_steps_from(AgentRole.COARSE),
                coarse_output=hyp,
            ),
            AgentRole.FINE: Trajectory(
                id="main-fine",
                agent_role=AgentRole.FINE,
                system_prompt="fine",
                user_query="精定位",
                image_path=image_path,
                steps=_steps_from(AgentRole.FINE),
                coarse_handoff=hyp,
                fine_output=submit,
            ),
            AgentRole.VERIFIER: Trajectory(
                id="main-ver",
                agent_role=AgentRole.VERIFIER,
                system_prompt="ver",
                user_query="验证",
                image_path=image_path,
                steps=_steps_from(AgentRole.VERIFIER),
                coarse_handoff=hyp,
                fine_handoff=submit,
                verifier_output=VerificationResult(
                    verdict="pass",
                    failed_checks=[],
                    suggested_recheck="",
                    return_to_agent=None,
                ),
            ),
        }

    def fake_reconstruct_rev(*args: Any, **kwargs: Any) -> list[Trajectory]:
        return []

    def fake_verify(
        traj: Trajectory,
        groundtruth: tuple[float, float],
        **kwargs: Any,
    ) -> TrajectoryVerificationReport:
        return TrajectoryVerificationReport(
            passed=True,
            quality_score=0.92,
            distance_error_km=0.1 if traj.agent_role == AgentRole.FINE else None,
            hard_fail_reasons=[],
            soft_warnings=[],
            leakage_detected=False,
        )

    return {
        "preprocess": fake_preprocess,
        "extract_keyframes": fake_extract,
        "detect_screen_actions": fake_detect,
        "build_all_agent_moves": fake_moves,
        "normalize_to_steps": fake_normalize,
        "generate_observations": fake_observations,
        "reconstruct_all_trajectories": fake_reconstruct_all,
        "reconstruct_revision_trajectories": fake_reconstruct_rev,
        "verify_and_score": fake_verify,
    }


def test_run_one_video_e2e_mock(cfg: Settings, tmp_path: Path) -> None:
    video = str(tmp_path / "sample_vid.mp4")
    Path(video).write_bytes(b"fake")
    hooks = _make_hooks()

    result = run_one_video(
        _video_input(video),
        settings=cfg,
        hooks=hooks,
    )
    assert result["status"] == "completed"
    assert result["entries_verified"] == 3
    assert result["entries_total"] == 3

    inter = Path(cfg.INTERMEDIATE_DIR) / "sample_vid"
    for name in (
        "manifest.json",
        "stage0_preprocess.json",
        "stage1_screen_actions.json",
        "stage2_moves.json",
        "stage3_normalized_steps.json",
        "stage4_observations.json",
        "stage5_trajectories.json",
        "stage6_verification.json",
        "stage7_entries.json",
    ):
        assert (inter / name).is_file(), name

    manifest = load_or_create_manifest("sample_vid", cfg)
    assert all(e.status == StageStatus.COMPLETED for e in manifest.stages)

    # 分片存在且含 resolved_latlng（FINE / VERIFIER）
    shard_fine = (
        Path(cfg.OUTPUT_DIR) / "shards" / "sample_vid_agent2.jsonl"
    ).read_text(encoding="utf-8")
    assert "resolved_latlng" in shard_fine
    assert "main-fine" in shard_fine


def test_checkpoint_resume_skips_completed(cfg: Settings, tmp_path: Path) -> None:
    video = str(tmp_path / "resume_vid.mp4")
    Path(video).write_bytes(b"fake")
    hooks = _make_hooks()
    call_counter = {"preprocess": 0}
    real_preprocess = hooks["preprocess"]

    def counting_preprocess(vin: VideoInput) -> PreprocessResult:
        call_counter["preprocess"] += 1
        return real_preprocess(vin)

    hooks["preprocess"] = counting_preprocess

    run_one_video(_video_input(video), settings=cfg, hooks=hooks)
    assert call_counter["preprocess"] == 1

    run_one_video(_video_input(video), settings=cfg, hooks=hooks)
    assert call_counter["preprocess"] == 1  # 第二次整段跳过
    assert is_video_fully_completed("resume_vid", cfg)


def test_force_rerun_from_invalidates(cfg: Settings, tmp_path: Path) -> None:
    video = str(tmp_path / "force_vid.mp4")
    Path(video).write_bytes(b"fake")
    hooks = _make_hooks()
    run_one_video(_video_input(video), settings=cfg, hooks=hooks)

    calls = {"verify": 0}
    base_verify = hooks["verify_and_score"]

    def counting_verify(traj: Trajectory, gt: tuple[float, float], **kw: Any) -> Any:
        calls["verify"] += 1
        return base_verify(traj, gt, **kw)

    hooks["verify_and_score"] = counting_verify
    run_one_video(
        _video_input(video),
        settings=cfg,
        hooks=hooks,
        force_rerun_from="stage6",
    )
    assert calls["verify"] >= 3


@pytest.mark.asyncio
async def test_batch_run_error_isolation_and_merge(
    cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from batch_run import batch_run_async

    ok_path = str(tmp_path / "ok_vid.mp4")
    bad_path = str(tmp_path / "bad_vid.mp4")
    Path(ok_path).write_bytes(b"ok")
    Path(bad_path).write_bytes(b"bad")

    hooks = _make_hooks()
    real_run = run_one_video

    def patched_run(
        video_input: VideoInput,
        *,
        video_id: Optional[str] = None,
        settings: Optional[Settings] = None,
        hooks: Optional[dict[str, Any]] = None,
        force_rerun_from: Optional[str] = None,
    ) -> dict[str, Any]:
        vid = video_id or video_id_from_path(video_input.video_path)
        if vid == "bad_vid":
            raise RuntimeError("simulated failure")
        return real_run(
            video_input,
            video_id=vid,
            settings=settings or cfg,
            hooks=hooks,
            force_rerun_from=force_rerun_from,
        )

    monkeypatch.setattr("batch_run.run_one_video", patched_run)

    summary = await batch_run_async(
        [_video_input(ok_path), _video_input(bad_path)],
        settings=cfg,
        max_attempts=1,
        skip_completed=False,
        merge=True,
        hooks=hooks,
    )
    assert summary["success_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["failed"][0]["video_id"] == "bad_vid"
    assert summary["merge_counts"]["agent1_coarse.jsonl"] == 1
    assert summary["merge_counts"]["agent2_fine.jsonl"] == 1
    assert summary["merge_counts"]["agent3_verifier.jsonl"] == 1

    final = (Path(cfg.OUTPUT_DIR) / "agent2_fine.jsonl").read_text(encoding="utf-8")
    row = json.loads(final.strip().splitlines()[0])
    assert row["agent_role"] == AgentRole.FINE.value
    assert any(
        "resolved_latlng" in m["content"]
        for m in row["messages"]
        if m["role"] == "tool"
    )


def test_batch_run_skips_completed(
    cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from batch_run import batch_run

    video = str(tmp_path / "skip_me.mp4")
    Path(video).write_bytes(b"x")
    hooks = _make_hooks()
    run_one_video(_video_input(video), settings=cfg, hooks=hooks)

    def boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("不应再次执行已完成视频")

    monkeypatch.setattr("batch_run.run_one_video", boom)
    summary = batch_run(
        [_video_input(video)],
        settings=cfg,
        max_attempts=1,
        skip_completed=True,
        merge=True,
        hooks=hooks,
    )
    assert summary["skip_count"] == 1
    assert summary["success_count"] == 0
    assert summary["failure_count"] == 0
