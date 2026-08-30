"""Evidence-linked human review sidecars; never filter or rewrite trajectories."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pipeline.schemas.audit import GeoTaskSpec
from pipeline.schemas.confidence import ConfidenceReport
from pipeline.schemas.tools import ToolParameterAudit
from pipeline.schemas.trajectory import Trajectory
from pipeline.schemas.transcript import TranscriptSegment


def timecode(seconds: float | None) -> str:
    if seconds is None:
        return "未记录"
    millis = round(max(0.0, seconds) * 1000)
    minutes, rest = divmod(millis, 60000)
    sec, ms = divmod(rest, 1000)
    return f"{minutes:02d}:{sec:02d}.{ms:03d}"


def _window(start: float, end: float) -> dict[str, Any]:
    return {"start": start, "end": end, "timecode": f"{timecode(start)}–{timecode(end)}"}


def _path_key(path: str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").casefold()


def _transcript_context(
    task: GeoTaskSpec, transcript: list[TranscriptSegment]
) -> list[dict[str, Any]]:
    # Adjacent context is labelled, not used as this question's answer evidence.
    indexed = sorted(enumerate(transcript), key=lambda pair: pair[1].start)
    overlap = [pair for pair in indexed if pair[1].end > task.time_start and pair[1].start < task.time_end]
    before = [pair for pair in indexed if pair[1].end <= task.time_start]
    after = [pair for pair in indexed if pair[1].start >= task.time_end]
    rows = []
    for role, pairs in (("preceding_context", before[-1:]), ("task_overlap", overlap), ("following_context", after[:1])):
        for index, segment in pairs:
            rows.append({"segment_index": index, "role": role, **_window(segment.start, segment.end), "text": segment.text})
    return rows


def _step_indices(evidence: str, count: int) -> list[int]:
    indices: set[int] = set()
    for match in re.finditer(r"(?:steps?|步骤|第)\s*([\d、,/／ ]+)(?:步)?", evidence, re.IGNORECASE):
        indices.update(int(number) for number in re.findall(r"\d+", match.group(1)))
    return sorted(index for index in indices if 1 <= index <= count)


def build_review_packet(
    *,
    report: ConfidenceReport,
    task: GeoTaskSpec,
    trajectory: Trajectory,
    transcript: list[TranscriptSegment],
    parameter_audits: list[ToolParameterAudit] | None = None,
    context_transcript: list[TranscriptSegment] | None = None,
    source_video_path: str | None = None,
) -> dict[str, Any]:
    """Build actionable findings, separating file facts from reviewer allegations."""
    selected = []
    for path in trajectory.image_paths:
        exact = next((a for a in task.frame_assessments if _path_key(a.image_path) == _path_key(path)), None)
        assessment = exact or next((a for a in task.frame_assessments if a.selected and Path(a.image_path).name == Path(path).name), None)
        selected.append({
            "image_path": str(Path(path).resolve()),
            "timestamp": assessment.timestamp if assessment else None,
            "timestamp_source": "assessment_path" if exact else "selected_assessment_basename" if assessment else "unrecorded",
            "file_exists": Path(path).is_file(),
            "assessment": assessment.model_dump(mode="json") if assessment else None,
        })
    # These are explicitly *unselected* candidates. Leakage here is not leakage in SFT input.
    candidates = [a.model_dump(mode="json") for a in task.frame_assessments if not a.selected]
    issues: list[dict[str, Any]] = []

    def add(code: str, origin: str, summary: str, guidance: str, *, evidence: str = "", frames: list[dict[str, Any]] | None = None, steps: list[int] | None = None, severity: str = "warning") -> None:
        indices = steps or []
        issues.append({
            "code": code, "origin": origin, "severity": severity,
            "verification": "metadata_fact" if origin == "artifact" else "requires_human_confirmation",
            "summary": summary, "evidence": evidence,
            "task_window": _window(task.time_start, task.time_end),
            "frames": frames or [], "step_indices": indices,
            "json_pointers": [f"/steps/{index - 1}/observation" for index in indices],
            "repair_guidance": guidance,
        })

    if not selected:
        add("no_selected_image", "artifact", "Stage 3/SFT 未记录任何输入图片", "先在本题时间窗内核对原始目标；参考候选帧和出示窗，但不要把地图核验图或答案帧直接补为原图。允许保留当前结果供人工审核，不中断流水线。", severity="error")
        add("stage2_image_provenance_unknown", "artifact", "没有 Stage 2 实际入模图片清单，不能断言模型未看图", "当前 Stage 2 在空图片列表时可能内部回退整视频概览帧，而 Stage 3/4 仍记录空图；先对齐三阶段实际使用的图片清单，再判断内容错配。")
    missing = [frame for frame in selected if not frame["file_exists"]]
    if missing:
        add("selected_file_missing", "artifact", "输入清单中的图片文件不存在", "恢复原文件或重新截取并同步 Stage 2/3/4 的图片路径，不以假路径代替。", frames=missing, severity="error")
    for frame in selected:
        assessment = frame["assessment"]
        if not assessment:
            continue
        if assessment.get("answer_leakage"):
            add("selected_answer_leakage", "stage15_assessment", "选中帧被标记为答案泄露", "人工确认泄露文字是否为原题固有线索；如为讲解揭晓内容，回到更早的原图出示时段，或在不破坏定位线索的前提下裁去答案区域。", evidence=assessment.get("reason", ""), frames=[frame], severity="error")
        if assessment.get("tutorial_overlay") or not assessment.get("clean_source"):
            add("selected_frame_packaging", "stage15_assessment", "原图可能正确，但带讲解覆盖或界面包装", "区分原生无人机 HUD 与后加字幕、箭头、社交界面；优先选同一原图更完整的出示帧，裁剪时保留定位所需边缘及阴影。不要仅凭水印就判成另一题。", evidence=assessment.get("reason", ""), frames=[frame])
        if float(assessment.get("quality_score", 0)) < 0.65:
            add("low_selected_frame_quality", "stage15_assessment", "选中帧质量评分低，需核对关键线索是否仍可辨认", "逐项检查原图主体、分辨率、遮挡和地标覆盖；不把低分本身视为选错图的证明。", evidence=assessment.get("reason", ""), frames=[frame])

    status = getattr(task.status, "value", task.status)
    if status == "needs_review":
        add("stage15_review_reason", "stage15_assessment", "Stage 1.5 留下待复核原因", "若原因是多个不相邻出示窗但只选一张，先判断是不是同一张原图重复出现；只有存在独立且缺失的视觉证据才补图，不强制凑图片数量。", evidence=task.status_reason)

    guides = {
        "image_trajectory_mismatch": "在记录的选帧时间与本题视觉简报之间逐项比对地标；换错题/镜头的图优先修正输入。字幕已有的地图返回不能仅因选图错配就再判为虚假 Observation。",
        "fabricated_observation": "逐步核对动作是否发生、返回值是否有来源；查看本题字幕及前后语境。明确杜撰的回执可改为 reasoning 或人工重写；不能仅因窄切片没有写出动作就断言未执行。",
        "hallucinated_precision": "查找原字幕/原图中的数值和单位；无来源则去掉精确值或保留为待验证推测，不生成新的坐标、距离或编号补洞。",
    }
    for gate in [*report.hard_gates, *report.soft_flags]:
        if gate.code == "task_needs_review":
            continue
        add(gate.code, "stage4_reviewer", "裁判报告的问题（待人工确认）", guides.get(gate.code, "核对裁判给出的证据和字段，保留原始输出，再决定修复方式；此记录不触发自动删除或重生成。"), evidence=gate.evidence, frames=selected if "image" in gate.code else None, steps=_step_indices(gate.evidence, len(trajectory.steps)), severity="error" if gate in report.hard_gates else "warning")

    for audit in parameter_audits or []:
        unavailable = any(issue.code == "input_schema_unavailable" for issue in audit.issues)
        if audit.readiness == "ready" and not unavailable:
            continue
        evidence = "；".join(f"{issue.field or '-'}: {issue.code}: {issue.message}" for issue in audit.issues)
        guidance = "；".join(action.guidance for action in audit.repair_actions)
        add("tool_contract_unverified" if unavailable else "tool_parameter_repair", "parameter_audit", f"step {audit.step_index}: {audit.tool}.{audit.operation} = {audit.readiness}", guidance or "补齐对应 operation 的 schema 与真实输入来源；schema 通过不代表后端工具已经实现或执行成功。", evidence=evidence, steps=[audit.step_index])
        issues[-1]["json_pointers"] = [f"/steps/{audit.step_index - 1}/action/params/inputs"]

    if report.judge_call_failed:
        add("judge_unavailable", "artifact", "语义审核未完成，不应把中性分当低质量证据", "保留样本，单独补审；不要重跑已成功的 Stage 2/3。")
    return {
        "schema_version": "manual_review_v1", "policy": "advisory_only_no_execution_gate",
        "task_id": task.task_id, "task_window": _window(task.time_start, task.time_end),
        "source_video_path": str(Path(source_video_path).resolve()) if source_video_path else None,
        "quality_score": report.quality_score, "audit_coverage": report.audit_coverage,
        "review_priority": report.review_priority, "quality_recommendation": report.decision,
        "notice": "分数是审核启发式，不是正确率；reject 是质量建议，文件仍保留。Tool 调用为蒸馏记录，不代表已执行真实地理工具。",
        "evaluation_notes": report.notes,
        "selected_frames": selected, "unselected_candidates": candidates,
        "proposed_show_source_windows": [_window(p.start, p.end) for p in task.process_intervals if p.role.value == "show_source"],
        "window_notice": "出示窗来自 Stage 1.5 模型，供定位复核，不保证其中必有正确原图。",
        "transcript_context": _transcript_context(task, context_transcript if context_transcript is not None else transcript),
        "issues": issues,
    }


def _file_link(path: str) -> str:
    return f"[{Path(path).name}](<{str(Path(path).resolve()).replace(chr(92), '/')}>)"


def render_review_markdown(packet: dict[str, Any]) -> str:
    lines = [f"# 人工审核：{packet['task_id']}", "", packet["notice"], "",
             f"题目范围：**{packet['task_window']['timecode']}**；分数 {packet['quality_score']:.4f}；coverage {packet['audit_coverage']:.4f}；建议 {packet['quality_recommendation']}。", ""]
    if packet.get("source_video_path"):
        lines.extend([f"原视频：{_file_link(packet['source_video_path'])}", ""])
    notes = str(packet.get("evaluation_notes") or "").strip() or "（无评价说明）"
    lines.extend(["## 评价说明", "", notes, "", "## 选图与候选窗", ""])
    if not packet["selected_frames"]:
        lines.append("Stage 3/SFT 未记录选中图；不要把以下候选图误认为已经输入模型。")
    for frame in packet["selected_frames"]:
        lines.append(f"- 选中帧 {timecode(frame['timestamp'])}：{_file_link(frame['image_path'])}；时间来源 {frame['timestamp_source']}。")
    windows = [window["timecode"] for window in packet["proposed_show_source_windows"]]
    lines.extend(["", "建议复核的模型出示窗：" + ("；".join(windows) or "未记录"), packet["window_notice"], "", "## 问题与修改指导", ""])
    for index, issue in enumerate(packet["issues"], start=1):
        stamps = "、".join(timecode(frame["timestamp"]) for frame in issue["frames"]) or "无精确选帧时间，按题目窗口定位"
        lines.extend([f"### {index}. {issue['summary']}", "",
                      f"- 类型：`{issue['code']}`；来源：`{issue['origin']}`；状态：`{issue['verification']}`。",
                      f"- 时间：{issue['task_window']['timecode']}；关联帧：{stamps}。",
                      f"- 原因：{issue['evidence'] or '见输入清单事实。'}",
                      f"- 修改建议：{issue['repair_guidance']}"])
        if issue["json_pointers"]:
            lines.append("- 轨迹位置：" + "、".join(f"`{pointer}`" for pointer in issue["json_pointers"]))
        lines.append("")
    lines.extend(["## 字幕与前后语境", "", "相邻语境仅帮助核对方法归属，不自动作为本题答案来源。", ""])
    for segment in packet["transcript_context"]:
        lines.extend([f"- {segment['timecode']} [{segment['role']}] {segment['text']}", ""])
    lines.extend(["## 未采用的候选帧", "", "以下帧均未被选择；候选帧含答案不等于最终输入泄露。", ""])
    for candidate in packet["unselected_candidates"]:
        lines.append(f"- {timecode(candidate['timestamp'])}：{_file_link(candidate['image_path'])}；{candidate['kind']}；答案泄露标记={candidate['answer_leakage']}。{candidate.get('reason', '')}")
    return "\n".join(lines) + "\n"


def write_review_packet(packet: dict[str, Any], report_path: str | Path) -> tuple[Path, Path]:
    base = Path(report_path)
    json_path = base.with_name(base.stem + ".review.json")
    md_path = base.with_name(base.stem + ".review.md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_review_markdown(packet), encoding="utf-8")
    return json_path, md_path
