"""标准图片地理定位 TAO 风格 few-shot（压缩摘录，适配当前 schema）。

源自训练样例外貌规范：Thought 只做图像地理推理 → Action → Observation，
禁止视频旁白叙事体。供 stage5 改写与 stage6 形态裁判共用。
"""

from __future__ import annotations

from pipeline.schemas import AgentRole

# ---------------------------------------------------------------------------
# 判定要点（短）
# ---------------------------------------------------------------------------

TAO_STYLE_RULES: str = (
    "标准图片地理定位 TAO Thought 必须：\n"
    "1. 主语是画面/线索（植被、建筑、文字、阴影、交通等），不是博主/求助者/粉丝/父亲；\n"
    "2. 说明为何调用本步工具；不得把本步 Observation 当已知；\n"
    "3. 禁止视频元叙事（片头标题、历时半年、故地重游、求助故事等）；\n"
    "4. COARSE：识别地理/人文特征 → 演绎排除 → 逐步收窄到国家/地区；禁止跳步；"
    "结论不得为精确坐标/最终 POI；\n"
    "5. FINE 可在证据（画面/Obs/user_query 线索）支持下尽早收窄到精确地点，"
    "禁止无依据粘贴真值，也不要在 Thought 里解释线索来源；\n"
    "6. VERIFIER 可复述 fine_handoff 候选做交叉验证，不得把真值当作已知正确答案。"
)

BAD_THOUGHT_EXAMPLES: tuple[str, ...] = (
    "BAD: 求助者希望找到父亲老照片的拍摄地点。",
    "BAD: 粉丝向我求助，让我帮忙找一下这张照片在哪儿拍的。",
    "BAD: 我足足花了半年时间，当我知道答案的那一刻起……",
    "BAD: 视频标题是『历时半年 探寻一段百年往事』，画面展示了……",
    "BAD: 查询结果显示该地点就是郑州黄河文化公园。（本步 Obs 尚未返回）",
    "BAD: 画面有骑楼，所以这就是越南。（单一弱特征跳步到国家）",
    "BAD: 搜索结果显示该地区位于华南。（COARSE 用检索结果直接给地区）",
)

# 每角色 1 段压缩正例（Thought→Action 一句，不塞全文 JSON）
_FEWSHOT_BY_ROLE: dict[AgentRole, str] = {
    AgentRole.COARSE: (
        "GOOD 示例（COARSE 递进）：\n"
        "Thought: 画面为城市街道：行道树叶片宽大浓密，偏热带阔叶；"
        "两侧为3-4层连续骑楼与拱廊，外墙米黄灰泥，风格接近法式殖民地建筑；"
        "车流以摩托车为主、靠右行驶。先放大立面上部确认建筑细节，"
        "再据此排除温带/靠左行驶地区。\n"
        "Action: {\"tool\": \"zoom_inspect\", \"params\": {\"bbox\": [0.0, 0.0, 1.0, 0.25]}}\n"
        "（Agent1 训练轨迹仅用 zoom_inspect/ocr/sun_position_calc"
        "及适配的动态特征观察 Tool；禁止 web_search/map_query；"
        "每步须特征→排除/收窄；禁止精确街道坐标作结论。）"
    ),
    AgentRole.FINE: (
        "GOOD 示例（FINE）：\n"
        "Thought: Agent1 已收窄到某国南部大城市，并提示路牌 OCR 不完整。"
        "先对路牌区域做高精度 OCR，获取完整路名后再检索与 map_query。"
        "若 user_query 已给地区线索或画面特征已足够，可尽早提出精确地点假设并核实。\n"
        "Action: {\"tool\": \"ocr\", \"params\": {\"bbox\": [0.6, 0.2, 0.35, 0.25]}}\n"
        "（须有图像/Obs/user_query 依据；禁止无依据粘贴真值；最后一步 submit_answer。）"
    ),
    AgentRole.VERIFIER: (
        "GOOD 示例（VERIFIER）：\n"
        "Thought: Agent2 给出候选坐标与地点名。先 map_query 反查该 latlng，"
        "核对解析地址是否与候选一致，再 web_search(verification) 交叉核对"
        "气候/建筑/交通/文字是否与图像自洽。\n"
        "Action: {\"tool\": \"map_query\", \"params\": {\"latlng\": [10.774, 106.7017]}}\n"
        "（可复述 fine_handoff 候选；不得声称已知最终真值答案。）"
    ),
}


def fewshot_block_for_role(agent_role: AgentRole) -> str:
    """返回写入 stage5/stage6 prompt 的短 few-shot + 反例。"""
    good = _FEWSHOT_BY_ROLE[agent_role]
    bad = "\n".join(BAD_THOUGHT_EXAMPLES)
    return (
        f"{TAO_STYLE_RULES}\n\n"
        f"{good}\n\n"
        f"反例（禁止）：\n{bad}"
    )


def tao_judge_checklist() -> str:
    """stage6 TAO 形态裁判用的判定清单。"""
    bad = "\n".join(BAD_THOUGHT_EXAMPLES)
    return f"{TAO_STYLE_RULES}\n\n若出现类似下列反例 → is_standard_geo_tao=false：\n{bad}"
