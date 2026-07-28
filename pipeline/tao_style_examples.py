"""标准图片地理定位 TAO 风格 few-shot（压缩摘录，适配当前 schema）。

源自训练样例外貌规范：Thought 只做图像地理推理 → Action → Observation，
禁止视频旁白叙事体。供 stage5 polish / judge rubric 使用（不进入训练 system prompt）。
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
    "4. COARSE：每个事实均须引用本视频来源声明，禁止自由看图发明；"
    "每一步必须明确排除地点/候选类别并递进；禁止跳步与试错无效步；"
    "组合地貌须写关系而非标签堆砌；工作范围可直接用但不得单凭它得最终 POI；"
    "不同位置观察（近处/远处）可并存并联合收窄地点候选，"
    "禁止用他处特征否定本处已建立结论；"
    "结论不得为精确坐标/最终 POI；命名自然区域进 summary/key_clues；\n"
    "5. FINE 可在证据（画面/Obs/user_query 线索）支持下尽早收窄到精确地点，"
    "禁止无依据粘贴真值，也不要在 Thought 里解释线索来源；\n"
    "6. VERIFIER 可复述 fine_handoff 候选做交叉验证，不得把真值当作已知正确答案。"
)

BAD_THOUGHT_EXAMPLES: tuple[str, ...] = (
    "BAD: 求助者希望找到父亲老照片的拍摄地点。",
    "BAD: 粉丝向我求助，让我帮忙找一下这张照片在哪儿拍的。",
    "BAD: 我足足花了半年时间，当我知道答案的那一刻起……",
    "BAD: 视频标题是『历时半年 探寻一段百年往事』，画面展示了……",
    "BAD: 查询结果显示该地点就是某公园。（本步 Obs 尚未返回）",
    "BAD: 画面有骑楼，所以这就是越南。（单一弱特征跳步到国家）",
    "BAD: 搜索结果显示该地区位于华南。（COARSE 用检索结果直接给地区）",
    "BAD: 已知线索是某城市，我对卫星图验证就是该城市。（把线索当答案验证）",
    "BAD: 画面有三个地貌标签，所以是某命名区域。（标签堆砌+地名自证）",
    "BAD: 不是给定城市市区，应该在该城市附近。（「附近」伪装成收窄）",
    "BAD: 调用太阳位置计算。（实际 Action 却是 compare_images）",
    "BAD: 远景可见某种新地貌与人文设施。（视频未提出的自由发明）",
    "BAD: 先放大看看有什么线索。（无排除对象的试错步）",
    "BAD: 远处对岸其实是河岸，所以之前近处高地的判断错了。"
    "（用他处特征撤销本处结论）",
)

# 每角色 1 段压缩正例（Thought→Action 一句，不塞全文 JSON）
_FEWSHOT_BY_ROLE: dict[AgentRole, str] = {
    AgentRole.COARSE: (
        "GOOD 示例（COARSE 最短排除链）：\n"
        "Thought: 工作范围已给定。视频来源事实 F1/F2 形成关系 R；"
        "据此排除与 R 冲突的候选类别 A 和缺少必要条件的候选 B。"
        "先检查来源事实指定的目标区域以确认 R。\n"
        "Action: {\"tool\": \"zoom_inspect\", \"params\": {\"bbox\": [0.0, 0.0, 1.0, 0.45]}}\n"
        "（只用本视频来源事实；每步必须明确排除；禁止自由发明与跳步。）\n"
        "GOOD 示例（多位置联合收窄）：\n"
        "Thought: 近处来源事实确认俯视屋顶（拍摄点），远处来源事实确认河岸平原；"
        "二者并存。联合排除缺少「高处俯视+对岸宽河」组合的候选，收窄到滨河高地一侧。\n"
        "Action: {\"tool\": \"zoom_inspect\", \"params\": {\"bbox\": [0.2, 0.1, 0.6, 0.4]}}\n"
        "（多作用域 observe 可同一步引用；禁止写成否定近处结论。）"
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
    """TAO 风格判定清单（供调试/人工审查；stage6 不再做形态裁判）。"""
    bad = "\n".join(BAD_THOUGHT_EXAMPLES)
    return f"{TAO_STYLE_RULES}\n\n若出现类似下列反例 → is_standard_geo_tao=false：\n{bad}"
