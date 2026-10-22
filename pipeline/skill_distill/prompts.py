"""技能蒸馏提示。只描述轨迹里已经发生的判断。"""

from __future__ import annotations

import json

from pipeline.skill_distill.schemas import SkillBank


def atomize_prompt(*, trajectory_text: str, operation_index: str) -> str:
    """让模型把一条轨迹切成规划句和工具句。"""
    return f"""你在从一条地理定位轨迹里抽取可复用技能。只依据下面的轨迹，不要补充轨迹里没有的地点、工具回执或坐标。

规划句只写一条收窄判断，并给出适用范围级别：country、region、place、point。
工具句必须使用下方目录中的 tool 与 operation，写清输入从轨迹的哪类信息来、回执里什么能当证据。目录里没有的调用不要编成工具句。
final_answer 不是工具技能。只在这一条轨迹里成立、换一张图就不成立的具体地名，不要写成不带条件的全局规则；把它留在适用条件里，或直接不输出。

可用 operation：
{operation_index}

轨迹：
{trajectory_text}
"""


def analyst_prompt(
    *,
    trajectory_text: str,
    drafts_json: str,
    frozen_bank_json: str,
) -> str:
    """让分析员对冻结库提补丁，而不是改写库本身。"""
    return f"""你是技能库分析员。下面的技能库是冻结副本，你只能输出补丁，不能重写整库。

补丁只有三种：
- add：草稿里有跨轨迹仍可能成立的新句子，且库中没有同义句。
- duplicate：草稿与某条已有句子同义，target_id 填那条句子的 id，不要另写一句。
- narrow：已有句子的适用条件过宽，target_id 指向它，并给出更窄的 applicability。

只出现一次、且依赖该图具体物体或地名的判断，不要 add。轨迹里没有的证据不要写进补丁。

冻结技能库：
{frozen_bank_json}

这条轨迹的草稿句：
{drafts_json}

轨迹：
{trajectory_text}
"""


def frozen_bank_json(bank: SkillBank) -> str:
    """分析员可见的库快照。不包含审核队列。"""
    visible = [
        {
            "id": sentence.id,
            "kind": sentence.kind,
            "text": sentence.text,
            "applicability": sentence.applicability,
            "level": sentence.level,
            "tool": sentence.tool,
            "operation": sentence.operation,
            "status": sentence.status,
        }
        for sentence in bank.sentences
    ]
    return json_dumps(visible)


def json_dumps(payload: object) -> str:
    """稳定的 UTF-8 JSON 文本。"""
    return json.dumps(payload, ensure_ascii=False)
