# Geoagent-data

从地理定位讲解视频蒸馏**单一 agent** ReAct 轨迹 JSONL（SFT）的数据工程流水线。

规格见 [SPEC.md](SPEC.md)（v3）。历史规格归档：`SPEC_legacy_v2.md`。

## 数据生成流程

1. **阶段1**：视频 → 带时间戳字幕  
2. **阶段1.5**：识别/拆分独立定位题并选择待定位图
3. **阶段2**：视频 + 字幕 → 自由 TAO（无统一 tool schema）
4. **阶段3**：tool 树归并 → 标准 JSONL（`geolocate_agent.jsonl`）
5. **阶段4**：生成置信度、问题分类和人工审核说明

真实 Tool Runtime Harness 是 Stage 3 后的可选重放/复核层，不属于数据蒸馏 Stage，也不会改写 Stage 2/3 Observation。

## 快速开始

```bash
# 依赖
uv sync   # 或 pip install -r requirements.txt

# 配置：复制 .env.example → .env，填入密钥；测试保持 ALLOW_REAL_API=false

# 单阶段
python run_stage1.py --video data/raw_videos/DEMO.mp4
python run_stage2.py --video data/raw_videos/DEMO.mp4 --transcript data/transcripts/DEMO.json
python run_stage3.py --freeform data/intermediate/DEMO/stage2_freeform_tao.json

# 全链路
python run_one_video.py --video data/raw_videos/DEMO.mp4

# 可选：关闭/配置真实 API 后，重放 Stage 3 Canonical Tool 调用
python run_tool_harness.py --trajectory data/intermediate/DEMO/tasks/TASK/stage3_trajectory.json

# 批量（jobs.json 为数组，每项含 video_path 等）
python batch_run.py --jobs jobs.json
```

## 测试

```bash
uv run pytest tests -q
```

禁止测试中调用真实付费 API。

## 产物

- 字幕：`data/transcripts/{id}.json`
- 中间件：`data/intermediate/{id}/stage{1,2,3}_*.json`
- 分片：`data/output/shards/{id}.jsonl`
- 合并：`data/output/geolocate_agent.jsonl`
- Tool 目录：`canonical_tool_catalog_v2.json`
- Tool 真实执行报告：`data/tool_runs/{safe_trajectory_id}_{id_hash}/harness_report.json`

Harness 的引用、恢复、安全和失败语义见 [docs/TOOL_RUNTIME_HARNESS.md](docs/TOOL_RUNTIME_HARNESS.md)，最终设计与实测结论见 [docs/TOOL_RUNTIME_HARNESS_FINAL_AUDIT_20260910.md](docs/TOOL_RUNTIME_HARNESS_FINAL_AUDIT_20260910.md)。
