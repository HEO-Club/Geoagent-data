# Geo Agent Dataset 流水线

从地理定位讲解视频生成三套 LoRA SFT 训练 JSONL（Agent1 粗定位 / Agent2 精定位 / Agent3 验证）。

完整契约见 [SPEC.md](SPEC.md)。

## 环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# 按需填写 API Key；测试环境保持 ALLOW_REAL_API=false
```

默认 LLM 为**通义千问（DashScope）**：在 `.env` 填写 `DASHSCOPE_API_KEY`，保持 `LLM_PROVIDER=qwen`。  
申请入口：https://bailian.console.aliyun.com/ → API-KEY。默认模型 `qwen3.7-plus`。

## 目录约定

| 路径 | 说明 |
|------|------|
| `data/raw_videos/` | 原始视频 |
| `data/transcripts/` | 带时间戳文字稿 JSON |
| `data/intermediate/{video_id}/` | 各阶段中间产物 + `manifest.json` |
| `data/output/shards/` | 每视频 JSONL 分片（并发安全） |
| `data/output/agent1_coarse.jsonl` 等 | 最终三份训练文件（批处理结束后合并） |

## 准备 groundtruth（地图查坐标）

若尚无精确坐标，可先从字幕推断地名并用 Nominatim 解析：

```bash
python prep_groundtruth.py --transcript data/transcripts/BV13m61BJEQC.json
# 或手动指定地名
python prep_groundtruth.py --transcript data/transcripts/BV13m61BJEQC.json --query 郑州黄河文化公园
```

成功后会打印 `gt: LAT,LNG`，可直接用于下方 `run_one_video.py --gt`。

## 单视频

文字稿为 `TranscriptSegment` 列表 JSON，例如：

```json
[
  {"start": 0.0, "end": 5.0, "text": "..."},
  {"start": 12.0, "end": 13.0, "text": "答案是……"}
]
```

```bash
python run_one_video.py \
  --video data/raw_videos/demo.mp4 \
  --transcript data/transcripts/demo.json \
  --gt 48.8584,2.2945 \
  --platform bilibili
```

断点续跑：同一 `video_id` 再次运行会跳过 `manifest.json` 中已 `completed` 且 input_hash 未变的阶段。

强制自某阶段重跑：

```bash
python run_one_video.py --video ... --transcript ... --gt LAT,LNG --force-from stage4
```

## 批处理

清单 `jobs.json`：

```json
[
  {
    "video_path": "data/raw_videos/a.mp4",
    "transcript_path": "data/transcripts/a.json",
    "groundtruth": [48.8584, 2.2945],
    "source_platform": "youtube"
  }
]
```

```bash
python batch_run.py --jobs jobs.json
```

行为要点：

- 按 `MAX_CONCURRENT_VIDEOS` 并发（asyncio）
- tenacity 重试瞬时失败
- 已全部 completed 的视频默认跳过
- 单视频失败不影响其他视频
- 全部结束后由**单 writer**合并 `shards/` → 三份最终 JSONL（禁止多协程直写最终文件）

## Tool 管理

```bash
python manage_tools.py list
python manage_tools.py promote <tool_name> --executor-ref pipeline.tools.<name>.execute
```

## 测试

测试中禁止真实付费 API（`ALLOW_REAL_API=false`）。

```bash
# 当前阶段（stage7 / orchestrator / batch）
python -m pytest tests/test_stage7_format.py tests/test_orchestrator_e2e.py -q

# 全量
python -m pytest -q
```
