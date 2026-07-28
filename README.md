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
# 按需填写 LLM API Key；测试环境保持 ALLOW_REAL_API=false
```

双通道 LLM（均支持多模态图片）：

| 通道 | 用途 | 默认 |
|------|------|------|
| `VLM_*` | stage3 之前（转录 / stage1 屏幕操作） | DashScope `qwen3.7-plus`（填 `DASHSCOPE_API_KEY`） |
| `LLM_*` | stage3+（normalize / Observation / stage5–6） | Moonshot `kimi-k3`（填 `MOONSHOT_API_KEY`；国内默认 `https://api.moonshot.cn/v1`） |

Kimi 申请：https://platform.kimi.ai/ → API Key。`KIMI_REASONING_EFFORT` 建议 `low`。

**Observation 一律由 LLM 按 `tool_registry.json` 中的 schema 合成**（关键帧 + 该步旁白），不调用真实搜索/地图/OCR 等 Tool API。

## 目录约定

| 路径 | 说明 |
|------|------|
| `data/raw_videos/` | 原始视频 |
| `data/transcripts/` | 带时间戳文字稿 JSON |
| `data/intermediate/{video_id}/` | 各阶段中间产物 + `manifest.json` |
| `data/output/shards/` | 每视频 JSONL 分片（并发安全） |
| `data/output/agent1_coarse.jsonl` 等 | 最终三份训练文件（批处理结束后合并） |

## 准备 groundtruth（地图查坐标）

若尚无精确坐标，可先从字幕推断地名并用 Nominatim 解析（仅离线辅助，不进入 Observation 管线）：

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
python manage_tools.py stats
python manage_tools.py register --from-json path/to/tool.json
```

Registry 仅保存 schema；无 draft/production 升档。运行时 stage3 亦可按 G 规则注册新 Tool。

## 测试

测试中禁止真实付费 API（`ALLOW_REAL_API=false`）；Observation 合成路径使用 mock LLM。

```bash
# 全量
python -m pytest -q
```
