# 融合 Stage 4 既有案例校准报告

## 数据范围

使用 `rerun_stage23_observation_gate_sonnet5_v2` 中 9 条既有 Stage 2/3 轨迹，结合：

- qwen3.5-omni-plus 现成字幕；
- 严格 Observation 直接证据审计；
- 当前 operation/input_schema 参数合同；
- 2026-08-22 最新 Stage 1.5 task 和选图记录；
- 旧 Stage 3 trajectory 实际使用的 image_paths。

采用三并发离线运行。外部 VLM 裁判因没有明确授权发送 Stage 2/3 派生轨迹而未调用，因此逻辑语义维度保持未覆盖状态。

## 校准过程

### 第一版：过度 hard reject

结果为 7 reject、2 parameter_repair。问题不是 Observation 或思维链突然变差，而是三项规则过严：

1. Stage 1.5 `needs_review` 被当作 hard fail；
2. 相同地点的不同中文写法按字符串不包含误判；
3. 旧轨迹图片与新 Stage 1.5 选图路径不一致被直接 hard fail。

### 第二版：正确路由但分数偏高

将 Stage 1.5 review 和图片版本不一致改为降分/人工路由，并用字符序列相似度处理地点写法后，得到 7 needs_review、2 parameter_repair、0 reject。但部分图片不一致样本仍有 0.85 左右，容易被误读为高质量。

### 最终版：增加软上限和 coverage 优先级

- trajectory image 与 Stage 1.5 selected image 不一致：质量分最高 0.75；
- Stage 1.5 task 仍需 review：最高 0.80；
- 未登记 Tool、未知 operation 或 invalid 参数：最高 0.70；
- coverage < 0.70 时优先 `needs_review`，不能只因参数可修而跳过语义/输入复核；
- 只有强地点冲突才 hard reject，相似措辞只降分并交给语义裁判。

## 最终离线结果

| 轨迹 | 分数 | coverage | 决策 | 主要原因 |
|---|---:|---:|---|---|
| 01_BV13m61BJEQC | 0.7500 | 0.8104 | needs_review | Stage1.5 review；新旧图片不一致；2个参数可修 |
| 02_BV1RC2XBdEnq | 0.8428 | 0.6990 | needs_review | 多题聚合缺 task 级对齐；2个参数可修 |
| 03_BV1JFp7zdEzH | 0.8400 | 0.6990 | needs_review | 多题聚合缺 task 级对齐；3个参数可修 |
| 04_BV1SbeqzoE5y | 0.7500 | 0.8104 | needs_review | Stage1.5 低质量嵌入图；新旧图片不一致 |
| 05_BV1ibbWzQEfN | 0.7000 | 0.8104 | needs_review | `field_site_visit` 无执行器；图片不一致 |
| 06_BV1CDtizwEhE | 0.7500 | 0.8104 | needs_review | 标题/讲解覆盖图；地点措辞待语义确认 |
| 08_BV1jjN2zdEQH | 0.7500 | 0.8104 | needs_review | 图片带字幕且与旧轨迹输入不同 |
| 09_BV15zyUY2EKs | 0.7500 | 0.8104 | needs_review | 聊天页内嵌图；8个参数可修 |
| 10_BV1ze2JY5EMV | 0.7500 | 0.8104 | needs_review | Stage1.5 所选树干帧与旧轨迹图片不一致 |

平均分 0.7648，平均 coverage 0.7856；9 条全部 needs_review、0 accept、0 reject。这与 2026-08-22 的人工结论一致：这些旧轨迹 Observation 已经过严格修复，但输入图片版本和 Stage1.5 选图仍不适合直接进入高质量 SFT。

## 合理性结论

最终版不再把“上游 review”“图片版本变化”“地点措辞差异”误判为不可恢复硬错误，同时仍能用严格 Observation 和参数合同抵抗 VLM 过度自信。质量分表示样本总体状况，decision 表示处理动作，coverage 表示结论充分程度，三者不能互相替代。

离线机器报告：`data/runs/fused_stage4_existing_best/summary.json`。

真实 VLM 语义裁判尚未运行；如需补齐，必须明确授权将对应关键帧、字幕和 Stage 2/3 轨迹发送至指定 HTTPS 端点。
