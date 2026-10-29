# location_verify：最终答案验证工具

## 定位

`location_verify.verify`（MCP 名 `location_verify__verify`）是对 `final_answer` 地点的独立复核工具，不属于数据蒸馏 Stage（Stage 1–4），也不会改写 Stage 2/3 已落盘的 Observation。它在答案地点打开街景、按多个朝向截图，用 `image_compare` 与待定位原图做程序化图像比对；再用 `ocr_read` 提取原图与街景截图的文字做一致性比对；两路证据融合输出 `verdict` 与 `confidence`。

验证只给出「正向确认」或「未确认」。弱图像匹配或文字缺失不等于答案错误；只有强图像匹配或相同有意义文字才判 `matched`。

## 输入与输出

输入：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `location` | string / `[lon,lat]` / `{lon,lat}` | 是 | 待验证的最终地点 |
| `reference_image` | string | 否 | 待定位原图，缺省 `$current_image` |
| `headings` | number[] | 否 | 街景朝向角，缺省 0/45/…/315 八个朝向 |
| `provider` | string | 否 | 街景后端 `mapillary`/`baidu`/`google` |
| `compare_method` | string | 否 | `image_compare` 方法，缺省 `auto` |

输出为 `Observation.result`，字段结构：

```text
operation / verdict / confidence / coverage / location / signals / applied / assumptions
  ├── location.input / location.resolved / location.match_level
  ├── signals.image.best_heading / inlier_ratio / match_count / score / conclusion / compared_headings
  └── signals.ocr.query_text / streetview_text / common / partial / conclusion
```

## 执行流程

1. 解析 `location`：地名交给 `geocode.geocode` 记录坐标与 `match_level`；坐标直接使用。
2. `streetview_query.open` 打开答案地点街景；`coverage=false` 时直接返回 `inconclusive`。
3. 对每个 `headings` 执行 `streetview_query.capture` 截图，再对每张截图与 `reference_image` 执行 `image_compare.compare`。
4. 取 `inlier_ratio`（其次 `match_count`）最高的朝向作为最相似方位，生成 `signals.image`。
5. `ocr_read.recognize` 提取原图文字 A 与全部街景截图文字 B，取交集并过滤单字/纯数字噪声，生成 `signals.ocr`。
6. 融合两路信号输出 `verdict` 与 `confidence`，随后删除本次保留的街景截图。

## 判定规则

| 条件 | verdict | confidence |
|---|---|---|
| 图像强匹配（`inlier_ratio ≥ 0.30` 且 `match_count ≥ 20`）或 OCR 存在相同有意义文字 | `matched` | 0.70–0.85 |
| 有街景覆盖，但两路均无强证据 | `no_positive_match` | 0.20–0.30 |
| 无街景覆盖或街景无法打开 | `inconclusive` | 0.10 |

`no_positive_match` 以及 `signals.ocr.conclusion` 的 `no_text`/`none` 只表示「未确认」，不是「答案错误」。`matched` 要求正向证据，不能由「未找到反例」推断得到。

## 当前限制

1. 街景与待定位图视角不同时特征匹配偏弱；弱匹配只作「未确认」，不能据此判错。
2. `ocr_read` 只提取原始文字，可能误识别；误识别文本只算 `partial`，不算 `match`。
3. 未接入视觉模型裁判；「街景与原图是否同一地点」的语义判断仍需人工或后续多模态模型。
4. `streetview_query.capture` 用后删除本地文件（许可约束），本工具以 `_RetainImageStore` 临时保留截图供 `image_compare`/`ocr_read` 引用，用完再删。
5. 无街景覆盖 ≠ 答案错误，只能判 `inconclusive` 并转卫星图或人工复核。

## 调用示例

```python
from tool import execute

observation = execute(
    "location_verify",
    "verify",
    purpose="验证最终定位答案",
    inputs={"location": "云南省丽江市永胜县涛源镇"},
    ctx=ctx,
)
```
