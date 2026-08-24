# Stage 4 置信度融合报告

## 融合结构

```text
学弟 Stage 4 编排与 VLM 全上下文裁判
                    +
operation / input_schema 参数合同
                    +
context_resolvable / repair_actions
                    +
严格 Observation 直接证据
                    +
轨迹–图片一致性与程序化门槛
                    +
audit_coverage
                    ↓
quality_score + review_priority
accept / provisional_pass / parameter_repair / needs_review / reject
```

Stage 3 只负责工具归并、参数规范和 `stage3_parameter_audit.json`；最终质量分统一由 Stage 4 写入，避免两套评分互相覆盖。

## 三并发对照基准

三个案例固定使用同一组高分 VLM 裁判结果，并发数为 3；区别只来自确定性审计。

| 案例 | 学弟原 Stage 4 | 融合 Stage 4 |
|---|---|---|
| 完整好样本 | 0.9525 / low | 0.9827 / accept / low / coverage=1.0 |
| 参数可修 | 0.9525 / low | 0.9557 / parameter_repair / medium / coverage=1.0 |
| 虚假 Observation | 0.9525 / low | 0.3000 / reject / high / `observation_direct_evidence` |

融合版保留 VLM 对逻辑和图片的优势，同时避免 VLM 漏判虚假回执或忽略参数缺口。裁判调用失败时不再统一记 0.5 并把未知当低质量，而是保留确定性得分并降低 `audit_coverage`。

报告：`data/runs/fused_stage4_benchmark/benchmark.json`。

## 当前边界

本轮使用合成/离线基准和全仓 mock 测试，没有把真实 Stage 2/3 轨迹发送给外部 Stage 4 裁判。真实 VLM 批量评分需另行明确授权对应关键帧、字幕和轨迹出站。
