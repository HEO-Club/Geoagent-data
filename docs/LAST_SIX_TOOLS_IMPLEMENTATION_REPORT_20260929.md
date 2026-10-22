# 后六个 Tool 平替方案实现与验证报告

日期：2026-09-29
代码基线：`origin/develop@a577c1f`，本地 `develop` 另含文档提交 `ead2a8d`

## 1. 最新 develop 状态

已刷新远程 `develop`。远程没有新增提交，本地代码已经包含远程最新实现；本地仅比远程多一项此前未推送的 API 采购与 Harness 路线图文档提交。因此本轮没有发生分支合并，也没有覆盖未跟踪材料。

当前远程最新代码的主体是：

- 合并 31 类 Tool、57 个 operation 的真实执行器；
- 统一 `ALLOW_REAL_TOOL_API` 外网闸门；
- 引入 ResultStore、ImageStore、Harness checkpoint/resume 与尝试审计；
- 增加网页与远程媒体 SSRF 防护；
- 实现 OSM、距离方位、空间过滤等本地地理后处理链；
- 为街景、卫星、天气等会话型 Tool 增加恢复机制；
- 保持真实外部服务不可用时显式失败，不生成伪 Observation。

修改前全量基线为 `760 passed, 6 warnings`。

## 2. Word 调研材料的判断

附件列出 12 类 Tool 问题。本轮按第 7 至第 12 项实现与加固：

1. `visibility_analysis`
2. `ocr_read.decode`
3. `web_page_read`
4. `flight_data_query`
5. `administrative_registry`
6. `osm_query`

材料方向总体合理，但以下内容不能直接照搬：

- 一维条码不需要默认交给云 OCR；当前 OpenCV 已能本地识别 EAN-8、EAN-13、UPC-A 和 UPC-E；
- `web_search` 只能作为独立证据检索步骤，不能伪装成行政档案或航班数据库；
- OpenSky 是 ADS-B 接收数据，不提供完整商业航班计划，也不能查询未来航迹；
- GeoNames 是通用聚合地名库，不等同于国家官方行政沿革档案；
- 公共 Overpass 只适合小规模查询，不能作为大批量生产后端。

## 3. visibility_analysis 与 OpenTopography

### 实现

采用显式上游获取而不是在 `visibility_analysis` 中隐藏下载：

```text
terrain_analysis
  -> 本地 DEM 或 OpenTopography GlobalDEM
  -> 保存真实 GeoTIFF artifact
  -> 返回 terrain_ref
  -> visibility_analysis 复用 terrain_ref
  -> 本地点到点测地线采样和遮挡判断
```

新增能力：

- `terrain_analysis` 在 Harness artifact 目录原子保存 GeoTIFF；
- Observation 同时返回 `artifacts.terrain_path` 和 `result.terrain_ref`；
- 同一 RuntimeContext 自动把 DEM 交给 `visibility_analysis`；
- checkpoint 复用时通过 Tool restorer 从 `terrain_ref` 恢复上下文；
- `visibility_analysis` 支持内存网格、本地栅格、共享 DemGrid 和 ResultStore 引用；
- canonical catalog 明确要求先调用 `terrain_analysis`，不允许编造 DEM 路径或隐藏下载。

配置：

```dotenv
ALLOW_REAL_TOOL_API=true
OPENTOPOGRAPHY_API_KEY=
TERRAIN_DEM_TYPE=COP30
```

当前本机 `.env` 中 `OPENTOPOGRAPHY_API_KEY` 尚未配置，因此在线 OpenTopography 调用未执行；本地 DEM 与完整 artifact 链已经跑通。官方开发者页：<https://opentopography.org/developers>

### 边界

- 当前是点到点 sightline，不是区域 viewshed；
- 裸地 DEM 不含树木和建筑，若需这些遮挡必须输入合适 DSM；
- 数据分辨率决定结论尺度；
- OpenTopography 商业软件集成需核对 Enterprise API 条款。

## 4. ocr_read.decode

### 实现

- 保留 OpenCV QRCodeDetector；
- 新增 OpenCV BarcodeDetector；
- 支持 `barcode/1d/ean/ean8/ean13/upc/upca/upce` 等输入别名；
- 输出规范化 `code_type`、`payload`、bbox 和 quad；
- QR 与一维码可以在同一次调用中共同识别；
- 未知码制仍进入 `unsupported_code_types`，不伪造结果；
- 最低 OpenCV 版本调整为 `opencv-python>=4.8`。

OpenCV 官方当前声明 BarcodeDetector 支持 EAN-8、EAN-13、UPC-A 和 UPC-E：<https://docs.opencv.org/5.0/tutorials/objdetect/barcode_detect_and_decode/barcode_detect_and_decode.html>

### 边界

Code 128、Code 39、PDF417 等不在当前 OpenCV BarcodeDetector 官方支持范围内。本轮没有为了这些码制引入云 OCR，也没有把 OCR 猜测包装成成功解码。

## 5. web_page_read

静态读取原本已具备 URL、重定向和 DNS 级 SSRF 防护；问题出在可选 Playwright 路径。动态页面可执行 JavaScript，若不拦截子请求，脚本可能访问 localhost、私网或云元数据地址。

新增：

- BrowserContext 级 `route("**/*")`；
- 每个 HTTP/HTTPS 子请求重新执行公网 DNS 校验；
- 私网请求以 `blockedbyclient` 中止；
- 禁用 Service Worker，防止请求绕过 route；
- 禁止自动下载；
- 页面最终 URL 再次验证；
- `data/blob/about` 等不发网的页面内部协议保留。

Playwright 官方的路由与 Service Worker 建议：<https://playwright.dev/python/docs/network>

登录页、人机验证和付费墙仍只返回限制状态，不尝试绕过。

## 6. flight_data_query

OpenSky 的能力边界保留不变：

- 匿名状态向量仅当前快照；
- 认证状态向量最多回溯一小时；
- 机场航班查询使用四字 ICAO 码；
- 航迹最多回溯 30 天；
- `/tracks` 是实验性端点；
- 返回稀疏实际航迹，不做直线插值；
- 不提供未来计划航班。

本轮没有把 `web_search` 塞入 `flight_data_query` 内部。需要网页证据时，应由 Planner 单独调用 `web_search`，并将其标为网页证据，不能假装成 OpenSky 记录。

新增错误分类：401、403、429、5xx、网络和超时分别映射为认证、权限、限流、Provider、网络和超时错误，使 Harness 能只重试真正可重试的情况。

OpenSky 官方 REST 文档：<https://openskynetwork.github.io/opensky-api/rest.html>

## 7. administrative_registry

现有后端包括 GeoNames 和本地官方 CSV/JSON dump。本轮修复默认 `auto` 模式的关键问题：此前即使已经配置本地官方档案，只要没有 GeoNames username 或外网闸门关闭，就会在加载官方档案前失败。

修复后：

- `registry=auto` 可以在没有 GeoNames 账号时独立使用本地官方 dump；
- `registry=geonames` 仍严格要求 `GEONAMES_USERNAME`；
- GeoNames status 15 按真实空结果处理；
- 18、19、20 映射为配额/限流；
- 10 映射为认证失败；
- 13、22 映射为 Provider 失败；
- 历史行政归属只接受含 `valid_from/valid_to` 的官方记录。

GeoNames 免费服务当前为每应用每日 10,000 credits、每小时 1,000 credits：<https://www.geonames.org/export/>

## 8. osm_query

现有结构化 Overpass 查询、结果归一化、本地 count 和 ResultStore 链保持不变。本轮增加：

- 实际生产执行器的 endpoint 校验；
- 默认只允许已知公共 Overpass HTTPS 主机；
- 自建或商业主机必须显式设置 `ALLOW_CUSTOM_TOOL_ENDPOINTS=true`；
- 拒绝明文 HTTP endpoint；
- 429、网络、超时、5xx 进入正确错误分类；
- 不自动轮换公共实例规避某一实例的限流。

公共 Overpass 只适合小规模、低频、可缓存查询；持续应用应自建实例或使用离线区域 PBF：<https://dev.overpass-api.de/overpass-doc/en/preface/commons.html>

## 9. 验证结果

- 修改前全量：`760 passed, 6 warnings`；
- 六 Tool + terrain 专项：`126 passed`；
- 加目录契约专项：`142 passed`；
- terrain -> visibility Harness artifact 链：`1 passed`；
- 修改后全量回归：`768 passed, 6 warnings`；
- `compileall` 通过；
- 修改文件的 Ruff F/I 检查通过；
- JSON catalog 与 Tool spec 可解析；
- `git diff --check` 通过。

现有 6 个 warning 均为 Pillow 或 rasterio 上游弃用提示，不影响本轮逻辑。

## 10. 未做的事情

- 未调用真实 OpenTopography，因为本机尚未配置 Key；
- 未绕过登录、人机验证或付费墙；
- 未把网页搜索结果伪装成航班或官方档案；
- 未给公共 Overpass 增加自动并发或跨实例轮换；
- 未把点到点 sightline 冒充区域 viewshed；
- 未推送远程，也未修改用户已有未跟踪材料。
