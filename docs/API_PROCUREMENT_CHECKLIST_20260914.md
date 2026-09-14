# GeoAgent Tool API 申请、购买与许可清单

日期：2026-09-14
范围：`canonical_tool_catalog_v2.json` 中 31 tool / 57 operation 的当前实现

## 1. 给汇报人的采购摘要

不是所有 Tool 都需要买 API。当前建议分三批：

| 优先级 | 服务 | 对应 Tool | 建议 |
|---|---|---|---|
| P0 | 高德 Web 服务 | `poi_search`、`geocode`、`route_query` | 国内地点/路线主链必需，先申请 Web 服务 Key；上线前确认企业认证、配额和数据使用范围 |
| P0 | Brave Search API | `web_search` | 网页证据检索主服务；新账户需选计划并生成 key |
| P0 | SerpApi Google Lens | `reverse_image_search` | 默认反向搜图后端；可先用免费额度，批量任务再升级 |
| P0 | Mapillary | `streetview_query` | 先注册开发者应用并拿 client access token；不是传统按次采购，但必须审查影像使用条款 |
| P0 | Copernicus Data Space | `satellite_imagery_query` | 免费注册并创建 OAuth client；注意处理配额与 Sentinel 数据许可 |
| P1 | OpenTopography | `terrain_analysis` | 本地无 DEM 时需要；学术免费 key，商业集成需 Enterprise key |
| P1 | GeoNames | `administrative_registry` | 原型可用免费 username；生产/SLA 购买 Premium Web Services 或使用授权本地 dump |
| P1 | Open-Meteo 商业计划 | `weather_archive_query` | 非商业原型可免 key；商业使用应购买 customer API 计划 |
| P1 | OpenSky 授权 | `flight_data_query` | OAuth client 可申请，但任何 operational use 还需书面许可，不能只把“免费账号”当授权 |
| 按需 | Google Cloud Vision Web Detection | `reverse_image_search` 可选后端 | SerpApi 之外的补充；需启用 Cloud Vision 和 billing |
| 按需 | OpenAI / Anthropic / 阿里云百炼 | `llm_query`，也可复用流水线模型 | 只买团队实际采用的一个或多个模型通道；当前工程优先复用已有 `LLM_*` 配置 |
| 按项目 | WMS/WFS/GeoServer 数据服务 | `map_layer_query` | 不是统一公共 API；需要单位自建 GeoServer 或采购有授权的图层服务和账号 |

## 2. 逐项申请与配置

### 2.1 高德开放平台 Web 服务

- Tool：`poi_search.poi_search/browse`、`geocode.geocode`、`route_query.route`
- 环境变量：`AMAP_WEB_KEY`（兼容 `AMAP_API_KEY`）
- 申请：注册/登录 → 应用管理 → 创建应用 → 添加 Key → 服务平台选择“Web 服务”。官方步骤：[创建应用和 Key](https://lbs.amap.com/api/webservice/create-project-and-key)。
- 配额：个人与企业认证配额不同；控制台“流量分析 → 配额管理”查看实际 QPS/日配额。官方说明：[流量限制](https://lbs.amap.com/api/webservice/guide/tools/flowlevel)、[认证与基础配额](https://lbs.amap.com/faq/account/certification/39670)。
- 购买：基础配额不够时进入服务升级页购买流量包或联系商务；页面给出的套餐与价格会变化，以控制台为准：[服务升级](https://lbs.amap.com/upgrade)。
- 风险：高德接口使用 GCJ-02；当前执行器已在边界转换 WGS84/GCJ-02。结果可否用于训练、批量缓存、再分发必须按账户协议单独确认。

### 2.2 Brave Search API

- Tool：`web_search.keyword_search/site_search`
- 环境变量：`BRAVE_SEARCH_API_KEY`（兼容 `BRAVE_API_KEY`）
- 申请：创建账号 → 激活 Search plan → API Keys → Add API Key。官方：[Quickstart](https://api-dashboard.search.brave.com/documentation/quickstart)。
- 当前公开价格：Search 为每 1,000 请求 5 美元，并有每月 5 美元 credit；价格随时可能调整，以 [Brave Search API](https://brave.com/search/api/) 为准。
- 重要许可：如果要存储完整或部分 API 结果用于训练/微调，必须选择明确授予存储权的计划。采购时要把“GeoAgent 数据集生产与留存”写进用途询问。

### 2.3 SerpApi Google Lens

- Tool：`reverse_image_search.search/search_crop` 默认后端
- 环境变量：`SERPAPI_API_KEY`（兼容 `SERPAPI_KEY`）
- 申请：在 SerpApi 注册；key 在 Dashboard 获取。官方：[Getting started](https://serpapi.com/blog/getting-started-with-serpapi-the-web-search-api/)。
- 当前公开价格：Free 250 searches/月；Starter 25 美元/月含 1,000 searches；更高套餐见 [SerpApi pricing](https://serpapi.com/pricing)。
- 采购建议：先用独立测试账号测量“每个视频平均反向搜图次数”，再按月峰值 × 1.3 预留量购买。确认 Google Lens 结果、缩略图和来源页信息的保存/训练权利。

### 2.4 Google Cloud Vision（可选）

- Tool：`reverse_image_search` 的可选 `google_vision` 后端
- 环境变量：`GOOGLE_VISION_API_KEY`
- 申请：创建 Google Cloud project → 绑定 billing → 启用 Cloud Vision API → Credentials 创建并限制 API key。官方：[Setup](https://docs.cloud.google.com/vision/docs/setup)、[Authentication](https://docs.cloud.google.com/vision/product-search/docs/auth)。
- 当前公开价格：每月前 1,000 units 免费；Web Detection 在 1,001–5,000,000 units 档为 3.50 美元/1,000 units。以 [Cloud Vision pricing](https://cloud.google.com/vision/pricing) 为准。
- 安全：限制 key 只能调用 Vision API，并尽量限制服务器出口 IP；不要将 key 放入 Tool inputs。

### 2.5 Mapillary

- Tool：`streetview_query.open/navigate/change_time/capture`
- 环境变量：`MAPILLARY_ACCESS_TOKEN`（兼容 `MAPILLARY_TOKEN`）
- 申请入口：[Mapillary Developer](https://www.mapillary.com/developer)。登录后创建应用并取得 client token；开发者页面可能要求 Meta/Mapillary 账号登录。
- 费用：当前项目不把它列为固定按次采购项；是否收费、调用限制和商业用途取决于账户/应用条款，正式使用前在开发者控制台核对。
- 许可重点：街景图片、元数据、派生截图和用于训练的权利不是同一个概念。只拿到 access token 不代表取得批量下载、长期留存或训练授权。

### 2.6 Copernicus Data Space Ecosystem

- Tool：`satellite_imagery_query.retrieve/change_time/oblique_view`
- 环境变量：`COPERNICUS_CLIENT_ID`、`COPERNICUS_CLIENT_SECRET`
- 申请：注册 CDSE → Dashboard User Settings → OAuth clients → Create → 复制 client id/secret。官方完整步骤：[Sentinel Hub authentication](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html)。
- 费用：CDSE 提供注册用户可用的开放免费数据与处理能力，但有配置容量/配额；参见 [CDSE APIs](https://documentation.dataspace.copernicus.eu/APIs.html)。超出公共配额或需要企业 SLA 时再询价，不应假设无限免费。
- 注意：secret 只显示一次，应存入 secret manager；token 可复用，不要每个请求都重新申请。

### 2.7 OpenTopography

- Tool：`terrain_analysis.terrain` 在没有本地 DEM 时
- 环境变量：`OPENTOPOGRAPHY_API_KEY`
- 申请：OpenTopography 注册 → My Account 申请免费 API key。官方：[OpenTopography for Developers](https://opentopography.org/developers)。
- 当前公开限制：学术免费 key 200 calls/24h，非学术免费 key 50 calls/24h；商业软件集成必须联系 `info@opentopography.org` 获取 Enterprise key。
- 订阅：公开页面列出 OT+ 30 美元/月或 330 美元/年；Enterprise 价格询价：[Subscriptions](https://opentopography.org/node/3447)。
- 采购建议：优先缓存/自建本地 DEM，OpenTopography 只补缺失区域，避免每个 Harness 重放重复下载。

### 2.8 GeoNames

- Tool：`administrative_registry.administrative/directory`
- 环境变量：`GEONAMES_USERNAME`
- 免费申请：注册普通 GeoNames 账号并在账号中启用 web services；免费服务公开限制为 10,000 credits/day、1,000/hour，需署名。官方：[GeoNames export/services](https://www.geonames.org/export/)。
- 生产购买：Premium Web Services 提供更高可用性/SLA；公开套餐从 40 欧元/年档起，按 credits 与 availability 分档，以 [Premium Web Services](https://www.geonames.org/commercial-webservices.html) 为准。
- 替代：若数据许可允许，可配置 `ADMINISTRATIVE_REGISTRY_OFFICIAL_PATH` 使用本地官方 dump，减少外网依赖。

### 2.9 Open-Meteo 与 NASA GIBS

- Tool：`weather_archive_query.weather/cloud_cover/snow_cover/refine_range`
- 当前实现：历史时序使用 Open-Meteo；图层使用 NASA GIBS WMS。
- 原型：Open-Meteo 免费端点无需 key，但只面向非商业使用且有限流；NASA GIBS 是公开标准 Web Service：[GIBS access basics](https://nasa-gibs.github.io/gibs-api-docs/access-basics/)。
- 商业购买：Open-Meteo 商业订阅会签发 API key，并改用 `customer-api.open-meteo.com`；Historical Weather 需要 Professional 或更高计划。官方：[Open-Meteo pricing](https://open-meteo.com/en/pricing)。当前代码尚未提供 customer endpoint/key 变量，购买后应增加适配再启用。
- 注意：天气模型数据通常要求 attribution；GIBS 各图层底层数据许可也要逐数据集核对。

### 2.10 OpenSky Network

- Tool：`flight_data_query.search/track/nearby_traffic`
- 环境变量：`OPENSKY_CLIENT_ID`、`OPENSKY_CLIENT_SECRET`
- 申请：注册并登录 OpenSky → Account → API Client → 创建 client id/secret；使用 OAuth2 client credentials 换取 token。官方：[OpenSky REST authentication](https://openskynetwork.github.io/opensky-api/rest.html)。
- 关键采购结论：OpenSky 条款明确，REST API 的任何 operational use（包括接入线上产品或自动化系统）需要书面许可。必须先联系 OpenSky 获得授权，不能仅凭免费 OAuth client 上线：[Terms of Use](https://opensky-network.org/about/terms-of-use)。
- 历史数据和研究高额度可能需要机构研究申请；个人免费账户不应被当作稳定生产 SLA。

### 2.11 OpenAI、Anthropic、阿里云百炼

- Tool：`llm_query.consult/enumerate`；也用于 Stage 1–3 模型流程
- 环境变量：按 Provider 使用 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY`/`LLM_ANTHROPIC_API_KEY`、`DASHSCOPE_API_KEY` 与对应 model/base URL。
- OpenAI：在 [API keys](https://platform.openai.com/api-keys) 创建 project key，在 [API pricing](https://developers.openai.com/api/docs/pricing) 按目标模型核价。建议单独 project、预算上限、服务账号 key，不与个人测试 key 混用。
- Anthropic：Claude Console Settings → API keys → Create key，可绑定 workspace 和失效期；官方：[Authentication](https://platform.claude.com/docs/en/manage-claude/authentication)、[Pricing](https://platform.claude.com/docs/en/about-claude/pricing)。
- 阿里云百炼：开通百炼本身不收费，模型调用通常按量计费；认证账户在免费额度用完后会转按量付费，可开启“免费额度用完即停”。官方：[百炼概览与计费](https://help.aliyun.com/zh/model-studio/what-is-model-studio/)、[模型价格](https://help.aliyun.com/zh/model-studio/model-pricing)。
- 采购建议：当前 `llm_query` 能复用已有 `LLM_*`，不要为它重复买一套账户。为 Harness 设置每 run token/金额预算后，再比较 OpenAI、Claude、Qwen 的真实任务成本。

### 2.12 WMS/WFS/GeoServer 图层服务

- Tool：`map_layer_query.load_layer`
- 环境变量：`MAP_LAYER_WMS_ENDPOINT`、`MAP_LAYER_WFS_ENDPOINT`、图层映射和可选 basic auth。
- 采购方式：此 Tool 没有一个统一供应商。可选：单位自建 GeoServer；向省市 GIS/自然资源部门申请授权服务；或购买商业 WMS/WFS。采购单必须同时包含 endpoint、允许图层名、账号、QPS、缓存/截图/训练权、SLA 与数据更新频率。
- 当前代码默认拒绝未配置 endpoint，因此必须先由数据负责人确定实际图层供应商。

## 3. 不需要购买 API 的 Tool

| Tool | 运行条件 |
|---|---|
| `image_edit`、`image_measure`、`image_compare` | Pillow/OpenCV，本地图片 |
| `ocr_read` | RapidOCR/ONNX Runtime，本地执行 |
| `media_metadata_read` | 本地 EXIF/文件信息 |
| `video_frame_extract` | 本地 FFmpeg；远程 URL 下载不需要 key，但受 SSRF/大小/外网闸门限制 |
| `osm_result_process`、`distance_bearing_calculator`、`spatial_filter` | Shapely/pyproj 本地算法 |
| `satellite_imagery_compare` | 比较已取得的影像；上游影像仍可能需要 CDSE |
| `visibility_analysis` | 本地 DEM/DSM；上游 DEM 不足时由 terrain 获取 |
| `solar_ephemeris`、`shadow_analysis` | pvlib NREL SPA 与本地几何 |
| `final_answer` | 纯终端合同 |
| `web_page_read` | 普通 HTTP/可选 Playwright，无专用 API key；仍需遵守目标站点 robots、条款和版权 |

以下服务也可免 key，但不等于可无限量或可任意用于训练：

- Wikimedia Commons Action API：遵守 2026 rate limits、User-Agent、Retry-After 和内容许可；高流量商业需求考虑 Wikimedia Enterprise：[Wikimedia API rate limits](https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits)。
- Overpass 公共实例：仅适合小规模/一次性用途；生产、商业或批量任务应自建或购买服务，并缓存/限流：[Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API)。
- 公共 Nominatim：绝对上限 1 request/s，不允许系统性/批量查询；当前代码默认拒绝公共实例，建议保持：[Nominatim Usage Policy](https://operations.osmfoundation.org/policies/nominatim/)。
- 英格兰 Planning Data：无需 key，数据通常按 OGL 发布，但 API 仍在 beta；国内建设/许可数据应使用明确授权 dump：[Planning Data docs](https://www.planning.data.gov.uk/docs)、[terms](https://www.planning.data.gov.uk/terms-and-conditions)。

## 4. 推荐采购顺序和验收材料

1. 先申请免费/测试凭证：高德、Brave、SerpApi、Mapillary、CDSE、OpenTopography、GeoNames、OpenSky。
2. 每个账号建立独立 dev/test project 或 application，限制 API、来源 IP、QPS 与月预算。
3. 先用 20 条黄金任务测量每类 Tool 的平均请求数、空结果率、429、时延和单任务费用。
4. 再决定正式付费档：搜索与反搜图按请求量；LLM 按 token；图层/航空按许可与 SLA；DEM 优先本地缓存。
5. 向供应商书面确认：是否允许缓存、截图、派生特征、数据集训练、内部共享、论文发布和商业化。
6. 凭证只写入本机 `.env` 或 secret manager；不得提交 Git、写入 trajectory、Observation、Harness report 或测试 fixture。

交付给学长的采购表至少应包含：供应商、Tool、测试账号负责人、正式账号主体、预计月请求、预算、QPS、许可用途、数据留存期、key 轮换日期、账单告警阈值和停服策略。
