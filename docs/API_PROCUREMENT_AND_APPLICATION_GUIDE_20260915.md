# GeoAgent 全量 API 申请、采购、配置与验收指南

核对日期：2026-09-15  
代码基线：`develop@a577c1f881bc4f64f41904792b8bc39f6c72ae70`  
范围：当前 `canonical_tool_catalog_v2.json` 的 31 类 Tool、57 个 operation，以及让 Stage 1–4 / Harness 真实运行所需的模型服务

> 本文是当前采购与申请的权威清单。供应商价格、配额和条款会变化，提交采购单前必须再次打开对应官方页面和账户控制台核对。密钥不得写进本文、聊天、Git、Trajectory、Observation、Harness report 或测试 fixture。

## 1. 最先要理解的三个结论

### 1.1 Tool 已实现，不等于已经具备真实外部数据

当前 Tool 的执行器、参数校验、Observation 合同、Harness 调用、重试与恢复骨架已经实现并通过全量测试。纯本地 Tool 可以立即运行；联网 Tool 还需要同时满足：

1. `ALLOW_REAL_TOOL_API=true`；
2. 对应供应商凭证已经配置；
3. 账号具有正确产品、配额、地域和许可；
4. 输入满足执行要求，例如卫星和地形 Tool 需要坐标或 bbox，不能只给自然语言地名；
5. 真实服务的网络、限流和账单状态正常。

### 1.2 当前应采购的是“已接入代码的后端”

以下服务在旧报告中曾被作为候选，但当前实现没有 adapter，**不要列入本轮采购**：

- openrouteservice：当前 `route_query` 只接高德或自建 OSRM；
- YouTube Data API：当前 `media_search` 只接 Wikimedia Commons；
- Firecrawl：当前 `web_page_read` 使用本地 HTTP，动态页面可用本地 Playwright；
- Europeana：当前 `media_search` 没有 Europeana adapter；
- Google Routes / Google Street View：当前没有接入，只保留未来替代方案意义。

如果以后决定增加这些供应商，应先实现、测试 adapter，再采购正式额度。

### 1.3 建议分为“必须先申请”“按需申请”“不用购买”三类

| 类别 | 服务 | 当前用途 | 建议 |
|---|---|---|---|
| P0 | 高德 Web 服务 | POI、地理编码、路线 | 中国任务主链，优先申请 |
| P0 | Brave Search API | 网页关键词/站内搜索 | 网页证据主入口，优先申请 |
| P0 | SerpApi Google Lens | 反向搜图 | 图像地点识别常用，优先申请 |
| P0 | Mapillary | 街景打开、移动、换时相、截图 | 需要开发者 token，优先申请 |
| P0 | Copernicus Data Space / Sentinel Hub APIs | 卫星检索和预览 | 免费注册与 OAuth client，优先申请 |
| P0 | 阿里云百炼、Anthropic、OpenAI、Moonshot 中至少一个 | Stage 1–4 与 `llm_query` | 复用已有模型通道，不要重复采购 |
| P1 | OpenTopography | 缺少本地 DEM 时下载高程 | 地形任务需要，按任务量选学术/商业方案 |
| P1 | GeoNames | 全球行政地名/目录查询 | 免费账号可起步，生产考虑 Premium 或本地 dump |
| P1 | Open-Meteo Customer API | 商业天气历史查询 | 非商业研发可先用公开端点，商业运行需计划 |
| P1 | OpenSky OAuth / 数据许可 | 航班搜索、轨迹、附近交通 | 研究用途先申请；生产/运营用途先书面授权 |
| 项目制 | WMS/WFS/GeoServer | 水系、地形、行政区、道路、历史地图 | 由具体数据单位提供，不是统一商品 API |
| 可选 | Google Cloud Vision Web Detection | 反向搜图备用 | 只有 SerpApi 不满足时再启用 |

## 2. 31 类 Tool / 57 operation 的完整凭证映射

| Tool | operation | 当前后端 | 是否要凭证/采购 | 关键说明 |
|---|---|---|---|---|
| `image_edit` | `enhance/crop/zoom` | Pillow/OpenCV | 否 | 本地图片变换 |
| `image_measure` | `measure` | Pillow/OpenCV | 否 | 没有真实比例尺时只应输出像素或相对量 |
| `image_compare` | `compare` | OpenCV | 否 | 本地图像比较 |
| `ocr_read` | `recognize/decode` | RapidOCR/OpenCV QR与EAN/UPC | 否 | 本地 OCR、二维码和常见商品一维码；语言模型不是事实来源 |
| `reverse_image_search` | `search/search_crop` | SerpApi Google Lens；Google Vision 备用 | 是 | 图片会上传第三方，必须审查数据许可和隐私 |
| `media_metadata_read` | `exif/file` | Pillow/OpenCV/FFmpeg | 否 | 本地 EXIF、媒体属性 |
| `web_search` | `keyword_search/site_search` | Brave Search API | 是 | 只返回标题、URL、摘要；正文再由 `web_page_read` 获取 |
| `web_page_read` | `open_result` | urllib / Playwright | 否 | 无专用 Key；仍受 robots、网站条款、版权与 SSRF 保护限制 |
| `media_search` | `video_search/photo_search` | Wikimedia Commons | 通常否 | 当前没有 YouTube/Europeana；必须保留每个文件的许可与署名 |
| `video_frame_extract` | `frame_retrieve` | FFmpeg | 否 | 本地视频无需 API；远程 URL 受外网闸门与 SSRF 检查 |
| `poi_search` | `poi_search/browse` | 高德 Web 服务 | 是 | 代码使用 `AMAP_WEB_KEY`；GCJ-02 在适配器边界转换到 WGS84 |
| `geocode` | `geocode` | 高德；可选自建 Nominatim | 高德要 Key；自建 Nominatim不要供应商 Key | 公共 Nominatim 默认被代码拒绝，不应用于批量生产 |
| `route_query` | `route` | 高德；可选自建 OSRM | 高德要 Key；自建 OSRM需服务器 | 当前未接 ORS/Google Routes |
| `map_layer_query` | `load_layer` | OGC WMS/WFS/GeoServer | 视数据源 | 需要 endpoint、图层名映射，可能还需 Basic Auth |
| `osm_query` | `query/count` | Overpass API | 公共低频免 Key | 批量应自建 Overpass 或使用离线 PBF/PostGIS |
| `osm_result_process` | `filter/export` | Python/Shapely | 否 | 本地处理共享 ResultStore 中的 OSM 结果 |
| `streetview_query` | `open/navigate/change_time/capture` | Mapillary Graph API | 是 | 当前未接 Google Street View |
| `satellite_imagery_query` | `retrieve/change_time/oblique_view` | Copernicus Data Space Sentinel Hub APIs | 是 | OAuth client；`oblique_view` 不是真斜摄影，只是正射预览的参数化视图 |
| `satellite_imagery_compare` | `compare_candidates/compare_time` | Rasterio/OpenCV | 计算无需 | 上游影像仍需合法取得 |
| `distance_bearing_calculator` | `distance/bearing` | pyproj.Geod | 否 | 本地 WGS84 椭球测地计算 |
| `visibility_analysis` | `sightline` | terrain_analysis GeoTIFF artifact + 本地 DEM 采样 | 计算无需 | 本地无 DEM 时由 terrain_analysis 用 OpenTopography Key 获取并保存 terrain_ref，visibility 不重复下载 |
| `terrain_analysis` | `terrain` | Rasterio/Horn；OpenTopography 下载 | 本地 DEM 时否；下载时要 Key | 纯地名不能执行，需 bbox 或中心点加半径 |
| `spatial_filter` | `geometry_filter` | Shapely/pyproj | 否 | 本地几何过滤，近距离使用局部投影 |
| `weather_archive_query` | `cloud_cover/weather/snow_cover/refine_range` | Open-Meteo Archive + NASA GIBS | 研发通常免 Key；商业需 Open-Meteo计划 | GIBS 图层只作为图像证据，不能凭像素伪造百分比 |
| `solar_ephemeris` | `sunset_time/sun_position` | pvlib NREL SPA | 否 | 本地算法，需要真实坐标、时间和时区 |
| `shadow_analysis` | `shadow_model` | pvlib + 本地几何 | 否 | 不是建筑/地形三维光线追踪 |
| `administrative_registry` | `administrative/directory` | GeoNames；本地官方 dump | GeoNames 要 username | 正式任务优先使用授权官方数据 dump |
| `infrastructure_registry` | `construction/permit` | 英格兰 Planning Data；本地授权 dump | 当前英格兰源免 Key；其他地区项目制 | 没有全球统一建设许可 API |
| `flight_data_query` | `search/track/nearby_traffic` | OpenSky REST/OAuth2 | 建议申请 | 运营、商业或长历史用途必须另核许可和数据产品 |
| `llm_query` | `consult/enumerate` | Anthropic 或 OpenAI-compatible | 是 | 外部模型输出只能是建议/候选，不能伪装成数据库 Observation |
| `final_answer` | `submit` | Harness/Agent runtime | 否 | 本地终端合同 |

## 3. P0：必须优先申请的服务

### 3.1 高德开放平台 Web 服务

**对应 Tool**

- `poi_search.poi_search`
- `poi_search.browse`
- `geocode.geocode`
- `route_query.route`

**当前代码配置**

```dotenv
ALLOW_REAL_TOOL_API=true
AMAP_WEB_KEY=
# 兼容别名：AMAP_API_KEY
# AMAP_TIMEOUT_SEC=30
```

代码默认访问高德 POI、地理编码/逆地理编码、驾车/步行/公交和骑行接口。高德返回 GCJ-02，当前 adapter 会在边界处与 WGS84 转换；调用方不应再次手工偏移。

**申请步骤**

1. 使用项目组织主体注册高德开放平台，不建议使用无法交接的个人临时账号；
2. 完成个人或企业认证；正式项目优先企业认证；
3. 在“应用管理”创建独立应用，例如 `geoagent-dev`；
4. 添加 Key，服务平台必须选择“Web 服务”；
5. 在控制台确认已允许地点搜索、地理编码和路径规划相关接口；
6. 为 dev/test/prod 创建不同应用或不同 Key；
7. 记录账户实际 QPS、日配额、并发、计费套餐和超限行为；
8. 流量不足时在服务升级/配额管理中购买流量包或联系商务；
9. 向供应商书面确认缓存、派生数据、论文展示、内部数据集及模型训练用途。

**官方入口**

- 创建应用和 Key：<https://lbs.amap.com/api/webservice/create-project-and-key>
- Web 服务文档：<https://lbs.amap.com/api/webservice/summary>
- 流量限制：<https://lbs.amap.com/api/webservice/guide/tools/flowlevel>
- 服务升级：<https://lbs.amap.com/upgrade>

**验收标准**

- 一个中文地址能正向编码为候选点；
- 一个 WGS84 坐标能逆向编码，回执明确 CRS；
- 一个 POI 类别查询能返回名称、地址、坐标和 provider ID；
- 驾车/步行路线返回路网距离、时长和可用折线；
- 错误 Key、超限、429/5xx 不得被包装成空成功；
- Harness 报告中不得出现 Key。

### 3.2 Brave Search API

**对应 Tool**：`web_search.keyword_search/site_search`。  
**环境变量**：`BRAVE_SEARCH_API_KEY`，兼容 `BRAVE_API_KEY`。

```dotenv
ALLOW_REAL_TOOL_API=true
BRAVE_SEARCH_API_KEY=
# BRAVE_SEARCH_TIMEOUT_SEC=30
```

**申请步骤**

1. 注册 Brave Search API Dashboard；
2. 选择 Search API 计划并完成账单设置；
3. 在 API Keys 页面创建 dev key；
4. 生产环境再创建独立 key；
5. 设置月预算、请求告警和组织账单联系人；
6. 记录当前计划允许的 QPS、月请求量、并发和结果存储权；
7. 如果结果要用于数据集、训练、长期缓存或再分发，必须购买明确包含相应用途许可的计划并书面确认。

官方在 2026-09-15 可见的公开页面显示 Search API 按每 1,000 次请求计价，并提供月度 credit；具体价格、QPS和存储权以采购当天页面与合同为准。

**官方入口**

- 产品与价格：<https://brave.com/search/api/>
- Quickstart：<https://api-dashboard.search.brave.com/documentation/quickstart>
- API Dashboard：<https://api.search.brave.com/app/keys>

**调用量估算**

```text
月请求数 = 月任务数 × 每任务平均 web_search 次数 × 重试放大系数
建议重试放大系数先取 1.10～1.20
预算 = 月请求数 / 1000 × 采购当天单价 - 可用 credit
```

### 3.3 SerpApi Google Lens

**对应 Tool**：`reverse_image_search.search/search_crop` 的默认后端。  
**环境变量**：`SERPAPI_API_KEY`，兼容 `SERPAPI_KEY`。

当前 adapter 会把本地图片压成不超过约 500 KiB 的 JPEG，先上传到 SerpApi Image API 获得 `image_id`，再发起 Google Lens 查询。采购评估时要注意：一次 Tool 调用在供应商侧可能包含“上传 + Lens 查询”两个网络动作，但 SerpApi 如何计 search 以其账单定义为准。

```dotenv
ALLOW_REAL_TOOL_API=true
SERPAPI_API_KEY=
# SERPAPI_TIMEOUT_SEC=30
```

**申请步骤**

1. 注册 SerpApi 并验证邮箱；
2. 在 Dashboard 获取开发 Key；
3. 确认所选计划可使用 Google Lens API 与 Image API；
4. 用 20～50 张具有代表性的完整图和裁剪图测量命中率、平均耗时及账单 search 数；
5. 用 `月图片任务数 × 每任务完整图/裁剪次数 × 重试系数` 估算套餐；
6. 正式账号设置账单告警与额度耗尽策略；
7. 明确图片上传、保留周期、Google Lens 结果缓存和训练用途条款。

**官方入口**

- Google Lens API：<https://serpapi.com/google-lens-api>
- Image API：<https://serpapi.com/image-api>
- 价格：<https://serpapi.com/pricing>
- Key 管理：<https://serpapi.com/manage-api-key>

官网当前可见 Free 和多个付费档。历史报告记录过 Free 250 searches/月、Starter 1,000 searches/月；价格和额度属于易变信息，正式采购必须以账户结算页为准，不应把本报告数字写死进预算合同。

### 3.4 Mapillary

**对应 Tool**：`streetview_query.open/navigate/change_time/capture`。  
**环境变量**：`MAPILLARY_ACCESS_TOKEN`，兼容 `MAPILLARY_TOKEN`。

```dotenv
ALLOW_REAL_TOOL_API=true
STREETVIEW_PROVIDER=mapillary
MAPILLARY_ACCESS_TOKEN=
# MAPILLARY_SEARCH_RADIUS_M=200
# MAPILLARY_IMAGE_LIMIT=100
```

当前 adapter 使用 `graph.mapillary.com` 搜索图像、读取图像字段、沿 sequence 导航并获取缩略图。它不是 Google Street View；数据覆盖、拍摄时间和图像质量取决于 Mapillary 用户贡献。

**申请步骤**

1. 注册/登录 Mapillary；
2. 进入开发者/API 页面创建应用；
3. 获取 client access token；
4. 为 dev/prod 分开应用或 token；
5. 确认应用能够读取 image、sequence、geometry、captured_at 和 thumbnail 等当前代码所需字段；
6. 记录 Graph API 限流和 token 生命周期；
7. 阅读 Mapillary Terms、API/data license，确认是否允许下载缩略图、保存多久、论文展示、派生特征与训练用途；
8. token 仅放服务器端，不发给浏览器或公开日志。

**官方入口**

- 开发者页面：<https://www.mapillary.com/developer>
- API 文档：<https://www.mapillary.com/developer/api-documentation/>
- Terms：<https://www.mapillary.com/terms>

**验收标准**

- `open` 能按坐标范围找到真实图像并返回拍摄时间；
- `navigate` 只在真实 sequence 中移动；
- `change_time` 找不到其他时相时明确失败/空候选，不伪造旧图；
- `capture` 生成受 ResultStore 管理的本地图像；
- checkpoint 恢复后会话可继续导航；
- 报告里不出现 token 或带 token 的完整 URL。

### 3.5 Copernicus Data Space Ecosystem / Sentinel Hub APIs

**对应 Tool**：`satellite_imagery_query.retrieve/change_time/oblique_view`。  
**环境变量**：`COPERNICUS_CLIENT_ID`、`COPERNICUS_CLIENT_SECRET`。

```dotenv
ALLOW_REAL_TOOL_API=true
SATELLITE_PROVIDER=copernicus
COPERNICUS_CLIENT_ID=
COPERNICUS_CLIENT_SECRET=
# SATELLITE_SEARCH_RADIUS_M=2000
# SATELLITE_PREVIEW_WIDTH=512
# SATELLITE_PREVIEW_HEIGHT=512
# SATELLITE_SCENE_LIMIT=50
```

当前代码通过 OAuth2 client credentials 获取 access token，使用 Catalog API 找场景，再用 Process API 生成预览。主要选择 Sentinel-2 L2A 等集合。`oblique_view` 只返回已有正射影像及 heading/tilt 参数，不等于获取真斜摄影产品。

**申请步骤**

1. 注册 Copernicus Data Space 账号并验证邮箱；
2. 登录 Sentinel Hub Dashboard；
3. 在 User Settings / OAuth clients 创建 OAuth client；
4. 保存 client ID 和仅显示一次的 client secret；
5. 将 secret 放入本地 `.env` 或生产 Secret Manager；
6. 核对当前账户 Processing Units、并发、Catalog/Process API 配额；
7. 使用一个小 bbox、短日期窗和 512×512 预览做验证；
8. 记录 Sentinel/Copernicus 数据许可与署名要求；
9. 不要把“可以下载”理解为可以无条件再分发第三方底图或融合产品。

**官方入口**

- APIs 总览：<https://documentation.dataspace.copernicus.eu/APIs.html>
- OAuth client 创建：<https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html>
- 配额/限制：<https://documentation.dataspace.copernicus.eu/Quotas.html>
- Dashboard：<https://shapps.dataspace.copernicus.eu/dashboard/>

## 4. P1：按任务量与用途申请

### 4.1 OpenTopography

**对应 Tool**：`terrain_analysis.terrain` 在没有本地 DEM 时的下载后端；`visibility_analysis` 可消费其上游 DEM。  
**环境变量**：`OPENTOPOGRAPHY_API_KEY`，兼容 `OPENTOPO_API_KEY`。

```dotenv
ALLOW_REAL_TOOL_API=true
OPENTOPOGRAPHY_API_KEY=
# TERRAIN_DEM_TYPE=COP30
# TERRAIN_MAX_BBOX_DEG=2
```

**申请与购买**

1. 注册 OpenTopography；
2. 在 MyOpenTopo 页面申请 API key；
3. 学术用户按 Academic access 使用；非学术/商业集成应评估 OT+ 或 Enterprise；
4. 生产或商业应用不要通过拆分账号规避配额，应联系官方获取企业 key；
5. 记录每个 DEM 产品自身的来源、分辨率、许可和引用；
6. 下载后优先合法缓存，避免对相同 bbox 重复计费/耗额度。

官方当前开发者页会区分学术、非学术/OT+和 Enterprise API 访问，并公布每日调用限制；配额和价格以申请当天页面为准。

**官方入口**

- API 与 key：<https://opentopography.org/developers>
- OT+：<https://opentopography.org/otplus>
- Global DEM API：<https://portal.opentopography.org/apidocs/#/Public/getGlobalDem>

### 4.2 GeoNames

**对应 Tool**：`administrative_registry.administrative/directory`。  
**环境变量**：`GEONAMES_USERNAME`，它是已启用 Web Service 的 GeoNames 用户名，不是密码。

```dotenv
ALLOW_REAL_TOOL_API=true
GEONAMES_USERNAME=
# ADMINISTRATIVE_REGISTRY_OFFICIAL_PATH=
```

**申请步骤**

1. 注册 GeoNames 用户；
2. 在账户页启用 free web services；
3. 把用户名写入 `GEONAMES_USERNAME`；
4. 查看账户 credit 使用情况；
5. 原型与低频任务可用免费层；
6. 如果需要更高日额度、SLA、商业稳定性或 HTTPS 高负载，购买 Premium Web Services；
7. 如果任务涉及官方行政沿革、许可证或法律边界，优先使用 `ADMINISTRATIVE_REGISTRY_OFFICIAL_PATH` 指向授权官方 dump，GeoNames 只能做候选/辅助。

**官方入口**

- 注册：<https://www.geonames.org/login>
- Web Services：<https://www.geonames.org/export/web-services.html>
- Premium：<https://www.geonames.org/commercial-webservices.html>

### 4.3 Open-Meteo Customer API

**对应 Tool**：`weather_archive_query` 的历史天气时序部分。NASA GIBS 图层仍可走公共服务。  
**当前公开开发端点不需要 Key**；商业运行应使用 Customer API 计划提供的 API key/专用域名，并相应扩展当前 adapter 配置。

当前代码默认：

```dotenv
WEATHER_ARCHIVE_OPENMETEO_ENDPOINT=https://archive-api.open-meteo.com/v1/archive
WEATHER_ARCHIVE_GIBS_ENDPOINT=https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi
```

**重要采购判断**

- 开源、个人和非商业研发可在公平使用范围内使用公开 API；
- 商业应用、稳定 SLA、较高调用量应选择 Open-Meteo Customer API；
- 当前 `weather_archive_query` 依赖 Historical Weather API，因此采购时必须确认所选计划包含 Historical API；官网当前计划矩阵并非每个档都包含历史 API；
- 报告必须注明数据是再分析/天气模型格点，不是照片现场测站的直接观测。

**官方入口**

- Historical Weather API：<https://open-meteo.com/en/docs/historical-weather-api>
- Pricing / Customer API：<https://open-meteo.com/en/pricing>
- Terms：<https://open-meteo.com/en/terms>

**代码差距**

购买 Customer API 后，应增加显式 `OPENMETEO_API_KEY` 或供应商指定认证配置，并用专用 endpoint 验收；当前代码只配置 endpoint，没有独立 key 字段。这属于“凭证接入小改动”，不是重新实现 Tool。

### 4.4 OpenSky Network

**对应 Tool**：`flight_data_query.search/track/nearby_traffic`。  
**环境变量**：`OPENSKY_CLIENT_ID`、`OPENSKY_CLIENT_SECRET`；不配 client 时只能使用允许匿名访问的当前状态能力。

当前 adapter 使用：

- `/states/all`：附近交通/当前状态；
- `/tracks/all`：单机轨迹；
- `/flights/departure`、`/flights/arrival`：机场起降搜索；
- OAuth2 client credentials token endpoint。

**申请步骤**

1. 注册 OpenSky Network 账户；
2. 在 Account / API Client 创建 OAuth client；
3. 保存 client ID/secret；
4. 核对当前 API credit、时间分辨率和历史窗口；
5. 研究用途按官方要求引用 OpenSky；
6. 任何 operational、商业、执法、安全或大规模历史用途，先向 OpenSky 申请书面许可或商业/研究数据产品；
7. 不把 ADS-B 接收数据当作全球完整航班计划数据库。

**官方入口**

- REST API 文档：<https://openskynetwork.github.io/opensky-api/rest.html>
- API Client：<https://opensky-network.org/my-opensky/account>
- 数据许可说明：<https://opensky-network.org/data/data-tools>
- Terms：<https://opensky-network.org/about/terms-of-use>

### 4.5 OGC WMS/WFS / GeoServer 数据服务

**对应 Tool**：`map_layer_query.load_layer`。这不是一个统一供应商，无法购买一个“全球 WMS Key”解决。

当前代码需要：

```dotenv
MAP_LAYER_PROVIDER=wms
MAP_LAYER_WMS_ENDPOINT=
MAP_LAYER_WFS_ENDPOINT=
MAP_LAYER_WMS_LAYERS={"hydrology":"workspace:rivers"}
MAP_LAYER_WFS_LAYERS=
MAP_LAYER_WMS_USER=
MAP_LAYER_WMS_PASSWORD=
```

**三种获得方式**

1. 单位已有 GeoServer：由管理员发 endpoint、允许图层名和只读账号；
2. 政府/科研机构 OGC 服务：签署数据使用协议，配置其 endpoint；
3. 商业 GIS 数据商：采购 WMS/WFS/API 套餐和 SLA。

**采购合同必须写明**

- 图层名与语义：水系、地形、行政区、道路、历史地图；
- 数据 CRS、精度、覆盖范围、更新频率；
- WMS 截图和 WFS 几何是否都允许；
- QPS、并发、单次 bbox、最大要素数；
- 是否允许缓存、裁剪、派生特征、论文、内部数据集和训练；
- Basic Auth/OAuth/IP 白名单方式；
- SLA、停服通知、联系人和版本变更策略。

## 5. LLM / ASR / VLM 账户：整个流水线至少需要一个

### 5.1 当前代码支持关系

| Provider | 使用位置 | 环境变量 | 当前结论 |
|---|---|---|---|
| 阿里云百炼/Qwen | Stage 1 ASR、VLM、Stage 2/3、`llm_query` 的 OpenAI-compatible 通道 | `DASHSCOPE_API_KEY` | 当前 `.env.example` 的默认主通道，最优先复用 |
| Anthropic | Stage 2/3、`llm_query` | `ANTHROPIC_API_KEY` 或 `LLM_ANTHROPIC_API_KEY` | 原生 Messages adapter |
| OpenAI | Stage 2/3、`llm_query` 的兼容通道 | `OPENAI_API_KEY` + 正确 `LLM_BASE_URL`/model | API 账单独立于 ChatGPT/Codex订阅 |
| Moonshot/Kimi | Stage 2/3、`llm_query` 的兼容通道 | `MOONSHOT_API_KEY` | 可选备用 |
| Google Gemini | 现有 Pipeline 可选通道 | `GOOGLE_API_KEY` | 不是当前 `llm_query` 专用默认后端 |

不要为 `llm_query` 单独再购买一套模型账号；它应复用 Pipeline 已选定的 provider。建议组织层面先选一个主通道，再保留一个故障切换通道。

### 5.2 共同申请与安全步骤

1. 使用项目/组织账号创建独立 project/workspace；
2. 绑定公司/实验室支付方式和账单联系人；
3. 创建服务端 key，不使用个人聊天产品 token；
4. dev/test/prod 分 key，设置月预算、硬限额或告警；
5. 固定 model 名称并记录 provider revision；
6. 核对输入视频帧、文本、坐标和网页内容是否会用于供应商训练，以及可选的数据保留设置；
7. 每个 Harness run 记录 input/output token、缓存 token、模型、重试和估算费用；
8. key 泄漏时立即吊销和轮换，而不是只删除 Git 文件。

### 5.3 官方申请入口

- 阿里云百炼获取 API Key：<https://help.aliyun.com/zh/model-studio/get-api-key>
- 阿里云百炼模型价格：<https://help.aliyun.com/zh/model-studio/model-pricing>
- Anthropic API Keys / Authentication：<https://platform.claude.com/docs/en/manage-claude/authentication>
- Anthropic Pricing：<https://platform.claude.com/docs/en/about-claude/pricing>
- OpenAI API Keys：<https://platform.openai.com/api-keys>
- OpenAI API Pricing：<https://developers.openai.com/api/docs/pricing>
- Gemini API Key：<https://ai.google.dev/gemini-api/docs/api-key>
- Gemini Pricing：<https://ai.google.dev/gemini-api/docs/pricing>
- Moonshot 开放平台：<https://platform.moonshot.cn/>

OpenAI Docs 当前将 API 定价按模型、输入/缓存输入/输出 token 和部分工具费用分别列出；ChatGPT/Codex 订阅不应被视为项目 API 额度。

### 5.4 预算公式

```text
单次任务模型费用
= Σ[(input_tokens / 1,000,000 × 输入单价)
   + (cached_input_tokens / 1,000,000 × 缓存输入单价)
   + (output_tokens / 1,000,000 × 输出单价)]

月模型预算
= 月任务数 × 单任务实测平均费用 × 1.20 容量余量
```

不要在没有实测 token 分布前按“每次对话价格”拍脑袋采购。先用 20～50 条黄金任务跑主模型，再按 P50/P95 token 和失败重试率核算。

## 6. 可选后端：Google Cloud Vision Web Detection

**对应 Tool**：`reverse_image_search` 的备用 `google_vision` 后端。  
**当前代码环境变量**：`GOOGLE_VISION_API_KEY`。

**申请步骤**

1. 创建独立 Google Cloud project；
2. 绑定 Billing；
3. 启用 Cloud Vision API；
4. 创建 API key；
5. 对 key 设置 API restriction，仅允许 Cloud Vision；
6. 对服务端使用场景配置适当来源/IP限制；
7. 配置 quota 和预算告警；
8. 使用 Web Detection 功能做真实验收。

Google Cloud 官方更推荐生产使用受控身份/服务账号；但当前代码只接 API key。如果决定长期使用 Vision，应增加 Application Default Credentials/service-account adapter，避免保留宽权限 API key。

**官方入口**

- Setup：<https://cloud.google.com/vision/docs/setup>
- Authentication：<https://cloud.google.com/vision/docs/authentication>
- Pricing：<https://cloud.google.com/vision/pricing>
- API key 安全：<https://cloud.google.com/docs/authentication/api-keys-best-practices>

## 7. 公共免 Key 服务：可以研发，不等于生产无限使用

### 7.1 Wikimedia Commons

- 当前 `media_search` 唯一后端；
- 通常无需 Key，但必须设置可识别 User-Agent；
- 每个文件有独立作者、许可、署名和使用限制；
- “API 能访问”不等于“图片可训练”；
- 高流量或企业需求应评估 Wikimedia Enterprise。

官方：<https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits>、<https://commons.wikimedia.org/wiki/Commons:API>

### 7.2 Overpass API

- `osm_query.query/count` 可使用公共实例；
- 公共实例只适合低频、小范围研发；
- 生产批量建议自建 Overpass、使用 Geofabrik PBF + PostGIS，或采购托管服务；
- 必须缓存、限制 bbox、限制返回量、控制并发并遵守 Retry-After。

官方：<https://wiki.openstreetmap.org/wiki/Overpass_API>、<https://dev.overpass-api.de/overpass-doc/en/preface/commons.html>

### 7.3 Nominatim

- 当前 `geocode` 仅在显式配置自建端点时建议使用；
- 公共 `nominatim.openstreetmap.org` 在代码中默认拒绝；
- 官方公共实例绝对上限为 1 request/s，并禁止重型/系统性批量地理编码；
- 生产应选高德、商业地理编码、自建 Nominatim 或离线数据库。

官方：<https://operations.osmfoundation.org/policies/nominatim/>

### 7.4 自建 OSRM

- `route_query` 可配置自建 `ROUTE_OSRM_ENDPOINT`；
- 公共 OSRM 默认被代码拒绝；
- 不购买 API key，但要承担服务器、地图更新、监控、备份和运维；
- 采购预算应列为基础设施成本，而不是 API 调用费。

官方：<https://project-osrm.org/docs/v5.24.0/api/>

### 7.5 NASA GIBS

- `weather_archive_query` 用于云、雪等 WMS 图层；
- `map_layer_query` 也可在配置后加载 GIBS 图层；
- 通常无需 Key；
- 图层时间、分辨率和可用日期由产品决定；
- 图层图像不能被当作逐像素定量测量结果，除非另有受验证的科学处理链。

官方：<https://nasa-gibs.github.io/gibs-api-docs/access-basics/>

### 7.6 英格兰 Planning Data

- 当前 `infrastructure_registry` 的结构化在线来源；
- 通常无需 Key，API仍需关注 beta/变更状态；
- 数据适用范围是英格兰，不能拿它覆盖中国或全球许可档案；
- 中国案例应配置单位授权 dump 或实现对应地区 connector。

官方：<https://www.planning.data.gov.uk/docs>、<https://www.planning.data.gov.uk/terms-and-conditions>

## 8. 完全不需要购买 API 的本地能力

以下 Tool 本体不需要供应商账号：

- 图像：`image_edit`、`image_measure`、`image_compare`；
- OCR/媒体：`ocr_read`、`media_metadata_read`、本地 `video_frame_extract`；
- 地理后处理：`osm_result_process`、`distance_bearing_calculator`、`spatial_filter`；
- 影像比较：`satellite_imagery_compare`，但影像来源仍要合法；
- 计算：`visibility_analysis`、`solar_ephemeris`、`shadow_analysis`；
- 终端：`final_answer`；
- 网页正文：`web_page_read` 本身无需专用 API，但必须遵守目标站点条款。

需要注意“上游数据成本”：例如 `visibility_analysis` 本地计算免费，但没有 DEM 时仍可能调用 OpenTopography；`satellite_imagery_compare` 免费，但没有卫星影像时仍需 Copernicus。

## 9. 推荐的实际申请顺序

### 第一批：立即申请，用于打通端到端最小链路

1. 高德 Web 服务 Key；
2. Brave Search API Key；
3. SerpApi Key；
4. Mapillary access token；
5. Copernicus OAuth client；
6. 复用现有阿里云百炼/Anthropic/OpenAI 中一个模型 Key。

这批可以打通典型链：

```text
任务/图片
  → OCR/图片增强（本地）
  → 反向搜图或网页搜索
  → 地理编码/POI
  → OSM/路线/街景/卫星
  → 本地距离、方位、空间过滤、影像比较
  → 证据汇总
  → final_answer
```

### 第二批：根据黄金任务缺口申请

1. OpenTopography；
2. GeoNames；
3. Open-Meteo 商业计划；
4. OpenSky；
5. 具体 WMS/WFS 数据服务。

### 第三批：只有主后端不满足时再申请

- Google Cloud Vision；
- 第二个 LLM provider；
- 未来新 adapter 对应的 Firecrawl、YouTube、Europeana、ORS、Google Maps 等。

## 10. `.env` 最小模板

以下只列变量名，不得把真实值提交 Git：

```dotenv
APP_ENV=development
ALLOW_REAL_API=true
ALLOW_REAL_TOOL_API=true

# 主模型通道：选一套
LLM_PROVIDER=qwen
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=<approved-model>
DASHSCOPE_API_KEY=<secret>

# P0 Tool providers
AMAP_WEB_KEY=<secret>
BRAVE_SEARCH_API_KEY=<secret>
SERPAPI_API_KEY=<secret>
MAPILLARY_ACCESS_TOKEN=<secret>
COPERNICUS_CLIENT_ID=<secret>
COPERNICUS_CLIENT_SECRET=<secret>

# P1 optional providers
OPENTOPOGRAPHY_API_KEY=<secret>
GEONAMES_USERNAME=<account-name>
OPENSKY_CLIENT_ID=<secret>
OPENSKY_CLIENT_SECRET=<secret>
```

如果使用 OpenAI-compatible 通道，必须让 `LLM_BASE_URL` 与实际 Key 的供应商匹配。当前 `llm_query` 会按 `DASHSCOPE_API_KEY → OPENAI_API_KEY → MOONSHOT_API_KEY` 顺序选择第一个非空 key；不要同时留下多套非空 key 却把 base URL 指向另一家，否则会造成难诊断的认证错误。更好的下一步是把 key 选择改为严格按 provider 映射。

## 11. 每个供应商必须完成的验收

### 11.1 凭证层

- 正常凭证能调用；
- 错误凭证得到明确 `authentication/permission` 类错误；
- 缺少凭证得到 `engine_unavailable`；
- 日志、异常和 Harness report 不含 Key、secret 或 bearer token；
- dev/prod key 能独立吊销。

### 11.2 数据层

- 回执包含 provider、source ID、时间、CRS/单位和必要许可字段；
- 空结果与网络失败严格区分；
- 坐标转换不会重复执行；
- 历史数据不会被当前数据替代；
- 图片/网页/地图结果不会被 LLM 自行补全为“真实 Observation”。

### 11.3 稳定性层

- 测量 P50/P95 延迟；
- 人为触发 429/超时，检查指数退避与 `Retry-After`；
- 非幂等/可能计费调用默认只尝试一次，确认后再开放重试；
- 额度耗尽时熔断，不继续无限请求；
- provider revision 改变后不错误复用旧 checkpoint。

### 11.4 许可层

采购负责人要为每个供应商留下书面结论：

- 是否允许长期缓存；
- 是否允许内部共享；
- 是否允许形成数据集；
- 是否允许用于训练/微调/评测；
- 是否允许论文和产品截图；
- 是否要求署名、回链或删除机制；
- 数据保留期限和用户删除要求；
- 商业化、跨境和敏感地理数据限制。

## 12. 给学长的采购表模板

| 字段 | 示例/要求 |
|---|---|
| Provider | Brave Search API |
| 对应 Tool | `web_search` |
| 申请主体 | 实验室/公司组织账号 |
| 负责人 | 姓名与工作邮箱 |
| 开发/生产项目 | 分离 |
| 当前计划 | 计划名、月额度、QPS |
| 预计月请求 | 基于黄金任务实测 |
| 月预算上限 | 金额与币种 |
| 超限行为 | 停止/降级/继续计费 |
| 环境变量 | 只写变量名，不写值 |
| Key 存储位置 | Secret Manager 条目名 |
| 创建日期 | YYYY-MM-DD |
| 轮换日期 | YYYY-MM-DD |
| 缓存/训练许可 | 供应商书面结论 |
| SLA/支持 | 套餐或联系人 |
| 验收人 | 工程负责人 |
| 验收结果 | 成功率、P95、429、费用 |

## 13. 当前最现实的采购结论

如果目标是尽快让 GeoAgent 的典型地理推理任务真实跑起来，第一张采购/申请单只需要包含：

1. 高德 Web 服务；
2. Brave Search API；
3. SerpApi；
4. Mapillary；
5. Copernicus Data Space OAuth client；
6. 已有主 LLM/ASR/VLM 供应商额度。

OpenTopography、GeoNames、Open-Meteo、OpenSky 和 WMS/WFS 应在黄金任务证明有真实缺口后补充。Google Vision 是反向搜图备用，不应与 SerpApi 同时高额采购。Firecrawl、YouTube、Europeana、openrouteservice 和 Google Street View 当前没有代码 adapter，不应进入本轮正式采购。
