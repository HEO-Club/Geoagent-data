# GeoAgent Tool API/凭证申请与交接指南（2026-09-11）

本报告以当前 31 类 Canonical Tool、57 个 operation 和已实现执行器为准。它区分：

- **纯本地，无需申请**：不访问外部服务；
- **公共服务免 Key**：仅适合低频研发，必须遵守限流、署名和缓存政策；
- **需要申请**：必须取得 API Key、Access Token、OAuth Client 或数据库凭证；
- **上游数据需要，计算无需**：算法本地执行，但卫星图、DEM、街景等输入需合法获取。

密钥只能由运行环境注入本机 `.env` 或密钥管理器，不进入 Tool inputs、Trajectory、HarnessReport、测试 fixture、聊天或 Git。

## 一、31 类 Tool 凭证需求总表

| Tool | 当前/建议后端 | 凭证 | 当前结论 |
|---|---|---:|---|
| `image_edit` | Pillow/OpenCV | 无需 | 已本地实现。 |
| `image_measure` | Pillow/OpenCV | 无需 | 已本地实现；无尺度时只输出像素/比例。 |
| `image_compare` | OpenCV | 无需 | 已本地实现。 |
| `ocr_read` | RapidOCR/OpenCV QR | 无需 | 已本地实现。 |
| `reverse_image_search` | SerpAPI Google Lens；Google Vision 备用 | **需要** | 已有 adapter；优先申请 SerpAPI。图片会发送给第三方。 |
| `media_metadata_read` | Pillow/OpenCV/FFmpeg | 无需 | 已本地实现。 |
| `web_search` | Brave Search API | **需要** | 建议第一批申请。 |
| `web_page_read` | 本地 HTTP/Playwright；Firecrawl 备用 | 本地无需；托管需 Key | 静态页面优先本地，动态/反爬再用 Firecrawl。 |
| `media_search` | YouTube Data API、Wikimedia Commons、Europeana | 混合 | YouTube 和 Europeana 需 Key；Wikimedia 通常免 Key。 |
| `video_frame_extract` | FFmpeg | 无需 | 下一阶段纯本地实现。 |
| `poi_search` | Nominatim/Overpass；高德补充 | 公共 OSM 免 Key；中国生产建议 Key | OSM 版本已实现；中国优先高德 Web 服务。 |
| `geocode` | Nominatim；高德/Google 备用 | 公共低频免 Key；生产建议 Key | Nominatim 已实现；不能用公共实例跑重型批量。 |
| `route_query` | openrouteservice、高德、Google Routes、OSRM 自建 | **需要或自建** | 全球优先 ORS，中国优先高德。 |
| `map_layer_query` | 公共 OGC WMS/WFS、NASA GIBS、高德/Google tiles | 视数据源 | 公共 OGC/GIBS 可免 Key；商业地图需 Key。 |
| `osm_query` | Overpass API/自建 Overpass | 公共低频免 Key | 已实现；批量建议自建或离线 extract。 |
| `osm_result_process` | Shapely + Python stdlib | **无需** | 已本地实现 `filter/export`；不支持伪造有损 `osm_xml`。 |
| `streetview_query` | Mapillary；Google Street View 备用 | **需要** | 优先 Mapillary；Google 缓存/训练许可需单独审批。 |
| `satellite_imagery_query` | Copernicus Data Space Sentinel Hub | **需要** | 申请 CDSE OAuth Client。 |
| `satellite_imagery_compare` | Rasterio/OpenCV | 计算无需 | 影像获取仍依赖 satellite provider。 |
| `distance_bearing_calculator` | pyproj.Geod | **无需** | 已本地实现 `distance/bearing`。route/磁北/图像轴缺数据时返回 needs_acquisition。 |
| `visibility_analysis` | GDAL viewshed + DEM | 计算无需；DEM 可能需 Key | 复用 CDSE/OpenTopography。 |
| `terrain_analysis` | GDAL/Rasterio + DEM | 计算无需；DEM 通常需 Key | CDSE 或 OpenTopography。 |
| `spatial_filter` | Shapely + pyproj；PostGIS 备用 | **本地无需** | 已本地实现 WGS84 几何关系与 AEQD 米制 near。 |
| `weather_archive_query` | Open-Meteo、NASA GIBS | 通常免 Key | 非商业/低频可先研发；生产需核对套餐和公平使用。 |
| `solar_ephemeris` | pvlib/astronomical algorithms | 无需 | 纯本地；需要真实坐标、日期、时区。 |
| `shadow_analysis` | pvlib + 本地几何 | 无需 | 纯本地；上游高度和表面几何必须真实。 |
| `administrative_registry` | GeoNames + 国家/地区官方档案 | GeoNames 需账号 username | 全球基础名称可先申请 GeoNames。 |
| `infrastructure_registry` | 各地区建设/许可档案 | 视地区 | 无全球统一 API，按案例建立 connector。 |
| `flight_data_query` | OpenSky OAuth2；商业历史库备用 | **建议申请** | REST 使用 client_id/client_secret；长历史权限另申请。 |
| `llm_query` | 百炼/Anthropic/OpenAI | **需要** | 项目已有模型适配；外部 LLM 结果只能作为建议/候选。 |
| `final_answer` | 本地合同 | 无需 | 不调用外部服务。 |

## 二、立即申请清单与步骤

### 1. SerpAPI

对应：`reverse_image_search.search/search_crop`。

官方入口：

- [Google Lens API](https://serpapi.com/google-lens-api)
- [Image API](https://serpapi.com/image-api)
- [API Key 管理](https://serpapi.com/manage-api-key)

申请步骤：

1. 注册 SerpAPI 账号并验证邮箱；
2. 在 Dashboard 选择开发/生产计划；
3. 打开 API Key 页面复制 Key；
4. 开发和生产使用不同 Key；
5. 在 `.env` 中配置，不粘贴到聊天。

```dotenv
ALLOW_REAL_TOOL_API=true
SERPAPI_API_KEY=
```

当前 Image API 会上传 JPG/PNG/WebP，并返回短期 `image_id`。申请 Key 不代表可以上传所有训练图片；每批数据仍需确认隐私、版权和第三方处理许可。

### 2. 高德开放平台 Web 服务 Key

对应：`geocode`、`poi_search`、`route_query`，以及中国区域部分 `map_layer_query`。

- [官方创建应用和 Key 教程](https://lbs.amap.com/api/webservice/create-project-and-key)
- [高德控制台](https://console.amap.com/dev/key/app)

步骤：

1. 登录/注册高德开放平台；
2. 按控制台要求完成开发者认证；
3. 应用管理 → 创建新应用；
4. 应用内点击“添加 Key”；
5. 服务平台必须选择“Web 服务”；
6. 设置配额告警和允许来源（若控制台支持）；
7. 记录账号归属、每日配额和到期/审核状态。

```dotenv
AMAP_API_KEY=
```

高德 Web 服务常返回 GCJ-02。Adapter 必须同时保存原始坐标、`provider_crs=GCJ-02`、WGS84 转换方法和转换后坐标，禁止直接与 OSM WGS84 混算。

### 3. Copernicus Data Space / Sentinel Hub OAuth Client

对应：`satellite_imagery_query`，并为 `terrain_analysis`、`visibility_analysis` 提供影像/DEM。

- [Copernicus Data Space](https://dataspace.copernicus.eu/)
- [Sentinel Hub OAuth 官方认证](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html)

步骤：

1. 注册并验证 CDSE 账号；
2. 登录 Sentinel Hub Dashboard；
3. User Settings → OAuth clients → Create；
4. 保存 Client ID 和只显示一次的 Client Secret；
5. 使用 client-credentials 换取短期 token；
6. 在 token 有效期内复用，不要每次 API 请求都重新取 token；
7. 记录账号套餐、processing units、并发和下载许可。

```dotenv
CDSE_CLIENT_ID=
CDSE_CLIENT_SECRET=
```

### 4. Mapillary Client Token

对应：`streetview_query`。

- [Developer Dashboard](https://www.mapillary.com/dashboard/developers)
- [Mapillary API 访问说明](https://help.mapillary.com/hc/en-us/articles/360010234680-Accessing-imagery-and-data-through-the-Mapillary-API)

步骤：

1. 登录 Mapillary/Meta 账号；
2. Developer Dashboard 创建应用；
3. 打开应用并查看 Client Token；
4. 服务端优先通过 Authorization header 使用；
5. 记录图片 ID、sequence、拍摄时间、作者/许可和 attribution。

```dotenv
MAPILLARY_ACCESS_TOKEN=
```

### 5. Brave Search API

对应：`web_search`。

- [官方 Quickstart](https://api-dashboard.search.brave.com/documentation/quickstart)
- [Dashboard](https://api-dashboard.search.brave.com/)

步骤：

1. 注册并验证邮箱；
2. 在 Plans 中激活 Search 计划；
3. Dashboard → API Keys → Add API Key；
4. 分别创建开发/生产 Key；
5. 请求时放在 `X-Subscription-Token` header；
6. 设置预算、用量监控和缓存。

```dotenv
BRAVE_SEARCH_API_KEY=
```

## 三、第二批申请

### openrouteservice

对应：`route_query` 全球路线；`distance_bearing_calculator` 不负责调用路线 API。

- [ORS API 与 Dashboard](https://api.openrouteservice.org/)
- [登录/注册](https://openrouteservice.org/log-in/)

注册并验证邮箱，登录 Dashboard，创建 Token/API Key，记录 profile、每日配额和矩阵/等时圈限制。

```dotenv
ORS_API_KEY=
```

### YouTube Data API v3

对应：`media_search.video_search`。

- [官方入门](https://developers.google.com/youtube/v3/getting-started)
- [凭证类型说明](https://developers.google.com/youtube/registering_an_application)

步骤：Google Cloud 创建项目 → API Library 启用 YouTube Data API v3 → Credentials 创建 API Key → Restrict key 仅允许 YouTube Data API → 设置 quota 监控。只读公开搜索可用 API Key；访问私有用户数据才需要 OAuth2。

```dotenv
YOUTUBE_API_KEY=
```

### Firecrawl

对应：`web_page_read` 的动态页面/批量正文提取备用后端。

- [Firecrawl v2 API](https://docs.firecrawl.dev/api-reference/v2-introduction)
- [Dashboard](https://www.firecrawl.dev/app/api-keys)

注册后在 Dashboard 获取 Key，请求使用 `Authorization: Bearer ...`。应只在本地读取失败且许可允许时启用；429 必须退避。

```dotenv
FIRECRAWL_API_KEY=
```

### Europeana

对应：`media_search.photo_search` 的欧洲文化遗产/历史图片来源。

- [官方 API key 申请说明](https://pro.europeana.eu/page/get-api)

自 2025-05 起，需注册 Europeana 账号、确认注册，再从账户区域申请和管理个人 Key；更高请求率可申请 project key。

```dotenv
EUROPEANA_API_KEY=
```

### OpenTopography

对应：`terrain_analysis`、`visibility_analysis` 的 DEM 获取。

- [官方开发者页](https://opentopography.org/developers)

注册 → My Account → Request API Key。免费 Key 存在每日调用和区域面积限制；商业软件集成需联系官方取得 Enterprise Key。

```dotenv
OPENTOPOGRAPHY_API_KEY=
```

### GeoNames

对应：`administrative_registry`。

- [官方 Web Service 文档](https://www.geonames.org/export/web-services.html)
- [账号注册](https://www.geonames.org/login)

注册并验证邮箱后，在账户页启用 Web Service。请求携带自己的 username；禁止使用文档中的 `demo` 账号跑测试或生产。

```dotenv
GEONAMES_USERNAME=
```

### OpenSky Network

对应：`flight_data_query`。

- [官方 REST/OAuth2 文档](https://openskynetwork.github.io/opensky-api/rest.html)
- [OpenSky Account](https://opensky-network.org/account)

登录账户 → Account 页创建 API Client → 保存 client_id/client_secret → 使用 OAuth2 client-credentials 换取 Bearer token。Basic Auth 用户名/密码已不再支持。长历史或机构研究访问需另行申请。

```dotenv
OPENSKY_CLIENT_ID=
OPENSKY_CLIENT_SECRET=
```

## 四、Google Cloud 只作为受限备用方案

可覆盖：Vision、Geocoding、Places、Routes、Elevation、Map Tiles、Street View Static。

- [Google Maps 凭证和限制对照](https://developers.google.com/maps/faq)
- [API Key 安全最佳实践](https://developers.google.com/maps/api-security-best-practices)
- [Cloud Vision 认证](https://docs.cloud.google.com/vision/docs/authentication)
- [Street View 使用政策](https://developers.google.com/maps/documentation/streetview/policies)

统一步骤：

1. Google Cloud 创建独立项目；
2. 绑定 Billing；
3. 只启用本项目需要的 API；
4. Credentials 创建 Key 或服务账号；
5. 服务端 Web Service Key 使用 IP restriction；
6. 再增加 API restriction，只允许指定 API；
7. 设置每日 quota 和预算告警；
8. 不复用 Gemini/Maps/Vision 的同一无限制 Key。

```dotenv
GOOGLE_VISION_API_KEY=
GOOGLE_MAPS_API_KEY=
```

特别提醒：Google Street View 通常禁止预取、索引、长期存储或缓存图像内容（panorama ID 有有限例外）。没有单独许可审查前，不应把 Google Street View 图片作为默认训练数据源。

## 五、公共免 Key 服务

### Nominatim

- [官方 Usage Policy](https://operations.osmfoundation.org/policies/nominatim/)

公共实例要求可识别 User-Agent、署名、缓存；绝对上限 1 request/s，重型/周期批量限制更严。GeoAgent 的 2000 条生产运行不应直接依赖公共实例，应使用高德/商业服务、自建 Nominatim 或离线数据。

### Overpass API

- [官方公共实例和配额说明](https://dev.overpass-api.de/overpass-doc/en/preface/commons.html)

公共实例适合小项目和手动/低频查询；高需求应自建或使用区域 PBF/Geofabrik extract。本项目必须按 query+scope 缓存并限制并发。

### Wikimedia Commons

- [Commons API](https://commons.wikimedia.org/wiki/Commons:API)

通常无需 Key，但每个文件许可不同。结果必须保存文件页 URL、作者、license、attribution 和“do not use/index”等版权状态，不能把“API 可访问”等同于“可用于训练”。

### NASA GIBS

- [GIBS Access Basics](https://nasa-gibs.github.io/gibs-api-docs/access-basics/)

WMS/WMTS/TWMS/XYZ 为公共标准服务，可用于 `map_layer_query` 和部分天气/云雪可视化。图层分辨率和时间维度是预先确定的；批量下载应先联系 Earthdata 支持。

### Open-Meteo

- [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)

非商业研发通常无需 Key，但公平使用、日请求量和商业条件需要在生产前确认。再分析格点不是照片点位实测站，输出必须保留模型/网格和时间分辨率。

## 六、LLM Provider

对应：`llm_query` 以及现有 Stage 1–4 模型调用。三者任选已获授权的一个，不必全部申请。

| Provider | 官方入口 | 本地变量 |
|---|---|---|
| 阿里云百炼 | [获取 API Key](https://help.aliyun.com/zh/model-studio/get-api-key/) | `DASHSCOPE_API_KEY` |
| Anthropic | [Claude API Quickstart](https://platform.claude.com/docs/en/get-started) | `ANTHROPIC_API_KEY` / `LLM_ANTHROPIC_API_KEY` |
| OpenAI | [API Keys](https://platform.openai.com/api-keys) | `OPENAI_API_KEY` |

外部 LLM 的返回只能标为模型建议、候选或分类结果，不得伪装成 OSM、地图、档案或搜索数据库的真实 Observation。

## 七、建议申请顺序

```text
P0  SerpAPI
P0  高德 Web 服务
P0  Copernicus Data Space OAuth Client
P0  Mapillary Client Token
P0  Brave Search API

P1  openrouteservice
P1  YouTube Data API
P1  Firecrawl
P1  Europeana

P2  OpenTopography
P2  GeoNames
P2  OpenSky

备用 Google Cloud Maps/Vision/Street View（先做许可与预算审查）
```

P0 的原因不是它们都应立即接入，而是它们覆盖当前真实轨迹中出现频率较高、且纯本地无法补出的反向搜图、中国地图、卫星图、街景和网页搜索证据。

## 八、交接模板

申请人只需按下面格式交接，不发送 Key：

```text
Provider：Brave Search API
账号归属：项目组织账号
已启用服务：Web Search
计划/配额：开发计划，每月/每秒限制已记录
环境变量：BRAVE_SEARCH_API_KEY
本机 .env：已配置
开发 Key 与生产 Key：已分离
API restriction/IP restriction：已配置或不适用
预算/告警：已设置
许可备注：只保存 URL、标题和短摘要；正文另由合法 reader 获取
```

## 九、安全和运行规则

1. 永不把明文 Key 发到聊天、Issue、日志、Trajectory、HarnessReport 或 Git。
2. 开发、生产和个人试验使用不同 Key/项目；能限制 API/IP/来源时必须限制。
3. OAuth token 在有效期内缓存，refresh/client secret 只进密钥管理器。
4. Harness CLI 每次 Provider/数据配置变化时更新 `--provider-revision`。
5. 所有外部 Tool 记录 provider、请求时间、真实参数、来源 ID、CRS/单位、署名、许可和失败码。
6. 429 使用 Retry-After/指数退避；预算耗尽进入熔断，不把错误转成空成功。
7. “申请成功”只代表能访问 API，不代表允许批量缓存、再分发或用于模型训练。
