# GeoAgent Tool API/凭证申请与交接指南

更新时间：2026-09-08

本文件区分三种状态：`无需申请` 表示纯本地计算；`公共服务免 Key` 表示可以低频试用但不适合直接批量跑数千条；`需要申请` 表示必须注册账号并取得 API Key、Access Token 或 OAuth Client。申请到的密钥只放本地 `.env` 或密钥管理器，不发送到聊天、不写入 Tool inputs、不提交 Git。

## 一、31 个 Tool 的凭证需求总表

| Tool | 当前/建议后端 | 是否需要申请 | 建议与交接结论 |
|---|---|---:|---|
| `image_edit` | Pillow | 无需申请 | 已本地实现并验证。 |
| `image_measure` | Pillow/OpenCV | 无需申请 | 已本地实现并验证；无真实尺度时只输出像素或比例。 |
| `image_compare` | OpenCV | 无需申请 | 已本地实现并验证。 |
| `ocr_read` | RapidOCR/OpenCV QR | 无需申请 | 已本地实现并验证。 |
| `reverse_image_search` | SerpAPI Google Lens；Google Vision 备用 | **需要申请** | 当前最优先申请 SerpAPI；两者只需先选一个。 |
| `media_metadata_read` | Pillow/OpenCV/FFmpeg | 无需申请 | 已本地实现并验证。 |
| `web_search` | Brave Search API | **需要申请** | 建议申请 Brave Search Key；当前 Tool 尚未接入。 |
| `web_page_read` | 本地 HTTP/Playwright；Firecrawl 备用 | 可本地；批量建议申请 | 静态网页可本地读取，动态和反爬页面可申请 Firecrawl。 |
| `media_search` | YouTube Data API、Wikimedia、Europeana | 视频搜索通常需要申请 | 优先申请 YouTube Data API Key；Wikimedia 通常免 Key。 |
| `video_frame_extract` | FFmpeg | 无需申请 | 纯本地。 |
| `poi_search` | 当前 Nominatim/Overpass；中国区域可加高德 | 公共 OSM 免 Key；批量/中国建议申请 | 已实现 OSM 版本并用 Mock 验证；生产批量建议高德 Web 服务 Key 或自建 OSM。 |
| `geocode` | 当前 Nominatim；中国区域可加高德 | 公共 Nominatim 免 Key；批量建议申请 | 已实现 Nominatim 版本并用 Mock 验证；公共实例最多低频研发。 |
| `route_query` | openrouteservice、高德、OSRM 自建 | **需要申请或自建** | 全球优先 openrouteservice，中国优先高德；当前尚未实现。 |
| `map_layer_query` | OGC WMS/WFS、GeoServer、各地图平台 | 视数据源而定 | 公共 OGC 可能免 Key；商业地图通常需要 Key。 |
| `osm_query` | Overpass API | 公共服务免 Key | 已实现并用 Mock 验证；批量应自建/托管，不要并发压公共实例。 |
| `osm_result_process` | GeoPandas/Shapely | 无需申请 | 纯本地，待实现。 |
| `streetview_query` | Mapillary；Google Street View 备用 | **需要申请** | 建议先申请 Mapillary Client Token。Google 数据许可需单独审查。 |
| `satellite_imagery_query` | Copernicus Data Space/Sentinel Hub | **需要申请** | 建议申请 CDSE 账号并创建 OAuth Client。 |
| `satellite_imagery_compare` | Rasterio/OpenCV | 无需申请 | 影像取得后本地比较；影像获取仍需要上一个 Tool。 |
| `distance_bearing_calculator` | pyproj.Geod | 无需申请 | 坐标由前置 Tool 补齐后纯本地计算。 |
| `visibility_analysis` | GDAL viewshed + DEM | 计算无需；DEM 获取可能需要 | 建议共用 Copernicus/OpenTopography 凭证。 |
| `terrain_analysis` | GDAL/Rasterio + DEM | 计算无需；DEM 获取通常需要 | Copernicus 或 OpenTopography 二选一。 |
| `spatial_filter` | GeoPandas/Shapely/PostGIS | 本地无需 | 若用远程 PostGIS，仅需项目内部数据库凭证，不是公网 API Key。 |
| `weather_archive_query` | Open-Meteo/NASA GIBS | 通常免 Key | 可先免 Key 研发；商业或高频使用再购买相应套餐。 |
| `solar_ephemeris` | pvlib | 无需申请 | 纯本地。 |
| `shadow_analysis` | pvlib + 本地几何 | 无需申请 | 纯本地。 |
| `administrative_registry` | GeoNames + 各国官方档案 | GeoNames 需注册 username | 全球基础地名可先申请 GeoNames；中国官方资料往往没有统一 API。 |
| `infrastructure_registry` | 各地区建设/许可档案 | 视地区而定 | 没有全球统一凭证，后续按案例地区建立 connector。 |
| `flight_data_query` | OpenSky OAuth2；商业历史库备用 | **建议申请** | 研究用途先申请 OpenSky Client ID/Secret；历史数据另申请权限。 |
| `llm_query` | Anthropic/OpenAI 或已授权兼容服务 | **需要申请** | 项目已有模型适配；密钥不能进入样本。 |
| `final_answer` | 本地合同校验 | 无需申请 | 纯本地。 |

## 二、建议立即申请的账号

### 1. SerpAPI：当前最高优先级

- 对应 Tool：`reverse_image_search.search/search_crop`
- 官方文档：https://serpapi.com/google-lens-api
- 注册/管理 Key：https://serpapi.com/manage-api-key
- 申请步骤：注册并验证邮箱；进入 Dashboard；选择可用计划；进入 API Key 页面复制 Key；只保存到本地 `.env`。
- 当前代码变量：

```dotenv
ALLOW_REAL_TOOL_API=true
SERPAPI_API_KEY=replace_me
ALLOW_CUSTOM_TOOL_ENDPOINTS=false
```

当前适配器会先把本地图片上传到 SerpAPI Image API，取得 `image_id` 后再请求 Google Lens。调用前必须确认图片允许发送到第三方。

### 2. 高德 Web 服务 Key：中国 POI/地理编码/路线的推荐补充

- 对应 Tool：`poi_search`、`geocode`、`route_query`，后续也可服务部分 `map_layer_query`
- 官方教程：https://lbs.amap.com/api/webservice/create-project-and-key
- 控制台：https://console.amap.com/dev/key/app
- 申请步骤：登录/注册高德开放平台；完成开发者认证；进入“应用管理”创建新应用；在应用中点击“添加 Key”；服务平台选择“Web 服务”；保存生成的 Key。
- 建议预留变量名：`AMAP_API_KEY`（当前代码尚未接入高德，申请后再实现 provider adapter）。
- 注意：高德通常返回 GCJ-02 坐标，项目内部 GIS 计算统一使用 WGS84，因此适配器必须显式标记 CRS，不能把两套坐标直接混用。

### 3. Copernicus Data Space OAuth Client：卫星图和 DEM 的共同入口

- 对应 Tool：`satellite_imagery_query`，并为 `terrain_analysis`、`visibility_analysis` 提供 DEM
- 注册入口：https://dataspace.copernicus.eu/
- 认证教程：https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html
- API 入门：https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/UserGuides/BeginnersGuide.html
- 申请步骤：注册并验证 CDSE 账号；登录后进入 Sentinel Hub Dashboard；打开 User Settings；进入 OAuth clients；创建 Client；立即保存显示的 Client ID 和 Client Secret；运行时用它们换取短期 Access Token，不要每次请求都重新取 Token。
- 建议预留变量：`CDSE_CLIENT_ID`、`CDSE_CLIENT_SECRET`。

### 4. Mapillary Client Token：开放街景优先方案

- 对应 Tool：`streetview_query`
- 开发者入口：https://www.mapillary.com/dashboard/developers
- 开发帮助：https://help.mapillary.com/hc/en-us/categories/17361771500956-Mapillary-for-Developers
- 申请步骤：使用 Meta/Mapillary 账号登录；进入 Developer Dashboard；创建应用；生成 Client Token；服务端请求使用 Authorization Header，不把 token 放进轨迹 JSON。
- 建议预留变量：`MAPILLARY_ACCESS_TOKEN`。

### 5. Brave Search API Key：网页搜索

- 对应 Tool：`web_search`
- Quickstart：https://api-dashboard.search.brave.com/documentation/quickstart
- Dashboard：https://api-dashboard.search.brave.com/
- 申请步骤：注册并验证邮箱；在 Plans 中启用计划；进入 API Keys；点击 Add API Key；保存 Key。请求通过 `X-Subscription-Token` Header 发送。
- 建议预留变量：`BRAVE_SEARCH_API_KEY`。

## 三、按后续开发阶段申请

### Firecrawl

- 对应 Tool：`web_page_read` 的动态网页/批量正文提取备用后端
- 文档与申请入口：https://docs.firecrawl.dev/introduction
- 步骤：创建免费账号；进入控制台生成 API Key；免费匿名调用只适合试用，更高限额使用 Key。
- 建议变量：`FIRECRAWL_API_KEY`。

### YouTube Data API

- 对应 Tool：`media_search.video_search`
- 官方教程：https://developers.google.com/youtube/v3/getting-started
- 步骤：进入 Google Cloud Console 创建项目；在 API Library 启用 YouTube Data API v3；进入 Credentials 创建 API Key；设置 API 限制和配额告警。
- 建议变量：`YOUTUBE_API_KEY`。

### openrouteservice

- 对应 Tool：`route_query`
- API/控制台：https://api.openrouteservice.org/
- 登录入口：https://openrouteservice.org/log-in/
- 步骤：注册并验证邮箱；登录 Developer Dashboard；创建 Token/API Key；记录免费配额；只在服务端保存。
- 建议变量：`ORS_API_KEY`。

### OpenTopography

- 对应 Tool：`terrain_analysis`、`visibility_analysis` 的 DEM 获取备用方案
- 官方开发者页：https://opentopography.org/developers
- 步骤：注册账号；进入 My Account；申请免费 API Key；确认学术/非学术每日配额和目标数据集许可。
- 建议变量：`OPENTOPOGRAPHY_API_KEY`。

### GeoNames

- 对应 Tool：`administrative_registry`
- 注册入口：https://www.geonames.org/login
- Web Service 说明：https://www.geonames.org/export/ws-overview.html
- 步骤：创建账号并验证邮箱；在账号页面启用免费 Web Service；调用时使用 username，而不是传统随机 API Key。
- 建议变量：`GEONAMES_USERNAME`。

### OpenSky Network

- 对应 Tool：`flight_data_query`
- 官方 REST 文档：https://openskynetwork.github.io/opensky-api/rest.html
- 步骤：注册并登录 OpenSky；进入 Account；创建 API Client；保存 Client ID 和 Client Secret；通过 OAuth2 client_credentials 换取 Bearer Token。若需要较长历史数据，另在账户中申请历史数据访问。
- 建议变量：`OPENSKY_CLIENT_ID`、`OPENSKY_CLIENT_SECRET`。

### Google Cloud 统一备用方案

- 可覆盖：Google Vision、Geocoding、Elevation、Routes、Street View
- 开通说明：https://developers.google.com/maps/get-started
- Vision 认证：https://docs.cloud.google.com/vision/docs/authentication
- 步骤：创建 Google Cloud Project；绑定 Billing；逐个启用所需 API；进入 Credentials 创建 Key 或服务账号；给 Key 添加 API 范围和服务器来源限制；设置预算和配额告警。
- 当前已支持变量：`GOOGLE_VISION_API_KEY`。其余 Maps 服务尚未接入，不要提前把 Key 写入 Tool inputs。

## 四、公共免 Key 服务的正确使用方式

### Nominatim

当前 `geocode` 和无范围的 POI 名称检索可以使用公共 Nominatim，不需要申请 Key，但必须设置可识别 User-Agent，公共实例绝对不适合并发跑 2000 条数据。

```dotenv
ALLOW_REAL_TOOL_API=true
GEOAGENT_USER_AGENT=Geoagent-data/0.1 (contact: team@example.com)
NOMINATIM_ENDPOINT=https://nominatim.openstreetmap.org
```

官方政策：https://operations.osmfoundation.org/policies/nominatim/

### Overpass API

当前 `osm_query` 和区域类别型 `poi_search` 可以使用公共 Overpass，不需要 Key。开发验证可使用官方公共实例，批量生产应自建、使用托管服务或提前缓存区域数据，不得用十路并发压公共实例。

```dotenv
ALLOW_REAL_TOOL_API=true
GEOAGENT_USER_AGENT=Geoagent-data/0.1 (contact: team@example.com)
OVERPASS_ENDPOINT=https://overpass-api.de/api/interpreter
OVERPASS_TIMEOUT_SEC=45
```

文档：https://wiki.openstreetmap.org/wiki/Overpass_API

## 五、密钥交接规则

1. 申请人只交接“Provider 名、账号归属、已启用 API、配额、到期时间和本地变量名”，不要把明文 Key 写进文档或群聊。
2. 每个 Provider 使用独立 Key，开发、批量生产分开，能设置 API restriction、IP restriction 或预算上限时必须设置。
3. `.env` 已被仓库 `.gitignore` 忽略；提交前仍要运行密钥扫描和 `git diff --cached` 检查。
4. 外部 Tool 返回必须保留 provider、真实请求参数、时间、来源 ID、CRS/单位、许可/署名和失败语义；网络失败不能转换成成功 Observation。
5. 申请完成不等于可以用于训练数据。反向搜图、街景、商业地图和卫星图仍需逐项确认上传、缓存、再分发和训练许可。
