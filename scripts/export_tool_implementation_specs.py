"""Export one display/review JSON for each Canonical Tool v2 definition.

The generated implementation sections are research/design notes, not claims that
the backends have been installed, authenticated, executed, or licensed for SFT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "canonical_tool_catalog_v2.json"
OUT_DIR = ROOT / "docs" / "tool_specs_v2"
RESEARCH_DATE = "2026-08-31"
RUNTIME_STATUS = {
    "image_edit": "runtime_verified_local",
    "image_measure": "runtime_verified_local",
    "image_compare": "runtime_verified_local",
    "ocr_read": "runtime_verified_local",
    "media_metadata_read": "runtime_verified_local",
    "reverse_image_search": "runtime_implemented_provider_mock_verified_live_key_pending",
    "geocode": "runtime_implemented_provider_mock_verified_live_service_pending",
    "osm_query": "runtime_implemented_provider_mock_verified_live_service_pending",
    "poi_search": "runtime_implemented_provider_mock_verified_live_service_pending",
}


def _backend(
    name: str,
    kind: str,
    role: str,
    url: str,
    auth: str = "无",
    *,
    license_note: str = "使用前核对软件与数据许可。",
) -> dict[str, str]:
    return {
        "name": name,
        "kind": kind,
        "role": role,
        "official_url": url,
        "authentication": auth,
        "license_note": license_note,
    }


PROFILES: dict[str, dict[str, Any]] = {
    "image_edit": {
        "architecture": "local_deterministic",
        "backends": [
            _backend("Pillow", "local_library", "亮度、对比度、清晰度、裁剪和缩放", "https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html"),
            _backend("OpenCV", "local_library", "插值缩放、仿射和透视变换", "https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html"),
        ],
        "execution_flow": ["解析真实 image 引用与 ROI", "执行确定性变换并保存新 image_id", "返回源图、参数、尺寸和处理日志"],
        "limitations": ["放大不能恢复原图不存在的细节", "生成式修复不得作为证据增强", "所有变换必须可复现且保留源图"],
        "mcp_strategy": "优先本地 Python executor；需要跨 Agent 使用时再用 MCP Python SDK 包装。",
    },
    "image_measure": {
        "architecture": "local_deterministic_with_calibration",
        "backends": [_backend("OpenCV calibration", "local_library", "像素、角度、透视和有参照尺度测量", "https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html")],
        "execution_flow": ["读取测量类型与 ROI", "验证 reference 是否包含真实尺度或标定", "输出数值、单位、误差与假设"],
        "limitations": ["无尺度时只能输出像素或相对比例", "未经标定不能将像素直接换成米", "透视和镜头畸变必须进入误差说明"],
        "mcp_strategy": "本地函数即可；禁止开放任意代码执行入口。",
    },
    "image_compare": {
        "architecture": "local_cv_then_optional_embedding",
        "backends": [_backend("OpenCV feature matching", "local_library", "特征点匹配、RANSAC 和单应性验证", "https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html")],
        "execution_flow": ["统一图像尺寸/颜色并限定区域", "计算匹配、配准或差异", "返回对应点、得分、叠加图与失败原因"],
        "limitations": ["相似不等于同一地点", "跨季节/视角/遮挡需降低置信度", "必须返回可检查的匹配证据而非只有结论"],
        "mcp_strategy": "本地 CV executor；深度特征模型只能作为可替换 backend。",
    },
    "ocr_read": {
        "architecture": "local_ocr_and_code_decoder",
        "backends": [
            _backend("PaddleOCR", "local_model", "中英文字检测、识别与框坐标", "https://www.paddleocr.ai/main/en/quick_start.html"),
            _backend("OpenCV QRCodeDetector", "local_library", "二维码检测与解码", "https://docs.opencv.org/master/javadoc/org/opencv/objdetect/QRCodeDetector.html"),
        ],
        "execution_flow": ["按 region 裁剪并保留原坐标映射", "选择语言/编码识别器", "返回原始文本、框、分数与候选"],
        "limitations": ["模型纠错必须和原始 OCR 分开", "低清文字不得自动补成确定地名", "二维码无结果不代表画面不存在编码"],
        "mcp_strategy": "本地服务复用模型进程，MCP 只传 image_id，不传任意文件路径。",
    },
    "reverse_image_search": {
        "architecture": "external_index_with_local_crop",
        "backends": [
            _backend("Google Cloud Vision Web Detection", "external_api", "匹配网页、完整/局部匹配图和视觉相似图", "https://docs.cloud.google.com/vision/docs/detecting-web", "Google Cloud 项目、启用 Vision、ADC/服务账号", license_note="调用前核对图片上传、结果留存和训练用途条款。"),
            _backend("SerpAPI Google Lens", "third_party_api", "第三方 Lens 结果接口", "https://serpapi.com/google-lens-api", "SerpAPI Key", license_note="不是 Google 官方 Lens API；需单独评估稳定性与许可。"),
        ],
        "execution_flow": ["必要时先生成局部 crop", "提交可授权图片并限制 top_k", "保存来源 URL、匹配类型和检索时间"],
        "limitations": ["搜索命中不是地点真值", "不得虚构已执行的检索回执", "敏感或无授权图片不上传第三方"],
        "mcp_strategy": "自建 provider adapter；密钥只在服务端环境变量中，不进入 Tool inputs。",
    },
    "media_metadata_read": {
        "architecture": "local_metadata_parser",
        "backends": [
            _backend("ExifTool", "local_cli", "EXIF、GPS、拍摄时间和设备字段", "https://exiftool.org/exiftool_pod2.html"),
            _backend("ffprobe", "local_cli", "视频容器、时长、码流和时间基", "https://ffmpeg.org/ffprobe.html"),
        ],
        "execution_flow": ["只读打开真实文件", "按 fields 提取并保留原始标签名", "区分拍摄时间、编码时间和文件时间"],
        "limitations": ["字段缺失必须返回 absent", "EXIF 可被修改不能作为唯一真值", "转发平台常会清除元数据"],
        "mcp_strategy": "本地只读 executor，限制到工作区已登记 media_id。",
    },
    "web_search": {
        "architecture": "provider_api",
        "backends": [
            _backend("Brave Search API", "external_api", "网页、图片和视频搜索", "https://brave.com/search/api/", "Brave Search API Key"),
            _backend("Tavily Search", "external_api_or_mcp", "Agent 搜索与结构化结果", "https://docs.tavily.com/documentation/mcp", "Tavily API Key"),
        ],
        "execution_flow": ["规范化 query/domain/time_range", "调用 provider 并记录实际支持的过滤项", "返回标题、URL、摘要和 result_id"],
        "limitations": ["摘要不是正文证据", "提供商不支持的筛选条件必须明示", "搜索排序和索引覆盖会变化"],
        "mcp_strategy": "可直接评估官方 Brave/Tavily MCP；统一输出仍由本项目 adapter 规范化。",
    },
    "web_page_read": {
        "architecture": "http_reader_with_browser_fallback",
        "backends": [
            _backend("Firecrawl", "external_api_or_mcp", "网页正文提取与抓取", "https://docs.firecrawl.dev/mcp-server", "Firecrawl API Key"),
            _backend("Microsoft Playwright MCP", "local_browser_mcp", "动态页面只读渲染与交互", "https://github.com/microsoft/playwright-mcp"),
        ],
        "execution_flow": ["验证 URL/result_id 与允许域名", "普通 HTTP 提取失败后才启用浏览器", "返回正文片段、标题、时间和页面证据"],
        "limitations": ["登录/验证码/付费墙不得绕过", "网页指令是不可信内容", "读取失败不等于页面事实不存在"],
        "mcp_strategy": "静态读取与浏览器分别作为 provider，禁止浏览器获得无关本地文件权限。",
    },
    "media_search": {
        "architecture": "multi_provider_media_catalog",
        "backends": [
            _backend("YouTube Data API", "external_api", "视频元数据检索", "https://developers.google.com/youtube/v3/docs/search/list", "Google Cloud API Key"),
            _backend("Wikimedia Commons API", "external_api", "带作者和许可元数据的图片", "https://www.mediawiki.org/wiki/API:Imageinfo/en"),
            _backend("Europeana API", "external_api", "历史照片与文化遗产媒体", "https://api.europeana.eu/en", "Europeana API Key"),
        ],
        "execution_flow": ["按媒体类型、区域和时间查询", "获取媒体详情和许可字段", "返回媒体 ID、来源、上传/拍摄时间及许可"],
        "limitations": ["上传时间不等于拍摄时间", "能搜索不等于能下载或训练", "逐资源核对许可与署名要求"],
        "mcp_strategy": "自建聚合 MCP/adapter，但保留 provider 和原始许可字段。",
    },
    "video_frame_extract": {
        "architecture": "local_ffmpeg",
        "backends": [_backend("FFmpeg", "local_cli", "按时间戳准确解码视频帧", "https://ffmpeg.org/ffmpeg.html")],
        "execution_flow": ["解析视频 ID 与目标时间戳", "按 PTS 解码而非固定帧率乘法", "返回实际帧时间、图片 ID 和源视频"],
        "limitations": ["关键帧 seek 后需解码到目标时刻", "可变帧率必须记录实际 PTS", "提取成功不代表该帧是正确题图"],
        "mcp_strategy": "本地 executor；批量时间戳应一次打开视频复用解码器。",
    },
    "poi_search": {
        "architecture": "regional_provider_with_osm_fallback",
        "backends": [
            _backend("高德 Web 服务 POI", "external_api_or_mcp", "国内关键词、周边和范围 POI", "https://lbs.amap.com/api/webservice/guide/api/search/", "高德 Web 服务 Key", license_note="核对地图数据存储、展示和训练许可。"),
            _backend("OpenStreetMap/Overpass", "open_data_api_or_local", "按标签查询开放地理对象", "https://dev.overpass-api.de/overpass-doc/en/"),
        ],
        "execution_flow": ["解析 area/query/category/radius", "调用区域适配 provider", "标准化名称、类别、坐标系、对象 ID 与来源"],
        "limitations": ["POI 覆盖不完整", "行政区中心不是拍摄点", "WGS84/GCJ02 等坐标系不得混用"],
        "mcp_strategy": "国内可评估高德官方 MCP；开放数据使用自建 Overpass adapter。",
    },
    "geocode": {
        "architecture": "provider_geocoder_or_self_host",
        "backends": [
            _backend("高德地理编码", "external_api", "国内地址与坐标互转", "https://lbs.amap.com/api/web-service/guide/georegeo", "高德 Web 服务 Key"),
            _backend("Nominatim", "self_host_or_public_limited", "OSM 地名搜索与逆地理编码", "https://nominatim.org/release-docs/latest/api/Overview/"),
        ],
        "execution_flow": ["按 direction 选择正向/逆向", "限制 area 并返回多个候选", "记录匹配层级、坐标系和 provider"],
        "limitations": ["公共 Nominatim 绝对上限 1 req/s 且禁止重批量", "地名中心点不等于精确目标", "跑批优先自建或商业服务"],
        "mcp_strategy": "统一 geocode executor；公共 Nominatim 不允许并发压测。",
    },
    "route_query": {
        "architecture": "routing_api_or_local_engine",
        "backends": [
            _backend("高德路径规划", "external_api", "国内驾车、步行等路径", "https://lbs.amap.com/api/web-service/guide/routes", "高德 Web 服务 Key"),
            _backend("OSRM", "self_host", "基于 OSM 的路线、匹配和距离", "https://project-osrm.org/docs/"),
        ],
        "execution_flow": ["解析起终点、途经点和交通方式", "调用对应路网快照", "返回路线几何、里程、时间和数据版本"],
        "limitations": ["道路距离不等于测地直线距离", "当前路网不能无说明证明历史路线", "不支持的交通方式不能静默改写"],
        "mcp_strategy": "高德 MCP 或自建 OSRM service；Tool 名保持统一。",
    },
    "map_layer_query": {
        "architecture": "ogc_service_or_local_geoserver",
        "backends": [_backend("GeoServer WMS/WFS", "self_host_or_external_ogc", "加载栅格地图和矢量属性", "https://docs.geoserver.org/latest/en/user/services/wms/reference/")],
        "execution_flow": ["先读 GetCapabilities/图层清单", "按 bbox、CRS、time、layer 请求", "区分地图图片与可计算矢量/栅格"],
        "limitations": ["WMS 图片不能替代真实几何", "TIME 支持取决于数据源", "必须保留 CRS、图层日期和许可"],
        "mcp_strategy": "自建 OGC adapter；不要给模型直接拼接不受限的服务 URL。",
    },
    "osm_query": {
        "architecture": "structured_query_compiler_plus_overpass",
        "backends": [
            _backend("Overpass API", "external_or_self_host_api", "按标签和空间关系查询 OSM", "https://dev.overpass-api.de/overpass-doc/en/"),
            _backend("Geofabrik extracts", "offline_dataset", "区域 PBF/GPKG 离线数据", "https://download.geofabrik.de/asia/china.html", license_note="遵守 ODbL 与署名要求。"),
        ],
        "execution_flow": ["优先把 area/tags/feature_types 编译成只读 Overpass QL", "限制范围、超时和结果数量", "返回对象 ID、标签、几何、数据时间和原查询"],
        "limitations": ["图片不能直接作为 Overpass 查询参数", "无结果不等于现实不存在", "只有原链提供代码时才接受 overpass_ql"],
        "mcp_strategy": "自建只读 MCP，禁止通用 Python/任意网络代码执行。",
    },
    "osm_result_process": {
        "architecture": "local_geodata_processing",
        "backends": [_backend("GeoPandas", "local_library", "过滤、去重和导出 GeoJSON/GPKG/Parquet", "https://docs.geopandas.org/en/latest/docs/user_guide/io.html")],
        "execution_flow": ["解析真实 source_result", "执行属性或空间过滤", "返回新结果 ID、保留/剔除数量与导出文件"],
        "limitations": ["不得把查询与后处理混为一个外部回执", "缺少属性时应报无法筛选", "必须保留源结果和过滤表达式"],
        "mcp_strategy": "本地 executor，source_result 使用不可伪造的运行时句柄。",
    },
    "streetview_query": {
        "architecture": "licensed_imagery_session",
        "backends": [
            _backend("Mapillary API", "external_api", "众包街景、序列、位置与拍摄时间", "https://help.mapillary.com/hc/en-us/articles/360010234680-Accessing-imagery-and-data-through-the-Mapillary-API", "Mapillary access token", license_note="逐项确认图像下载、保留和训练使用范围。"),
            _backend("Google Street View", "external_api_restricted", "全景打开、邻接和视角截图", "https://developers.google.com/maps/documentation/streetview/request-streetview", "Google Maps Platform Key", license_note="通用条款限制批量提取、缓存及用于训练/测试模型；无额外授权不作为默认 SFT 数据源。"),
        ],
        "execution_flow": ["open 建立真实 panorama/session", "navigate/change_time 只能使用实际存在的邻接和日期", "capture 返回视角、版权、拍摄日期与图片 ID"],
        "limitations": ["网站历史滑块不等于 API 可指定任意年份", "无历史覆盖必须返回 no_coverage", "会话和 panorama ID 不得编造"],
        "mcp_strategy": "自建受许可约束的 session adapter；每个返回保留 provider attribution。",
    },
    "satellite_imagery_query": {
        "architecture": "stac_catalog_plus_process_api",
        "backends": [
            _backend("Copernicus Data Space Sentinel Hub", "external_api", "STAC 检索、影像处理和下载", "https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview.html", "Copernicus 账户、OAuth client_id/client_secret"),
            _backend("Planet Data API", "commercial_api", "商业高分影像目录与资产", "https://docs.planet.com/develop/apis/data/", "Planet 账户/OAuth或受支持的 API 凭证", license_note="下载和训练使用取决于商业合同。"),
        ],
        "execution_flow": ["Catalog 先按 AOI/日期/云量查真实 acquisition", "选择实际可用资产后 Process/下载", "记录 scene_id、日期、分辨率、CRS、处理脚本和许可"],
        "limitations": ["resolution_m 不能创造不存在的分辨率", "云量/拼接/季节会影响判断", "oblique_view 渲染不得冒充真实斜拍照片"],
        "mcp_strategy": "自建 OAuth token 缓存的 API adapter；不要每次请求新 token。",
    },
    "satellite_imagery_compare": {
        "architecture": "local_geospatial_alignment_and_compare",
        "backends": [_backend("Rasterio", "local_library", "影像重投影、重采样和空间对齐", "https://rasterio.readthedocs.io/en/stable/topics/reproject.html")],
        "execution_flow": ["解析真实候选 scene IDs", "统一 CRS、范围、像元尺度和有效掩膜", "生成差异图、统计和可核对叠加图"],
        "limitations": ["云、阴影、水位和季节会产生伪变化", "不同分辨率候选必须报告不对称", "变化检测结论需要人工或其他证据解释"],
        "mcp_strategy": "本地计算 executor；影像获取仍属于 satellite_imagery_query。",
    },
    "distance_bearing_calculator": {
        "architecture": "local_geodesic_math",
        "backends": [_backend("pyproj.Geod", "local_library", "椭球测地距离与正反方位角", "https://pyproj4.github.io/pyproj/stable/api/geod.html")],
        "execution_flow": ["验证点坐标与 CRS", "按 mode 选择测地/投影/三维方法", "返回数值、单位、方法和输入点"],
        "limitations": ["经纬度不能直接欧氏相减当米", "道路距离应交给 route_query", "上游坐标不实会产生精确但错误结果"],
        "mcp_strategy": "纯本地确定性函数，适合直接内嵌流水线。",
    },
    "visibility_analysis": {
        "architecture": "local_dem_viewshed",
        "backends": [_backend("GDAL viewshed", "local_cli_or_library", "DEM 可视域与最小可见高度", "https://gdal.org/en/stable/programs/gdal_raster_viewshed.html")],
        "execution_flow": ["解析 observer/target 和高度", "选择覆盖区域且投影正确的 DEM/DSM", "输出视线剖面、遮挡位置或 viewshed raster"],
        "limitations": ["裸地 DEM 不包含建筑树木", "投影与高度基准必须明确", "数据分辨率决定可验证尺度"],
        "mcp_strategy": "本地 GIS worker；限制最大范围和计算资源。",
    },
    "terrain_analysis": {
        "architecture": "local_dem_processing",
        "backends": [
            _backend("GDAL DEM tools", "local_cli_or_library", "坡度、坡向、阴影和地形指标", "https://gdal.org/en/stable/programs/gdaldem.html"),
            _backend("OpenTopography", "external_data_api", "DEM 数据获取", "https://opentopography.org/developers", "OpenTopography API Key", license_note="数据集、配额和商业使用条件按账户核对。"),
        ],
        "execution_flow": ["获取/解析真实 DEM", "统一垂直/水平单位和 CRS", "计算 metrics 或 path 剖面并返回数据版本"],
        "limitations": ["低分辨率 DEM 不支持精细局部结论", "高程基准与单位差异需校正", "地形分析不能自动包含城市障碍物"],
        "mcp_strategy": "本地 GDAL worker；数据下载由独立受控 provider 完成。",
    },
    "spatial_filter": {
        "architecture": "local_geopandas_or_postgis",
        "backends": [
            _backend("GeoPandas", "local_library", "小规模相交、包含、缓冲和筛选", "https://docs.geopandas.org/en/latest/docs/user_guide/io.html"),
            _backend("PostGIS ST_DWithin", "database", "大规模索引距离过滤", "https://postgis.net/docs/ST_DWithin.html", "数据库凭证"),
        ],
        "execution_flow": ["解析 source_result 与 geometry", "统一 CRS 并验证 distance_m 单位", "执行 relation 并返回保留对象与关系证据"],
        "limitations": ["标注点不能代替真实对象几何", "geometry 与 geography 单位不同", "空间筛选不补造缺失属性"],
        "mcp_strategy": "本地/数据库 executor；只允许参数化查询。",
    },
    "weather_archive_query": {
        "architecture": "reanalysis_api_plus_remote_sensing_layer",
        "backends": [
            _backend("Open-Meteo Historical Weather", "external_api", "历史气象和再分析变量", "https://open-meteo.com/en/docs/historical-weather-api"),
            _backend("NASA GIBS", "external_ogc_api", "时序云雪等遥感图层", "https://nasa-gibs.github.io/gibs-api-docs/access-basics/"),
        ],
        "execution_flow": ["根据变量选择气象序列或遥感图层", "查询 area/time_range 并记录格点/图层", "refine_range 只对真实返回序列做筛选"],
        "limitations": ["再分析格点不是照片点位实测站", "云量与积雪变量定义不同", "时区和日期边界必须统一"],
        "mcp_strategy": "自建 provider adapter；缓存按数据许可和请求日期管理。",
    },
    "solar_ephemeris": {
        "architecture": "local_astronomy_math",
        "backends": [_backend("pvlib", "local_library", "太阳高度、方位和日出日落", "https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.get_solarposition.html")],
        "execution_flow": ["解析候选地点、日期时间和时区", "计算太阳位置或日落", "返回算法、角度、时间和输入假设"],
        "limitations": ["时区错误会直接改变结论", "候选地点计算不能反向证明地点", "历史历法/地平线遮挡需另行处理"],
        "mcp_strategy": "纯本地函数；无需外部 API。",
    },
    "shadow_analysis": {
        "architecture": "local_geometry_using_solar_ephemeris",
        "backends": [_backend("pvlib + local geometry", "local_library", "太阳位置与阴影方向/长度模型", "https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.get_solarposition.html")],
        "execution_flow": ["调用太阳位置", "验证物体高度、地面和相机/图像尺度", "计算阴影候选并返回假设与误差"],
        "limitations": ["未知物高、地面坡度和相机姿态时通常多解", "简单公式仅适用于竖直物体和平地", "不得从模糊阴影补精确时间"],
        "mcp_strategy": "本地组合 executor，不需要新外部服务。",
    },
    "administrative_registry": {
        "architecture": "gazetteer_plus_official_regional_sources",
        "backends": [
            _backend("GeoNames", "external_api_or_offline_dump", "全球地名、别名和行政层级", "https://www.geonames.org/export/ws-overview.html", "GeoNames username"),
            _backend("国家地名信息库/地方民政资料", "official_web_or_import", "中国标准地名与行政资料", "https://app.www.gov.cn/govdata/gov/202212/25/495555/article.html", "按具体来源", license_note="很多来源没有统一公开 API，需要人工导入与逐来源许可。"),
        ],
        "execution_flow": ["按 registry 和 area 选数据源", "查询标准名、别名和层级", "保留有效时间与官方出处"],
        "limitations": ["当前行政归属不能自动外推到历史年份", "地名库质量与覆盖不同", "未查到不能证明名称不存在"],
        "mcp_strategy": "按国家/地区做 connector，统一输出 provenance 和 validity interval。",
    },
    "infrastructure_registry": {
        "architecture": "regional_registry_connectors",
        "backends": [
            _backend("全国建筑市场监管公共服务平台/地方住建公示", "official_web_or_import", "国内建设项目、许可和公示线索", "https://jzsc.mohurd.gov.cn/home", "可能需要网站查询/地方接口", license_note="未确认覆盖全国历史设施的统一公开 API。"),
            _backend("Planning Data England", "official_open_api", "英格兰规划、建筑和约束数据", "https://www.planning.data.gov.uk/docs"),
        ],
        "execution_flow": ["根据地区选择官方 registry", "按项目/设施/时间查询并保存原文", "区分许可、开工、竣工和开放日期"],
        "limitations": ["没有全球统一接口", "公开覆盖不全，查无记录不等于不存在", "历史文件可能只能人工导入"],
        "mcp_strategy": "每个地区单独 connector；不建议用一个无来源的通用网页搜索冒充 registry。",
    },
    "flight_data_query": {
        "architecture": "adsb_api_with_optional_commercial_archive",
        "backends": [
            _backend("OpenSky Network", "external_api", "状态向量、航班和航迹", "https://github.com/openskynetwork/opensky-api/blob/master/docs/free/rest.rst", "OpenSky OAuth2 client_id/client_secret"),
            _backend("Flightradar24 API", "commercial_api", "商业航班与历史位置", "https://fr24api.flightradar24.com/docs/faq", "订阅/API 凭证"),
        ],
        "execution_flow": ["按 flight/date/area 选择 search/track/nearby", "统一 UTC、坐标和高度字段", "返回原始采样点、覆盖说明和最近距离所需数据"],
        "limitations": ["ADS-B 覆盖和采样不完整", "计划航班不等于实际航迹", "气压高度与几何高度不可混用"],
        "mcp_strategy": "自建 API adapter；限制时间窗口、区域和隐私用途。",
    },
    "llm_query": {
        "architecture": "existing_model_adapter",
        "backends": [
            _backend("Anthropic Messages API", "external_api", "外部模型咨询和候选枚举", "https://platform.claude.com/docs/en/api/overview", "Anthropic API Key 或已授权兼容中转凭证"),
            _backend("MCP Python SDK", "local_protocol_sdk", "把受控本地/外部能力暴露为 MCP", "https://py.sdk.modelcontextprotocol.io/"),
        ],
        "execution_flow": ["最小化 context 并选择明确模型", "结构化返回建议/候选", "记录 provider、实际模型、请求版本和失败状态"],
        "limitations": ["LLM 建议不是搜索或数据库回执", "中转别名不自动证明上游真实模型身份", "普通内部思考不应为凑 Tool 而调用"],
        "mcp_strategy": "复用现有 pipeline.llm；密钥仅保存在本地配置，不进入样本。",
    },
    "final_answer": {
        "architecture": "local_contract_validator",
        "backends": [_backend("GeoAgent local validator", "local_code", "校验并提交 location 字符串或有序数组", "https://github.com/kesizar/Geoagent-data")],
        "execution_flow": ["确认它是末步且 observation=null", "校验 location 类型、非空和题目数量", "记录最终答案，不做新的地理推理"],
        "limitations": ["格式通过不等于地点正确", "不得补造精细坐标", "多题答案必须保持讲解顺序"],
        "mcp_strategy": "保留为本地终端 Tool，不需要外部 API 或 MCP。",
    },
}


COMMON_EXECUTOR_CONTRACT = {
    "required_result_fields": [
        "status",
        "provider_or_algorithm",
        "request_parameters_used",
        "retrieved_at",
        "source_result_id_or_raw_ref",
        "provenance",
        "warnings",
    ],
    "status_values": [
        "success",
        "no_result",
        "no_coverage",
        "parameter_missing",
        "unsupported",
        "permission_denied",
        "provider_error",
    ],
    "rules": [
        "no_result 与 no_coverage、provider_error 必须分开",
        "任何精确数值必须带单位、方法和源数据",
        "调用失败不得生成看似成功的 Observation",
        "密钥、令牌和内部文件系统路径不得进入训练样本",
    ],
}


def _parameter_guide(operation: dict[str, Any]) -> list[dict[str, Any]]:
    schema = operation.get("input_schema") or {}
    guides: list[dict[str, Any]] = []
    for field in schema.get("fields") or []:
        guides.append(
            {
                "name": field.get("name"),
                "type": field.get("type"),
                "required": bool(field.get("required")),
                "requirement_level": field.get("requirement_level"),
                "purpose": field.get("description"),
                "how_to_obtain": field.get("acquisition_hint"),
                "context_sources": field.get("context_sources") or [],
                "context_default": field.get("context_default"),
                "allowed_values": field.get("allowed_values") or [],
                "minimum": field.get("minimum"),
                "maximum": field.get("maximum"),
                "example": field.get("example"),
                "aliases": field.get("aliases") or [],
                "validation_note": (
                    "execution 级字段缺失时不得执行；按 how_to_obtain 先获取真实值。"
                    if field.get("requirement_level") == "execution"
                    else "可选字段缺失不应自动判为不可执行。"
                ),
            }
        )
    return guides


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    trees = catalog.get("trees") or []
    names = [tree["canonical"]["name"] for tree in trees]
    missing = sorted(set(names) - set(PROFILES))
    extra = sorted(set(PROFILES) - set(names))
    if missing or extra:
        raise SystemExit(f"profile/catalog mismatch missing={missing} extra={extra}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for ordinal, tree in enumerate(trees, start=1):
        canonical = tree["canonical"]
        name = canonical["name"]
        profile = PROFILES[name]
        parameter_guides = {
            operation["name"]: _parameter_guide(operation)
            for operation in canonical.get("operations") or []
        }
        payload = {
            "schema_version": "geoagent_tool_implementation_spec_v1",
            "ordinal": ordinal,
            "research_date": RESEARCH_DATE,
            "research_status": RUNTIME_STATUS.get(
                name,
                "design_researched_not_runtime_implemented",
            ),
            "catalog_source": "canonical_tool_catalog_v2.json",
            "canonical_tool": tree,
            "parameter_guide": parameter_guides,
            "implementation_research": profile,
            "recommended_executor_contract": COMMON_EXECUTOR_CONTRACT,
            "new_tool_review_hint": {
                "covered_execution_boundary": canonical.get("executor"),
                "do_not_create_when": [
                    "只是对象、地区、查询词或参数不同",
                    "现有 Tool 的某个 operation 与本次动作共享同一执行器",
                    "只是基于已有结果进行比较、筛选、总结或推理",
                ],
                "consider_new_only_when": [
                    "现有31类执行器及所有 operation 均不能执行该动作",
                    "存在独立、可实现、可测试的真实后端边界",
                    "多个独立样本反复出现相同能力缺口",
                    "能够提供完整 schema、许可评估、失败语义和测试证据",
                ],
            },
        }
        path = OUT_DIR / f"{ordinal:02d}_{name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs.append(path)
    if len(outputs) != 31:
        raise SystemExit(f"expected 31 outputs, got {len(outputs)}")
    print(json.dumps({"count": len(outputs), "out_dir": str(OUT_DIR)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
