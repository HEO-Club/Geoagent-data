"""flight_data_query 执行器测试；禁止真实付费 API。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

from tool import execute
from tool.contract import Observation, RuntimeContext
from tool.flight_data_query import _query as query_mod
from tool.flight_data_query._query import BBox


class FakeFlightProvider:
    """测试替身：记录 OpenSky 风格 flights / states / tracks 请求。"""

    name = "opensky"
    authenticated = True
    max_state_lookback_sec: int | None = None
    max_track_lookback_sec: int | None = None

    def __init__(
        self,
        *,
        flights: list[dict[str, Any]] | None = None,
        states: dict[str, Any] | None = None,
        track: dict[str, Any] | None = None,
    ) -> None:
        self.flights = flights if flights is not None else _sample_flights()
        self.states = states if states is not None else _sample_states()
        self.track = track if track is not None else _sample_track()
        self.departure_calls: list[tuple[str, int, int]] = []
        self.arrival_calls: list[tuple[str, int, int]] = []
        self.state_calls: list[tuple[BBox, int | None]] = []
        self.track_calls: list[tuple[str, int]] = []

    def fetch_departures(self, airport: str, begin: int, end: int) -> list[dict[str, Any]]:
        self.departure_calls.append((airport, begin, end))
        return list(self.flights)

    def fetch_arrivals(self, airport: str, begin: int, end: int) -> list[dict[str, Any]]:
        self.arrival_calls.append((airport, begin, end))
        return list(self.flights)

    def fetch_states(self, *, bbox: BBox, time_unix: int | None) -> dict[str, Any]:
        self.state_calls.append((bbox, time_unix))
        return dict(self.states)

    def fetch_track(self, *, icao24: str, time_unix: int) -> dict[str, Any]:
        self.track_calls.append((icao24, time_unix))
        payload = dict(self.track)
        payload["icao24"] = icao24
        return payload


def _sample_flights() -> list[dict[str, Any]]:
    return [
        {
            "icao24": "3c675a",
            "firstSeen": 1577836800,
            "estDepartureAirport": "EDDF",
            "lastSeen": 1577844000,
            "estArrivalAirport": "EGLL",
            "callsign": "DLH123  ",
            "confirmed_location": "MUST NOT LEAK",
            "raw_content": "FULL FLIGHT JSON MUST NOT LEAK",
        },
        {
            "icao24": "abc123",
            "firstSeen": 1577836900,
            "estDepartureAirport": "EDDF",
            "lastSeen": 1577844100,
            "estArrivalAirport": "LFPG",
            "callsign": "AFR456",
        },
    ]


def _sample_states() -> dict[str, Any]:
    return {
        "time": 1577836800,
        "confirmed_location": "MUST NOT LEAK",
        "states": [
            [
                "3c675a",
                "DLH123  ",
                "Germany",
                1577836790,
                1577836800,
                8.57,
                50.03,
                10668.0,
                False,
                230.0,
                270.0,
                0.0,
                None,
                11000.0,
                "1000",
                False,
                0,
                4,
            ],
            [
                "abc123",
                "AFR456",
                "France",
                1577836795,
                1577836800,
                8.60,
                50.05,
                9800.0,
                False,
                210.0,
                80.0,
                -5.0,
                None,
                10100.0,
                None,
                False,
                0,
                4,
            ],
        ],
    }


def _sample_track() -> dict[str, Any]:
    return {
        "icao24": "3c675a",
        "startTime": 1577836800,
        "endTime": 1577840400,
        "callsign": "DLH123",
        "confirmed_location": "MUST NOT LEAK",
        "raw_content": "FULL TRACK JSON MUST NOT LEAK",
        "path": [
            [1577836800, 50.03, 8.57, 300.0, 10.0, False],
            [1577838600, 50.50, 4.00, 10668.0, 280.0, False],
        ],
    }


def _nested_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(value)
        for item in value.values():
            found.update(_nested_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_nested_keys(item))
    return found


def _call(
    operation: str,
    *,
    inputs: dict[str, Any] | None = None,
    provider: FakeFlightProvider | None = None,
    ctx: RuntimeContext | None = None,
) -> Observation:
    engine = provider if provider is not None else FakeFlightProvider()
    runtime = ctx if ctx is not None else RuntimeContext()
    runtime.extras["flight_data_query_provider"] = engine
    return execute(
        "flight_data_query",
        operation,
        purpose="查询航班档案",
        inputs=inputs or {},
        ctx=runtime,
    )


def test_empty_inputs_are_missing_input() -> None:
    for operation in ("search", "track", "nearby_traffic"):
        observation = execute(
            "flight_data_query",
            operation,
            purpose="scaffold",
            inputs={},
        )
        assert observation.ok is False
        assert observation.error_code == "missing_input"


def test_search_flight_number_without_scope_is_missing_input() -> None:
    observation = _call("search", inputs={"flight_number": "DLH123"})
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "机场" in observation.error or "区域" in observation.error


def test_nearby_missing_time_range_is_missing_input() -> None:
    observation = _call(
        "nearby_traffic",
        inputs={"area": {"bbox": [8.4, 49.9, 8.8, 50.2]}},
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"


def test_place_name_without_geometry_is_missing_input() -> None:
    observation = _call(
        "search",
        inputs={"area": "郑州市", "flight_number": "DLH123"},
    )
    assert observation.ok is False
    assert observation.error_code == "missing_input"
    assert observation.error is not None
    assert "geocode" in observation.error


def test_search_icao24_filters_airport_flights() -> None:
    provider = FakeFlightProvider()
    observation = _call(
        "search",
        inputs={"airports": "EDDF", "date": "2020-01-01", "icao24": "3c675a"},
        provider=provider,
    )
    assert observation.ok is True
    assert observation.result is not None
    flights = observation.result["flights"]
    assert [item["icao24"] for item in flights] == ["3c675a"]
    assert observation.result["applied"]["icao24"] == "3c675a"


def test_iata_airport_is_invalid_input() -> None:
    observation = _call(
        "search",
        inputs={"airports": "FRA", "date": "2020-01-01"},
    )
    assert observation.ok is False
    assert observation.error_code == "invalid_input"
    assert observation.error is not None
    assert "ICAO" in observation.error


def test_search_airports_filters_callsign_and_keeps_actual_flight() -> None:
    provider = FakeFlightProvider()
    observation = _call(
        "search",
        inputs={
            "airports": "EDDF",
            "date": "2020-01-01",
            "flight_number": "DLH 123",
        },
        provider=provider,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert provider.departure_calls
    assert provider.arrival_calls
    assert provider.departure_calls[0][0] == "EDDF"
    assert observation.result["record_kind"] == "actual_flight"
    assert observation.result["provider"] == "opensky"
    assert observation.result["data_source"] == "adsb_reception"
    flights = observation.result["flights"]
    assert len(flights) == 1
    assert flights[0]["icao24"] == "3c675a"
    assert flights[0]["est_departure_airport"] == "EDDF"
    assert flights[0]["est_arrival_airport"] == "EGLL"
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert "raw_content" not in keys
    assert any("计划航班" in item for item in observation.result["assumptions"])
    assert any("distance_bearing_calculator" in item for item in observation.result["assumptions"])


def test_search_route_keeps_origin_destination_pair() -> None:
    provider = FakeFlightProvider()
    observation = _call(
        "search",
        inputs={"route": "EDDF-EGLL", "date": "2020-01-01"},
        provider=provider,
    )
    assert observation.ok is True
    assert observation.result is not None
    flights = observation.result["flights"]
    assert len(flights) == 1
    assert flights[0]["est_arrival_airport"] == "EGLL"
    assert observation.result["applied"]["route"] == {
        "origin": "EDDF",
        "destination": "EGLL",
    }


def test_track_icao24_waypoints_are_not_interpolated() -> None:
    provider = FakeFlightProvider()
    observation = _call(
        "track",
        inputs={"icao24": "3c675a", "date": "2020-01-01T00:00:00Z"},
        provider=provider,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["record_kind"] == "actual_track"
    assert provider.track_calls
    assert provider.track_calls[0][0] == "3c675a"
    track = observation.result["tracks"][0]
    points = track["waypoints"]
    assert len(points) == 2
    assert all(point["interpolated"] is False for point in points)
    assert all(point["geo_altitude_m"] is None for point in points)
    assert points[0]["baro_altitude_m"] == 300.0
    keys = _nested_keys(observation.result)
    assert "confirmed_location" not in keys
    assert "raw_content" not in keys
    assert any("气压高度" in item for item in observation.result["assumptions"])


def test_track_empty_path_is_ok_without_invented_points() -> None:
    provider = FakeFlightProvider(track={"icao24": "3c675a", "path": []})
    observation = _call(
        "track",
        inputs={"icao24": "3c675a", "date": "2020-01-01T00:00:00Z"},
        provider=provider,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["tracks"][0]["waypoints"] == []


def test_track_area_only_is_state_snapshot() -> None:
    provider = FakeFlightProvider()
    observation = _call(
        "track",
        inputs={"area": {"bbox": [8.4, 49.9, 8.8, 50.2]}},
        provider=provider,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["record_kind"] == "state_snapshot"
    assert observation.result["count"] == 2
    assert provider.state_calls
    assert provider.track_calls == []


def test_nearby_traffic_snapshot_splits_baro_and_geo() -> None:
    provider = FakeFlightProvider()
    observation = _call(
        "nearby_traffic",
        inputs={
            "area": {"bbox": [8.4, 49.9, 8.8, 50.2]},
            "time_range": "2020-01-01",
            "radius_km": 20,
        },
        provider=provider,
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["record_kind"] == "state_snapshot"
    assert observation.result["count"] == 2
    assert provider.state_calls
    bbox, _time_unix = provider.state_calls[0]
    assert bbox.west < bbox.east
    row = observation.result["aircraft"][0]
    assert row["baro_altitude_m"] == 10668.0
    assert row["geo_altitude_m"] == 11000.0
    assert any("快照" in item for item in observation.result["assumptions"])


def test_anonymous_historical_states_are_not_rewritten_to_live() -> None:
    provider = FakeFlightProvider()
    provider.authenticated = False
    provider.max_state_lookback_sec = 0
    observation = _call(
        "nearby_traffic",
        inputs={
            "area": {"bbox": [8.4, 49.9, 8.8, 50.2]},
            "time_range": "2020-01-01",
        },
        provider=provider,
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "不能改写成 live" in observation.error
    assert provider.state_calls == []


def test_track_beyond_thirty_days_is_not_rewritten_to_live() -> None:
    provider = FakeFlightProvider()
    provider.max_track_lookback_sec = 30 * 24 * 3600
    observation = _call(
        "track",
        inputs={"icao24": "3c675a", "date": "2018-06-01T12:00:00Z"},
        provider=provider,
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "不能改写成 live" in observation.error
    assert provider.track_calls == []


def test_allow_real_api_false_without_provider_is_unavailable() -> None:
    observation = execute(
        "flight_data_query",
        "search",
        purpose="闸门",
        inputs={"airports": "EDDF", "date": "2020-01-01"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is False
    assert observation.error_code == "engine_unavailable"
    assert observation.error is not None
    assert "ALLOW_REAL_API" in observation.error


def test_opensky_http_query_shape(monkeypatch: Any) -> None:
    calls: list[str] = []

    class _FakeHttpResponse:
        def __init__(self, payload: bytes) -> None:
            self._body = payload

        def read(self) -> bytes:
            return self._body

        def __enter__(self) -> _FakeHttpResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request: Any, timeout: float | None = None) -> _FakeHttpResponse:
        del timeout
        url = str(request.full_url)
        calls.append(url)
        parsed = urlparse(url)
        if parsed.path.endswith("/flights/departure") or parsed.path.endswith("/flights/arrival"):
            return _FakeHttpResponse(b"[]")
        return _FakeHttpResponse(b"{}")

    monkeypatch.setenv("ALLOW_REAL_API", "true")
    monkeypatch.setattr(query_mod.urllib.request, "urlopen", fake_urlopen)
    observation = execute(
        "flight_data_query",
        "search",
        purpose="组装 OpenSky",
        inputs={"airports": "EDDF", "date": "2020-01-01"},
        ctx=RuntimeContext(),
    )
    assert observation.ok is True
    assert observation.result is not None
    assert observation.result["flights"] == []
    departure = [url for url in calls if "/flights/departure" in url]
    arrival = [url for url in calls if "/flights/arrival" in url]
    assert departure and arrival
    params = parse_qs(urlparse(departure[0]).query)
    assert params["airport"] == ["EDDF"]
    assert "begin" in params
    assert "end" in params


def test_nearby_live_window_omits_time_param() -> None:
    provider = FakeFlightProvider()
    provider.max_state_lookback_sec = 0
    now = datetime.now(timezone.utc)
    observation = _call(
        "nearby_traffic",
        inputs={
            "area": {"bbox": [8.4, 49.9, 8.8, 50.2]},
            "time_range": now.strftime("%Y-%m-%d"),
        },
        provider=provider,
    )
    assert observation.ok is True
    assert provider.state_calls
    assert provider.state_calls[0][1] is None


def test_authenticated_state_lookback_one_hour(monkeypatch: Any) -> None:
    provider = FakeFlightProvider()
    provider.max_state_lookback_sec = 3600
    frozen = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(query_mod, "_utcnow", lambda: frozen)
    too_old = (frozen - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    failed = _call(
        "nearby_traffic",
        inputs={
            "area": {"bbox": [8.4, 49.9, 8.8, 50.2]},
            "time_range": too_old,
        },
        provider=provider,
    )
    assert failed.ok is False
    assert failed.error_code == "engine_unavailable"
    assert failed.error is not None
    assert "不能改写成 live" in failed.error
    recent = (frozen - timedelta(minutes=10)).isoformat().replace("+00:00", "Z")
    ok = _call(
        "nearby_traffic",
        inputs={
            "area": {"bbox": [8.4, 49.9, 8.8, 50.2]},
            "time_range": recent,
        },
        provider=provider,
    )
    assert ok.ok is True
    assert provider.state_calls
    assert provider.state_calls[-1][1] is not None
