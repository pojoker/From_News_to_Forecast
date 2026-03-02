#!/usr/bin/env python3
"""
Real-time flight tracker for Middle East airspace.

Features:
1) One-shot CLI snapshot (once)
2) Continuous CLI monitor (watch)
3) Local web dashboard (serve)

Data sources:
- OpenSky Network states API
- adsb.lol public API
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
ADSB_LOL_POINT_URL = "https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{radius_nm}"
USER_AGENT = "MiddleEastFlightTracker/1.0"
FEET_PER_METER = 3.28084
METER_PER_FOOT = 0.3048
KMH_PER_MPS = 3.6
MPS_PER_KNOT = 0.514444
MPS_PER_FPM = 0.00508

SUPPORTED_PROVIDERS = ("opensky", "adsb-lol")


class FlightDataError(RuntimeError):
    """Raised when flight data fetch fails."""


@dataclass(frozen=True)
class Bounds:
    """Geographic bounding box."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def to_query_params(self) -> dict[str, str]:
        return {
            "lamin": f"{self.min_lat:.4f}",
            "lamax": f"{self.max_lat:.4f}",
            "lomin": f"{self.min_lon:.4f}",
            "lomax": f"{self.max_lon:.4f}",
        }


REGION_PRESETS: dict[str, Bounds] = {
    # Covers Gulf, Levant, Arabian Peninsula, Iran, Iraq, Syria.
    "middle-east": Bounds(min_lat=12.0, max_lat=42.0, min_lon=24.0, max_lon=64.0),
    # Core Gulf area.
    "gulf": Bounds(min_lat=18.0, max_lat=33.0, min_lon=44.0, max_lon=60.0),
    # East Mediterranean + Levant.
    "levant": Bounds(min_lat=28.0, max_lat=38.5, min_lon=31.0, max_lon=40.5),
}


@dataclass
class FlightState:
    icao24: str
    callsign: str
    origin_country: str
    time_position: int | None
    last_contact: int | None
    longitude: float | None
    latitude: float | None
    baro_altitude_m: float | None
    on_ground: bool
    velocity_mps: float | None
    true_track_deg: float | None
    vertical_rate_mps: float | None
    geo_altitude_m: float | None
    squawk: str | None
    spi: bool
    position_source: int | None

    def to_payload(self, now_ts: int, trail: list[dict[str, float | int]]) -> dict[str, Any]:
        return {
            "icao24": self.icao24,
            "callsign": self.callsign.strip() or "N/A",
            "origin_country": self.origin_country.strip() or "Unknown",
            "latitude": self.latitude,
            "longitude": self.longitude,
            "baro_altitude_m": self.baro_altitude_m,
            "baro_altitude_ft": _to_feet(self.baro_altitude_m),
            "geo_altitude_m": self.geo_altitude_m,
            "geo_altitude_ft": _to_feet(self.geo_altitude_m),
            "velocity_mps": self.velocity_mps,
            "velocity_kmh": _to_kmh(self.velocity_mps),
            "true_track_deg": self.true_track_deg,
            "vertical_rate_mps": self.vertical_rate_mps,
            "on_ground": self.on_ground,
            "last_contact": self.last_contact,
            "last_seen_seconds": _safe_age_seconds(self.last_contact, now_ts),
            "position_source": self.position_source,
            "squawk": self.squawk,
            "spi": self.spi,
            "trail": trail,
        }


def _to_feet(meters: float | None) -> int | None:
    if meters is None:
        return None
    return int(round(meters * FEET_PER_METER))


def _feet_to_meters(feet: float | None) -> float | None:
    if feet is None:
        return None
    return float(feet) * METER_PER_FOOT


def _to_kmh(mps: float | None) -> int | None:
    if mps is None:
        return None
    return int(round(mps * KMH_PER_MPS))


def _knots_to_mps(knots: float | None) -> float | None:
    if knots is None:
        return None
    return float(knots) * MPS_PER_KNOT


def _fpm_to_mps(feet_per_minute: float | None) -> float | None:
    if feet_per_minute is None:
        return None
    return float(feet_per_minute) * MPS_PER_FPM


def _safe_age_seconds(last_contact: int | None, now_ts: int) -> int | None:
    if last_contact is None:
        return None
    return max(0, now_ts - last_contact)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_provider_order(raw_value: str) -> list[str]:
    if not raw_value.strip():
        raise ValueError("--providers cannot be empty")

    normalized: list[str] = []
    seen: set[str] = set()
    for token in raw_value.split(","):
        provider = token.strip().lower().replace("_", "-")
        if provider == "adsblol":
            provider = "adsb-lol"
        if provider not in SUPPORTED_PROVIDERS:
            supported = ", ".join(SUPPORTED_PROVIDERS)
            raise ValueError(f"Unsupported provider '{provider}'. Supported: {supported}")
        if provider not in seen:
            normalized.append(provider)
            seen.add(provider)

    if not normalized:
        raise ValueError("--providers must include at least one valid provider")
    return normalized


def request_json(url: str, *, timeout: int, provider: str) -> dict[str, Any]:
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT},
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise FlightDataError(f"{provider} returned status code: {response.status}")
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raise FlightDataError(f"{provider} request failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise FlightDataError(f"{provider} network error: {exc.reason}") from exc
    except TimeoutError as exc:
        raise FlightDataError(f"{provider} request timed out") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FlightDataError(f"{provider} returned invalid JSON") from exc


def parse_state_row(row: list[Any]) -> FlightState | None:
    """Parse one row from OpenSky states data."""
    if not isinstance(row, list) or len(row) < 17:
        return None

    icao24 = str(row[0] or "").strip()
    if not icao24:
        return None

    return FlightState(
        icao24=icao24,
        callsign=str(row[1] or "").strip(),
        origin_country=str(row[2] or "").strip(),
        time_position=_safe_int(row[3]),
        last_contact=_safe_int(row[4]),
        longitude=_safe_float(row[5]),
        latitude=_safe_float(row[6]),
        baro_altitude_m=_safe_float(row[7]),
        on_ground=bool(row[8]),
        velocity_mps=_safe_float(row[9]),
        true_track_deg=_safe_float(row[10]),
        vertical_rate_mps=_safe_float(row[11]),
        geo_altitude_m=_safe_float(row[13]),
        squawk=str(row[14]).strip() if row[14] is not None else None,
        spi=bool(row[15]),
        position_source=_safe_int(row[16]),
    )


def fetch_opensky_states(bounds: Bounds, timeout: int) -> tuple[int, list[FlightState]]:
    """Fetch live flight states from OpenSky in the given bounds."""
    query = urlencode(bounds.to_query_params())
    payload = request_json(
        f"{OPENSKY_STATES_URL}?{query}",
        timeout=timeout,
        provider="OpenSky",
    )

    server_time = _safe_int(payload.get("time")) or int(time.time())
    rows = payload.get("states") or []
    flights: list[FlightState] = []
    for row in rows:
        parsed = parse_state_row(row)
        if parsed is not None:
            flights.append(parsed)

    flights.sort(key=lambda item: item.last_contact or 0, reverse=True)
    return server_time, flights


def _is_inside_bounds(lat: float | None, lon: float | None, bounds: Bounds) -> bool:
    if lat is None or lon is None:
        return True
    return bounds.min_lat <= lat <= bounds.max_lat and bounds.min_lon <= lon <= bounds.max_lon


def _bbox_size_km(bounds: Bounds) -> tuple[float, float]:
    mid_lat = (bounds.min_lat + bounds.max_lat) / 2.0
    lat_km = abs(bounds.max_lat - bounds.min_lat) * 111.32
    lon_km = abs(bounds.max_lon - bounds.min_lon) * 111.32 * max(
        0.2, math.cos(math.radians(mid_lat))
    )
    return max(1e-6, lat_km), max(1e-6, lon_km)


def _required_radius_nm(height_km: float, width_km: float, rows: int, cols: int) -> float:
    step_h = height_km / (rows - 1) if rows > 1 else height_km
    step_w = width_km / (cols - 1) if cols > 1 else width_km
    # Extra margin keeps edge/cell overlap more stable.
    margin_km = 35.0
    return (0.5 * math.hypot(step_h, step_w) + margin_km) / 1.852


def plan_adsblol_queries(
    bounds: Bounds,
    *,
    max_queries: int,
    max_radius_nm: int = 250,
) -> list[tuple[float, float, int]]:
    budget = max(1, min(max_queries, 64))
    height_km, width_km = _bbox_size_km(bounds)

    best_cover: tuple[int, float, int, int] | None = None
    best_effort: tuple[int, float, int, int] | None = None
    for rows in range(1, budget + 1):
        max_cols = budget // rows
        for cols in range(1, max_cols + 1):
            query_count = rows * cols
            radius_nm = _required_radius_nm(height_km, width_km, rows, cols)
            candidate = (query_count, radius_nm, rows, cols)

            if radius_nm <= max_radius_nm:
                if (
                    best_cover is None
                    or candidate[0] < best_cover[0]
                    or (candidate[0] == best_cover[0] and candidate[1] < best_cover[1])
                ):
                    best_cover = candidate
            elif best_effort is None or candidate[1] < best_effort[1]:
                best_effort = candidate

    selected = best_cover or best_effort
    if selected is None:
        center_lat = (bounds.min_lat + bounds.max_lat) / 2.0
        center_lon = (bounds.min_lon + bounds.max_lon) / 2.0
        return [(center_lat, center_lon, max_radius_nm)]

    _, needed_radius_nm, rows, cols = selected
    radius_nm = max(40, min(max_radius_nm, int(math.ceil(needed_radius_nm))))

    if rows == 1:
        lat_values = [(bounds.min_lat + bounds.max_lat) / 2.0]
    else:
        lat_step = (bounds.max_lat - bounds.min_lat) / (rows - 1)
        lat_values = [bounds.min_lat + (lat_step * i) for i in range(rows)]

    if cols == 1:
        lon_values = [(bounds.min_lon + bounds.max_lon) / 2.0]
    else:
        lon_step = (bounds.max_lon - bounds.min_lon) / (cols - 1)
        lon_values = [bounds.min_lon + (lon_step * j) for j in range(cols)]

    queries: list[tuple[float, float, int]] = []
    for lat in lat_values:
        for lon in lon_values:
            queries.append((round(lat, 4), round(lon, 4), radius_nm))
    return queries


def parse_adsblol_aircraft(row: dict[str, Any], now_ts: int) -> FlightState | None:
    if not isinstance(row, dict):
        return None

    icao24 = str(row.get("hex") or "").strip().lower()
    if not icao24:
        return None

    callsign = str(row.get("flight") or "").strip()
    if not callsign:
        callsign = str(row.get("r") or "").strip()

    lat = _safe_float(row.get("lat"))
    lon = _safe_float(row.get("lon"))

    alt_baro_raw = row.get("alt_baro")
    baro_altitude_m: float | None = None
    on_ground = False
    if isinstance(alt_baro_raw, str) and alt_baro_raw.strip().lower() == "ground":
        on_ground = True
    else:
        baro_altitude_m = _feet_to_meters(_safe_float(alt_baro_raw))

    geo_altitude_m = _feet_to_meters(_safe_float(row.get("alt_geom")))
    speed_mps = _knots_to_mps(_safe_float(row.get("gs")))
    true_track_deg = _safe_float(row.get("track"))
    geom_rate_fpm = _safe_float(row.get("geom_rate"))
    if geom_rate_fpm is None:
        geom_rate_fpm = _safe_float(row.get("baro_rate"))
    vertical_rate_mps = _fpm_to_mps(geom_rate_fpm)

    if not on_ground and (_safe_float(row.get("gs")) or 0.0) <= 1.0 and baro_altitude_m is None:
        on_ground = True

    seen_seconds = _safe_float(row.get("seen"))
    seen_pos_seconds = _safe_float(row.get("seen_pos"))
    last_contact = now_ts - int(round(seen_seconds)) if seen_seconds is not None else now_ts
    time_position = (
        now_ts - int(round(seen_pos_seconds)) if seen_pos_seconds is not None else last_contact
    )

    squawk = str(row.get("squawk")).strip() if row.get("squawk") is not None else None
    spi = bool(row.get("spi"))

    return FlightState(
        icao24=icao24,
        callsign=callsign,
        origin_country="Unknown",
        time_position=max(0, time_position),
        last_contact=max(0, last_contact),
        longitude=lon,
        latitude=lat,
        baro_altitude_m=baro_altitude_m,
        on_ground=on_ground,
        velocity_mps=speed_mps,
        true_track_deg=true_track_deg,
        vertical_rate_mps=vertical_rate_mps,
        geo_altitude_m=geo_altitude_m,
        squawk=squawk,
        spi=spi,
        position_source=None,
    )


def fetch_adsblol_states(
    bounds: Bounds,
    timeout: int,
    *,
    max_queries: int = 36,
) -> tuple[int, list[FlightState]]:
    queries = plan_adsblol_queries(bounds, max_queries=max_queries)
    source_time = int(time.time())
    merged: dict[str, FlightState] = {}
    errors: list[str] = []

    for lat, lon, radius_nm in queries:
        url = ADSB_LOL_POINT_URL.format(
            lat=f"{lat:.4f}",
            lon=f"{lon:.4f}",
            radius_nm=radius_nm,
        )
        try:
            payload = request_json(url, timeout=timeout, provider="adsb.lol")
        except FlightDataError as exc:
            errors.append(str(exc))
            continue

        current_ts_ms = _safe_int(payload.get("now")) or _safe_int(payload.get("ctime"))
        current_ts = int(current_ts_ms / 1000) if current_ts_ms is not None else int(time.time())
        source_time = max(source_time, current_ts)

        aircraft = payload.get("ac") or []
        if not isinstance(aircraft, list):
            continue
        for row in aircraft:
            parsed = parse_adsblol_aircraft(row, now_ts=current_ts)
            if parsed is None:
                continue
            if not _is_inside_bounds(parsed.latitude, parsed.longitude, bounds):
                continue

            previous = merged.get(parsed.icao24)
            if previous is None or (parsed.last_contact or 0) >= (previous.last_contact or 0):
                merged[parsed.icao24] = parsed

    if not merged and errors and len(errors) == len(queries):
        raise FlightDataError(
            f"adsb.lol failed for all {len(queries)} requests. Last error: {errors[-1]}"
        )

    flights = sorted(merged.values(), key=lambda item: item.last_contact or 0, reverse=True)
    return source_time, flights


def fetch_states_with_failover(
    bounds: Bounds,
    *,
    timeout: int,
    providers: list[str],
    adsb_max_queries: int,
) -> tuple[int, list[FlightState], str, list[str]]:
    failures: list[str] = []
    for provider in providers:
        try:
            if provider == "opensky":
                source_time, flights = fetch_opensky_states(bounds, timeout=timeout)
            elif provider == "adsb-lol":
                source_time, flights = fetch_adsblol_states(
                    bounds,
                    timeout=timeout,
                    max_queries=adsb_max_queries,
                )
            else:
                failures.append(f"{provider}: unsupported provider")
                continue
            return source_time, flights, provider, failures
        except FlightDataError as exc:
            failures.append(f"{provider}: {exc}")

    joined = " | ".join(failures) if failures else "no provider configured"
    raise FlightDataError(f"All providers failed. {joined}")


class MiddleEastFlightTracker:
    """Tracker wrapper with short trail history."""

    def __init__(
        self,
        bounds: Bounds,
        timeout: int = 15,
        history_points: int = 12,
        providers: list[str] | None = None,
        adsb_max_queries: int = 36,
    ) -> None:
        self.bounds = bounds
        self.timeout = timeout
        self.history_points = max(2, history_points)
        self.providers = providers or ["opensky", "adsb-lol"]
        self.adsb_max_queries = max(1, min(adsb_max_queries, 64))
        self._trails: dict[str, deque[dict[str, float | int]]] = defaultdict(
            lambda: deque(maxlen=self.history_points)
        )

    def snapshot(self) -> dict[str, Any]:
        source_time, flights, source_provider, provider_failures = fetch_states_with_failover(
            self.bounds,
            timeout=self.timeout,
            providers=self.providers,
            adsb_max_queries=self.adsb_max_queries,
        )
        now_ts = int(time.time())
        active_icao: set[str] = set()

        for flight in flights:
            active_icao.add(flight.icao24)
            if flight.latitude is None or flight.longitude is None:
                continue
            self._trails[flight.icao24].append(
                {
                    "lat": round(flight.latitude, 5),
                    "lon": round(flight.longitude, 5),
                    "ts": flight.last_contact or source_time,
                }
            )

        self._evict_stale_trails(active_icao=active_icao, now_ts=now_ts)

        flight_payloads = [
            flight.to_payload(now_ts=now_ts, trail=list(self._trails.get(flight.icao24, ())))
            for flight in flights
        ]
        return {
            "fetched_at": now_ts,
            "source_time": source_time,
            "source_provider": source_provider,
            "provider_failures": provider_failures,
            "bounds": {
                "min_lat": self.bounds.min_lat,
                "max_lat": self.bounds.max_lat,
                "min_lon": self.bounds.min_lon,
                "max_lon": self.bounds.max_lon,
            },
            "flight_count": len(flight_payloads),
            "flights": flight_payloads,
        }

    def _evict_stale_trails(self, active_icao: set[str], now_ts: int) -> None:
        stale_after_seconds = 3600
        for icao24 in list(self._trails.keys()):
            trail = self._trails[icao24]
            if not trail:
                del self._trails[icao24]
                continue
            last_ts = int(trail[-1].get("ts", 0))
            if icao24 not in active_icao and now_ts - last_ts > stale_after_seconds:
                del self._trails[icao24]


def filter_and_limit_flights(
    flights: list[dict[str, Any]],
    *,
    limit: int,
    country_filter: str = "",
) -> list[dict[str, Any]]:
    country_filter = country_filter.strip().lower()
    if country_filter:
        filtered = [
            item for item in flights if country_filter in item.get("origin_country", "").lower()
        ]
    else:
        filtered = flights
    return filtered[: max(1, limit)]


def format_value(value: Any, placeholder: str = "-") -> str:
    if value is None:
        return placeholder
    return str(value)


def format_table(rows: list[list[str]], headers: list[str]) -> str:
    col_count = len(headers)
    widths = [len(header) for header in headers]

    for row in rows:
        for idx in range(col_count):
            widths[idx] = max(widths[idx], len(row[idx]))

    def _format_row(cells: list[str]) -> str:
        return " | ".join(cells[i].ljust(widths[i]) for i in range(col_count))

    divider = "-+-".join("-" * widths[i] for i in range(col_count))
    lines = [_format_row(headers), divider]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


def render_cli_snapshot(snapshot: dict[str, Any], max_results: int) -> str:
    flights = filter_and_limit_flights(snapshot.get("flights", []), limit=max_results)
    headers = [
        "Callsign",
        "Country",
        "Latitude",
        "Longitude",
        "Altitude(ft)",
        "Speed(km/h)",
        "Heading(deg)",
        "OnGround",
        "LastSeen(s)",
    ]
    rows: list[list[str]] = []
    for flight in flights:
        rows.append(
            [
                format_value(flight.get("callsign")),
                format_value(flight.get("origin_country")),
                f"{flight['latitude']:.4f}" if flight.get("latitude") is not None else "-",
                f"{flight['longitude']:.4f}" if flight.get("longitude") is not None else "-",
                format_value(flight.get("baro_altitude_ft")),
                format_value(flight.get("velocity_kmh")),
                (
                    f"{flight['true_track_deg']:.0f}"
                    if flight.get("true_track_deg") is not None
                    else "-"
                ),
                "Y" if flight.get("on_ground") else "N",
                format_value(flight.get("last_seen_seconds")),
            ]
        )

    title_ts = datetime.fromtimestamp(snapshot["fetched_at"], tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    total = snapshot.get("flight_count", 0)
    shown = len(flights)
    provider = snapshot.get("source_provider", "unknown")
    head = (
        f"Middle East live snapshot | Source: {provider} | Updated: {title_ts} "
        f"| Total {total}, Showing {shown}"
    )
    failures = snapshot.get("provider_failures") or []
    if failures:
        head = f"{head}\nFallback notes: {' ; '.join(failures)}"
    if not rows:
        return f"{head}\nNo flights to display."
    return f"{head}\n{format_table(rows, headers)}"


def write_json_output(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def clear_terminal() -> None:
    print("\033[2J\033[H", end="")


@dataclass
class SnapshotCache:
    tracker: MiddleEastFlightTracker
    min_refresh_seconds: float
    default_limit: int
    _snapshot: dict[str, Any] | None = None
    _last_fetch_monotonic: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_snapshot(self, force_refresh: bool = False) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            need_fetch = (
                force_refresh
                or self._snapshot is None
                or (now - self._last_fetch_monotonic) >= self.min_refresh_seconds
            )
            if need_fetch:
                self._snapshot = self.tracker.snapshot()
                self._last_fetch_monotonic = now
            return self._snapshot


def build_api_payload(
    snapshot: dict[str, Any],
    *,
    limit: int,
    country_filter: str,
) -> dict[str, Any]:
    flights = filter_and_limit_flights(
        snapshot.get("flights", []),
        limit=limit,
        country_filter=country_filter,
    )
    return {
        "fetched_at": snapshot.get("fetched_at"),
        "source_time": snapshot.get("source_time"),
        "source_provider": snapshot.get("source_provider"),
        "provider_failures": snapshot.get("provider_failures", []),
        "flight_count": len(snapshot.get("flights", [])),
        "shown_count": len(flights),
        "bounds": snapshot.get("bounds"),
        "country_filter": country_filter,
        "flights": flights,
    }


def _safe_positive_int(value: str, default: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def build_dashboard_html(bounds: Bounds, refresh_seconds: float) -> str:
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Middle East Live Flight Tracker</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #10141a;
      --panel: #161d26;
      --text: #e7edf5;
      --muted: #9fb0c3;
      --accent: #2ec4b6;
      --grid: #223142;
      --warn: #ff9f1c;
      --danger: #ff4d6d;
    }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 16px;
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 1.5rem;
    }
    .meta {
      color: var(--muted);
      margin-bottom: 16px;
      font-size: 0.95rem;
    }
    .panel {
      background: var(--panel);
      border: 1px solid #263649;
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 16px;
    }
    canvas {
      width: 100%;
      max-width: 100%;
      border-radius: 8px;
      border: 1px solid #2a3b50;
      background: #0c1016;
      display: block;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }
    th, td {
      border-bottom: 1px solid #27374a;
      text-align: left;
      padding: 8px 6px;
      white-space: nowrap;
    }
    th {
      color: #c6d5e6;
      position: sticky;
      top: 0;
      background: #1a2330;
    }
    .table-wrap {
      max-height: 360px;
      overflow: auto;
    }
    .ok { color: var(--accent); }
    .warn { color: var(--warn); }
    .danger { color: var(--danger); }
  </style>
</head>
<body>
  <div class="container">
    <h1>Middle East Live Flight Tracker</h1>
    <div id="meta" class="meta">Loading flight data...</div>
    <div class="panel">
      <canvas id="radar" width="1120" height="540"></canvas>
    </div>
    <div class="panel">
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Callsign</th>
              <th>Country</th>
              <th>Lat/Lon</th>
              <th>Altitude(ft)</th>
              <th>Speed(km/h)</th>
              <th>Heading</th>
              <th>OnGround</th>
              <th>LastSeen(s)</th>
            </tr>
          </thead>
          <tbody id="tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const bounds = __BOUNDS__;
    const refreshMs = __REFRESH_MS__;
    const canvas = document.getElementById("radar");
    const ctx = canvas.getContext("2d");
    const meta = document.getElementById("meta");
    const tbody = document.getElementById("tbody");

    function project(lat, lon) {
      const x = ((lon - bounds.min_lon) / (bounds.max_lon - bounds.min_lon)) * canvas.width;
      const y = canvas.height - ((lat - bounds.min_lat) / (bounds.max_lat - bounds.min_lat)) * canvas.height;
      return [x, y];
    }

    function drawGrid() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#0c1016";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.strokeStyle = "#223142";
      ctx.lineWidth = 1;
      const latSteps = 6;
      const lonSteps = 8;
      for (let i = 0; i <= latSteps; i++) {
        const y = (i / latSteps) * canvas.height;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
      }
      for (let i = 0; i <= lonSteps; i++) {
        const x = (i / lonSteps) * canvas.width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
      }

      ctx.fillStyle = "#7f91a8";
      ctx.font = "12px sans-serif";
      ctx.fillText(`Latitude ${bounds.min_lat} ~ ${bounds.max_lat}`, 10, 20);
      ctx.fillText(`Longitude ${bounds.min_lon} ~ ${bounds.max_lon}`, 10, 40);
    }

    function drawFlights(flights) {
      drawGrid();
      for (const flight of flights) {
        const trail = Array.isArray(flight.trail) ? flight.trail : [];
        if (trail.length > 1) {
          ctx.strokeStyle = "rgba(46,196,182,0.7)";
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          for (let i = 0; i < trail.length; i++) {
            const [x, y] = project(trail[i].lat, trail[i].lon);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }

        if (flight.latitude == null || flight.longitude == null) continue;
        const [x, y] = project(flight.latitude, flight.longitude);

        ctx.beginPath();
        ctx.fillStyle = flight.on_ground ? "#ff9f1c" : "#2ec4b6";
        ctx.arc(x, y, flight.on_ground ? 3 : 4, 0, Math.PI * 2);
        ctx.fill();

        const label = (flight.callsign || "N/A").trim();
        if (label) {
          ctx.fillStyle = "#e7edf5";
          ctx.font = "11px monospace";
          ctx.fillText(label, x + 6, y - 6);
        }
      }
    }

    function renderTable(flights) {
      tbody.innerHTML = "";
      for (const flight of flights) {
        const tr = document.createElement("tr");
        const cells = [
          (flight.callsign || "N/A").trim(),
          flight.origin_country || "Unknown",
          (flight.latitude == null || flight.longitude == null)
            ? "-"
            : `${flight.latitude.toFixed(4)}, ${flight.longitude.toFixed(4)}`,
          flight.baro_altitude_ft == null ? "-" : String(flight.baro_altitude_ft),
          flight.velocity_kmh == null ? "-" : String(flight.velocity_kmh),
          flight.true_track_deg == null ? "-" : `${Math.round(flight.true_track_deg)} deg`,
          flight.on_ground ? "Y" : "N",
          flight.last_seen_seconds == null ? "-" : String(flight.last_seen_seconds),
        ];

        for (const value of cells) {
          const td = document.createElement("td");
          td.textContent = value;
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
    }

    async function refresh() {
      try {
        const response = await fetch("/api/flights?limit=250");
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        drawFlights(payload.flights || []);
        renderTable(payload.flights || []);
        const dt = new Date((payload.fetched_at || 0) * 1000);
        const provider = payload.source_provider || "unknown";
        const fallback = (payload.provider_failures || []).join(" ; ");
        meta.innerHTML = `Status: <span class="ok">online</span> | Source: ${provider} | Flights: ${payload.flight_count || 0} | Shown: ${payload.shown_count || 0} | Updated: ${dt.toUTCString()} | Refresh: ${Math.round(refreshMs/1000)}s${fallback ? ` | Fallback: ${fallback}` : ""}`;
      } catch (error) {
        meta.innerHTML = `Status: <span class="danger">error</span> | ${error.message}`;
      }
    }

    drawGrid();
    refresh();
    setInterval(refresh, refreshMs);
  </script>
</body>
</html>
"""
    html = html.replace("__BOUNDS__", json.dumps(bounds.__dict__, ensure_ascii=False))
    html = html.replace("__REFRESH_MS__", str(max(1000, int(refresh_seconds * 1000))))
    return html


def make_handler(cache: SnapshotCache) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "MiddleEastFlightTracker/1.0"

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, status: int, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                page = build_dashboard_html(
                    bounds=cache.tracker.bounds,
                    refresh_seconds=cache.min_refresh_seconds,
                )
                self._send_html(HTTPStatus.OK, page)
                return

            if parsed.path == "/api/flights":
                params = parse_qs(parsed.query)
                limit = _safe_positive_int(
                    params.get("limit", [str(cache.default_limit)])[0],
                    cache.default_limit,
                )
                country_filter = params.get("country", [""])[0]
                force_refresh = params.get("force", ["0"])[0] == "1"
                try:
                    snapshot = cache.get_snapshot(force_refresh=force_refresh)
                except FlightDataError as exc:
                    if cache._snapshot is not None:
                        payload = build_api_payload(
                            cache._snapshot,
                            limit=limit,
                            country_filter=country_filter,
                        )
                        payload["warning"] = f"Serving cached snapshot: {exc}"
                        self._send_json(HTTPStatus.OK, payload)
                    else:
                        self._send_json(
                            HTTPStatus.SERVICE_UNAVAILABLE,
                            {"error": str(exc), "flights": []},
                        )
                    return

                payload = build_api_payload(
                    snapshot,
                    limit=limit,
                    country_filter=country_filter,
                )
                self._send_json(HTTPStatus.OK, payload)
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not Found"})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            # Skip access logs to keep terminal clean.
            return

    return DashboardHandler


def resolve_bounds(region: str, bbox: list[float] | None) -> Bounds:
    if bbox:
        if len(bbox) != 4:
            raise ValueError("--bbox requires 4 numbers: min_lat max_lat min_lon max_lon")
        min_lat, max_lat, min_lon, max_lon = bbox
        if min_lat >= max_lat or min_lon >= max_lon:
            raise ValueError("invalid bounds: min values must be lower than max values")
        return Bounds(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)
    return REGION_PRESETS[region]


def add_region_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--region",
        choices=sorted(REGION_PRESETS.keys()),
        default="middle-east",
        help="Region preset (default: middle-east)",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LAT", "MAX_LAT", "MIN_LON", "MAX_LON"),
        help="Custom bounds that override --region",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="API timeout in seconds",
    )
    parser.add_argument(
        "--history-points",
        type=int,
        default=12,
        help="Number of trail points to retain per flight",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="opensky,adsb-lol",
        help=(
            "Ordered provider list, comma-separated. "
            "Supported: opensky, adsb-lol (default: opensky,adsb-lol)"
        ),
    )
    parser.add_argument(
        "--adsb-max-queries",
        type=int,
        default=36,
        help="Maximum fallback requests used by adsb-lol for large areas",
    )


def build_tracker_from_args(args: argparse.Namespace, bounds: Bounds) -> MiddleEastFlightTracker:
    return MiddleEastFlightTracker(
        bounds=bounds,
        timeout=args.timeout,
        history_points=args.history_points,
        providers=parse_provider_order(args.providers),
        adsb_max_queries=args.adsb_max_queries,
    )


def cmd_once(args: argparse.Namespace) -> int:
    bounds = resolve_bounds(args.region, args.bbox)
    tracker = build_tracker_from_args(args, bounds)
    snapshot = tracker.snapshot()
    print(render_cli_snapshot(snapshot, max_results=args.max_results))
    if args.json_out:
        write_json_output(args.json_out, snapshot)
        print(f"\nSnapshot written to: {args.json_out}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    bounds = resolve_bounds(args.region, args.bbox)
    tracker = build_tracker_from_args(args, bounds)

    try:
        while True:
            try:
                snapshot = tracker.snapshot()
                clear_terminal()
                print(render_cli_snapshot(snapshot, max_results=args.max_results))
                if args.json_out:
                    write_json_output(args.json_out, snapshot)
            except FlightDataError as exc:
                clear_terminal()
                print(f"Data fetch failed: {exc}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped live tracking.")
        return 0


def cmd_serve(args: argparse.Namespace) -> int:
    bounds = resolve_bounds(args.region, args.bbox)
    tracker = build_tracker_from_args(args, bounds)
    cache = SnapshotCache(
        tracker=tracker,
        min_refresh_seconds=max(2.0, args.interval),
        default_limit=args.max_results,
    )
    handler_cls = make_handler(cache)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)

    print(
        f"Dashboard started: http://{args.host}:{args.port} "
        f"(refresh interval ~{max(2.0, args.interval):.1f}s)"
    )
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time Middle East flight tracker with provider failover",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    once_parser = subparsers.add_parser("once", help="Fetch a single snapshot")
    add_region_args(once_parser)
    once_parser.add_argument("--max-results", type=int, default=20, help="Max rows in CLI output")
    once_parser.add_argument("--json-out", type=str, default="", help="Write full snapshot as JSON")
    once_parser.set_defaults(func=cmd_once)

    watch_parser = subparsers.add_parser("watch", help="Continuously refresh in terminal")
    add_region_args(watch_parser)
    watch_parser.add_argument("--interval", type=float, default=15.0, help="Refresh interval in seconds")
    watch_parser.add_argument("--max-results", type=int, default=20, help="Max rows in CLI output")
    watch_parser.add_argument("--json-out", type=str, default="", help="Write JSON on each refresh")
    watch_parser.set_defaults(func=cmd_watch)

    serve_parser = subparsers.add_parser("serve", help="Start local dashboard server")
    add_region_args(serve_parser)
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Listen host")
    serve_parser.add_argument("--port", type=int, default=8787, help="Listen port")
    serve_parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Minimum backend refresh interval in seconds",
    )
    serve_parser.add_argument("--max-results", type=int, default=250, help="Max flights in API response")
    serve_parser.set_defaults(func=cmd_serve)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except FlightDataError as exc:
        print(f"Error: {exc}")
        return 2
    except ValueError as exc:
        print(f"Argument error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
