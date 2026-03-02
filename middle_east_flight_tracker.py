#!/usr/bin/env python3
"""
Real-time flight tracker for Middle East airspace.

Features:
1) One-shot CLI snapshot (once)
2) Continuous CLI monitor (watch)
3) Local web dashboard (serve)

Data source: OpenSky Network states API
https://opensky-network.org/apidoc/rest.html
"""

from __future__ import annotations

import argparse
import json
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
USER_AGENT = "MiddleEastFlightTracker/1.0"
FEET_PER_METER = 3.28084
KMH_PER_MPS = 3.6


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


def _to_kmh(mps: float | None) -> int | None:
    if mps is None:
        return None
    return int(round(mps * KMH_PER_MPS))


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
    request = Request(
        f"{OPENSKY_STATES_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise FlightDataError(f"OpenSky returned status code: {response.status}")
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raise FlightDataError(f"OpenSky request failed: HTTP {exc.code}") from exc
    except URLError as exc:
        raise FlightDataError(f"OpenSky network error: {exc.reason}") from exc
    except TimeoutError as exc:
        raise FlightDataError("OpenSky request timed out") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FlightDataError("OpenSky returned invalid JSON") from exc

    server_time = _safe_int(payload.get("time")) or int(time.time())
    rows = payload.get("states") or []
    flights: list[FlightState] = []
    for row in rows:
        parsed = parse_state_row(row)
        if parsed is not None:
            flights.append(parsed)

    flights.sort(key=lambda item: item.last_contact or 0, reverse=True)
    return server_time, flights


class MiddleEastFlightTracker:
    """Tracker wrapper with short trail history."""

    def __init__(self, bounds: Bounds, timeout: int = 15, history_points: int = 12) -> None:
        self.bounds = bounds
        self.timeout = timeout
        self.history_points = max(2, history_points)
        self._trails: dict[str, deque[dict[str, float | int]]] = defaultdict(
            lambda: deque(maxlen=self.history_points)
        )

    def snapshot(self) -> dict[str, Any]:
        source_time, flights = fetch_opensky_states(self.bounds, self.timeout)
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
    head = f"Middle East live snapshot | Updated: {title_ts} | Total {total}, Showing {shown}"
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
        meta.innerHTML = `Status: <span class="ok">online</span> | Flights: ${payload.flight_count || 0} | Shown: ${payload.shown_count || 0} | Updated: ${dt.toUTCString()} | Refresh: ${Math.round(refreshMs/1000)}s`;
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


def cmd_once(args: argparse.Namespace) -> int:
    bounds = resolve_bounds(args.region, args.bbox)
    tracker = MiddleEastFlightTracker(
        bounds=bounds,
        timeout=args.timeout,
        history_points=args.history_points,
    )
    snapshot = tracker.snapshot()
    print(render_cli_snapshot(snapshot, max_results=args.max_results))
    if args.json_out:
        write_json_output(args.json_out, snapshot)
        print(f"\nSnapshot written to: {args.json_out}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    bounds = resolve_bounds(args.region, args.bbox)
    tracker = MiddleEastFlightTracker(
        bounds=bounds,
        timeout=args.timeout,
        history_points=args.history_points,
    )

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
    tracker = MiddleEastFlightTracker(
        bounds=bounds,
        timeout=args.timeout,
        history_points=args.history_points,
    )
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
        description="Real-time Middle East flight tracker (OpenSky API)",
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
