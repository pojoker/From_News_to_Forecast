import middle_east_flight_tracker as tracker
import pytest


def test_resolve_bounds_with_preset() -> None:
    bounds = tracker.resolve_bounds("middle-east", None)
    assert bounds.min_lat == 12.0
    assert bounds.max_lat == 42.0
    assert bounds.min_lon == 24.0
    assert bounds.max_lon == 64.0


def test_resolve_bounds_with_custom_bbox() -> None:
    bounds = tracker.resolve_bounds("middle-east", [10.0, 20.0, 30.0, 40.0])
    assert bounds.min_lat == 10.0
    assert bounds.max_lat == 20.0
    assert bounds.min_lon == 30.0
    assert bounds.max_lon == 40.0


def test_parse_state_row_success() -> None:
    row = [
        "abc123",
        "UAE101 ",
        "United Arab Emirates",
        1710000000,
        1710000012,
        54.3773,
        24.4539,
        10668.0,
        False,
        250.0,
        90.0,
        0.0,
        None,
        10800.0,
        "1234",
        False,
        0,
    ]
    flight = tracker.parse_state_row(row)
    assert flight is not None
    assert flight.icao24 == "abc123"
    assert flight.callsign == "UAE101"
    assert flight.origin_country == "United Arab Emirates"
    assert flight.longitude == 54.3773
    assert flight.latitude == 24.4539
    assert flight.on_ground is False


def test_parse_state_row_rejects_invalid() -> None:
    assert tracker.parse_state_row(["only", "two"]) is None
    assert tracker.parse_state_row(["", "", "", 0, 0, 0, 0, 0, False, 0, 0, 0, None, 0, None, False, 0]) is None


def test_filter_and_limit_flights_country_filter() -> None:
    flights = [
        {"callsign": "UAE101", "origin_country": "United Arab Emirates"},
        {"callsign": "QTR800", "origin_country": "Qatar"},
        {"callsign": "AIC120", "origin_country": "India"},
    ]
    filtered = tracker.filter_and_limit_flights(flights, limit=10, country_filter="qat")
    assert len(filtered) == 1
    assert filtered[0]["callsign"] == "QTR800"


def test_build_api_payload_counts() -> None:
    snapshot = {
        "fetched_at": 1710000010,
        "source_time": 1710000000,
        "bounds": {"min_lat": 12.0, "max_lat": 42.0, "min_lon": 24.0, "max_lon": 64.0},
        "flights": [
            {"callsign": "UAE101", "origin_country": "United Arab Emirates"},
            {"callsign": "QTR800", "origin_country": "Qatar"},
            {"callsign": "AIC120", "origin_country": "India"},
        ],
    }
    payload = tracker.build_api_payload(snapshot, limit=2, country_filter="")
    assert payload["flight_count"] == 3
    assert payload["shown_count"] == 2
    assert len(payload["flights"]) == 2


def test_parse_provider_order_normalizes_and_deduplicates() -> None:
    providers = tracker.parse_provider_order("OpenSky, adsb_lol, opensky")
    assert providers == ["opensky", "adsb-lol"]


def test_parse_provider_order_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        tracker.parse_provider_order("opensky,foo")


def test_parse_adsblol_aircraft_unit_conversion() -> None:
    row = {
        "hex": "70c107",
        "flight": "OMA672  ",
        "alt_baro": 35000,
        "alt_geom": 36700,
        "gs": 525.0,
        "track": 93.6,
        "baro_rate": 640,
        "lat": 22.713959,
        "lon": 51.710549,
        "squawk": "3357",
        "spi": 0,
        "seen": 1.6,
        "seen_pos": 4.8,
    }
    flight = tracker.parse_adsblol_aircraft(row, now_ts=1710000000)
    assert flight is not None
    assert flight.icao24 == "70c107"
    assert flight.callsign == "OMA672"
    assert tracker._to_feet(flight.baro_altitude_m) == 35000
    assert tracker._to_feet(flight.geo_altitude_m) == 36700
    assert tracker._to_kmh(flight.velocity_mps) == 972
    assert flight.last_contact == 1709999998
    assert flight.time_position == 1709999995


def test_plan_adsblol_queries_respects_budget() -> None:
    bounds = tracker.Bounds(min_lat=12.0, max_lat=42.0, min_lon=24.0, max_lon=64.0)
    queries = tracker.plan_adsblol_queries(bounds, max_queries=8)
    assert 1 <= len(queries) <= 8
    assert all(len(item) == 3 for item in queries)


def test_fetch_states_with_failover_switches_to_backup(monkeypatch: pytest.MonkeyPatch) -> None:
    bounds = tracker.Bounds(min_lat=12.0, max_lat=42.0, min_lon=24.0, max_lon=64.0)

    def fake_open(*args, **kwargs):
        raise tracker.FlightDataError("opensky unavailable")

    def fake_adsb(*args, **kwargs):
        return 1710000001, []

    monkeypatch.setattr(tracker, "fetch_opensky_states", fake_open)
    monkeypatch.setattr(tracker, "fetch_adsblol_states", fake_adsb)

    source_time, flights, provider, failures = tracker.fetch_states_with_failover(
        bounds,
        timeout=10,
        providers=["opensky", "adsb-lol"],
        adsb_max_queries=10,
    )
    assert source_time == 1710000001
    assert flights == []
    assert provider == "adsb-lol"
    assert len(failures) == 1
    assert failures[0].startswith("opensky:")
