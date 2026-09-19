#!/usr/bin/env python3
"""
weather_drift_tracker.py

Logs a daily (or multiple-times-daily) snapshot of the NWS forecast for
each tracked city and target date, so BAZI's weather module can see how
much the forecast moved as settlement approaches.

Tracks BOTH:
  - temp_high_f / temp_low_f  (for temperature bucket contracts)
  - rain_prob_pct             (for rain / no-rain contracts)

Env vars required (set as GitHub Actions secrets):
  SUPABASE_URL   - e.g. https://hgzmfdupcfeavsgqmcaa.supabase.co
  SUPABASE_KEY   - service role or anon key with insert rights on
                   weather_forecast_snapshots

Run with no args to snapshot "today" and "tomorrow" for all cities.
Pass --days N to also snapshot N days out (useful for the 5-7 day drift
you care about): e.g. `python weather_drift_tracker.py --days 0 1 2 3 5 7`
"""

import os
import sys
import json
import argparse
import datetime as dt
from typing import Optional

import requests

NWS_USER_AGENT = "bazi-weather-drift-tracker (contact: arbaazshakil-dev)"

# Downtown-ish coordinates for each tracked city.
CITIES = {
    "NYC":     (40.7128, -74.0060),
    "CHICAGO": (41.8781, -87.6298),
    "DENVER":  (39.7392, -104.9903),
    "MIAMI":   (25.7617, -80.1918),
    "LA":      (34.0522, -118.2437),
}

NWS_HEADERS = {"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"}


def get_grid_forecast_url(lat: float, lon: float) -> dict:
    """Resolve lat/lon -> NWS office/grid + forecast URL."""
    r = requests.get(
        f"https://api.weather.gov/points/{lat},{lon}",
        headers=NWS_HEADERS,
        timeout=15,
    )
    r.raise_for_status()
    props = r.json()["properties"]
    return {
        "forecast_url": props["forecast"],
        "office": props.get("gridId"),
        "grid_x": props.get("gridX"),
        "grid_y": props.get("gridY"),
    }


def fetch_periods(forecast_url: str) -> list:
    r = requests.get(forecast_url, headers=NWS_HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()["properties"]["periods"]


def find_periods_for_date(periods: list, target_date: dt.date) -> dict:
    """
    NWS 'forecast' periods alternate day/night, each ~12h, named things like
    'Tuesday', 'Tuesday Night'. Match by startTime's date, split into the
    daytime period (-> high) and nighttime period (-> low + rain prob is
    taken from whichever period we have; day period usually carries the
    afternoon rain chance which is what most Kalshi rain markets settle on).
    """
    day_period = None
    night_period = None
    for p in periods:
        start = dt.datetime.fromisoformat(p["startTime"]).date()
        if start != target_date:
            continue
        if p.get("isDaytime"):
            day_period = p
        else:
            night_period = p
    return {"day": day_period, "night": night_period}


def extract_metrics(day_period: Optional[dict], night_period: Optional[dict]) -> dict:
    temp_high = day_period["temperature"] if day_period else None
    temp_low = night_period["temperature"] if night_period else None

    # Prefer daytime precipitation chance (most rain/no-rain Kalshi markets
    # are framed as "does it rain during the day"); fall back to night.
    rain_prob = None
    for p in (day_period, night_period):
        if p and p.get("probabilityOfPrecipitation", {}).get("value") is not None:
            rain_prob = p["probabilityOfPrecipitation"]["value"]
            break

    short_forecast = (day_period or night_period or {}).get("shortForecast")
    raw = day_period or night_period

    return {
        "temp_high_f": temp_high,
        "temp_low_f": temp_low,
        "rain_prob_pct": rain_prob,
        "short_forecast": short_forecast,
        "raw_period": raw,
    }


def insert_snapshot(supabase_url: str, supabase_key: str, row: dict) -> None:
    resp = requests.post(
        f"{supabase_url}/rest/v1/weather_forecast_snapshots",
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        data=json.dumps(row, default=str),
        timeout=15,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"Supabase insert failed ({resp.status_code}): {resp.text}")


def run(days_out_list: list[int]) -> None:
    supabase_url = os.environ["SUPABASE_URL"].rstrip("/")
    supabase_key = os.environ["SUPABASE_KEY"]

    today = dt.date.today()
    now = dt.datetime.now(dt.timezone.utc)

    for city, (lat, lon) in CITIES.items():
        try:
            grid = get_grid_forecast_url(lat, lon)
            periods = fetch_periods(grid["forecast_url"])
        except Exception as e:
            print(f"[{city}] FAILED to fetch NWS data: {e}", file=sys.stderr)
            continue

        for offset in days_out_list:
            target_date = today + dt.timedelta(days=offset)
            found = find_periods_for_date(periods, target_date)
            if not found["day"] and not found["night"]:
                # NWS 'forecast' endpoint only covers ~7 days out; skip silently
                # if this city/offset isn't in range yet.
                continue

            metrics = extract_metrics(found["day"], found["night"])
            row = {
                "city": city,
                "target_date": target_date.isoformat(),
                "snapshot_time": now.isoformat(),
                "days_out": offset,
                "temp_high_f": metrics["temp_high_f"],
                "temp_low_f": metrics["temp_low_f"],
                "rain_prob_pct": metrics["rain_prob_pct"],
                "short_forecast": metrics["short_forecast"],
                "nws_office": grid["office"],
                "grid_x": grid["grid_x"],
                "grid_y": grid["grid_y"],
                "raw_period": metrics["raw_period"],
            }

            try:
                insert_snapshot(supabase_url, supabase_key, row)
                print(
                    f"[{city}] {target_date} (+{offset}d): "
                    f"high={metrics['temp_high_f']}F low={metrics['temp_low_f']}F "
                    f"rain={metrics['rain_prob_pct']}% -> logged"
                )
            except Exception as e:
                print(f"[{city}] {target_date} insert FAILED: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 5, 7],
        help="Days-out offsets to snapshot (default: 0 1 2 3 5 7)",
    )
    args = parser.parse_args()
    run(args.days)


if __name__ == "__main__":
    main()
