import requests
import math
from datetime import datetime, timezone, timedelta
from supabase import create_client
import os

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

CITIES = {
    "NYC":     {"kalshi_series": "KXHIGHNY",  "nws_station": "KNYC", "bias": -1.0, "default_std": 3.0},
    "CHICAGO": {"kalshi_series": "KXHIGHCHI", "nws_station": "KMDW", "bias": 0.0,  "default_std": 3.0},
    "DENVER":  {"kalshi_series": "KXHIGHDEN", "nws_station": "KDEN", "bias": 0.0,  "default_std": 5.0},
    "MIAMI":   {"kalshi_series": "KXHIGHMIA", "nws_station": "KMIA", "bias": -3.0, "default_std": 2.5},
    "LA":      {"kalshi_series": "KXHIGHLAX", "nws_station": "KLAX", "bias": 0.0,  "default_std": 2.5},
}

# ---------- Forecast pulling ----------

def get_nws_forecast(station_id):
    """Get today's forecasted high temp for a station via NWS gridpoint API."""
    points_url = f"https://api.weather.gov/stations/{station_id}"
    r = requests.get(points_url, headers={"User-Agent": "weather-betting-bot"})
    r.raise_for_status()
    coords = r.json()["geometry"]["coordinates"]  # [lon, lat]
    lon, lat = coords[0], coords[1]

    grid_url = f"https://api.weather.gov/points/{lat},{lon}"
    grid_r = requests.get(grid_url, headers={"User-Agent": "weather-betting-bot"})
    grid_r.raise_for_status()
    forecast_url = grid_r.json()["properties"]["forecast"]

    fc_r = requests.get(forecast_url, headers={"User-Agent": "weather-betting-bot"})
    fc_r.raise_for_status()
    periods = fc_r.json()["properties"]["periods"]

    for p in periods:
        if p["isDaytime"]:
            return p["temperature"]
    return periods[0]["temperature"]

# ---------- Probability math ----------

def normal_cdf(x, mean, std):
    return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))

def bracket_probability(mean, std, floor_strike, cap_strike):
    if floor_strike is None:
        return normal_cdf(cap_strike + 0.5, mean, std)
    if cap_strike is None:
        return 1 - normal_cdf(floor_strike - 0.5, mean, std)
    return normal_cdf(cap_strike + 0.5, mean, std) - normal_cdf(floor_strike - 0.5, mean, std)

# ---------- Kalshi market access ----------

def get_kalshi_markets(series_ticker, status="open"):
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={series_ticker}&status={status}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json().get("markets", [])

def get_event_ticker_for_date(series_ticker, dt):
    date_part = dt.strftime("%y%b%d").upper()  # e.g. 26SEP19
    return f"{series_ticker}-{date_part}"

def infer_settled_temp(finalized_markets):
    """
    Infer the actual settled high temperature from a set of finalized bracket
    markets for one event. Looks for the bracket where result == 'yes' and
    returns its midpoint (or the boundary value for open-ended brackets).
    """
    for m in finalized_markets:
        if m.get("result") == "yes":
            floor_strike = m.get("floor_strike")
            cap_strike = m.get("cap_strike")
            if floor_strike is not None and cap_strike is not None:
                return (floor_strike + cap_strike) / 2
            if floor_strike is not None:
                return floor_strike + 1  # "or above" bracket, approximate
            if cap_strike is not None:
                return cap_strike - 1  # "or below" bracket, approximate
    return None

# ---------- Calibration ----------

def record_yesterdays_outcome(city, series_ticker):
    """
    Look up yesterday's finalized markets for this city, infer the settled
    temperature, and log it against the forecast we made yesterday (if we
    have one stored) so we can calibrate our std going forward.
    """
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    event_ticker = get_event_ticker_for_date(series_ticker, yesterday)

    finalized = get_kalshi_markets(series_ticker, status="finalized")
    todays_finalized = [m for m in finalized if m.get("event_ticker") == event_ticker]

    if not todays_finalized:
        print(f"No finalized markets yet for {city} ({event_ticker}), skipping outcome recording")
        return

    actual_high = infer_settled_temp(todays_finalized)
    if actual_high is None:
        print(f"Could not infer settled temp for {city} ({event_ticker})")
        return

    # Find our stored forecast for yesterday's event
    existing = supabase.table("weather_predictions") \
        .select("raw_forecast, corrected_forecast, created_at") \
        .eq("city", city) \
        .like("ticker", f"{event_ticker}%") \
        .order("created_at", desc=False) \
        .limit(1) \
        .execute()

    if not existing.data:
        print(f"No stored forecast found for {city} ({event_ticker}), skipping")
        return

    forecasted_high = existing.data[0]["corrected_forecast"]
    error = actual_high - forecasted_high

    # Avoid duplicate outcome rows for the same city/date
    already_logged = supabase.table("weather_outcomes") \
        .select("id") \
        .eq("city", city) \
        .eq("event_date", yesterday.date().isoformat()) \
        .execute()

    if already_logged.data:
        print(f"Outcome already logged for {city} on {yesterday.date().isoformat()}")
        return

    supabase.table("weather_outcomes").insert({
        "city": city,
        "event_date": yesterday.date().isoformat(),
        "forecasted_high": forecasted_high,
        "actual_high": actual_high,
        "error": error,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()
    print(f"Logged outcome for {city}: forecast {forecasted_high}, actual {actual_high}, error {error}")

def get_calibrated_std(city, default_std, lookback_days=14, min_samples=5):
    """
    Compute the std of recent forecast errors for a city. Falls back to the
    default guess if there isn't enough history yet.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
    rows = supabase.table("weather_outcomes") \
        .select("error") \
        .eq("city", city) \
        .gte("event_date", cutoff) \
        .execute()

    errors = [r["error"] for r in rows.data if r["error"] is not None]

    if len(errors) < min_samples:
        return default_std, len(errors)

    mean_error = sum(errors) / len(errors)
    variance = sum((e - mean_error) ** 2 for e in errors) / len(errors)
    std = math.sqrt(variance)

    # Guard against a degenerate (near-zero) std from a lucky short streak
    std = max(std, 1.0)
    return std, len(errors)

# ---------- Main run ----------

def run():
    results = []
    for city, cfg in CITIES.items():
        try:
            # 1. Try to record yesterday's outcome (safe no-op if already logged or not ready)
            record_yesterdays_outcome(city, cfg["kalshi_series"])

            # 2. Get today's forecast
            raw_forecast = get_nws_forecast(cfg["nws_station"])
            corrected_mean = raw_forecast + cfg["bias"]

            # 3. Use calibrated std if we have enough history, else fall back to default
            std, sample_count = get_calibrated_std(city, cfg["default_std"])
            print(f"{city}: using std={std:.2f} (from {sample_count} historical samples)")

            all_markets = get_kalshi_markets(cfg["kalshi_series"])
            todays_event = get_event_ticker_for_date(cfg["kalshi_series"], datetime.now(timezone.utc))
            markets = [m for m in all_markets if m.get("event_ticker") == todays_event]

            if not markets:
                print(f"No markets found for {city} matching event {todays_event}")
                continue

            for m in markets:
                floor_strike = m.get("floor_strike")
                cap_strike = m.get("cap_strike")
                our_prob = bracket_probability(corrected_mean, std, floor_strike, cap_strike)

                yes_ask = float(m.get("yes_ask_dollars", 0) or 0)
                yes_bid = float(m.get("yes_bid_dollars", 0) or 0)
                market_prob = (yes_ask + yes_bid) / 2 if (yes_ask and yes_bid) else yes_ask

                edge = our_prob - market_prob

                row = {
                    "city": city,
                    "ticker": m["ticker"],
                    "raw_forecast": raw_forecast,
                    "corrected_forecast": corrected_mean,
                    "floor_strike": floor_strike,
                    "cap_strike": cap_strike,
                    "our_probability": round(our_prob, 4),
                    "market_probability": round(market_prob, 4),
                    "edge": round(edge, 4),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                results.append(row)
        except Exception as e:
            print(f"Error processing {city}: {e}")

    if results:
        supabase.table("weather_predictions").insert(results).execute()
        print(f"Inserted {len(results)} rows")
    else:
        print("No results to insert")

if __name__ == "__main__":
    run()
