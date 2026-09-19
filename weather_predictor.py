import requests
import math
from datetime import datetime, timezone
from supabase import create_client
import os

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

CITIES = {
    "NYC":     {"kalshi_series": "KXHIGHNY",  "nws_station": "KNYC", "bias": -1.0, "std": 3.0},
    "CHICAGO": {"kalshi_series": "KXHIGHCHI", "nws_station": "KMDW", "bias": 0.0,  "std": 3.0},
    "DENVER":  {"kalshi_series": "KXHIGHDEN", "nws_station": "KDEN", "bias": 0.0,  "std": 5.0},
    "MIAMI":   {"kalshi_series": "KXHIGHMIA", "nws_station": "KMIA", "bias": -3.0, "std": 2.5},
    "LA":      {"kalshi_series": "KXHIGHLAX", "nws_station": "KLAX", "bias": 0.0,  "std": 2.5},
}

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

    # First daytime period = today's high
    for p in periods:
        if p["isDaytime"]:
            return p["temperature"]
    return periods[0]["temperature"]

def normal_cdf(x, mean, std):
    return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))

def bracket_probability(mean, std, floor_strike, cap_strike):
    """Probability temp falls within [floor, cap] under a Normal(mean, std) model."""
    if floor_strike is None:
        return normal_cdf(cap_strike + 0.5, mean, std)
    if cap_strike is None:
        return 1 - normal_cdf(floor_strike - 0.5, mean, std)
    return normal_cdf(cap_strike + 0.5, mean, std) - normal_cdf(floor_strike - 0.5, mean, std)

def get_kalshi_markets(series_ticker):
    """Fetch today's active markets for a given Kalshi series."""
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={series_ticker}&status=open"
    r = requests.get(url)
    r.raise_for_status()
    return r.json().get("markets", [])

def get_todays_event_ticker(series_ticker):
    """Build today's event_ticker, e.g. KXHIGHNY-26SEP19, to filter markets to today only."""
    today = datetime.now(timezone.utc)
    date_part = today.strftime("%y%b%d").upper()  # e.g. 26SEP19
    return f"{series_ticker}-{date_part}"

def run():
    results = []
    for city, cfg in CITIES.items():
        try:
            raw_forecast = get_nws_forecast(cfg["nws_station"])
            corrected_mean = raw_forecast + cfg["bias"]
            std = cfg["std"]

            all_markets = get_kalshi_markets(cfg["kalshi_series"])

            # Only keep today's event for this city (skip tomorrow/other days mixed into the same series)
            todays_event = get_todays_event_ticker(cfg["kalshi_series"])
            markets = [m for m in all_markets if m.get("event_ticker") == todays_event]

            if not markets:
                print(f"No markets found for {city} matching event {todays_event} (found {len(all_markets)} total in series)")
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
