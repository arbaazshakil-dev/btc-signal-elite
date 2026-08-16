"""BAZI V12.1 checkpoint tracker for KXBTC15M.

Runs separately from the main V12 worker. It records independent predictions
near 8m, 6m, 4m and 2m remaining, stores them in bazi_checkpoints, and resolves
each one from Kalshi's official result. It does NOT update the main bazi_model.
"""
from __future__ import annotations

import math
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone

import requests

CB = "https://api.exchange.coinbase.com"
KRAKEN = "https://api.kraken.com/0/public"
KALSHI = "https://external-api.kalshi.com/trade-api/v2"
SERIES = os.getenv("KALSHI_SERIES", "KXBTC15M")
PASSES = max(1, int(os.getenv("BAZI_PASSES", "90")))
PASS_INTERVAL = max(15, int(os.getenv("BAZI_PASS_INTERVAL", "20")))
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
HEADERS = {"User-Agent": "BAZI-V12.1-Checkpoint/1.0", "Accept": "application/json"}

CHECKPOINTS = {
    "8m": (450, 510),
    "6m": (330, 390),
    "4m": (210, 270),
    "2m": (90, 150),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def get(url, params=None):
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def sb_headers(prefer=None):
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


def sb_get(table, params=None):
    r = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", params=params, headers=sb_headers(), timeout=15)
    r.raise_for_status()
    return r.json()


def sb_post(table, payload):
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        json=payload,
        headers=sb_headers("return=representation"),
        timeout=15,
    )
    if r.status_code == 409:
        return []
    r.raise_for_status()
    return r.json() if r.content else []


def sb_patch(table, params, payload):
    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/{table}",
        params=params,
        json=payload,
        headers=sb_headers("return=representation"),
        timeout=15,
    )
    r.raise_for_status()
    return r.json() if r.content else []


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def market_expiry(market):
    return parse_time(market.get("close_time") or market.get("latest_expiration_time"))


def market_target(market):
    for key in ("floor_strike", "cap_strike"):
        try:
            v = float(market.get(key))
            if v > 1000:
                return v
        except (TypeError, ValueError):
            pass
    text = " ".join(str(market.get(k, "")) for k in ("functional_strike", "subtitle", "yes_sub_title", "title"))
    nums = re.findall(r"\$?([0-9]{2,3}(?:,[0-9]{3})+(?:\.[0-9]+)?)", text)
    return float(nums[-1].replace(",", "")) if nums else None


def yes_probability(market):
    vals = []
    for key in ("yes_bid_dollars", "yes_ask_dollars"):
        try:
            v = float(market.get(key))
            if 0 <= v <= 1:
                vals.append(v)
        except (TypeError, ValueError):
            pass
    if vals:
        return sum(vals) / len(vals)
    try:
        v = float(market.get("last_price_dollars"))
        return v if 0 <= v <= 1 else None
    except (TypeError, ValueError):
        return None


def ema(values, span):
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    out = float(values[0])
    for value in values[1:]:
        out = alpha * float(value) + (1.0 - alpha) * out
    return out


def stdev(values, default=0.0):
    return statistics.stdev(values) if len(values) >= 2 else default


def hret(closes, minutes):
    if len(closes) <= minutes or closes[-minutes - 1] <= 0:
        return 0.0
    return math.log(closes[-1] / closes[-minutes - 1])


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-clamp(x, -30, 30)))


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def market_snapshot():
    ticker = get(f"{CB}/products/BTC-USD/ticker")
    candles = get(f"{CB}/products/BTC-USD/candles", {"granularity": 60})
    book = get(f"{CB}/products/BTC-USD/book", {"level": 2})

    rows = sorted(candles, key=lambda x: x[0])[-61:]
    lows = [float(x[1]) for x in rows]
    highs = [float(x[2]) for x in rows]
    closes = [float(x[4]) for x in rows]
    rets = [math.log(b / a) for a, b in zip(closes, closes[1:]) if a > 0 and b > 0]

    bids = book.get("bids", [])
    asks = book.get("asks", [])
    bid_depth = sum(float(x[1]) for x in bids[:25])
    ask_depth = sum(float(x[1]) for x in asks[:25])
    imbalance = clamp((bid_depth - ask_depth) / max(bid_depth + ask_depth, 1e-9), -1, 1)
    best_bid = float(bids[0][0]) if bids else float(ticker["price"])
    best_ask = float(asks[0][0]) if asks else float(ticker["price"])
    spread_bps = 10000.0 * max(0.0, best_ask - best_bid) / max((best_bid + best_ask) / 2.0, 1e-9)

    try:
        k = get(f"{KRAKEN}/Ticker", {"pair": "XBTUSD"})
        kraken = float(next(iter(k["result"].values()))["c"][0])
    except Exception:
        kraken = None

    return {
        "cb": float(ticker["price"]),
        "kraken": kraken,
        "lows": lows,
        "highs": highs,
        "closes": closes,
        "rets": rets,
        "book": imbalance,
        "spread_bps": spread_bps,
    }


def classify_checkpoint(seconds):
    for label, (lo, hi) in CHECKPOINTS.items():
        if lo <= seconds <= hi:
            return label
    return None


def make_prediction(market, snap, seconds):
    closes = snap["closes"]
    rets = snap["rets"]
    vol_short = stdev(rets[-10:], 0.0006)
    vol_long = stdev(rets[-30:], vol_short)
    vol = clamp(0.65 * vol_short + 0.35 * vol_long, 0.00015, 0.0045)

    m1 = clamp(hret(closes, 1) / max(vol, 1e-9), -3, 3)
    m3 = clamp(hret(closes, 3) / max(vol * math.sqrt(3), 1e-9), -3, 3)
    m5 = clamp(hret(closes, 5) / max(vol * math.sqrt(5), 1e-9), -3, 3)
    m10 = clamp(hret(closes, 10) / max(vol * math.sqrt(10), 1e-9), -3, 3)

    ema_gap = clamp((ema(closes[-20:], 5) - ema(closes[-40:], 15)) / max(snap["cb"] * vol, 1e-9), -3, 3)
    lo15 = min(snap["lows"][-15:])
    hi15 = max(snap["highs"][-15:])
    range_pos = clamp(2.0 * ((snap["cb"] - lo15) / max(hi15 - lo15, 1e-9)) - 1.0, -1, 1)

    target = market_target(market)
    kp = yes_probability(market)
    sigma = max(snap["cb"] * vol * math.sqrt(max(seconds, 1) / 60.0), snap["cb"] * 0.00008)
    gap_vol = 0.0 if target is None else clamp((snap["cb"] - target) / sigma, -5, 5)

    kraken_basis = 0.0 if snap["kraken"] is None else clamp((snap["kraken"] - snap["cb"]) / max(snap["cb"] * vol, 1e-9), -3, 3)
    drift_z = clamp(0.28*m3 + 0.12*m5 + 0.10*m10 + 0.12*ema_gap + 0.10*snap["book"] + 0.06*kraken_basis, -0.9, 0.9)
    p_target = normal_cdf(gap_vol + drift_z) if target is not None else 0.5
    p_trend = sigmoid(0.30*m3 + 0.18*m5 + 0.11*m10 + 0.15*ema_gap + 0.10*snap["book"] + 0.05*kraken_basis + 0.05*range_pos)
    p_up = 0.74*p_target + 0.26*p_trend if target is not None else p_trend
    if kp is not None:
        p_up = 0.92*p_up + 0.08*kp

    cap = 0.94
    if target is None:
        cap = min(cap, 0.80)
    if snap["kraken"] is None:
        cap = min(cap, 0.86)
    if snap["spread_bps"] > 5:
        cap = min(cap, 0.83)

    p_up = clamp(p_up, 1.0-cap, cap)
    side = "UP" if p_up >= 0.5 else "DOWN"
    confidence = p_up if side == "UP" else 1.0-p_up
    decision = "CONFIDENT" if confidence >= 0.80 and target is not None and snap["spread_bps"] <= 5 else ("LEAN" if confidence >= 0.64 else "WAIT")

    features = {
        "gap_vol": round(gap_vol, 6),
        "mom_1": round(m1, 6),
        "mom_3": round(m3, 6),
        "mom_5": round(m5, 6),
        "mom_10": round(m10, 6),
        "ema_gap": round(ema_gap, 6),
        "range_pos": round(range_pos, 6),
        "book": round(snap["book"], 6),
        "spread_bps": round(snap["spread_bps"], 6),
        "kraken_basis": round(kraken_basis, 6),
    }
    return side, confidence, decision, features, target, kp


def already_logged(ticker, checkpoint):
    rows = sb_get("bazi_checkpoints", {
        "market_ticker": f"eq.{ticker}",
        "checkpoint": f"eq.{checkpoint}",
        "select": "id",
        "limit": "1",
    })
    return bool(rows)


def discover_and_predict():
    data = get(f"{KALSHI}/markets", {"series_ticker": SERIES, "status": "open", "limit": 100})
    now = datetime.now(timezone.utc)
    candidates = []
    for market in data.get("markets", []):
        expiry = market_expiry(market)
        if not expiry:
            continue
        seconds = int((expiry - now).total_seconds())
        checkpoint = classify_checkpoint(seconds)
        if checkpoint:
            candidates.append((seconds, checkpoint, market, expiry))

    pending = [x for x in candidates if not already_logged(x[2].get("ticker", ""), x[1])]
    if not pending:
        print("No new checkpoint to record this pass.")
        return

    snap = market_snapshot()
    for seconds, checkpoint, market, expiry in pending:
        ticker = market.get("ticker")
        if not ticker:
            continue
        side, confidence, decision, features, target, kp = make_prediction(market, snap, seconds)
        payload = {
            "market_ticker": ticker,
            "series_ticker": SERIES,
            "checkpoint": checkpoint,
            "seconds_remaining": seconds,
            "expiry_at": expiry.isoformat(),
            "target": target,
            "coinbase_now": snap["cb"],
            "kraken_now": snap["kraken"],
            "kalshi_yes_probability": kp,
            "predicted_side": side,
            "probability": round(confidence, 6),
            "decision": decision,
            "features": features,
            "model_version": "v12.1-checkpoint",
        }
        created = sb_post("bazi_checkpoints", payload)
        print(("Logged" if created else "Already logged"), ticker, checkpoint, decision, side, f"{confidence:.1%}", f"{seconds}s_to_expiry")


def resolve_finished():
    now = datetime.now(timezone.utc).isoformat()
    rows = sb_get("bazi_checkpoints", {
        "resolved_at": "is.null",
        "expiry_at": f"lt.{now}",
        "select": "id,market_ticker,predicted_side,checkpoint",
    })
    for row in rows:
        try:
            market = get(f"{KALSHI}/markets/{row['market_ticker']}").get("market", {})
            result = str(market.get("result", "")).lower()
            if result not in ("yes", "no"):
                continue
            outcome = "UP" if result == "yes" else "DOWN"
            changed = sb_patch("bazi_checkpoints", {"id": f"eq.{row['id']}", "resolved_at": "is.null"}, {
                "outcome_side": outcome,
                "correct": outcome == row["predicted_side"],
                "resolution_source": "KALSHI_OFFICIAL_RESULT",
                "resolved_at": datetime.now(timezone.utc).isoformat(),
            })
            if changed:
                print("Resolved", row["market_ticker"], row["checkpoint"], outcome)
        except Exception as exc:
            print("Resolution retry later", row["market_ticker"], exc)


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    failures = 0
    for pass_no in range(1, PASSES + 1):
        print(f"BAZI V12.1 checkpoint pass {pass_no}/{PASSES} at {datetime.now(timezone.utc).isoformat()}")
        try:
            resolve_finished()
            discover_and_predict()
        except Exception as exc:
            failures += 1
            print(f"Pass {pass_no} failed: {exc}", file=sys.stderr)
        if pass_no < PASSES:
            time.sleep(PASS_INTERVAL)
    if failures == PASSES:
        raise RuntimeError("All BAZI V12.1 checkpoint passes failed")


if __name__ == "__main__":
    main()
