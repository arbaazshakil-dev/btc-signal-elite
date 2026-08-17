"""BAZI V12 accuracy-focused unattended KXBTC15M learner.

Changes from V11:
- richer multi-horizon momentum and acceleration features
- EMA/VWAP/range/volume regime features
- short-vs-long volatility regime feature
- Coinbase top-of-book spread + depth imbalance
- Kraken basis confirmation
- target distance normalized by expected remaining volatility
- ensemble of target-distance, trend, mean-reversion, and learned components
- learned component is introduced gradually to reduce overfitting on a tiny sample
- conservative confidence calibration until enough resolved examples exist

Read-only market access: this worker contains no order endpoint and cannot trade.
Required environment: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY.
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
PASSES = max(1, int(os.getenv("BAZI_PASSES", "4")))
PASS_INTERVAL = max(15, int(os.getenv("BAZI_PASS_INTERVAL", "45")))
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

HEADERS = {"User-Agent": "BAZI-V12-Research/1.0", "Accept": "application/json"}

# Single threshold used both to penalize confidence on a wide spread and to
# gate the CONFIDENT decision, so the two checks can't disagree with each other.
MAX_TRADEABLE_SPREAD_BPS = 4.0

# Preferred prediction anchor: 8-12 minutes remaining. Predicting at a
# consistent point in the contract's life gives cleaner training examples
# than predicting whenever a poll happens to first see the contract.
PRIMARY_WINDOW = (480, 720)
FALLBACK_MIN_SECONDS = 60
FALLBACK_MAX_SECONDS = 900

FEATURES = (
    "gap_vol",
    "mom_1", "mom_3", "mom_5", "mom_10", "mom_15",
    "accel",
    "vol_ratio",
    "book",
    "spread_bps",
    "ema_gap",
    "vwap_gap",
    "range_pos",
    "volume_z",
    "kraken_agree",
    "kraken_basis",
    "kalshi_market",
    "time_frac",
)

SCALES = {
    "gap_vol": 3.0,
    "mom_1": 1.5, "mom_3": 1.5, "mom_5": 1.5, "mom_10": 1.5, "mom_15": 1.5,
    "accel": 1.5,
    "vol_ratio": 1.0,
    "book": 1.0,
    "spread_bps": 5.0,
    "ema_gap": 1.5,
    "vwap_gap": 1.5,
    "range_pos": 1.0,
    "volume_z": 2.0,
    "kraken_agree": 1.0,
    "kraken_basis": 2.0,
    "kalshi_market": 1.0,
    "time_frac": 1.0,
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def get(url, params=None):
    response = requests.get(url, params=params, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.json()


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
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        params=params,
        headers=sb_headers(),
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def sb_post(table, payload, prefer="return=representation"):
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{table}",
        json=payload,
        headers=sb_headers(prefer),
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
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def market_expiry(market):
    return parse_time(market.get("close_time") or market.get("latest_expiration_time"))


def market_target(market):
    for key in ("floor_strike", "cap_strike"):
        value = market.get(key)
        if value not in (None, ""):
            try:
                value = float(value)
                if value > 1000:
                    return value
            except (TypeError, ValueError):
                pass

    text = " ".join(
        str(market.get(k, ""))
        for k in ("functional_strike", "subtitle", "yes_sub_title", "title")
    )
    nums = re.findall(r"\$?([0-9]{2,3}(?:,[0-9]{3})+(?:\.[0-9]+)?)", text)
    return float(nums[-1].replace(",", "")) if nums else None


def _strike_value(market, key):
    value = market.get(key)
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 1000 else None


def classify_market(market):
    """Determine the contract's settlement condition and threshold price(s).

    Treating every market as a floor/"above" contract (the old behavior)
    silently mis-scores any market where YES actually means "below" or
    "within a range" -- this makes that distinction explicit instead of
    guessing.

    Returns (kind, info):
      "above"   info = floor price;  YES resolves if price >= floor
      "below"   info = cap price;    YES resolves if price <= cap
      "range"   info = (floor, cap); YES resolves if floor <= price <= cap
      "unknown" info = text-parsed price with unclear direction (rare)
    """
    floor = _strike_value(market, "floor_strike")
    cap = _strike_value(market, "cap_strike")

    if floor is not None and cap is not None:
        return "range", (floor, cap)
    if floor is not None:
        return "above", floor
    if cap is not None:
        return "below", cap

    text = " ".join(
        str(market.get(k, ""))
        for k in ("functional_strike", "subtitle", "yes_sub_title", "title")
    )
    nums = re.findall(r"\$?([0-9]{2,3}(?:,[0-9]{3})+(?:\.[0-9]+)?)", text)
    if nums:
        return "unknown", float(nums[-1].replace(",", ""))
    return "unknown", None


def yes_probability(market):
    vals = []
    for key in ("yes_bid_dollars", "yes_ask_dollars"):
        try:
            value = float(market.get(key))
            if 0 <= value <= 1:
                vals.append(value)
        except (TypeError, ValueError):
            pass

    if vals:
        return sum(vals) / len(vals)

    try:
        value = float(market.get("last_price_dollars"))
        return value if 0 <= value <= 1 else None
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


def safe_stdev(values, default=0.0):
    return statistics.stdev(values) if len(values) >= 2 else default


def zscore(value, values):
    if len(values) < 3:
        return 0.0
    sd = safe_stdev(values)
    if sd <= 1e-12:
        return 0.0
    return (value - statistics.mean(values)) / sd


def horizon_return(closes, minutes):
    if len(closes) <= minutes or closes[-minutes - 1] <= 0:
        return 0.0
    return math.log(closes[-1] / closes[-minutes - 1])


def market_snapshot():
    ticker = get(f"{CB}/products/BTC-USD/ticker")
    candles = get(f"{CB}/products/BTC-USD/candles", {"granularity": 60})
    book = get(f"{CB}/products/BTC-USD/book", {"level": 2})

    rows = sorted(candles, key=lambda x: x[0])[-61:]
    # Coinbase candle row: [time, low, high, open, close, volume]
    lows = [float(x[1]) for x in rows]
    highs = [float(x[2]) for x in rows]
    opens = [float(x[3]) for x in rows]
    closes = [float(x[4]) for x in rows]
    volumes = [float(x[5]) for x in rows]

    rets = [
        math.log(b / a)
        for a, b in zip(closes, closes[1:])
        if a > 0 and b > 0
    ]

    bids = book.get("bids", [])
    asks = book.get("asks", [])
    bid_depth = sum(float(x[1]) for x in bids[:25])
    ask_depth = sum(float(x[1]) for x in asks[:25])
    imbalance = clamp(
        (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1e-9), -1, 1
    )

    best_bid = float(bids[0][0]) if bids else float(ticker["price"])
    best_ask = float(asks[0][0]) if asks else float(ticker["price"])
    mid = max((best_bid + best_ask) / 2.0, 1e-9)
    spread_bps = 10000.0 * max(0.0, best_ask - best_bid) / mid

    try:
        k = get(f"{KRAKEN}/Ticker", {"pair": "XBTUSD"})
        kraken = float(next(iter(k["result"].values()))["c"][0])
    except Exception:
        kraken = None

    return {
        "cb": float(ticker["price"]),
        "kraken": kraken,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "rets": rets,
        "book": imbalance,
        "spread_bps": spread_bps,
    }


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-clamp(x, -30, 30)))


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_model():
    rows = sb_get("bazi_model", {"id": "eq.1", "select": "weights,examples"})
    if not rows:
        raise RuntimeError("bazi_model is missing; run bazi_supabase_schema.sql")
    weights = rows[0].get("weights") or {}
    return weights, int(rows[0].get("examples") or 0)


def settled_accuracy():
    """Return overall settled accuracy when enough rows exist.

    This is intentionally simple and only used to shrink displayed confidence
    if the model has recently proven less accurate than its confidence suggests.
    """
    try:
        rows = sb_get(
            "bazi_predictions",
            {
                "resolved_at": "not.is.null",
                "select": "correct",
                "order": "resolved_at.desc",
                "limit": "200",
            },
        )
        vals = [bool(r["correct"]) for r in rows if r.get("correct") is not None]
        if len(vals) < 50:
            return None, len(vals)
        return sum(vals) / len(vals), len(vals)
    except Exception:
        return None, 0


def make_prediction(market, snap, seconds):
    closes = snap["closes"]
    highs = snap["highs"]
    lows = snap["lows"]
    volumes = snap["volumes"]
    rets = snap["rets"]

    vol_short = safe_stdev(rets[-10:], 0.0006)
    vol_long = safe_stdev(rets[-30:], vol_short)
    vol = clamp(0.65 * vol_short + 0.35 * vol_long, 0.00015, 0.0045)

    raw_m1 = horizon_return(closes, 1)
    raw_m3 = horizon_return(closes, 3)
    raw_m5 = horizon_return(closes, 5)
    raw_m10 = horizon_return(closes, 10)
    raw_m15 = horizon_return(closes, 15)

    mom1 = clamp(raw_m1 / max(vol, 1e-9), -3, 3)
    mom3 = clamp(raw_m3 / max(vol * math.sqrt(3), 1e-9), -3, 3)
    mom5 = clamp(raw_m5 / max(vol * math.sqrt(5), 1e-9), -3, 3)
    mom10 = clamp(raw_m10 / max(vol * math.sqrt(10), 1e-9), -3, 3)
    mom15 = clamp(raw_m15 / max(vol * math.sqrt(15), 1e-9), -3, 3)

    accel = clamp(mom3 - mom10, -3, 3)
    vol_ratio = clamp(vol_short / max(vol_long, 1e-9) - 1.0, -1.5, 2.0)

    ema_fast = ema(closes[-20:], 5)
    ema_slow = ema(closes[-40:], 15)
    ema_gap = clamp(
        (ema_fast - ema_slow) / max(snap["cb"] * vol, 1e-9), -3, 3
    )

    recent_volumes = volumes[-20:]
    weighted_px = [
        (h + l + c) / 3.0
        for h, l, c in zip(highs[-20:], lows[-20:], closes[-20:])
    ]
    denom = sum(recent_volumes)
    vwap = (
        sum(p * v for p, v in zip(weighted_px, recent_volumes)) / denom
        if denom > 0
        else closes[-1]
    )
    vwap_gap = clamp(
        (snap["cb"] - vwap) / max(snap["cb"] * vol, 1e-9), -3, 3
    )

    lo15 = min(lows[-15:]) if lows else snap["cb"]
    hi15 = max(highs[-15:]) if highs else snap["cb"]
    range_pos = clamp(
        2.0 * ((snap["cb"] - lo15) / max(hi15 - lo15, 1e-9)) - 1.0, -1, 1
    )

    volume_z = clamp(zscore(volumes[-1], volumes[-21:-1]), -3, 3)

    kind, strike_info = classify_market(market)
    kp = yes_probability(market)

    # Expected remaining 1-sigma move, scaled from one-minute realized volatility.
    sigma = max(
        snap["cb"] * vol * math.sqrt(max(seconds, 1) / 60.0),
        snap["cb"] * 0.00008,
    )

    # target/gap_vol are always expressed in "favors YES" terms so the
    # ensemble and learned model don't need to know the contract kind.
    if kind == "above":
        target = strike_info
        gap_vol = clamp((snap["cb"] - target) / sigma, -5, 5)
    elif kind == "below":
        target = strike_info
        gap_vol = clamp((target - snap["cb"]) / sigma, -5, 5)
    elif kind == "range":
        floor, cap_strike = strike_info
        target = (floor + cap_strike) / 2.0
        gap_vol = clamp(
            min((cap_strike - snap["cb"]) / sigma, (snap["cb"] - floor) / sigma),
            -5,
            5,
        )
    else:
        target = strike_info  # text-parsed price, direction unclear
        gap_vol = 0.0

    if snap["kraken"] is None:
        kraken_basis = 0.0
        kraken_agree = 0.0
    else:
        kraken_basis = clamp(
            (snap["kraken"] - snap["cb"]) / max(snap["cb"] * vol, 1e-9),
            -3,
            3,
        )
        trend_sign = raw_m3 if abs(raw_m3) > 1e-12 else raw_m1
        kraken_agree = 1.0 if trend_sign * (snap["kraken"] - snap["cb"]) >= 0 else -1.0

    feats = {
        "gap_vol": gap_vol,
        "mom_1": mom1,
        "mom_3": mom3,
        "mom_5": mom5,
        "mom_10": mom10,
        "mom_15": mom15,
        "accel": accel,
        "vol_ratio": vol_ratio,
        "book": snap["book"],
        "spread_bps": clamp(snap["spread_bps"], 0, 25),
        "ema_gap": ema_gap,
        "vwap_gap": vwap_gap,
        "range_pos": range_pos,
        "volume_z": volume_z,
        "kraken_agree": kraken_agree,
        "kraken_basis": kraken_basis,
        "kalshi_market": 0.0 if kp is None else 2.0 * kp - 1.0,
        "time_frac": clamp(seconds / 900.0, 0, 1),
        "contract_kind": kind,
    }

    # Component 1: contract geometry. This dominates when the strike direction is known.
    if kind in ("above", "below", "range"):
        # Small drift adjustment; deliberately capped to avoid chasing noise.
        # Sign flips for "below" contracts: upward momentum works against YES.
        drift_sign = -1.0 if kind == "below" else 1.0
        drift_z = drift_sign * clamp(
            0.25 * mom3 + 0.12 * mom10 + 0.10 * ema_gap + 0.08 * snap["book"],
            -0.8,
            0.8,
        )
        if kind == "range":
            floor, cap_strike = strike_info
            z_hi = clamp((cap_strike - snap["cb"]) / sigma - drift_z, -5, 5)
            z_lo = clamp((floor - snap["cb"]) / sigma - drift_z, -5, 5)
            p_target = clamp(normal_cdf(z_hi) - normal_cdf(z_lo), 0.0, 1.0)
        else:
            p_target = normal_cdf(gap_vol + drift_z)
    else:
        # Unknown strike direction: stay neutral rather than guess a side.
        p_target = 0.5

    # Component 2: continuation/trend.
    trend_score = (
        0.30 * mom3
        + 0.18 * mom5
        + 0.10 * mom10
        + 0.13 * ema_gap
        + 0.08 * vwap_gap
        + 0.08 * snap["book"]
        + 0.05 * kraken_basis
        + 0.04 * accel
    )
    p_trend = sigmoid(trend_score)

    # Component 3: controlled mean reversion. More weight only in quiet regimes.
    quiet = clamp(1.0 - max(0.0, vol_ratio), 0.0, 1.0)
    revert_score = -0.22 * vwap_gap - 0.10 * range_pos
    p_revert = sigmoid(revert_score)

    # Base ensemble. Kalshi is a confirmation input, never the primary price feed.
    if target is not None:
        base = 0.68 * p_target + 0.26 * p_trend + 0.06 * quiet * p_revert
        norm = 0.68 + 0.26 + 0.06 * quiet
        base /= norm
    else:
        base = 0.82 * p_trend + 0.18 * quiet * p_revert
        norm = 0.82 + 0.18 * quiet
        base /= norm

    if kp is not None:
        # Less market leakage than V11: use Kalshi only as a small ensemble vote.
        base = 0.90 * base + 0.10 * kp

    weights, n = load_model()
    learned_score = weights.get("bias", 0.0)
    for f in FEATURES:
        learned_score += (
            weights.get(f, 0.0)
            * float(feats.get(f, 0.0))
            / SCALES[f]
        )
    learned = sigmoid(learned_score)

    # Tiny samples are noisy. Do not let 30-50 observations dominate the system.
    if n < 50:
        learned_blend = 0.0
    elif n < 100:
        learned_blend = 0.05
    elif n < 250:
        learned_blend = 0.10 + 0.10 * (n - 100) / 150.0
    else:
        learned_blend = min(0.38, 0.20 + 0.18 * (n - 250) / 1000.0)

    base = (1.0 - learned_blend) * base + learned_blend * learned

    # Data-quality and spread penalties.
    cap = 0.94
    if target is None:
        cap = min(cap, 0.80)
    if snap["kraken"] is None:
        cap = min(cap, 0.86)
    if snap["spread_bps"] > MAX_TRADEABLE_SPREAD_BPS:
        cap = min(cap, 0.84)

    # If realized accuracy is poor, shrink probability toward 50%.
    recent_acc, recent_n = settled_accuracy()
    if recent_acc is not None and recent_n >= 50:
        reliability = clamp((recent_acc - 0.50) / 0.20, 0.35, 1.0)
        base = 0.5 + (base - 0.5) * reliability

    p_up = clamp(base, 1.0 - cap, cap)
    side = "UP" if p_up >= 0.5 else "DOWN"
    confidence = p_up if side == "UP" else 1.0 - p_up

    # Require stronger proof for an actionable pick.
    if (
        confidence >= 0.80
        and kind != "unknown"
        and snap["kraken"] is not None
        and snap["spread_bps"] <= MAX_TRADEABLE_SPREAD_BPS
    ):
        decision = "CONFIDENT"
    elif confidence >= 0.64:
        decision = "LEAN"
    else:
        decision = "WAIT"

    return side, confidence, decision, feats, target, kp


def already_logged(ticker):
    rows = sb_get(
        "bazi_predictions",
        {
            "market_ticker": f"eq.{ticker}",
            "select": "id",
            "limit": "1",
        },
    )
    return bool(rows)


def _eligible(seconds):
    lo, hi = PRIMARY_WINDOW
    if lo <= seconds <= hi:
        return True
    # Fallback net: only log outside the primary window if the contract has
    # already passed it (a poll gap caused a miss), never jump the gun on a
    # contract that hasn't reached the window yet.
    if FALLBACK_MIN_SECONDS <= seconds < lo:
        return True
    return False


def discover_and_predict():
    """Capture each open KXBTC15M contract once, anchored to 8-12 minutes
    remaining when possible so predictions land at a consistent point in the
    contract's life. Falls back to logging immediately if that window was
    missed due to a poll gap, so no contract goes unpredicted."""
    data = get(
        f"{KALSHI}/markets",
        {"series_ticker": SERIES, "status": "open", "limit": 100},
    )
    now = datetime.now(timezone.utc)
    candidates = []

    for market in data.get("markets", []):
        expiry = market_expiry(market)
        if not expiry:
            continue
        seconds = int((expiry - now).total_seconds())
        if seconds > FALLBACK_MAX_SECONDS:
            continue
        if _eligible(seconds):
            candidates.append((seconds, market, expiry))

    candidates.sort(key=lambda item: item[0])

    if not candidates:
        print("No KXBTC15M contract currently in the 1–15 minute capture window.")
        return

    pending = [
        item
        for item in candidates
        if not already_logged(item[1].get("ticker", ""))
    ]
    if not pending:
        print("Eligible KXBTC15M contract(s) already logged.")
        return

    snap = market_snapshot()

    for seconds, market, expiry in pending:
        ticker = market.get("ticker")
        if not ticker:
            continue

        side, p, decision, features, target, kp = make_prediction(
            market, snap, seconds
        )
        payload = {
            "market_ticker": ticker,
            "series_ticker": SERIES,
            "expiry_at": expiry.isoformat(),
            "target": target,
            "coinbase_now": snap["cb"],
            "kraken_now": snap["kraken"],
            "kalshi_yes_probability": kp,
            "predicted_side": side,
            "probability": round(p, 6),
            "decision": decision,
            "features": features,
            "model_version": "v12",
        }
        created = sb_post("bazi_predictions", payload)
        print(
            ("Logged" if created else "Already logged"),
            ticker,
            decision,
            side,
            f"{p:.1%}",
            f"{seconds}s_to_expiry",
        )


def train(row, outcome_up):
    weights, n = load_model()
    x = row.get("features") or {}

    score = weights.get("bias", 0.0)
    for f in FEATURES:
        score += (
            weights.get(f, 0.0)
            * float(x.get(f, 0.0))
            / SCALES[f]
        )

    error = int(outcome_up) - sigmoid(score)

    # Slower than V11 at the beginning: protects against overfitting.
    lr = max(0.008, 0.045 / math.sqrt(1.0 + n / 40.0))
    l2 = 0.003

    weights["bias"] = clamp(weights.get("bias", 0.0) + lr * error, -3, 3)

    for f in FEATURES:
        value = float(x.get(f, 0.0)) / SCALES[f]
        old = weights.get(f, 0.0)
        weights[f] = clamp(old * (1.0 - lr * l2) + lr * error * value, -3, 3)

    sb_patch(
        "bazi_model",
        {"id": "eq.1"},
        {
            "weights": weights,
            "examples": n + 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def resolve_finished():
    now = datetime.now(timezone.utc).isoformat()
    rows = sb_get(
        "bazi_predictions",
        {
            "resolved_at": "is.null",
            "expiry_at": f"lt.{now}",
            "select": "id,market_ticker,predicted_side,features",
        },
    )

    for row in rows:
        try:
            market = get(f"{KALSHI}/markets/{row['market_ticker']}").get("market", {})
            result = str(market.get("result", "")).lower()
            status = str(market.get("status", "")).lower()

            if result not in ("yes", "no"):
                print("Awaiting official result", row["market_ticker"], status)
                continue

            outcome = "UP" if result == "yes" else "DOWN"
            changed = sb_patch(
                "bazi_predictions",
                {"id": f"eq.{row['id']}", "resolved_at": "is.null"},
                {
                    "outcome_side": outcome,
                    "correct": outcome == row["predicted_side"],
                    "resolution_source": "KALSHI_OFFICIAL_RESULT",
                    "resolved_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            if changed:
                train(row, outcome == "UP")
                print("Resolved and learned", row["market_ticker"], outcome)
        except Exception as exc:
            print("Resolution retry later", row["market_ticker"], exc)


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    failures = 0
    for pass_no in range(1, PASSES + 1):
        print(
            f"BAZI V12 pass {pass_no}/{PASSES} "
            f"at {datetime.now(timezone.utc).isoformat()}"
        )
        try:
            resolve_finished()
            discover_and_predict()
        except Exception as exc:
            failures += 1
            print(f"Pass {pass_no} failed: {exc}", file=sys.stderr)

        if pass_no < PASSES:
            time.sleep(PASS_INTERVAL)

    if failures == PASSES:
        raise RuntimeError("All BAZI V12 passes failed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"BAZI V12 worker failed: {exc}", file=sys.stderr)
        raise
