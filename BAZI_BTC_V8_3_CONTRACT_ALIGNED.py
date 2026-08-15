
import time
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import requests
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG
# -----------------------------
KRAKEN = "https://api.kraken.com/0/public"
COINBASE = "https://api.exchange.coinbase.com"

FEATURES = [
    "r1",
    "r3",
    "r5",
    "ema_gap",
    "rsi",
    "vol_z",
    "range_pct",
    "body_pct",
]

st.set_page_config(
    page_title="BAZI BTC V8.3 Contract-Aligned",
    page_icon="₿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {padding: 1rem .8rem 2rem; max-width: 760px;}
    h1 {font-size: 1.8rem;}
    .signal {
        border-radius: 22px;
        padding: 22px;
        text-align: center;
        margin: 8px 0 14px;
    }
    .up {background:#0d2b1b; border:2px solid #35c46a;}
    .down {background:#321414; border:2px solid #ef6262;}
    .wait {background:#2d2610; border:2px solid #d4a72c;}
    .call {font-size:2.15rem; font-weight:800; margin:0;}
    .conf {font-size:1.15rem; font-weight:700; margin-top:6px;}
    .small {opacity:.78; font-size:.88rem;}
    .card {
        padding:14px;
        border-radius:16px;
        background:#17243b;
        margin:8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# API
# -----------------------------
def api(endpoint, params=None):
    r = requests.get(
        f"{KRAKEN}/{endpoint}",
        params=params or {},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()

    if data.get("error"):
        raise RuntimeError(str(data["error"]))

    return data["result"]


@st.cache_data(ttl=10)
def get_ohlc(pair="XBTUSD"):
    result = api("OHLC", {"pair": pair, "interval": 1})
    key = [k for k in result if k != "last"][0]

    df = pd.DataFrame(
        result[key],
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "count",
        ],
    )

    for c in ["open", "high", "low", "close", "vwap", "volume", "count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    return df.dropna().reset_index(drop=True)


@st.cache_data(ttl=5)
def get_ticker(pair="XBTUSD"):
    result = api("Ticker", {"pair": pair})
    key = list(result.keys())[0]
    return float(result[key]["c"][0])


@st.cache_data(ttl=5)
def get_book(pair="XBTUSD"):
    result = api("Depth", {"pair": pair, "count": 100})
    key = list(result.keys())[0]
    book = result[key]

    bid_notional = sum(float(x[0]) * float(x[1]) for x in book["bids"])
    ask_notional = sum(float(x[0]) * float(x[1]) for x in book["asks"])

    total = bid_notional + ask_notional
    imbalance = (bid_notional - ask_notional) / total if total else 0.0

    return imbalance, bid_notional, ask_notional



# -----------------------------
# COINBASE PRIMARY FEED
# Public market-data endpoints; no API key required.
# -----------------------------
def coinbase_api(path, params=None):
    headers = {"User-Agent": "BAZI-BTC/7.0", "Accept": "application/json"}
    r = requests.get(
        f"{COINBASE}{path}",
        params=params or {},
        headers=headers,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=10)
def get_coinbase_ohlc(product_id="BTC-USD", batches=3):
    """
    Pull ~15 hours of 1-minute Coinbase candles in <=300-candle batches.
    Coinbase returns [time, low, high, open, close, volume], newest first.
    """
    end = datetime.now(timezone.utc)
    frames = []
    for i in range(batches):
        batch_end = end - timedelta(minutes=300 * i)
        batch_start = batch_end - timedelta(minutes=299)
        rows = coinbase_api(
            f"/products/{product_id}/candles",
            {
                "granularity": 60,
                "start": batch_start.isoformat(),
                "end": batch_end.isoformat(),
            },
        )
        if not rows:
            continue
        df = pd.DataFrame(
            rows,
            columns=["time", "low", "high", "open", "close", "volume"],
        )
        frames.append(df)

    if not frames:
        raise RuntimeError("Coinbase returned no candle data.")

    df = pd.concat(frames, ignore_index=True)
    for c in ["low", "high", "open", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = (
        df.sort_values("time")
        .drop_duplicates("time")
        .dropna()
        .reset_index(drop=True)
    )
    # Keep old feature schema compatible.
    df["vwap"] = df["close"]
    df["count"] = 0.0
    return df[["time", "open", "high", "low", "close", "vwap", "volume", "count"]]


@st.cache_data(ttl=5)
def get_coinbase_ticker(product_id="BTC-USD"):
    data = coinbase_api(f"/products/{product_id}/ticker")
    return float(data["price"])


@st.cache_data(ttl=5)
def get_coinbase_book(product_id="BTC-USD"):
    book = coinbase_api(f"/products/{product_id}/book", {"level": 2})
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    bid_notional = sum(float(x[0]) * float(x[1]) for x in bids)
    ask_notional = sum(float(x[0]) * float(x[1]) for x in asks)
    total = bid_notional + ask_notional
    imbalance = (bid_notional - ask_notional) / total if total else 0.0
    return imbalance, bid_notional, ask_notional


def kraken_confirmation(pair="XBTUSD"):
    """Secondary exchange confirmation only. Failure does not stop BAZI."""
    try:
        kdf = add_features(get_ohlc(pair))
        kimb, _, _ = get_book(pair)
        ema = float(kdf["ema_gap"].iloc[-1])
        r5 = float(kdf["r5"].iloc[-1])
        votes = [
            1 if ema > 0 else -1,
            1 if r5 > 0 else -1,
            1 if kimb > 0 else -1,
        ]
        score = sum(votes)
        return ("UP" if score > 0 else "DOWN"), float(kimb)
    except Exception:
        return None, None


def current_15m_expiry(tz_name="America/Los_Angeles"):
    """
    Return the next quarter-hour boundary and seconds remaining.
    Example: 11:07 -> expiry 11:15, ~8 minutes remaining.
    """
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    minute = now.minute
    next_quarter = ((minute // 15) + 1) * 15

    if next_quarter >= 60:
        expiry = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        expiry = now.replace(minute=next_quarter, second=0, microsecond=0)

    seconds_remaining = max(0.0, (expiry - now).total_seconds())
    return seconds_remaining, expiry




def train_expiry_return_model(x, horizon):
    """
    Forecast the actual forward BTC return at expiry rather than merely
    classifying current price as above/below a target.
    """
    h = max(1, int(horizon))
    future_return = x["close"].shift(-h) / x["close"] - 1.0
    mask = x[FEATURES].notna().all(axis=1) & future_return.notna()
    if int(mask.sum()) < 250:
        raise RuntimeError("Not enough Coinbase candles for expiry forecast.")

    model = Pipeline([
        ("scale", StandardScaler()),
        ("reg", Ridge(alpha=8.0)),
    ])
    model.fit(x.loc[mask, FEATURES], future_return.loc[mask])

    pred = float(model.predict(x[FEATURES].iloc[[-1]])[0])

    # Residual error gives us a data-driven uncertainty band.
    fitted = model.predict(x.loc[mask, FEATURES])
    residuals = future_return.loc[mask].to_numpy() - fitted
    resid_sigma = float(np.std(residuals[-300:], ddof=1)) if len(residuals) > 30 else 0.001
    resid_sigma = max(resid_sigma, 0.00015)
    return model, pred, resid_sigma


def expiry_forecast(x, current_price, target_price, seconds_remaining, book_imbalance):
    """
    True forecast layer:
    1) predict forward return to expiry from historical Coinbase candles;
    2) make only small live adjustments;
    3) produce a forecast BTC expiry price;
    4) compare THAT future-price distribution with Price to Beat.
    """
    minutes = max(seconds_remaining / 60.0, 1.0 / 60.0)
    h = max(1, min(15, int(round(minutes))))

    _, predicted_return, resid_sigma = train_expiry_return_model(x, h)

    # Live information nudges the forecast; it cannot simply flip the output
    # because spot crossed the target.
    r1 = float(x["r1"].iloc[-1]) if pd.notna(x["r1"].iloc[-1]) else 0.0
    r3 = float(x["r3"].iloc[-1]) if pd.notna(x["r3"].iloc[-1]) else 0.0
    momentum_nudge = np.clip(0.12*r1 + 0.06*(r3/3.0), -0.00035, 0.00035)
    book_nudge = np.clip(book_imbalance, -0.15, 0.15) * 0.00012
    adjusted_return = float(np.clip(predicted_return + momentum_nudge + book_nudge, -0.02, 0.02))

    forecast_price = float(current_price * (1.0 + adjusted_return))

    # Scale residual uncertainty to the fractional horizon.
    sigma_return = max(resid_sigma * math.sqrt(max(minutes, 1.0) / h), 0.00012)
    threshold_return = target_price / current_price - 1.0
    z = (threshold_return - adjusted_return) / sigma_return
    p_above = float(np.clip(0.5 * math.erfc(z / math.sqrt(2.0)), 0.001, 0.999))

    return {
        "p_above": p_above,
        "p_below": 1.0-p_above,
        "forecast_price": forecast_price,
        "forecast_return": adjusted_return,
        "raw_forecast_return": predicted_return,
        "sigma_return": sigma_return,
        "required_move_dollars": target_price-current_price,
        "required_move_pct": threshold_return,
        "minutes_remaining": minutes,
        "forecast_horizon": h,
    }


def target_probability(x, current_price, target_price, seconds_remaining, book_imbalance):
    """
    Estimate probability BTC finishes ABOVE the contract target at expiry.

    Core behavior:
    - Distance to target is explicit.
    - Remaining time shrinks every refresh.
    - Recent realized 1m volatility determines how feasible the required move is.
    - Short momentum/order-book pressure provide only modest drift adjustments.
    """
    mins = max(seconds_remaining / 60.0, 1.0 / 60.0)

    r1 = x["close"].pct_change().dropna()
    recent = r1.tail(120)
    sigma_1m = float(recent.std(ddof=1)) if len(recent) >= 20 else 0.0008
    sigma_1m = max(sigma_1m, 0.00015)

    # Robust short-term drift, deliberately capped.
    drift_1m = float(recent.tail(20).mean()) if len(recent) >= 10 else 0.0
    momentum = float(x["r5"].iloc[-1]) / 5.0 if pd.notna(x["r5"].iloc[-1]) else 0.0
    ema_drift = float(x["ema_gap"].iloc[-1]) / 8.0 if pd.notna(x["ema_gap"].iloc[-1]) else 0.0
    book_drift = float(np.clip(book_imbalance, -0.15, 0.15)) * 0.00045

    mu_1m = 0.35 * drift_1m + 0.35 * momentum + 0.20 * ema_drift + 0.10 * book_drift
    mu_1m = float(np.clip(mu_1m, -0.0015, 0.0015))

    log_gap = math.log(max(target_price, 1e-9) / max(current_price, 1e-9))
    expected_log_return = mu_1m * mins
    total_sigma = sigma_1m * math.sqrt(mins)

    z = (log_gap - expected_log_return) / max(total_sigma, 1e-9)
    # Normal survival function without scipy.
    p_above = 0.5 * math.erfc(z / math.sqrt(2.0))
    p_above = float(np.clip(p_above, 0.001, 0.999))

    return {
        "p_above": p_above,
        "p_below": 1.0 - p_above,
        "sigma_1m": sigma_1m,
        "mu_1m": mu_1m,
        "required_move_dollars": target_price - current_price,
        "required_move_pct": (target_price / current_price - 1.0) if current_price else 0.0,
        "minutes_remaining": mins,
    }


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def add_features(df):
    x = df.copy()

    x["r1"] = x["close"].pct_change(1)
    x["r3"] = x["close"].pct_change(3)
    x["r5"] = x["close"].pct_change(5)

    ema5 = x["close"].ewm(span=5, adjust=False).mean()
    ema20 = x["close"].ewm(span=20, adjust=False).mean()
    x["ema_gap"] = ema5 / ema20 - 1

    delta = x["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    x["rsi"] = x["rsi"].fillna(50).clip(0, 100)

    vol_mean = x["volume"].rolling(30).mean()
    vol_std = x["volume"].rolling(30).std()
    x["vol_z"] = (
        (x["volume"] - vol_mean)
        / vol_std.replace(0, np.nan)
    )

    x["range_pct"] = (
        (x["high"] - x["low"])
        / x["close"]
    )

    x["body_pct"] = (
        (x["close"] - x["open"])
        / x["open"]
    )

    return x.replace([np.inf, -np.inf], np.nan)


def make_model():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


# -----------------------------
# MODEL / VALIDATION
# -----------------------------
def make_target(x, horizon):
    future_return = (
        x["close"].shift(-horizon)
        / x["close"]
        - 1
    )
    y = (future_return > 0).astype(int)

    mask = (
        x[FEATURES].notna().all(axis=1)
        & future_return.notna()
    )

    return y, future_return, mask


def train_final_model(x, horizon):
    y, _, mask = make_target(x, horizon)

    if int(mask.sum()) < 250:
        raise RuntimeError(
            "Not enough valid historical candles to train the model."
        )

    model = make_model()
    model.fit(
        x.loc[mask, FEATURES],
        y.loc[mask],
    )

    return model


def walk_forward_validation(
    x,
    horizon,
    min_train=250,
    max_tests=180,
):
    y, future_return, mask = make_target(x, horizon)
    valid_idx = np.where(mask.values)[0]

    if len(valid_idx) < min_train + 40:
        return None

    test_start = max(
        min_train,
        len(valid_idx) - max_tests,
    )

    preds = []
    actuals = []
    probs = []

    for j in range(test_start, len(valid_idx)):
        test_i = valid_idx[j]
        train_positions = valid_idx[:j]

        if len(train_positions) < min_train:
            continue

        model = make_model()

        model.fit(
            x.iloc[train_positions][FEATURES],
            y.iloc[train_positions],
        )

        p_up = float(
            model.predict_proba(
                x.iloc[[test_i]][FEATURES]
            )[0, 1]
        )

        preds.append(int(p_up >= 0.5))
        actuals.append(int(y.iloc[test_i]))
        probs.append(p_up)

    if not preds:
        return None

    accuracy = accuracy_score(actuals, preds)

    return {
        "accuracy": float(accuracy),
        "n": int(len(preds)),
        "avg_confidence": float(
            np.mean(
                np.maximum(
                    probs,
                    1 - np.array(probs),
                )
            )
        ),
    }


def similar_setup_stats(
    x,
    horizon,
    current_row,
    neighbors=80,
):
    y, _, mask = make_target(x, horizon)
    hist = x.loc[mask, FEATURES].copy()

    if len(hist) < 100:
        return None

    means = hist.mean()
    stds = hist.std().replace(0, 1)

    z_hist = (hist - means) / stds
    z_now = (
        current_row.iloc[0] - means
    ) / stds

    dist = np.sqrt(
        ((z_hist - z_now) ** 2).mean(axis=1)
    )

    nearest_idx = dist.nsmallest(
        min(neighbors, len(dist))
    ).index

    outcomes = y.loc[nearest_idx]

    p_up = float(outcomes.mean())

    return {
        "p_up": p_up,
        "n": int(len(outcomes)),
    }


# -----------------------------
# MARKET REGIME
# -----------------------------
def regime_info(x, book_imbalance):
    rsi = float(x["rsi"].iloc[-1])
    ema_gap = float(x["ema_gap"].iloc[-1])
    r5 = float(x["r5"].iloc[-1])
    vol_z = float(
        x["vol_z"].iloc[-1]
        if pd.notna(x["vol_z"].iloc[-1])
        else 0
    )

    choppy = (
        abs(ema_gap) < 0.00025
        and abs(r5) < 0.0007
        and 42 <= rsi <= 58
    )

    if choppy or abs(book_imbalance) < 0.02:
        flip_risk = "HIGH"
    elif (
        abs(book_imbalance) < 0.05
        or abs(ema_gap) < 0.0004
    ):
        flip_risk = "MEDIUM"
    else:
        flip_risk = "LOW"

    return {
        "choppy": choppy,
        "flip_risk": flip_risk,
        "rsi": rsi,
        "ema_gap": ema_gap,
        "r5": r5,
        "vol_z": vol_z,
    }


# -----------------------------
# UI
# -----------------------------
st.title("₿ BAZI BTC V8.3 — CONTRACT ALIGNED")
st.caption(
    "Coinbase forecast + Kalshi/CF settlement-reference safety • auto 15-minute expiry"
)

with st.sidebar:
    product_id = st.text_input(
        "Coinbase product",
        "BTC-USD",
    ).upper()

    pair = st.text_input(
        "Kraken confirmation pair",
        "XBTUSD",
    ).upper()

    st.markdown("### 15m contract")
    target_price = st.number_input(
        "Price to beat",
        min_value=1.0,
        value=63000.0,
        step=1.0,
        format="%.2f",
    )

    settlement_reference = st.number_input(
        "Contract NOW / settlement-reference price",
        min_value=0.0,
        value=0.0,
        step=0.01,
        format="%.2f",
        help=(
            "Enter the live NOW price shown on the same 15-minute contract. "
            "BAZI will not issue ABOVE/BELOW without it. Coinbase is used only "
            "to estimate the FUTURE MOVE from that contract reference."
        ),
    )

    basis_tolerance = st.number_input(
        "Feed disagreement tolerance ($)",
        min_value=0.50,
        value=20.00,
        step=0.50,
        format="%.2f",
        help=(
            "Coinbase and the contract reference can differ because they are different feeds. "
            "A normal basis no longer flips the prediction. This only blocks unusually large discrepancies."
        ),
    )

    horizon = st.slider(
        "Generic model horizon (secondary only)",
        1,
        15,
        8,
    )

    min_confidence = st.slider(
        "Minimum lean confidence",
        0.50,
        0.70,
        0.55,
        0.01,
        help="Below this BAZI shows WAIT. Strong picks use a separate confirmation test."
    )

    refresh_seconds = st.slider(
        "Refresh seconds",
        10,
        60,
        20,
    )

try:
    df = get_coinbase_ohlc(product_id)
    x = add_features(df)

    model = train_final_model(
        x,
        horizon,
    )

    price = get_coinbase_ticker(product_id)

    book_imbalance, bid_depth, ask_depth = (
        get_coinbase_book(product_id)
    )
    kraken_direction, kraken_book_imbalance = kraken_confirmation(pair)
    seconds_remaining, expiry_local = current_15m_expiry()
    target = expiry_forecast(
        x,
        price,
        float(target_price),
        seconds_remaining,
        book_imbalance,
    )

    # CONTRACT-ALIGNED FORECAST:
    # Forecast the MOVE using Coinbase, but anchor that move to the contract's
    # own NOW/reference price. This prevents Coinbase-vs-contract basis from
    # incorrectly turning an ABOVE contract into BELOW (or vice versa).
    settlement_ref_available = float(settlement_reference) > 0.0
    settlement_basis = (
        float(settlement_reference) - price
        if settlement_ref_available else 0.0
    )

    if settlement_ref_available:
        # Coinbase model supplies only the expected return/move.
        # The contract reference supplies the starting price.
        settlement_forecast_price = float(settlement_reference) * (1.0 + target["forecast_return"])

        # Volatility scales from the contract reference too.
        sigma_dollars = max(float(settlement_reference) * target["sigma_return"], 0.01)
        z_settle = (float(target_price) - settlement_forecast_price) / sigma_dollars
        settle_p_above = float(np.clip(
            0.5 * math.erfc(z_settle / math.sqrt(2.0)),
            0.001,
            0.999,
        ))
        target["p_above"] = settle_p_above
        target["p_below"] = 1.0 - settle_p_above
    else:
        settlement_forecast_price = float("nan")
        # No contract reference = no valid contract-side probability.
        target["p_above"] = 0.5
        target["p_below"] = 0.5

    target["settlement_forecast_price"] = settlement_forecast_price
    target["settlement_basis"] = settlement_basis
    target["settlement_ref_available"] = settlement_ref_available

    current_row = x[FEATURES].iloc[[-1]]

    raw_p_up = float(
        model.predict_proba(
            current_row
        )[0, 1]
    )

    # Order-book adjustment is deliberately capped.
    live_adjustment = float(
        np.clip(
            book_imbalance * 0.08,
            -0.08,
            0.08,
        )
    )

    model_p_up = float(
        np.clip(
            raw_p_up + live_adjustment,
            0.01,
            0.99,
        )
    )

    validation = walk_forward_validation(
        x,
        horizon,
    )

    similar = similar_setup_stats(
        x,
        horizon,
        current_row,
    )

    regime = regime_info(
        x,
        book_imbalance,
    )

    # Blend model with nearest historical setups.
    if similar is not None and similar["n"] >= 30:
        final_p_up = (
            0.75 * model_p_up
            + 0.25 * similar["p_up"]
        )
    else:
        final_p_up = model_p_up

    final_p_up = float(
        np.clip(final_p_up, 0.01, 0.99)
    )

    generic_direction = "UP" if final_p_up >= 0.5 else "DOWN"
    generic_confidence = max(final_p_up, 1 - final_p_up)

    # CONTRACT DIRECTION is the primary decision now.
    direction = "ABOVE" if target["p_above"] >= 0.5 else "BELOW"
    confidence = max(target["p_above"], target["p_below"])

    # Three-level decision system.
    # A confidence number alone is not enough for a strong pick; BAZI also checks
    # whether short-term momentum and the live order book point the same way.
    # Confirmation must refer to the TARGET direction, not generic UP/DOWN.
    target_up = direction == "ABOVE"
    momentum_up = regime["ema_gap"] > 0 and regime["r5"] > 0
    momentum_down = regime["ema_gap"] < 0 and regime["r5"] < 0
    momentum_agrees = momentum_up if target_up else momentum_down
    book_agrees = (book_imbalance > 0.01) if target_up else (book_imbalance < -0.01)

    similar_agrees = False
    similar_conf = 0.50
    if similar is not None and similar.get("n", 0) >= 30:
        similar_direction = "UP" if similar["p_up"] >= 0.5 else "DOWN"
        similar_conf = max(similar["p_up"], 1 - similar["p_up"])
        desired_generic = "UP" if target_up else "DOWN"
        similar_agrees = similar_direction == desired_generic

    kraken_agrees = (
        kraken_direction == ("UP" if target_up else "DOWN")
        if kraken_direction else False
    )

    validation_ok = validation is not None and validation["accuracy"] >= 0.55

    confirmations = sum([momentum_agrees, book_agrees, similar_agrees, kraken_agrees])

    # Settlement safety. In the final 90 seconds, Coinbase alone is not enough
    # because the contract settles from a different benchmark.
    late_contract = seconds_remaining <= 90
    settlement_safe = settlement_ref_available

    # A Coinbase/contract basis is expected and must NOT determine ABOVE/BELOW.
    # Only block if the difference is unusually large, suggesting stale/wrong input.
    basis_abs = abs(settlement_basis)
    feed_conflict = settlement_ref_available and basis_abs >= float(basis_tolerance)

    # Standard strong pick outside the final 90 seconds.
    strong_confirm = (
        confidence >= 0.75
        and not regime["choppy"]
        and regime["flip_risk"] != "HIGH"
        and confirmations >= 2
        and validation_ok
        and settlement_safe
        and not feed_conflict
    )

    # Late-contract mode: once the actual settlement reference is supplied,
    # target distance + countdown can dominate generic 8m validation.
    if late_contract and settlement_ref_available and not feed_conflict:
        ref_gap = float(settlement_reference) - float(target_price)
        sigma_dollars = max(price * target["sigma_return"], 0.01)
        reversal_sigma = abs(ref_gap) / sigma_dollars
        late_strong = (
            confidence >= 0.80
            and reversal_sigma >= 1.0
            and confirmations >= 1
        )
        if late_strong:
            strong_confirm = True

    forced_wait_reason = None
    if seconds_remaining <= 0:
        forced_wait_reason = "CONTRACT EXPIRED"
    elif not settlement_ref_available:
        forced_wait_reason = "CONTRACT NOW PRICE REQUIRED"
    elif feed_conflict:
        forced_wait_reason = "PRICE FEED DISAGREEMENT"
    elif confidence < min_confidence:
        forced_wait_reason = "LOW CONFIDENCE"
    elif regime["choppy"] and confidence < 0.75:
        forced_wait_reason = "CHOPPY MARKET"

    if forced_wait_reason:
        signal_level = "WAIT"
        status = f"WAIT — {forced_wait_reason}"
        css = "wait"
        emoji = "🟡"
        confirmation = "WAIT"
    elif strong_confirm:
        signal_level = "CONFIDENT"
        status = f"{direction} TARGET — {confidence * 100:.1f}%"
        css = "up" if direction == "ABOVE" else "down"
        emoji = "🟢" if direction == "ABOVE" else "🔴"
        confirmation = "CONFIDENT PICK"
    else:
        signal_level = "LEAN"
        status = f"LEAN {direction} TARGET — {confidence * 100:.1f}%"
        css = "wait"
        emoji = "🟠"
        confirmation = "LEAN — NOT A STRONG PICK"

    st.markdown(
        f"""
        <div class="signal {css}">
          <div class="small">CURRENT STATUS</div>
          <div class="call">{emoji} {status}</div>
          <div class="conf">
            {int(seconds_remaining//60):02d}:{int(seconds_remaining%60):02d} remaining
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "BTC",
        f"${price:,.2f}",
    )

    c2.metric(
        "Order Book",
        f"{book_imbalance * 100:+.2f}%",
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    st.write("### Confirmation")
    st.write(
        f"**Status:** {confirmation}"
    )
    st.write(
        f"**COINBASE FORECAST AT EXPIRY:** ${target['forecast_price']:,.2f}"
    )
    st.write(
        f"**Forecast move from now:** {target['forecast_return']*100:+.3f}%"
    )
    if settlement_ref_available:
        current_gap = float(settlement_reference) - float(target_price)
        current_side = "ABOVE" if current_gap >= 0 else "BELOW"
        forecast_gap = settlement_forecast_price - float(target_price)
        forecast_side = "ABOVE" if forecast_gap >= 0 else "BELOW"
        st.write(
            f"**CONTRACT POSITION NOW:** {current_side} target by ${abs(current_gap):,.2f}"
        )
        st.write(
            f"**FORECAST AT EXPIRY:** {forecast_side} target by ${abs(forecast_gap):,.2f}"
        )
    if settlement_ref_available:
        st.write(
            f"**Kalshi / settlement reference NOW:** ${float(settlement_reference):,.2f}"
        )
        st.write(
            f"**Coinbase ↔ settlement basis:** {settlement_basis:+.2f}"
        )
        st.write(
            f"**SETTLEMENT-ADJUSTED FORECAST:** ${settlement_forecast_price:,.2f}"
        )
        if feed_conflict:
            st.error(
                f"⚠️ POSSIBLE STALE/WRONG REFERENCE: Coinbase and the contract NOW price "
                f"differ by ${basis_abs:.2f}, beyond your ${float(basis_tolerance):.2f} tolerance. "
                f"Verify the contract NOW value."
            )
    else:
        st.warning(
            "Settlement reference not entered. Coinbase can be used for forecasting, "
            "but BAZI will block confident picks inside the final 90 seconds."
        )
    st.write(
        f"**Contract forecast:** ABOVE {target['p_above']*100:.1f}% • "
        f"BELOW {target['p_below']*100:.1f}%"
    )
    st.write(
        f"**Generic 8m direction (secondary):** "
        f"{generic_direction} {generic_confidence*100:.1f}%"
    )
    st.write(
        f"**Price to beat:** ${float(target_price):,.2f}"
    )
    st.write(
        f"**Distance to target:** {target['required_move_dollars']:+,.2f} "
        f"({target['required_move_pct']*100:+.3f}%)"
    )
    st.write(
        f"**Countdown:** {int(seconds_remaining//60):02d}:{int(seconds_remaining%60):02d}"
    )
    st.write(
        f"**Auto expiry:** {expiry_local.strftime('%I:%M:%S %p')} local"
    )
    st.write(
        f"**Settlement safety:** {'PASS' if settlement_safe and not feed_conflict else 'BLOCKED'}"
    )
    if late_contract:
        st.info(
            "⏱️ Late Contract Mode: final 90 seconds. Settlement-reference price "
            "takes priority over Coinbase spot for target position."
        )
    st.write(
        f"**Signal level:** {signal_level}"
    )
    st.write(
        f"**Momentum agrees:** {'YES' if momentum_agrees else 'NO'}"
    )
    st.write(
        f"**Order book agrees:** {'YES' if book_agrees else 'NO'}"
    )
    st.write(
        f"**Kraken confirms target side:** "
        f"{'YES' if kraken_agrees else ('NO' if kraken_direction else 'UNAVAILABLE')}"
    )
    st.write(
        f"**Similar setups agree:** {'YES' if similar_agrees else 'NO'}"
    )
    st.write(
        f"**75% confidence gate:** {'PASS' if confidence >= 0.75 else 'FAIL'}"
    )
    st.write(
        f"**Validation gate (≥55%):** {'PASS' if validation_ok else 'FAIL'}"
    )
    st.write(
        f"**Flip risk:** "
        f"{regime['flip_risk']}"
    )
    st.write(
        f"**Market regime:** "
        f"{'CHOPPY' if regime['choppy'] else 'TRENDING / ACTIVE'}"
    )

    if validation is not None:
        st.write(
            f"**Walk-forward hit rate:** "
            f"{validation['accuracy']*100:.1f}% "
            f"over {validation['n']} tests"
        )
    else:
        st.write(
            "**Walk-forward hit rate:** "
            "not enough history"
        )

    if similar is not None:
        hist_direction = (
            "UP"
            if similar["p_up"] >= 0.5
            else "DOWN"
        )
        hist_conf = max(
            similar["p_up"],
            1 - similar["p_up"],
        )

        st.write(
            f"**Similar historical setups:** "
            f"{hist_direction} "
            f"{hist_conf*100:.1f}% "
            f"· n={similar['n']}"
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    st.write("### Target feasibility")
    st.write(
        f"BTC must move **{target['required_move_dollars']:+,.2f}** "
        f"to finish at the target."
    )
    st.write(
        f"Forecast uncertainty: **{target['sigma_return']*100:.3f}%**"
    )
    st.write(
        f"Forecast horizon used: **{target['forecast_horizon']} minute(s)**"
    )
    if seconds_remaining < 180:
        st.info(
            "Late-contract mode: target distance dominates. As the clock approaches zero, "
            "the probability of an unclosed gap collapses automatically."
        )

    st.write("### Market structure")
    st.write(
        f"**Momentum:** "
        f"{'Bullish' if regime['ema_gap'] > 0 else 'Bearish'}"
    )
    st.write(
        f"**RSI:** {regime['rsi']:.1f}"
    )
    st.write(
        f"**Volume Z-score:** "
        f"{regime['vol_z']:+.2f}"
    )
    st.write(
        f"**1-min return:** "
        f"{x['r1'].iloc[-1]*100:+.3f}%"
    )
    st.write(
        f"**5-min return:** "
        f"{x['r5'].iloc[-1]*100:+.3f}%"
    )
    st.write(
        f"**EMA gap:** "
        f"{regime['ema_gap']*100:+.4f}%"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    st.write("### Order-book depth")
    st.write(
        f"Bid depth: **${bid_depth:,.0f}**"
    )
    st.write(
        f"Ask depth: **${ask_depth:,.0f}**"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.line_chart(
        df.tail(180).set_index("time")["close"],
        height=260,
    )

    st.caption(
        "Primary data: Coinbase Exchange public market feed • Kraken used only as secondary confirmation"
    )

    st.caption(
        "Signal rules: WAIT below the lean threshold; LEAN for a directional edge without enough confirmation; "
        "CONFIDENT PICK when probability and independent momentum/order-book/history checks agree. "
        "V8.3 anchors the contract forecast to the contract NOW/reference price. Coinbase contributes forecasted movement and market features, not the starting settlement level."
    )

except Exception as e:
    st.error(
        f"Market data/model error: {e}"
    )

time.sleep(refresh_seconds)
st.rerun()
