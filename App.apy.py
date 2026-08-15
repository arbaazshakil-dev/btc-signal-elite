
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG
# -----------------------------
KRAKEN = "https://api.kraken.com/0/public"

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
    page_title="BAZI BTC V6 Signal",
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
st.title("₿ BAZI BTC V6 Signal")
st.caption(
    "8-minute research model with validation, similar-setup analysis, and 3-level WAIT / LEAN / CONFIDENT filtering"
)

with st.sidebar:
    pair = st.text_input(
        "Kraken pair",
        "XBTUSD",
    ).upper()

    horizon = st.slider(
        "Prediction window (minutes)",
        1,
        30,
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
    df = get_ohlc(pair)
    x = add_features(df)

    model = train_final_model(
        x,
        horizon,
    )

    price = get_ticker(pair)

    book_imbalance, bid_depth, ask_depth = (
        get_book(pair)
    )

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

    direction = (
        "UP"
        if final_p_up >= 0.5
        else "DOWN"
    )

    confidence = max(
        final_p_up,
        1 - final_p_up,
    )

    # Three-level decision system.
    # A confidence number alone is not enough for a strong pick; BAZI also checks
    # whether short-term momentum and the live order book point the same way.
    momentum_up = regime["ema_gap"] > 0 and regime["r5"] > 0
    momentum_down = regime["ema_gap"] < 0 and regime["r5"] < 0
    momentum_agrees = momentum_up if direction == "UP" else momentum_down
    book_agrees = (book_imbalance > 0.01) if direction == "UP" else (book_imbalance < -0.01)

    similar_agrees = False
    similar_conf = 0.50
    if similar is not None and similar.get("n", 0) >= 30:
        similar_direction = "UP" if similar["p_up"] >= 0.5 else "DOWN"
        similar_conf = max(similar["p_up"], 1 - similar["p_up"])
        similar_agrees = similar_direction == direction

    # V6.2: a green/red CONFIDENT PICK is reserved for genuinely high model
    # confidence. Supporting signals can confirm a 75%+ prediction, but they
    # can no longer promote a 60-66% lean into a confident pick.
    validation_ok = validation is not None and validation["accuracy"] >= 0.55

    strong_confirm = (
        confidence >= 0.75
        and not regime["choppy"]
        and regime["flip_risk"] != "HIGH"
        and momentum_agrees
        and book_agrees
        and similar_agrees
        and similar_conf >= 0.58
        and validation_ok
    )

    if confidence < min_confidence or (regime["choppy"] and confidence < 0.75):
        signal_level = "WAIT"
        status = "WAIT — NO TRADE"
        css = "wait"
        emoji = "🟡"
        confirmation = "WAIT"
    elif strong_confirm:
        signal_level = "CONFIDENT"
        status = f"{direction} — {confidence * 100:.1f}%"
        css = "up" if direction == "UP" else "down"
        emoji = "🟢" if direction == "UP" else "🔴"
        confirmation = "CONFIDENT PICK"
    else:
        signal_level = "LEAN"
        status = f"LEAN {direction} — {confidence * 100:.1f}%"
        css = "wait"
        emoji = "🟠"
        confirmation = "LEAN — NOT A STRONG PICK"

    st.markdown(
        f"""
        <div class="signal {css}">
          <div class="small">CURRENT STATUS</div>
          <div class="call">{emoji} {status}</div>
          <div class="conf">
            {horizon}-minute horizon
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
        f"**Model lean:** "
        f"{'UP' if model_p_up >= 0.5 else 'DOWN'} "
        f"{max(model_p_up, 1-model_p_up)*100:.1f}%"
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
        "Data source: Kraken public REST API"
    )

    st.caption(
        "Signal rules: WAIT below the lean threshold; LEAN for a directional edge without enough confirmation; "
        "CONFIDENT PICK when probability and independent momentum/order-book/history checks agree. "
        "Model confidence and historical hit rates are diagnostics, not guaranteed probabilities of profit."
    )

except Exception as e:
    st.error(
        f"Market data/model error: {e}"
    )

time.sleep(refresh_seconds)
st.rerun()
